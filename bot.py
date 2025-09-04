# coding: utf-8
# Fantasy Football Discord Bot — GLOBAL-only cmds, ESPN snapshot+grades, assistant memory (OpenAI Threads).
# NOTE: keep `import logging` directly above logging.basicConfig(...) per user requirement.

import os
import re
import asyncio
import datetime as dt
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
load_dotenv()  # read .env for DISCORD_TOKEN, ESPN creds, OpenAI, toggles

# import logging must remain directly above logging.basicConfig — DO NOT MOVE
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fantasy-bot")

# -------- Discord / Bot config --------
INTENTS = discord.Intents.default()
ALLOW_MESSAGE_MEMORY = os.getenv("ALLOW_MESSAGE_MEMORY", "false").lower() == "true"
INTENTS.message_content = True if ALLOW_MESSAGE_MEMORY else False

BOT_TOKEN = os.getenv("DISCORD_TOKEN", "REPLACE_ME")

# -------- ESPN config (public league-friendly) --------
def _getenv_str(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("none", "null"):
        return None
    return v

def _detect_league_id() -> Optional[str]:
    # Try a few common names, then parse from URL
    for key in ("LEAGUE_ID", "ESPN_LEAGUE_ID", "LEAGUEID"):
        v = _getenv_str(key)
        if v and v.isdigit():
            return v
    url = _getenv_str("LEAGUE_URL") or _getenv_str("ESPN_LEAGUE_URL")
    if url:
        try:
            q = parse_qs(urlparse(url).query)
            lid = q.get("leagueId") or q.get("leagueid")
            if lid:
                lid_val = str(lid[0]).strip()
                if lid_val.isdigit():
                    return lid_val
        except Exception:
            pass
    return None

def _detect_year(default_year: str = "2025") -> int:
    y = _getenv_str("YEAR")
    if y and y.isdigit():
        return int(y)
    url = _getenv_str("LEAGUE_URL") or _getenv_str("ESPN_LEAGUE_URL")
    if url:
        try:
            q = parse_qs(urlparse(url).query)
            sid = q.get("seasonId") or q.get("season") or []
            if sid:
                s = str(sid[0]).strip()
                if s.isdigit():
                    return int(s)
        except Exception:
            pass
    return int(default_year)

LEAGUE_ID = _detect_league_id()
YEAR = _detect_year("2025")

# Public league mode: cookies not needed
ESPN_S2 = None
SWID = None

ESPn_AVAILABLE = True
try:
    from espn_api.football import League  # pip install espn_api
except Exception as e:
    ESPn_AVAILABLE = False
    logger.warning("espn_api unavailable: %s", e)

# -------- OpenAI Assistants/Threads config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
OPENAI_THREAD_ID = os.getenv("OPENAI_THREAD_ID")

OpenAI_AVAILABLE = True
try:
    from openai import AsyncOpenAI   # pip install openai
except Exception as e:
    OpenAI_AVAILABLE = False
    logger.warning("openai client unavailable: %s", e)

# ---------- League Snapshot (fixed lineup counts) ----------
LEAGUE_SNAPSHOT: Dict[str, dict] = {
    "lineup": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "FLEX": 1},
    "meta": {"last_refresh": None, "league_id": LEAGUE_ID, "year": YEAR},
    "teams": {},    # team_id -> {"name": str, "manager": str, "players": [player_ids...]}
    "players": {},  # player_id -> {"name": str, "team": "KC", "position": "RB", "projections": float}
    "draft": [],    # {"pick": int, "team_name": str, "player_id": str}
}

def format_lineup_counts(snapshot: dict) -> str:
    parts = [f"{pos}: {cnt}" for pos, cnt in snapshot.get("lineup", {}).items()]
    return "\n".join(parts)

def _league_ready() -> Tuple[bool, Optional[str]]:
    if not ESPn_AVAILABLE:
        return False, "Python package 'espn_api' is not installed."
    if not LEAGUE_ID:
        return False, "Missing LEAGUE_ID env variable (set LEAGUE_ID=digits or LEAGUE_URL with leagueId=...)."
    return True, None

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "?"

def _norm_slot_to_pos(slot: Optional[str]) -> Optional[str]:
    """Map ESPN slot labels to our fixed positions."""
    if not slot:
        return None
    s = slot.upper()
    if "QB" in s: return "QB"
    if "RB" in s and "WR" not in s: return "RB"
    if "WR" in s and "RB" not in s: return "WR"
    if "TE" in s: return "TE"
    if "K" == s or "PK" in s: return "K"
    # Ignore DST/DEF since lineup spec excludes it
    return None

async def _fill_weekly_projections(lg, players_map: Dict[str, dict]) -> None:
    """Populate projections via box scores (most reliable)."""
    try:
        week = getattr(lg, "current_week", None)
        if not week:
            # Fallback heuristic: try 1 if pre-season, or 17 if late season
            week = 1
        boxscores = lg.box_scores(week=week)
        updated = 0
        for bs in boxscores:
            for side in ("home_lineup", "away_lineup"):
                lineup = getattr(bs, side, []) or []
                for bp in lineup:
                    pid = _safe_str(getattr(bp, "playerId", None))
                    proj = getattr(bp, "projected_points", None)
                    slot = _norm_slot_to_pos(getattr(bp, "slot_position", None))
                    if not pid:
                        continue
                    if pid not in players_map:
                        # Create a minimal entry if not present
                        players_map[pid] = {
                            "name": getattr(getattr(bp, "player", None), "name", f"Player {pid}"),
                            "team": getattr(getattr(bp, "player", None), "proTeamAbbreviation", "") or "",
                            "position": slot or "",
                            "projections": 0.0,
                        }
                    if proj is None:
                        # last resort fallbacks on some installs
                        proj = getattr(bp, "projected_total_points", None)
                    if isinstance(proj, (int, float)):
                        if float(proj) > float(players_map[pid].get("projections") or 0.0):
                            players_map[pid]["projections"] = float(proj)
                            updated += 1
                    # If we learned a better position, keep it
                    if slot and not players_map[pid].get("position"):
                        players_map[pid]["position"] = slot
        logger.info("Filled weekly projections for %s players (week=%s)", updated, week)
    except Exception as e:
        logger.info("Box score projections not available: %s", e)

async def refresh_snapshot(force: bool = False) -> None:
    """Populate LEAGUE_SNAPSHOT from ESPN API."""
    ok, why = _league_ready()
    if not ok:
        logger.warning("League not ready: %s", why)
        return
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        lg = League(league_id=int(LEAGUE_ID), year=YEAR)

        # Teams / players
        teams_map: Dict[str, dict] = {}
        players_map: Dict[str, dict] = {}
        for t in lg.teams:
            team_id = getattr(t, "team_id", None) or getattr(t, "teamId", None) or _safe_str(t)
            team_name = getattr(t, "team_name", "Team")
            manager = getattr(getattr(t, "owner", None), "nickname", None) or getattr(t, "owner", None) or "Manager"
            roster_ids: List[str] = []
            for p in getattr(t, "roster", []):
                pid = _safe_str(getattr(p, "playerId", None) or getattr(p, "id", None) or getattr(p, "name", ""))
                team_abbrev = getattr(p, "proTeam", "") or getattr(p, "proTeamAbbreviation", "") or ""
                pos = getattr(p, "position", "") or (getattr(p, "eligibleSlots", [""])[0] if getattr(p, "eligibleSlots", None) else "")
                # initial projection attempts (will be overwritten by boxscore fill)
                proj = None
                # common fallbacks across espn_api versions
                for attr in ("projected_points", "projected_total_points", "avg_points", "points"):
                    if hasattr(p, attr):
                        val = getattr(p, attr)
                        if isinstance(val, (int, float)):
                            proj = float(val)
                            break
                if proj is None:
                    proj = 0.0
                players_map[pid] = {
                    "name": getattr(p, "name", f"Player {pid}"),
                    "team": team_abbrev,
                    "position": (pos or "").upper(),
                    "projections": proj,
                }
                roster_ids.append(pid)
            teams_map[str(team_id)] = {"name": team_name, "manager": _safe_str(manager), "players": roster_ids}

        # Draft
        draft_list = []
        try:
            if getattr(lg, "draft", None):
                for pick in lg.draft:
                    try:
                        overall = getattr(pick, "overall_pick", None) or getattr(pick, "pick", None)
                        team_name = getattr(getattr(pick, "team", None), "team_name", None) or "Team"
                        pid = _safe_str(getattr(getattr(pick, "player", None), "playerId", None) or getattr(pick, "playerId", None))
                        draft_list.append({"pick": int(overall) if overall else len(draft_list)+1,
                                           "team_name": team_name, "player_id": pid})
                    except Exception:
                        continue
        except Exception as e:
            logger.info("Draft not available: %s", e)

        # Strong projection fill from the current week box scores
        await _fill_weekly_projections(lg, players_map)

        LEAGUE_SNAPSHOT["teams"] = teams_map
        LEAGUE_SNAPSHOT["players"] = players_map
        LEAGUE_SNAPSHOT["draft"] = draft_list
        LEAGUE_SNAPSHOT["meta"]["last_refresh"] = now
        logger.info("League snapshot refreshed at %s (teams=%d, players=%d)", now, len(teams_map), len(players_map))
    except Exception as e:
        logger.exception("Failed to refresh league snapshot: %s", e)

# --------- Grading Logic ---------
def grade_from_zscore(z: float) -> str:
    if z >= 1.25: return "A+"
    if z >= 0.75: return "A"
    if z >= 0.35: return "A-"
    if z >= 0.10: return "B+"
    if z >= -0.10: return "B"
    if z >= -0.35: return "B-"
    if z >= -0.60: return "C+"
    if z >= -0.85: return "C"
    if z >= -1.10: return "C-"
    if z >= -1.35: return "D+"
    if z >= -1.60: return "D"
    return "F"

def compute_team_projection(team: dict) -> float:
    """Sum projections for best starting lineup (based on fixed counts + FLEX from RB/WR/TE)."""
    lineup = LEAGUE_SNAPSHOT.get("lineup", {})
    pid_list = team.get("players", [])
    buckets = {"QB": [], "RB": [], "WR": [], "TE": [], "K": []}
    flex_pool = []
    for pid in pid_list:
        p = LEAGUE_SNAPSHOT["players"].get(pid)
        if not p:
            continue
        pos = (p.get("position") or "").upper()
        proj = float(p.get("projections") or 0.0)
        if pos in buckets:
            buckets[pos].append(proj)
        if pos in ("RB", "WR", "TE"):
            flex_pool.append(proj)
    total = 0.0
    for pos, need in lineup.items():
        if pos == "FLEX":
            continue
        arr = sorted(buckets.get(pos, []), reverse=True)
        total += sum(arr[: int(need or 0)])
    flex_need = int(lineup.get("FLEX", 0) or 0)
    if flex_need > 0 and flex_pool:
        flex_pool.sort(reverse=True)
        total += sum(flex_pool[:flex_need])
    return total

def compute_team_breakdown(team: dict) -> Dict[str, float]:
    """Return projected points by slot (QB, RB, WR, TE, FLEX, K)."""
    lineup = LEAGUE_SNAPSHOT.get("lineup", {})
    pid_list = team.get("players", [])
    pos_map = {"QB": [], "RB": [], "WR": [], "TE": [], "K": []}
    flex_pool = []
    for pid in pid_list:
        p = LEAGUE_SNAPSHOT["players"].get(pid)
        if not p:
            continue
        pos = (p.get("position") or "").upper()
        proj = float(p.get("projections") or 0.0)
        if pos in pos_map:
            pos_map[pos].append(proj)
        if pos in ("RB", "WR", "TE"):
            flex_pool.append(proj)
    breakdown = {}
    for pos, need in lineup.items():
        if pos == "FLEX":
            continue
        vals = sorted(pos_map.get(pos, []), reverse=True)
        breakdown[pos] = sum(vals[:int(need or 0)])
    if lineup.get("FLEX", 0):
        flex_pool.sort(reverse=True)
        breakdown["FLEX"] = sum(flex_pool[: int(lineup["FLEX"])])
    return breakdown

def compute_grades() -> List[Tuple[str, float, float, str]]:
    totals = []
    for team_id, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        totals.append((t.get("name", f"Team {team_id}"), compute_team_projection(t)))
    if not totals:
        return []
    vals = [v for (_, v) in totals]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, (len(vals) - 1))
    sd = var ** 0.5 if var > 0 else 1.0
    out = []
    for name, v in sorted(totals, key=lambda x: x[1], reverse=True):
        z = (v - mean) / sd if sd else 0.0
        out.append((name, v, z, grade_from_zscore(z)))
    return out

# ---------- OpenAI Assistant Memory ----------
class DiscordOpenAIInterface:
    def __init__(self):
        self.enabled = bool(OpenAI_AVAILABLE and OPENAI_API_KEY and OPENAI_ASSISTANT_ID and OPENAI_THREAD_ID)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if self.enabled else None
        self.assistant_id = OPENAI_ASSISTANT_ID
        self.thread_id = OPENAI_THREAD_ID
        self._active_run = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.queue_lock = asyncio.Lock()

    @staticmethod
    def thread_format(message: discord.Message) -> str:
        return f"{message.author.display_name} said: {message.clean_content}"

    async def add_to_thread(self, message_text: str, role: str = "user") -> None:
        if not self.enabled:
            return
        await self.message_queue.put((message_text, role))
        await self._process_queue()

    async def _process_queue(self) -> None:
        if not self.enabled:
            return
        if self.queue_lock.locked():
            return
        while not self.message_queue.empty():
            message_text, role = await self.message_queue.get()
            try:
                await self.client.beta.threads.messages.create(
                    thread_id=self.thread_id, role=role, content=message_text
                )
            except Exception as e:
                logger.error("OpenAI queue error: %s", e)
            finally:
                self.message_queue.task_done()

    async def get_reply(self) -> str:
        if not self.enabled:
            return "Assistant not configured."
        async with self.queue_lock:
            try:
                self._active_run = await self.client.beta.threads.runs.create(
                    thread_id=self.thread_id, assistant_id=self.assistant_id
                )
                while self._active_run.status in ("in_progress", "queued"):
                    self._active_run = await self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread_id, run_id=self._active_run.id
                    )
                    await asyncio.sleep(1)
                if self._active_run.status == "completed":
                    messages = await self.client.beta.threads.messages.list(thread_id=self.thread_id)
                    return messages.data[0].content[0].text.value
                return f"(assistant run failed: {self._active_run.status})"
            except Exception as e:
                logger.exception("OpenAI run failed: %s", e)
                return "(assistant error)"

# ---------- Tiny HTTP health server (Render) ----------
def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/health", "/ping"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, fmt, *args):
            return
    server = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health server bound to 0.0.0.0:%s", port)
    return server

# ---------- Bot ----------
class FantasyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=INTENTS, help_command=None)
        self.synced = False
        self.assistant = DiscordOpenAIInterface()

    async def setup_hook(self) -> None:
        try:
            # Force global sync (guild=None ensures it's global only)
            commands_list = await self.tree.sync(guild=None)
            self.synced = True
            logger.info(f"✅ Global slash commands synced ({len(commands_list)}): {[c.name for c in commands_list]}")
        except Exception as e:
            logger.exception("❌ Failed to sync global commands: %s", e)

bot = FantasyBot()

# ------ Optional message memory ------
@bot.event
async def on_message(message: discord.Message):
    if not ALLOW_MESSAGE_MEMORY:
        return
    if message.author.bot:
        return
    if bot.assistant.enabled:
        await bot.assistant.add_to_thread(bot.assistant.thread_format(message), role="user")
        if (bot.user in message.mentions) and (not message.mention_everyone):
            reply = await bot.assistant.get_reply()
            await message.channel.send(reply)
    await bot.process_commands(message)

# --------------- Utilities -----------------
def ok_embed(title: str, description: str = "") -> discord.Embed:
    embed = discord.Embed(title=title, description=description, color=0x2ecc71)
    embed.set_footer(text="Fantasy Bot")
    return embed

def err_embed(msg: str) -> discord.Embed:
    return discord.Embed(title="Error", description=msg, color=0xe74c3c)

def join_lines(lines: List[str]) -> str:
    return "\n".join(lines)

# ---------- Commands ----------
@bot.tree.command(name="sync_cleanup", description="Clear stale guild commands after switching back to global-only.")
@app_commands.default_permissions(manage_guild=True)
async def sync_cleanup(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await bot.tree.sync(guild=None)  # globals authoritative
        await interaction.followup.send(
            embed=ok_embed("Sync Cleanup", "Global commands re-synced. Stale guild commands should clear shortly."),
            ephemeral=True
        )
    except Exception as e:
        await interaction.followup.send(embed=err_embed(f"Sync failed: {e}"), ephemeral=True)

@bot.tree.command(name="ask", description="Ask something (league-aware).")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()

    # Helper: compute projections league-wide
    totals = []
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        totals.append((t.get("name", f"Team {tid}"), compute_team_projection(t)))
    totals = [(n, float(v)) for (n, v) in totals]
    totals.sort(key=lambda x: x[1], reverse=True)

    # If we have no teams or no projections, tell the user what's up
    if not totals:
        await interaction.followup.send(
            "I don't have league data yet. Make sure **LEAGUE_ID** (or **LEAGUE_URL**) is set and the league is public."
        )
        return

    qlow = question.lower()
    wants_rankings = any(
        k in qlow
        for k in (
            "who is going to win",
            "who's going to win",
            "who will win",
            "prediction",
            "predictions",
            "top 3",
            "top three",
            "power rank",
            "rank the teams",
        )
    )

    # If the question smells like a prediction request, answer directly from projections
    if wants_rankings:
        top_n = 3
        top = totals[:top_n]
        if all(v == 0.0 for (_, v) in top):
            await interaction.followup.send(
                "Projections for this scoring period aren't available yet from ESPN. Try again later, or use **/roster**."
            )
            return

        lines = [f"**Top {len(top)} by projected starting lineup**"]
        for i, (name, pts) in enumerate(top, start=1):
            lines.append(f"{i}. **{name}** — {pts:.1f} projected")
        lines.append(f"Snapshot: {LEAGUE_SNAPSHOT['meta'].get('last_refresh','never')}")
        await interaction.followup.send("\n".join(lines))
        return

    # Otherwise, if Assistant is configured, let it answer with league context
    if bot.assistant.enabled:
        # Compact league summary the assistant can use
        lineup_info = ", ".join(f"{k}:{v}" for k, v in LEAGUE_SNAPSHOT.get("lineup", {}).items())
        top_preview = "\n".join(f"- {name}: {pts:.1f}" for name, pts in totals[:8])
        context = (
            "League snapshot context:\n"
            f"Lineup counts → {lineup_info}\n"
            "Top teams by projected starting lineup:\n"
            f"{top_preview}\n"
            "Answer the user's question using this context. If the question asks for predictions, "
            "reference the projected totals."
        )

        # Push context + user question, then get reply
        await bot.assistant.add_to_thread(context, role="system")
        await bot.assistant.add_to_thread(question, role="user")
        reply = await bot.assistant.get_reply()
        await interaction.followup.send(reply)
        return

    # Fallback if Assistant is not configured
    lineup_info = "\n".join(f"{k}: {v}" for k, v in LEAGUE_SNAPSHOT.get("lineup", {}).items())
    lines = [
        f"**Q:** {question}",
        "**Top 5 by projected starting lineup:**",
        *(f"- {name}: {pts:.1f}" for name, pts in totals[:5]),
        "",
        "**Lineup (fixed counts):**",
        lineup_info,
        f"Snapshot: {LEAGUE_SNAPSHOT['meta'].get('last_refresh','never')}",
        "",
        "_Tip: ask '**who will win**' or '**top 3 predictions**' to get ranked picks._",
    ]
    await interaction.followup.send("\n".join(lines))


@bot.tree.command(name="roster", description="Show a team's roster.")
@app_commands.describe(team="Team name or manager (partial ok)")
async def roster(interaction: discord.Interaction, team: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    match_team_id = None
    tlower = team.lower()
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        if tlower in (t.get("name", "") or "").lower() or tlower in (t.get("manager", "") or "").lower():
            match_team_id = tid
            break
    if not match_team_id:
        await interaction.followup.send(f"Couldn't find a team matching **{team}**.")
        return
    t = LEAGUE_SNAPSHOT["teams"][match_team_id]
    lines = [f"**{t['name']}** (Mgr: {t['manager']})"]
    for pid in t.get("players", []):
        p = LEAGUE_SNAPSHOT["players"].get(pid)
        if not p:
            continue
        lines.append(f"- {p['name']} ({p['position']}, {p['team']}) proj: {p['projections']:.2f}")
    await interaction.followup.send(join_lines(lines))

@bot.tree.command(name="whohas", description="Find which team has a player.")
@app_commands.describe(player="Player name (partial ok)")
async def whohas(interaction: discord.Interaction, player: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    pl = player.lower()
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        for pid in t.get("players", []):
            p = LEAGUE_SNAPSHOT["players"].get(pid, {})
            if pl in (p.get("name", "") or "").lower():
                await interaction.followup.send(f"**{p.get('name','?')}** is on **{t.get('name','Team')}**.")
                return
    await interaction.followup.send(f"Couldn't find **{player}** in the snapshot.")

@bot.tree.command(name="draftboard", description="Show the draft board with player names and team abbrev.")
async def draftboard(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    picks = LEAGUE_SNAPSHOT.get("draft", [])
    if not picks:
        await interaction.followup.send("No draft data available.")
        return
    lines = ["**Draft Board**"]
    for item in sorted(picks, key=lambda x: x.get("pick", 0)):
        pid = item.get("player_id")
        p = LEAGUE_SNAPSHOT["players"].get(pid, {"name": f"Player {pid}", "team": ""})
        team_abbrev = f" ({p['team']})" if p.get("team") else ""
        lines.append(f"{item.get('pick', '?')}. **{item.get('team_name', 'Team')}** → {p['name']}{team_abbrev}")
    await interaction.followup.send(join_lines(lines))

# A forgiving alias for typos: /draftboards
@bot.tree.command(name="draftboards", description="Alias of /draftboard.")
async def draftboards(interaction: discord.Interaction):
    await draftboard.callback(interaction)

@bot.tree.command(name="grades", description="Roster grades with slot breakdown.")
async def grades(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    results = []
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        breakdown = compute_team_breakdown(t)
        total = sum(breakdown.values())
        results.append((t["name"], breakdown, total))
    if not results:
        await interaction.followup.send("No league data available.")
        return
    totals = [r[2] for r in results]
    mean = sum(totals) / len(totals)
    sd = (sum((x - mean) ** 2 for x in totals) / max(1, len(totals) - 1)) ** 0.5
    lines = ["**Roster Grades**"]
    for name, breakdown, total in sorted(results, key=lambda x: x[2], reverse=True):
        z = (total - mean) / sd if sd else 0
        grade = grade_from_zscore(z)
        parts = ", ".join([f"{k}:{v:.1f}" for k, v in breakdown.items()])
        lines.append(f"- **{name}** → {grade} (proj {total:.1f}, z={z:+.2f}) [{parts}]")
    await interaction.followup.send("\n".join(lines))

# Alias for /grades
@app_commands.command(name="draft_grades", description="Alias of /grades.")
async def draft_grades(interaction: discord.Interaction):
    await grades.callback(interaction)
bot.tree.add_command(draft_grades)

@bot.tree.command(name="proj", description="Alias of /projections.")
async def proj(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send("Use **/grades** or **/roster** for projections by team.")

@bot.tree.command(name="projections", description="Show team projections league-wide.")
async def projections(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    totals = []
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        totals.append((t.get("name", f"Team {tid}"), compute_team_projection(t)))
    if not totals:
        await interaction.followup.send("No data available. Check your ESPN configuration or if projections are available yet.")
        return
    totals.sort(key=lambda x: x[1], reverse=True)
    lines = ["**Projected Starting Lineup Totals**"]
    for name, v in totals[:12]:
        lines.append(f"- {name}: {v:.1f}")
    await interaction.followup.send(join_lines(lines))

@bot.tree.command(name="status", description="Show ESPN connection status & snapshot info.")
async def status(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    ready, why = _league_ready()
    lines = [
        f"League ID: {LEAGUE_ID or 'missing'} | Year: {YEAR}",
        f"Snapshot last refresh: {LEAGUE_SNAPSHOT['meta'].get('last_refresh', 'never')}",
        f"Teams cached: {len(LEAGUE_SNAPSHOT.get('teams', {}))} | Players cached: {len(LEAGUE_SNAPSHOT.get('players', {}))}",
        f"Ready: {'yes' if ready else 'no'}{f' — {why}' if not ready else ''}",
    ]
    await interaction.followup.send("\n".join(lines), ephemeral=True)

# -------- Background: live refresh --------
@tasks.loop(minutes=10)
async def auto_refresh_snapshot():
    """Refresh ESPN data every 10m on game days, every 2h otherwise (you chose 10m cadence)."""
    now = dt.datetime.now(dt.timezone.utc)
    dow = now.weekday()  # Monday=0, Sunday=6
    # Tue (1), Wed (2), Fri (4), Sat (5) → only refresh at even hours
    if dow in (1, 2, 4, 5):
        if now.hour % 2 != 0:
            return
    await refresh_snapshot()

@auto_refresh_snapshot.before_loop
async def before_auto_refresh():
    await bot.wait_until_ready()

@tasks.loop(minutes=30)
async def heartbeat():
    logger.info("Heartbeat: bot alive at %s", dt.datetime.now(dt.timezone.utc).isoformat())

@heartbeat.before_loop
async def before_heartbeat():
    await bot.wait_until_ready()

# --------- Startup ---------
@bot.event
async def on_ready():
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")
    logger.info("Config check: LEAGUE_ID=%r YEAR=%s", LEAGUE_ID, YEAR)

    if not heartbeat.is_running():
        heartbeat.start()
    if not auto_refresh_snapshot.is_running():
        auto_refresh_snapshot.start()

    await refresh_snapshot(force=True)

    try:
        commands_list = await bot.tree.fetch_commands()
        logger.info(f"🔎 Slash commands loaded ({len(commands_list)}): {[c.name for c in commands_list]}")
    except Exception as e:
        logger.exception("❌ Failed to fetch commands: %s", e)

if __name__ == "__main__":
    if BOT_TOKEN == "REPLACE_ME":
        logger.warning("DISCORD_TOKEN env var not set. Please set it before running.")
    else:
        _server = start_health_server()  # bind a port for Render Web Service
        bot.run(BOT_TOKEN)
