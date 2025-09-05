# coding: utf-8
# Fantasy Football Discord Bot — GLOBAL-only cmds, ESPN snapshot+grades, assistant memory (OpenAI Threads).
# NOTE: keep `import logging` directly above logging.basicConfig(...) per user requirement.

import os
import re
import io
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
# Slash commands still work either way; message memory (on_message) needs Message Content intent:
ALLOW_MESSAGE_MEMORY = os.getenv("ALLOW_MESSAGE_MEMORY", "false").lower() == "true"
if ALLOW_MESSAGE_MEMORY:
    INTENTS.message_content = True
else:
    INTENTS.message_content = False

BOT_TOKEN = os.getenv("DISCORD_TOKEN", "REPLACE_ME")

# -------- ESPN config --------
def _getenv_str(name: str) -> Optional[str]:
    """Normalize env strings: strip whitespace; map '', 'none', 'null' -> None."""
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("none", "null"):
        return None
    return v

def _detect_league_id() -> Optional[str]:
    """Find LEAGUE_ID from multiple env names or parse it from a league URL."""
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
    """Pick YEAR; prefer explicit env, else parse seasonId from URL, else default."""
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
                proj = 0.0
                try:
                    stats = getattr(p, "stats", []) or []
                    for s in stats:
                        if getattr(s, "projected_points", None) is not None:
                            proj = float(s.projected_points)
                            break
                except Exception:
                    pass
                players_map[pid] = {"name": getattr(p, "name", f"Player {pid}"),
                                    "team": team_abbrev,
                                    "position": pos,
                                    "projections": proj}
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

        LEAGUE_SNAPSHOT["teams"] = teams_map
        LEAGUE_SNAPSHOT["players"] = players_map
        LEAGUE_SNAPSHOT["draft"] = draft_list
        LEAGUE_SNAPSHOT["meta"]["last_refresh"] = now
        logger.info("League snapshot refreshed at %s", now)
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
    for pos, need in LEAGUE_SNAPSHOT.get("lineup", {}).items():
        if pos == "FLEX":
            continue
        vals = sorted(pos_map.get(pos, []), reverse=True)
        breakdown[pos] = sum(vals[:int(need or 0)])
    if LEAGUE_SNAPSHOT.get("lineup", {}).get("FLEX", 0):
        flex_pool.sort(reverse=True)
        breakdown["FLEX"] = sum(flex_pool[: int(LEAGUE_SNAPSHOT["lineup"]["FLEX"])])
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
    """
    Remembers messages by pushing them into an OpenAI Thread and replies when bot is mentioned.
    Uses an asyncio Queue + Lock to avoid runs being executed against the wrong latest message.
    """
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

# ---------- Tiny HTTP health server (for Render Web Service) ----------
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

# ------ Optional memory over regular messages (requires ALLOW_MESSAGE_MEMORY=true) ------
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

def league_context_for_ai(max_teams: int = 8) -> str:
    """
    Small, neutral context for the assistant: lineup rules + a short view of the league.
    Keeps it generic; no canned answers.
    """
    lineup = ", ".join(f"{k}:{v}" for k, v in LEAGUE_SNAPSHOT.get("lineup", {}).items())
    last = LEAGUE_SNAPSHOT["meta"].get("last_refresh", "never")

    # Build a lightweight top list (if projections exist)
    totals = []
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        total = compute_team_projection(t)
        totals.append((t.get("name", f"Team {tid}"), float(total)))
    totals.sort(key=lambda x: x[1], reverse=True)

    top_lines = []
    if totals and any(v > 0 for _, v in totals):
        for name, v in totals[:max_teams]:
            top_lines.append(f"- {name}: {v:.1f}")
    else:
        for tid, t in list(LEAGUE_SNAPSHOT.get("teams", {}).items())[:max_teams]:
            top_lines.append(f"- {t.get('name', f'Team {tid}')}")

    context = (
        "You are a helpful fantasy football assistant inside a Discord server.\n"
        "You have access to a brief league snapshot to ground your answers.\n\n"
        f"Lineup rules: {lineup}\n"
        f"Last refresh: {last}\n"
        "Teams overview:\n" + "\n".join(top_lines)
    )
    return context

def chunk_text(s: str, limit: int = 1900) -> List[str]:
    """Split a long string into Discord-safe chunks."""
    chunks: List[str] = []
    text = s
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks

async def send_long_followup(interaction: discord.Interaction, content: str, attach_name: Optional[str] = None):
    """Safely send long content via followup; chunk and optionally attach a file."""
    if len(content) <= 1900:
        await interaction.followup.send(content)
        return
    for chunk in chunk_text(content):
        await interaction.followup.send(chunk)
    if attach_name:
        bio = io.BytesIO(content.encode("utf-8"))
        bio.seek(0)
        await interaction.followup.send(file=discord.File(bio, filename=attach_name))

# ---------- Commands ----------
@bot.tree.command(name="sync_cleanup", description="Clear stale guild commands after switching back to global-only.")
@app_commands.default_permissions(manage_guild=True)
async def sync_cleanup(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        await bot.tree.sync(guild=None)  # globals authoritative
        await interaction.followup.send(embed=ok_embed("Sync Cleanup", "Global commands re-synced. Stale guild commands should clear shortly."))
    except Exception as e:
        await interaction.followup.send(embed=err_embed(f"Sync failed: {e}"))

@bot.tree.command(name="ask", description="Ask anything — generic assistant reply with light league context.")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()

    if bot.assistant.enabled:
        try:
            context = league_context_for_ai(max_teams=8)
            await bot.assistant.add_to_thread(context, role="system")
            await bot.assistant.add_to_thread(question, role="user")
            reply = await bot.assistant.get_reply()
            if not isinstance(reply, str) or not reply.strip():
                reply = "I couldn’t generate a reply just now."
            await send_long_followup(interaction, reply.strip())
            return
        except Exception as e:
            logger.exception("Assistant /ask failed: %s", e)
            await interaction.followup.send("Assistant error. Please try again in a moment.")
            return

    missing = []
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    if not OPENAI_ASSISTANT_ID: missing.append("OPENAI_ASSISTANT_ID")
    if not OPENAI_THREAD_ID: missing.append("OPENAI_THREAD_ID")
    msg = (
        "Generic chat is not enabled because the OpenAI Assistant isn’t configured.\n"
        "Set these env vars and redeploy: " + (", ".join(missing) if missing else "(missing configuration)")
    )
    await interaction.followup.send(msg)

@bot.tree.command(name="roster", description="Show a team's roster.")
@app_commands.describe(team="Team name or manager (partial ok)")
async def roster(interaction: discord.Interaction, team: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    match_team_id = None
    tlower = team.lower()
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        if tlower in t.get("name", "").lower() or tlower in t.get("manager", "").lower():
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
    await send_long_followup(interaction, "\n".join(lines))

@bot.tree.command(name="whohas", description="Find which team has a player.")
@app_commands.describe(player="Player name (partial ok)")
async def whohas(interaction: discord.Interaction, player: str):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    pl = player.lower()
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        for pid in t.get("players", []):
            p = LEAGUE_SNAPSHOT["players"].get(pid, {})
            if pl in p.get("name", "").lower():
                await interaction.followup.send(f"**{p.get('name','?')}** is on **{t.get('name','Team')}**.")
                return
    await interaction.followup.send(f"Couldn't find **{player}** in the snapshot.")

# --- Fixed /draftboard: defer + chunk + optional limit ---
@bot.tree.command(name="draftboard", description="Show the draft board with player names and team abbrev.")
@app_commands.describe(limit="Max picks to show (optional)")
async def draftboard(interaction: discord.Interaction, limit: Optional[int] = None):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()
    picks = LEAGUE_SNAPSHOT.get("draft", [])
    if not picks:
        await interaction.followup.send("No draft data available.")
        return
    ordered = sorted(picks, key=lambda x: x.get("pick", 0))
    if limit is not None and isinstance(limit, int) and limit > 0:
        ordered = ordered[:limit]

    lines = ["**Draft Board**"]
    for item in ordered:
        pid = item.get("player_id")
        p = LEAGUE_SNAPSHOT["players"].get(pid, {"name": f"Player {pid}", "team": ""})
        team_abbrev = f" ({p['team']})" if p.get("team") else ""
        lines.append(f"{item.get('pick', '?')}. **{item.get('team_name', 'Team')}** → {p['name']}{team_abbrev}")

    text = "\n".join(lines)
    await send_long_followup(interaction, text, attach_name="draftboard.txt")

# Helpers for projections commentary
def _league_pos_averages() -> Dict[str, float]:
    """Compute league-average projected points by slot."""
    sums: Dict[str, float] = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "K": 0.0, "FLEX": 0.0}
    n = 0
    for _, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        breakdown = compute_team_breakdown(t)
        if breakdown:
            n += 1
            for k in sums:
                sums[k] += float(breakdown.get(k, 0.0))
    if n == 0:
        return sums
    return {k: v / n for k, v in sums.items()}

def _strength_line(name: str, breakdown: Dict[str, float], avg: Dict[str, float]) -> str:
    strong = [pos for pos, val in breakdown.items() if avg.get(pos, 0.0) > 0 and val >= avg[pos] * 1.15 and val > 0]
    weak = [pos for pos, val in breakdown.items() if avg.get(pos, 0.0) > 0 and val <= avg[pos] * 0.85]
    parts = []
    if strong:
        parts.append("strong at " + "/".join(strong))
    if weak:
        parts.append("light at " + "/".join(weak))
    if not parts:
        return f"{name}: balanced roster."
    return f"{name}: " + "; ".join(parts) + "."

# --- Upgraded /projections with commentary ---
@bot.tree.command(name="projections", description="Show team projections league-wide (with commentary).")
async def projections(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await refresh_snapshot()

    totals = []
    breakdowns: Dict[str, Dict[str, float]] = {}
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        b = compute_team_breakdown(t)
        breakdowns[t.get("name", f"Team {tid}")] = b
        totals.append((t.get("name", f"Team {tid}"), sum(b.values())))
    if not totals:
        await interaction.followup.send("No data available. Check your league visibility and LEAGUE_ID/LEAGUE_URL.")
        return

    totals.sort(key=lambda x: x[1], reverse=True)
    avg = _league_pos_averages()

    top8 = totals[:8]
    lines = ["**Projected Starting Lineup Totals**", *(f"- {name}: {v:.1f}" for name, v in top8)]
    lines.append("")
    lines.append("**Quick Commentary**")

    for name, _v in totals[:3]:
        bl = breakdowns.get(name, {})
        lines.append(f"- {_strength_line(name, bl, avg)}")

    # If assistant is set up, add a short fun blurb
    if bot.assistant.enabled:
        try:
            context = (
                "Create a short, punchy blurb (2–3 sentences) reacting to these projections. "
                "Keep it light and avoid inventing specific past matchups. Focus on positional strengths.\n\n"
                "Top teams and totals:\n" + "\n".join(f"{n}: {v:.1f}" for n, v in totals[:5])
            )
            await bot.assistant.add_to_thread(context, role="system")
            await bot.assistant.add_to_thread("Give me a fun but grounded summary.", role="user")
            blurb = await bot.assistant.get_reply()
            lines.append("")
            lines.append(blurb.strip())
        except Exception as e:
            logger.info("Assistant commentary failed: %s", e)

    lines.append(f"\nSnapshot: {LEAGUE_SNAPSHOT['meta'].get('last_refresh','never')}")
    await send_long_followup(interaction, "\n".join(lines))

# /recap and alias /weekly_report (placeholder stubs; can wire box score parsing later)
@bot.tree.command(name="recap", description="Alias of /weekly_report.")
async def recap(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send("Weekly recap will be implemented with box score parsing.")

@bot.tree.command(name="weekly_report", description="Weekly report.")
async def weekly_report(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await recap.callback(interaction)

# Optional: quick status command for debugging
@bot.tree.command(name="status", description="Show ESPN connection status & snapshot info.")
async def status(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    ready, why = _league_ready()
    lines = [
        f"League ID: {LEAGUE_ID or 'missing'} | Year: {YEAR}",
        f"Snapshot last refresh: {LEAGUE_SNAPSHOT['meta'].get('last_refresh', 'never')}",
        f"Ready: {'yes' if ready else 'no'}{f' — {why}' if not ready else ''}",
    ]
    await interaction.followup.send("\n".join(lines))

# -------- Background: live refresh --------
@tasks.loop(minutes=10)
async def auto_refresh_snapshot():
    """Refresh ESPN data every 10m on game days, every 2h otherwise."""
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

# Provide a default heartbeat too
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

    # Refresh ESPN snapshot immediately
    await refresh_snapshot(force=True)

    # Debug: log which commands are registered
    try:
        commands_list = await bot.tree.fetch_commands()
        logger.info(f"🔎 Slash commands loaded ({len(commands_list)}): {[c.name for c in commands_list]}")
    except Exception as e:
        logger.exception("❌ Failed to fetch commands: %s", e)

if __name__ == "__main__":
    if BOT_TOKEN == "REPLACE_ME":
        logger.warning("DISCORD_TOKEN env var not set. Please set it before running.")
    else:
        # Bind a tiny HTTP server so Render Web Service sees an open port
        _server = start_health_server()
        bot.run(BOT_TOKEN)
