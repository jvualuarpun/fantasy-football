import os, asyncio, datetime, hashlib, json
from typing import Optional, List, Dict, Tuple, Set
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from openai import AsyncOpenAI
from espn_api.football import League
import httpx

# --------------------- Load config ---------------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LEAGUE_ID = int(os.getenv("ESPN_LEAGUE_ID"))
YEAR = int(os.getenv("ESPN_YEAR"))
SWID = os.getenv("ESPN_SWID")
S2 = os.getenv("ESPN_S2")

ANNOUNCE_CHANNEL_ID = int(os.getenv("ANNOUNCE_CHANNEL_ID", "0"))
TZ = ZoneInfo("America/Chicago")

# --------------------- Discord setup -------------------
intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)
ai = AsyncOpenAI(api_key=OPENAI_API_KEY)

announce_channel: Optional[discord.TextChannel] = None

# --------------------- ESPN helpers --------------------
def league() -> League:
    # Fresh object each call to avoid stale data
    return League(league_id=LEAGUE_ID, year=YEAR, swid=SWID, espn_s2=S2)

def find_team(l: League, name: str):
    q = (name or "").strip().lower()
    for t in l.teams:
        if q in t.team_name.lower():
            return t
    return None

def matchup_bullets(scoreboard, week: int) -> str:
    lines = [f"Week {week} matchups:"]
    for gm in scoreboard:
        try:
            home = gm.home_team.team_name
            away = gm.away_team.team_name
        except Exception:
            continue
        hs = getattr(gm, "home_score", 0.0) or 0.0
        as_ = getattr(gm, "away_score", 0.0) or 0.0
        lines.append(f"- {home} ({hs:.2f}) vs {away} ({as_:.2f})")
    return "\n".join(lines)

async def llm_summary(title: str, bullets: str) -> str:
    system = (
        "You are a witty but respectful fantasy football analyst. "
        "Given league bullets, write a concise 5–10 line recap calling out close games, blowouts, and standouts. "
        "No profanity, no personal insults."
    )
    resp = await ai.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"system","content":system},{"role":"user","content":f"{title}\n{bullets}"}],
    )
    out = []
    for item in resp.output:
        for c in getattr(item, "content", []):
            if c.type == "output_text":
                out.append(c.text)
    return ("\n".join(out)).strip() or "No analysis available."

# --------------------- Draft helpers --------------------
def _draft_endpoint() -> str:
    # ESPN endpoint that includes draft details for the league
    return f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}?view=mDraftDetail"

async def fetch_draft_json() -> dict:
    # Use cookies so private leagues work
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.get(_draft_endpoint(), cookies={"swid": SWID, "espn_s2": S2})
        r.raise_for_status()
        return r.json()

def pick_line(p: dict, team_lookup: Dict[int, str]) -> str:
    r = p.get("roundId")
    rp = p.get("roundPickNumber")
    op = p.get("overallPickNumber")
    pid = p.get("playerId")
    t_id = p.get("teamId")
    name = p.get("playerName") or f"Player {pid}"
    team_name = team_lookup.get(t_id, f"Team {t_id}")
    return f"Round {r}, Pick {rp} (#{op}): **{name}** → **{team_name}**"

# --------------------- Projections (kona) helpers --------------------
def _kona_endpoint() -> str:
    # Combine views so we have teams, rosters, players, and settings
    base = f"https://fantasy.espn.com/apis/v3/games/ffl/seasons/{YEAR}/segments/0/leagues/{LEAGUE_ID}"
    views = ["mTeam", "mRoster", "kona_player_info", "mSettings"]
    return base + "?" + "&".join([f"view={v}" for v in views])

async def fetch_kona() -> dict:
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(_kona_endpoint(), cookies={"swid": SWID, "espn_s2": S2})
        r.raise_for_status()
        return r.json()

# Slot category constants used by ESPN
SLOT_QB = 0
SLOT_RB = 2
SLOT_WR = 4
SLOT_TE = 6
SLOT_K  = 17
SLOT_DST = 16
SLOT_FLEX = 23   # RB/WR/TE
SLOT_BENCH = 20
SLOT_IR = 21

FLEX_ELIGIBLE = {SLOT_RB, SLOT_WR, SLOT_TE}

def _extract_player_projections(players: list) -> Dict[int, float]:
    """Return map of playerId -> projected points (best available projection)."""
    proj = {}
    for p in players or []:
        pl = p.get("player") or {}
        pid = pl.get("id")
        best = 0.0
        for st in pl.get("stats", []):
            # statSourceId 1 is projections in ESPN; appliedTotal is fantasy points
            if st.get("statSourceId") == 1:
                val = float(st.get("appliedTotal") or 0.0)
                if val > best:
                    best = val
        if pid is not None:
            proj[pid] = best
    return proj

def _extract_player_meta(players: list):
    names = {}
    elig = {}
    for p in players or []:
        pl = p.get("player") or {}
        pid = pl.get("id")
        nm = (pl.get("fullName") or pl.get("name")) or "Unknown Player"
        slots = set(pl.get("eligibleSlots") or [])
        if pid is not None:
            names[pid] = nm
            elig[pid] = slots
    return names, elig

def _extract_rosters_and_slots(data: dict):
    # Build team rosters as list of playerIds; and slot counts for starting lineup
    teams = data.get("teams") or []
    roster_settings = ((data.get("settings") or {}).get("rosterSettings") or {})
    lineup_counts = {}
    for sc in roster_settings.get("slotCategoryItems", []):
        cid = sc.get("slotCategoryId")
        ct = sc.get("num") or 0
        lineup_counts[cid] = ct
    team_rosters = {}
    for t in teams:
        tid = t.get("id")
        roster = []
        for e in ((t.get("roster") or {}).get("entries") or []):
            pid = (e.get("playerPoolEntry") or {}).get("id")
            if pid is not None:
                roster.append(pid)
        if tid is not None:
            team_rosters[tid] = roster
    team_names = {t.get("id"): (t.get("location","") + " " + t.get("nickname","")).strip() for t in teams}
    return team_rosters, lineup_counts, team_names

def _pick_best_lineup(roster_pids: list, player_proj: Dict[int, float], player_elig: Dict[int, set], lineup_counts: Dict[int, int]):
    """Greedy: fill each starting slot with highest projection eligible player, then flex, then compute bench."""
    remaining = set(roster_pids)
    starters = []
    def take_for(slot_id, count):
        nonlocal remaining, starters
        for _ in range(count):
            candidates = [pid for pid in remaining if slot_id in player_elig.get(pid, set())]
            if not candidates:
                continue
            best = max(candidates, key=lambda pid: player_proj.get(pid, 0.0))
            starters.append(best)
            remaining.remove(best)
    # Primary slots
    for sid in [SLOT_QB, SLOT_RB, SLOT_WR, SLOT_TE, SLOT_DST, SLOT_K]:
        c = lineup_counts.get(sid, 0)
        if c:
            take_for(sid, c)
    # Flex slots (RB/WR/TE)
    cflex = lineup_counts.get(SLOT_FLEX, 0)
    for _ in range(cflex):
        candidates = [pid for pid in remaining if player_elig.get(pid, set()) & FLEX_ELIGIBLE]
        if candidates:
            best = max(candidates, key=lambda pid: player_proj.get(pid, 0.0))
            starters.append(best)
            remaining.remove(best)
    bench = list(remaining)
    return starters, bench

# --------------------- Caches for alerts ----------------
SEEN_ACTIVITY_KEYS: Set[str] = set()
PLAYER_STATUS_CACHE: Dict[int, str] = {}  # playerId -> injuryStatus
POSTED_FINAL_GAMES: Set[str] = set()      # ESPN event id for NFL finals, to dedupe

# --------------------- Slash commands ------------------
@tree.command(name="help", description="Show bot commands")
async def help_cmd(interaction: discord.Interaction):
    msg = (
        "**Fantasy Bot Commands**\n"
        "• `/standings` — current standings\n"
        "• `/weekly_report week:<#>` — AI recap for a week\n"
        "• `/matchup team_name:<text> week:<#>` — show a team’s matchup\n"
        "• `/projections week:<#>` — all matchups with projections\n"
        "• `/team team_name:<text>` — team card (record/PF/PA/rank)\n"
        "• `/injuries [team_name]` — injuries for all or one team\n"
        "• `/power_ranks` — simple power rankings\n"
        "• `/player player_name:<text>` — search rosters + top 256 free agents\n"
        "• `/ask prompt:<text>` — ask ChatGPT anything\n"
        "• `/draftboard` — draft board by round\n"
        "• `/draft_grades` — grades based on projections (starters weighted more)\n"
        "\nAutomatic posts: pre-kickoff weekly report (Sun), NFL finals, trades/waivers, injury changes, and Tuesday recap."
    )
    await interaction.response.send_message(msg, ephemeral=True)

@tree.command(name="standings", description="Show current standings")
async def standings(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        l = league()
        teams = sorted(l.teams, key=lambda t: t.standing)
        lines = [f"{t.standing}. {t.team_name} ({t.wins}-{t.losses}-{t.ties})  PF:{t.points_for:.1f}  PA:{t.points_against:.1f}"
                 for t in teams]
        await interaction.followup.send("**Standings:**\n" + "\n".join(lines))
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="weekly_report", description="AI recap for a week (e.g., /weekly_report 3)")
async def weekly_report_cmd(interaction: discord.Interaction, week: int):
    await interaction.response.defer()
    try:
        l = league()
        sb = l.scoreboard(week=week)
        if not sb:
            return await interaction.followup.send("No games found for that week.")
        bullets = matchup_bullets(sb, week)
        analysis = await llm_summary(f"ESPN League {LEAGUE_ID}", bullets)
        await interaction.followup.send(f"```\n{bullets}\n```\n**AI breakdown:**\n{analysis}")
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="matchup", description="Show a matchup for a team and week")
async def matchup(interaction: discord.Interaction, team_name: str, week: int):
    await interaction.response.defer()
    try:
        l = league()
        sb = l.scoreboard(week=week)
        if not sb:
            return await interaction.followup.send("No games found for that week.")
        q = team_name.lower().strip()
        for gm in sb:
            try:
                home = gm.home_team.team_name
                away = gm.away_team.team_name
            except Exception:
                continue
            if q in (home.lower(), away.lower()):
                hs = getattr(gm, "home_score", 0.0) or 0.0
                as_ = getattr(gm, "away_score", 0.0) or 0.0
                hp = getattr(gm, "home_projected", None)
                ap = getattr(gm, "away_projected", None)
                proj = f" | Proj: {hp:.1f} - {ap:.1f}" if hp is not None and ap is not None else ""
                return await interaction.followup.send(
                    f"Week {week}: **{home} {hs:.2f}** vs **{away} {as_:.2f}**{proj}"
                )
        await interaction.followup.send("Couldn’t find that team this week. Use the exact ESPN team name.")
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="projections", description="All matchups with projections for a week")
async def projections(interaction: discord.Interaction, week: int):
    await interaction.response.defer()
    try:
        l = league()
        sb = l.scoreboard(week=week)
        if not sb:
            return await interaction.followup.send("No games found for that week.")
        lines = [f"**Week {week} Projections**"]
        for gm in sb:
            try:
                home = gm.home_team.team_name
                away = gm.away_team.team_name
            except Exception:
                continue
            hp = getattr(gm, "home_projected", None)
            ap = getattr(gm, "away_projected", None)
            if hp is not None and ap is not None:
                lines.append(f"- {home} {hp:.1f} vs {away} {ap:.1f}")
            else:
                lines.append(f"- {home} vs {away} (no projections available)")
        await interaction.followup.send("\n".join(lines))
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="team", description="Show a team's card (record, PF/PA, rank)")
async def team_card(interaction: discord.Interaction, team_name: str):
    await interaction.response.defer()
    try:
        l = league()
        t = find_team(l, team_name)
        if not t:
            return await interaction.followup.send("Team not found. Use the exact name shown in ESPN.")
        msg = (f"**{t.team_name}**\n"
               f"Rank: {t.standing}\n"
               f"Record: {t.wins}-{t.losses}-{t.ties}\n"
               f"PF: {t.points_for:.1f} | PA: {t.points_against:.1f}")
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="injuries", description="Show injuries (all teams or filter by team)")
async def injuries_cmd(interaction: discord.Interaction, team_name: Optional[str] = None):
    await interaction.response.defer()
    try:
        l = league()
        teams = [find_team(l, team_name)] if team_name else l.teams
        teams = [t for t in teams if t]
        if not teams:
            return await interaction.followup.send("Team not found.")
        lines: List[str] = []
        for t in teams:
            injured = []
            for p in t.roster:
                status = getattr(p, "injuryStatus", None)
                injured_flag = getattr(p, "injured", False)
                detail = getattr(p, "injuryStatusDetail", None)
                if injured_flag or (status and status != "ACTIVE"):
                    pos = getattr(p, "position", "")
                    injured.append(f"{p.name} ({pos}) - {status or 'INJ'}{f' | {detail}' if detail else ''}")
            if injured:
                lines.append(f"**{t.team_name}**:")
                lines.extend([f"- {x}" for x in injured])
        if not lines:
            return await interaction.followup.send("No reported injuries found on current rosters.")
        await interaction.followup.send("\n".join(lines))
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="power_ranks", description="Simple power rankings (wins + points for)")
async def power_ranks(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        l = league()
        rankings = []
        for t in l.teams:
            score = (t.wins * 2.0) + (t.points_for / 100.0)
            rankings.append((score, t))
        rankings.sort(key=lambda x: x[0], reverse=True)
        lines = ["**Power Rankings (2*Wins + PF/100)**"]
        for i, (score, t) in enumerate(rankings, start=1):
            lines.append(f"{i}. {t.team_name} — {score:.2f} (W-L-T {t.wins}-{t.losses}-{t.ties}, PF {t.points_for:.1f})")
        await interaction.followup.send("\n".join(lines))
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="player", description="Look up a player on rosters + free agents (top 256)")
async def player_lookup(interaction: discord.Interaction, player_name: str):
    await interaction.response.defer()
    try:
        l = league()
        q = player_name.lower().strip()
        hits = []

        # 1) Search rosters
        for t in l.teams:
            for p in t.roster:
                if q in p.name.lower():
                    pos = getattr(p, "position", "N/A")
                    pro = getattr(p, "proTeam", "N/A")
                    pts = getattr(p, "total_points", None) or getattr(p, "points", None)
                    status = getattr(p, "injuryStatus", None)
                    injured = getattr(p, "injured", False)
                    info = f"**{p.name}** ({pos}, {pro}) on **{t.team_name}**"
                    if pts is not None:
                        info += f" — Season Pts: {pts}"
                    if injured or (status and status != 'ACTIVE'):
                        info += f" — Injury: {status}"
                    hits.append(info)

        # 2) Search top 256 free agents
        try:
            fas = l.free_agents(size=256)  # across positions
            for p in fas:
                if q in p.name.lower():
                    pos = getattr(p, "position", "N/A")
                    pro = getattr(p, "proTeam", "N/A")
                    pts = getattr(p, "total_points", None) or getattr(p, "points", None)
                    info = f"**{p.name}** ({pos}, {pro}) — Free Agent"
                    if pts is not None:
                        info += f" — Season Pts: {pts}"
                    hits.append(info)
        except Exception:
            pass  # some seasons/league types restrict FA views

        if not hits:
            return await interaction.followup.send("No matches on rosters or top free agents.")
        await interaction.followup.send("\n".join(hits[:20]))
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="ask", description="Ask ChatGPT anything")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    try:
        resp = await ai.responses.create(model="gpt-4o-mini", input=[{"role":"user","content":prompt}])
        out = []
        for item in resp.output:
            for c in getattr(item, "content", []):
                if c.type == "output_text":
                    out.append(c.text)
        text = ("\n".join(out)).strip() or "No response."
        if len(text) > 1800:
            text = text[:1800] + "\n...(truncated)"
        await interaction.followup.send(text)
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="draft_grades", description="Grade teams based on ESPN projections (starters weighted)")
async def draft_grades(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        data = await fetch_kona()
        players = data.get("players") or []
        player_proj = _extract_player_projections(players)
        player_names, player_elig = _extract_player_meta(players)
        team_rosters, lineup_counts, team_names = _extract_rosters_and_slots(data)

        results = []
        for tid, roster in team_rosters.items():
            starters, bench = _pick_best_lineup(roster, player_proj, player_elig, lineup_counts)
            s_total = sum(player_proj.get(pid, 0.0) for pid in starters)
            b_total = sum(player_proj.get(pid, 0.0) for pid in bench)
            score = s_total + 0.3 * b_total
            results.append((score, s_total, b_total, tid, starters, bench))

        if not results:
            return await interaction.followup.send("No roster/projection data available yet. Try after your draft completes.")

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[0][0] or 1.0
        lines = ["**Draft Grades (ESPN projections, starters weighted 1.0, bench 0.3)**"]
        for rank, (score, s_total, b_total, tid, starters, bench) in enumerate(results, start=1):
            pct = (score / top) * 100.0
            grade = "A+" if pct >= 95 else "A" if pct >= 90 else "A-" if pct >= 85 else "B+" if pct >= 80 else "B" if pct >= 75 else "C" if pct >= 65 else "D"
            lines.append(f"{rank}. {team_names.get(tid, f'Team {tid}')}: **{grade}** — Score {score:.1f} (Starters {s_total:.1f} | Bench {b_total:.1f})")
        msg = "\n".join(lines)
        if len(msg) > 1900:
            msg = msg[:1900] + "\n...(truncated)"
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

@tree.command(name="draftboard", description="Show the draft board (by round)")
async def draftboard(interaction: discord.Interaction):
    await interaction.response.defer()
    try:
        data = await fetch_draft_json()
        dd = (data or {}).get("draftDetail") or {}
        picks = dd.get("picks") or []
        if not picks:
            return await interaction.followup.send("No draft data found yet.")
        l = league()
        team_lookup = {t.team_id: t.team_name for t in l.teams}
        picks.sort(key=lambda x: (x.get("roundId", 0), x.get("roundPickNumber", 0)))
        lines = []
        cur_round = None
        for p in picks:
            r = p.get("roundId")
            if r != cur_round:
                cur_round = r
                lines.append(f"\n**Round {r}**")
            lines.append("• " + pick_line(p, team_lookup))
        msg = "\n".join(lines)
        if len(msg) > 1900:
            msg = msg[:1900] + "\n...(truncated)"
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"Error: `{e}`")

# --------------------- Background tasks ----------------
async def post_to_channel(text: str):
    if announce_channel:
        await announce_channel.send(text)

# A) Auto weekly report (every day at 11:00 CT; only runs on Sunday)
@tasks.loop(time=datetime.time(hour=11, minute=0, tzinfo=TZ))
async def weekly_report_auto():
    now = datetime.datetime.now(TZ)
    if now.weekday() != 6:  # Sunday=6
        return
    l = league()
    week = getattr(l, "nfl_week", None) or getattr(l, "current_week", None)
    if not week:
        return
    sb = l.scoreboard(week=week)
    if not sb:
        return
    bullets = matchup_bullets(sb, week)
    analysis = await llm_summary(f"ESPN League {LEAGUE_ID}", bullets)
    await post_to_channel(f"**Pre-kickoff Weekly Report (Week {week})**\n```\n{bullets}\n```\n**AI breakdown:**\n{analysis}")

# B) End-of-week fantasy recap (Tuesday 9:00am CT)
@tasks.loop(time=datetime.time(hour=9, minute=0, tzinfo=TZ))
async def tuesday_recap():
    now = datetime.datetime.now(TZ)
    if now.weekday() != 1:  # Tuesday=1
        return
    l = league()
    week = (getattr(l, "nfl_week", None) or getattr(l, "current_week", None)) - 1
    if not week or week < 1:
        return
    sb = l.scoreboard(week=week)
    if not sb:
        return
    bullets = matchup_bullets(sb, week)
    analysis = await llm_summary(f"ESPN League {LEAGUE_ID}", bullets)
    await post_to_channel(f"**Final Recap (Week {week})**\n```\n{bullets}\n```\n**AI breakdown:**\n{analysis}")

# C) Transactions watcher (trades/waivers/free-agent adds) — every 5 minutes
@tasks.loop(minutes=5)
async def transactions_watch():
    try:
        l = league()
        for msg_type in ("TRADED", "WAIVER", "FA"):
            acts = l.recent_activity(size=25, msg_type=msg_type)
            for a in acts:
                key_src = f"{a.date}:{[(getattr(t,'team_name','?'),act,getattr(p,'name','?'),bid) for (t,act,p,bid) in a.actions]}"
                key = hashlib.sha1(key_src.encode()).hexdigest()
                if key in SEEN_ACTIVITY_KEYS:
                    continue
                SEEN_ACTIVITY_KEYS.add(key)
                lines = []
                for (t, action, p, bid) in a.actions:
                    team = getattr(t, "team_name", "Team")
                    player = getattr(p, "name", "Player")
                    piece = f"{team} {action} {player}"
                    if bid:
                        piece += f" (bid: {bid})"
                    lines.append(f"- {piece}")
                await post_to_channel(f"**{msg_type.title()} update:**\n" + "\n".join(lines))
    except Exception:
        pass  # stay quiet if ESPN throttles or no activity

# D) Injury watcher (on-roster status changes) — hourly
@tasks.loop(hours=1)
async def injury_watch():
    try:
        l = league()
        changes = []
        for t in l.teams:
            for p in t.roster:
                pid = getattr(p, "playerId", None)
                if pid is None:
                    continue
                new = getattr(p, "injuryStatus", None) or "ACTIVE"
                old = PLAYER_STATUS_CACHE.get(pid)
                if old is None:
                    PLAYER_STATUS_CACHE[pid] = new
                elif new != old:
                    PLAYER_STATUS_CACHE[pid] = new
                    pos = getattr(p, "position", "")
                    changes.append(f"- {p.name} ({pos}, {t.team_name}) → **{new}** (was {old})")
        if changes:
            await post_to_channel("**Injury status changes:**\n" + "\n".join(changes[:20]))
    except Exception:
        pass

# E) NFL finals watcher — every 5 minutes on Thu/Sun/Mon
NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def _is_nfl_day(dt: datetime.datetime) -> bool:
    # Thu (3), Sun (6), Mon (0)
    return dt.weekday() in (3, 6, 0)

@tasks.loop(minutes=5)
async def nfl_finals_watch():
    now = datetime.datetime.now(TZ)
    if not _is_nfl_day(now):
        return
    try:
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.get(NFL_SCOREBOARD)
            data = r.json()
        events = data.get("events", [])
        for ev in events:
            eid = ev.get("id")
            comp = (ev.get("competitions") or [{}])[0]
            status = ((comp.get("status") or {}).get("type") or {})
            is_final = bool(status.get("completed"))
            if not (eid and is_final):
                continue
            if eid in POSTED_FINAL_GAMES:
                continue
            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue
            a = teams[0]; b = teams[1]
            a_name = a.get("team", {}).get("displayName", "Team A")
            b_name = b.get("team", {}).get("displayName", "Team B")
            a_score = a.get("score", "0")
            b_score = b.get("score", "0")
            short = status.get("shortDetail", "Final")
            msg = f"**NFL Final:** {a_name} {a_score} — {b_name} {b_score}  ({short})"
            await post_to_channel(msg)
            POSTED_FINAL_GAMES.add(eid)
    except Exception:
        pass

# --------------------- Bot lifecycle -------------------
@bot.event
async def on_ready():
    global announce_channel
    announce_channel = bot.get_channel(ANNOUNCE_CHANNEL_ID)
    print(f"Logged in as {bot.user} | Announce channel: {announce_channel}")

    # Register slash commands
    await tree.sync()

    # Start background tasks
    weekly_report_auto.start()
    tuesday_recap.start()
    transactions_watch.start()
    injury_watch.start()
    nfl_finals_watch.start()

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)

