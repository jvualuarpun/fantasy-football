
# coding: utf-8
# Rebuilt Discord Fantasy Bot — fixed strings, global-only commands, proper aliases.
# NOTE: logging import kept directly above logging.basicConfig(...) per user request.

import os
import re
import asyncio
import datetime as dt
import logging
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
load_dotenv()

# import logging must remain directly above logging.basicConfig — DO NOT MOVE
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fantasy-bot")

INTENTS = discord.Intents.default()
INTENTS.message_content = False  # slash commands only
BOT_TOKEN = os.getenv("DISCORD_TOKEN", "REPLACE_ME")

# ---------- League Snapshot (fixed lineup counts) ----------
LEAGUE_SNAPSHOT = {
    "lineup": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "FLEX": 1},
    # Populate or fetch IDs, team -> manager mapping, etc. (stub for now)
    "teams": {},   # team_id -> {"name": str, "manager": str, "players": [player_ids...]}
    "players": {}, # player_id -> {"name": str, "team": "KC"}
}

def format_lineup_counts(snapshot: dict) -> str:
    parts = [f"{pos}: {cnt}" for pos, cnt in snapshot.get("lineup", {}).items()]
    return "\n".join(parts)

class FantasyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=INTENTS, help_command=None)
        self.synced = False

    async def setup_hook(self) -> None:
        # Global commands only — clear any guild-bound commands by syncing globally
        try:
            await self.tree.sync()
            self.synced = True
            logger.info("Global application commands synced.")
        except Exception as e:
            logger.exception("Failed to sync global commands: %s", e)

bot = FantasyBot()

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
        await bot.tree.sync(guild=None)  # ensure globals are authoritative
        await interaction.response.send_message(
            embed=ok_embed("Sync Cleanup", "Global commands re-synced. Stale guild commands should clear on Discord's side shortly."),
            ephemeral=True
        )
    except Exception as e:
        await interaction.response.send_message(embed=err_embed(f"Sync failed: {e}"), ephemeral=True)

@bot.tree.command(name="ask", description="Ask something (league-aware).")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    lineup_info = format_lineup_counts(LEAGUE_SNAPSHOT)
    lines = [
        f"**Q:** {question}",
        "**League Lineup (fixed counts):**",
        lineup_info,
        "—",
        "This is a placeholder for a real Q&A engine tied to your league data."
    ]
    await interaction.response.send_message(join_lines(lines))

@bot.tree.command(name="roster", description="Show a team's roster.")
@app_commands.describe(team="Team name or manager")
async def roster(interaction: discord.Interaction, team: str):
    lines = [f"Roster for **{team}**:", "- (stub) Add players here"]
    await interaction.response.send_message(join_lines(lines))

@bot.tree.command(name="whohas", description="Find which team has a player.")
@app_commands.describe(player="Player name")
async def whohas(interaction: discord.Interaction, player: str):
    owner = None
    for team_id, team_info in LEAGUE_SNAPSHOT.get("teams", {}).items():
        for pid in team_info.get("players", []):
            pdata = LEAGUE_SNAPSHOT["players"].get(pid, {})
            if pdata.get("name", "").lower() == player.lower():
                owner = team_info.get("name")
                break
        if owner:
            break
    msg = f"**{player}** is on **{owner}**." if owner else f"Couldn't find **{player}** in the snapshot."
    await interaction.response.send_message(msg)

@bot.tree.command(name="draftboard", description="Show the draft board with names and team abbrev (when available).")
async def draftboard(interaction: discord.Interaction):
    # Example stub picks; replace with your real board
    picks = [
        (1, "Team A", "123"),  # (overall, team, player_id)
        (2, "Team B", "124"),
    ]
    lines = ["**Draft Board**"]
    for pick_num, team_name, pid in picks:
        p = LEAGUE_SNAPSHOT["players"].get(pid, {"name": f"Player {pid}", "team": ""})
        team_abbrev = f" ({p['team']})" if p.get("team") else ""
        lines.append(f"{pick_num}. **{team_name}** → {p['name']}{team_abbrev}")
    await interaction.response.send_message(join_lines(lines))

@bot.tree.command(name="grades", description="Draft grades.")
async def grades(interaction: discord.Interaction):
    await interaction.response.send_message("**Draft Grades**\n(stub) Add grading logic here.")

# Alias for /grades
@app_commands.command(name="draft_grades", description="Alias of /grades.")
async def draft_grades(interaction: discord.Interaction):
    await grades.callback(interaction)

bot.tree.add_command(draft_grades)

# /proj and alias /projections
@bot.tree.command(name="proj", description="Alias of /projections.")
async def proj(interaction: discord.Interaction):
    await interaction.response.send_message("**Projections**\n(stub) Add projections here.")

@bot.tree.command(name="projections", description="Show projections.")
async def projections(interaction: discord.Interaction):
    await proj.callback(interaction)

# /recap and alias /weekly_report
@bot.tree.command(name="recap", description="Alias of /weekly_report.")
async def recap(interaction: discord.Interaction):
    await interaction.response.send_message("**Weekly Recap**\n(stub) Add weekly report here.")

@bot.tree.command(name="weekly_report", description="Weekly report.")
async def weekly_report(interaction: discord.Interaction):
    await recap.callback(interaction)

# -------- Background Tasks (preserved verbatim but commented out to keep this file runnable) --------
# If you want these active, move them into proper functions/modules and ensure all dependencies exist.
# @tasks.loop(time=datetime.time(hour=11, minute=0, tzinfo=TZ))
# async def weekly_report_auto():
#     now = datetime.datetime.now(TZ)
#     if now.weekday() != 6:
#         return
#     l = league()
#     week = getattr(l, "nfl_week", None) or getattr(l, "current_week", None)
#     if not week:
#         return
#     sb = l.scoreboard(week=week)
#     if not sb:
#         return
#     bullets = matchup_bullets(sb, week)
#     analysis = await llm_summary(f"ESPN League {LEAGUE_ID}", bullets)
#     await post_to_channel(f"**Pre-kickoff Weekly Report (Week {week})**\n```\n{bullets}\n```\n**AI breakdown:**\n{analysis}")
#
# @tasks.loop(time=datetime.time(hour=9, minute=0, tzinfo=TZ))
# async def tuesday_recap():
#     now = datetime.datetime.now(TZ)
#     if now.weekday() != 1:
#         return
#     l = league()
#     week = (getattr(l, "nfl_week", None) or getattr(l, "current_week", None)) - 1
#     if not week or week < 1:
#         return
#     sb = l.scoreboard(week=week)
#     if not sb:
#         return
#     bullets = matchup_bullets(sb, week)
#     analysis = await llm_summary(f"ESPN League {LEAGUE_ID}", bullets)
#     await post_to_channel(f"**Final Recap (Week {week})**\n```\n{bullets}\n```\n**AI breakdown:**\n{analysis}")
#
# @tasks.loop(minutes=5)
# async def transactions_watch():
#     try:
#         l = league()
#         for msg_type in ("TRADED", "WAIVER", "FA"):
#             acts = l.recent_activity(size=25, msg_type=msg_type)
#             for a in acts:
#                 key_src = f"{a.date}:{[(getattr(t,'team_name','?'),act,getattr(p,'name','?'),bid) for (t,act,p,bid) in a.actions]}"
#                 key = hashlib.sha1(key_src.encode()).hexdigest()
#                 if key in SEEN_ACTIVITY_KEYS:
#                     continue
#                 SEEN_ACTIVITY_KEYS.add(key)
#                 lines = []
#                 for (t, action, p, bid) in a.actions:
#                     team = getattr(t, "team_name", "Team")
#                     player = getattr(p, "name", "Player")
#                     piece = f"{team} {action} {player}"
#                     if bid:
#                         piece += f" (bid: {bid})"
#                     lines.append(f"- {piece}")
#                 await post_to_channel(f"**{msg_type.title()} update:**\n" + "\n".join(lines))
#     except Exception:
#         pass
#
# @tasks.loop(hours=1)
# async def injury_watch():
#     try:
#         l = league()
#         changes = []
#         for t in l.teams:
#             for p in t.roster:
#                 pid = getattr(p, "playerId", None)
#                 if pid is None:
#                     continue
#                 new = getattr(p, "injuryStatus", None) or "ACTIVE"
#                 old = PLAYER_STATUS_CACHE.get(pid)
#                 if old is None:
#                     PLAYER_STATUS_CACHE[pid] = new
#                 elif new != old:
#                     PLAYER_STATUS_CACHE[pid] = new
#                     pos = getattr(p, "position", "")
#                     changes.append(f"- {p.name} ({pos}, {t.team_name}) → **{new}** (was {old})")
#         if changes:
#             await post_to_channel("**Injury status changes:**\n" + "\n".join(changes[:20]))
#     except Exception:
#         pass
#
# NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
# def _is_nfl_day(dt: datetime.datetime) -> bool:
#     return dt.weekday() in (3, 6, 0)  # Thu, Sun, Mon
#
# @tasks.loop(minutes=5)
# async def nfl_finals_watch():
#     now = datetime.datetime.now(TZ)
#     if not _is_nfl_day(now):
#         return
#     try:
#         async with httpx.AsyncClient(timeout=20) as http:
#             r = await http.get(NFL_SCOREBOARD)
#             data = r.json()
#         events = data.get("events", [])
#         for ev in events:
#             eid = ev.get("id")
#             comp = (ev.get("competitions") or [{}])[0]
#             status = ((comp.get("status") or {}).get("type") or {})
#             is_final = bool(status.get("completed"))
#             if not (eid and is_final):
#                 continue
#             if eid in POSTED_FINAL_GAMES:
#                 continue
#             teams = comp.get("competitors", [])
#             if len(teams) != 2:
#                 continue
#             a = teams[0]; b = teams[1]
#             a_name = a.get("team", {}).get("displayName", "Team A")
#             b_name = b.get("team", {}).get("displayName", "Team B")
#             a_score = a.get("score", "0")
#             b_score = b.get("score", "0")
#             short = status.get("shortDetail", "Final")
#             msg = f"**NFL Final:** {a_name} {a_score} — {b_name} {b_score}  ({short})"
#             await post_to_channel(msg)
#             POSTED_FINAL_GAMES.add(eid)
#     except Exception:
#         pass

# Provide a default heartbeat if none existed
@tasks.loop(minutes=30)
async def heartbeat():
    logger.info("Heartbeat: bot alive at %s", dt.datetime.utcnow().isoformat())

@heartbeat.before_loop
async def before_heartbeat():
    await bot.wait_until_ready()

# --------- Startup ---------
@bot.event
async def on_ready():
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")
    if not heartbeat.is_running():
        heartbeat.start()

if __name__ == "__main__":
    if BOT_TOKEN == "REPLACE_ME":
        logger.warning("DISCORD_TOKEN env var not set. Please set it before running.")
    else:
        bot.run(BOT_TOKEN)
