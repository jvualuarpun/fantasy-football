# coding: utf-8
# Fantasy Football Discord Bot — ESPN snapshot+grades, assistant memory (OpenAI Threads), alerts (scores/injuries).
# NOTE: keep `import logging` directly above logging.basicConfig(...) per user requirement.

import os
import re
import io
import json
import asyncio
import datetime as dt
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import contextlib

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()  # read .env for DISCORD_TOKEN, ESPN creds, OpenAI, toggles

# import logging must remain directly above logging.basicConfig — DO NOT MOVE
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fantasy-bot")

# Quieten noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# -------- Discord / Bot config --------
INTENTS = discord.Intents.default()
ALLOW_MESSAGE_MEMORY = os.getenv("ALLOW_MESSAGE_MEMORY", "false").lower() == "true"
if ALLOW_MESSAGE_MEMORY:
    INTENTS.message_content = True
else:
    INTENTS.message_content = False

BOT_TOKEN = os.getenv("DISCORD_TOKEN", "REPLACE_ME")

# -------- Alerts config (scores/injuries) --------
ALERT_CHANNEL_ID = int(os.getenv("ALERT_CHANNEL_ID", "0") or 0)
POST_SCORES = os.getenv("POST_SCORES", "true").lower() == "true"
POST_INJURIES = os.getenv("POST_INJURIES", "true").lower() == "true"
STATE_PATH = os.getenv("STATE_PATH", "state.json")
# Injury recency + volume caps
SCORE_POST_LIMIT       = int(os.getenv("SCORE_POST_LIMIT", "12"))
INJURY_POST_LIMIT      = int(os.getenv("INJURY_POST_LIMIT", "24"))
INJURY_PRIORITY_LIMIT  = int(os.getenv("INJURY_PRIORITY_LIMIT", "16"))
INJURY_MAX_DAYS        = int(os.getenv("INJURY_MAX_DAYS", "3"))
INJURY_MAX_DAYS_Q      = int(os.getenv("INJURY_MAX_DAYS_Q", "7"))

# NEW: Top-N injury limiter
INJURY_LIMIT_TO_TOP_N = os.getenv("INJURY_LIMIT_TO_TOP_N", "true").lower() == "true"
INJURY_TOP_N = int(os.getenv("INJURY_TOP_N", "200"))

TOP_PLAYER_NAMES: set[str] = set()

def refresh_top_players() -> None:
    """Build Top-N players by projection from LEAGUE_SNAPSHOT."""
    try:
        players = list(LEAGUE_SNAPSHOT.get("players", {}).values())
        ranked = sorted(players, key=lambda p: float(p.get("projections") or 0.0), reverse=True)
        topn = ranked[:INJURY_TOP_N]
        TOP_PLAYER_NAMES.clear()
        for p in topn:
            nm = (p.get("name") or "").lower().strip()
            if nm:
                TOP_PLAYER_NAMES.add(nm)
        logger.info("Top players list rebuilt (%s names)", len(TOP_PLAYER_NAMES))
    except Exception as e:
        logger.warning("Failed to refresh top players: %s", e)


# httpx for ESPN site requests
try:
    import httpx
    _HTTPX = True
except Exception as e:
    _HTTPX = False
    logger.warning("httpx unavailable: %s", e)

# -------- ESPN config (for /snapshot etc.) --------
def _getenv_str(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.lower() in ("none", "null"):
        return None
    return v

def _detect_league_id() -> Optional[str]:
    for key in ("LEAGUE_ID", "ESPN_LEAGUE_ID", "LEAGUEID"):
        v = _getenv_str(key)
        if v and v.isdigit():
            return v
    url = _getenv_str("LEAGUE_URL") or _getenv_str("ESPN_LEAGUE_URL")
    if url:
        with contextlib.suppress(Exception):
            q = parse_qs(urlparse(url).query)
            lid = q.get("leagueId") or q.get("leagueid")
            if lid:
                s = str(lid[0]).strip()
                if s.isdigit():
                    return s
    return None

def _detect_year(default_year: str = "2025") -> int:
    y = _getenv_str("YEAR")
    if y and y.isdigit():
        return int(y)
    url = _getenv_str("LEAGUE_URL") or _getenv_str("ESPN_LEAGUE_URL")
    if url:
        with contextlib.suppress(Exception):
            q = parse_qs(urlparse(url).query)
            sid = q.get("seasonId") or q.get("season") or []
            if sid:
                s = str(sid[0]).strip()
                if s.isdigit():
                    return int(s)
    return int(default_year)

LEAGUE_ID = _detect_league_id()
YEAR = _detect_year("2025")
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
    "teams": {},
    "players": {},
    "draft": [],
}

# --- Week + stats helpers (paste right after LEAGUE_SNAPSHOT) ---

def _detect_current_week_from_league(lg) -> int:
    # Try env override first: WEEK=##
    try:
        w = int(os.getenv("WEEK", "0"))
        if w > 0:
            return w
    except Exception:
        pass
    # Fall back to league property
    for attr in ("current_week", "currentWeek"):
        if hasattr(lg, attr):
            try:
                w = int(getattr(lg, attr))
                if w > 0:
                    return w
            except Exception:
                pass
    return 1  # safe default

def _extract_week_points(p, week: int, source_id: int) -> float:
    """
    Read a player's total for a given week from espn_api stats.
      source_id 0 = ACTUAL, 1 = PROJECTED
    Tries common attribute names used by espn_api.
    """
    try:
        for s in getattr(p, "stats", []) or []:
            sp = (
                getattr(s, "scoring_period", None)
                or getattr(s, "scoringPeriodId", None)
                or getattr(s, "scoringPeriod", None)
            )
            try:
                sp_i = int(sp)
            except Exception:
                sp_i = None
            if sp_i == int(week):

                sid = (
                    getattr(s, "stat_source_id", None)
                    or getattr(s, "statSourceId", None)
                    or getattr(s, "source_id", None)
                )
                if sid == source_id:
                    val = (
                        getattr(s, "applied_total", None)
                        or getattr(s, "appliedTotal", None)
                        or getattr(s, "points", None)
                        or getattr(s, "projected_points", None)
                    )
                    return float(val or 0.0)
    except Exception:
        pass
    return 0.0

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

def _collect_weekly_points(lg, week: int) -> Dict[str, Dict[str, float]]:
    """
    Build { player_key: {"pts_week": x, "proj_week": y} } from box scores for a week.
    Keys include:
      - the numeric/string playerId when available
      - a fallback 'name:<lowercased name>' key (for slots where espn_api lacks an id)
    """
    out: Dict[str, Dict[str, float]] = {}
    try:
        boxes = lg.box_scores(week=week)
    except Exception as e:
        logger.warning("box_scores(%s) failed: %s", week, e)
        return out

    def _ingest_lineup(lineup):
        for bp in lineup or []:
            # try to pull id & name from BoxPlayer
            raw_id = (
                getattr(bp, "playerId", None)
                or getattr(getattr(bp, "player", None), "playerId", None)
                or getattr(getattr(bp, "player", None), "id", None)
            )
            name = (
                getattr(bp, "name", None)
                or getattr(bp, "playerName", None)
                or getattr(getattr(bp, "player", None), "name", None)
            )

            pts = 0.0
            proj = 0.0
            with contextlib.suppress(Exception):
                pts = float(getattr(bp, "points", 0.0) or 0.0)
            with contextlib.suppress(Exception):
                proj = float(getattr(bp, "projected_points", 0.0) or 0.0)

            # store under id if present
            if raw_id is not None:
                out[_safe_str(raw_id)] = {"pts_week": pts, "proj_week": proj}
            # also store a by-name key (helps when id is missing/mismatched)
            if name:
                out[f"name:{name.lower().strip()}"] = {"pts_week": pts, "proj_week": proj}

    try:
        for bs in boxes or []:
            _ingest_lineup(getattr(bs, "home_lineup", []))
            _ingest_lineup(getattr(bs, "away_lineup", []))
    except Exception as e:
        logger.warning("Failed parsing box_scores lineups: %s", e)

    logger.info("Collected weekly box points for %d player keys (week %s)", len(out), week)
    return out

async def refresh_snapshot(force: bool = False) -> None:
    ok, why = _league_ready()
    if not ok:
        logger.warning("League not ready: %s", why)
        return
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        lg = League(league_id=int(LEAGUE_ID), year=YEAR)
        current_week = _detect_current_week_from_league(lg)
        LEAGUE_SNAPSHOT["meta"]["current_week"] = current_week

        logger.info("refresh_snapshot: current_week=%s", current_week)

        # Pull weekly points/projections from box scores FIRST
        week_pts_map = _collect_weekly_points(lg, current_week)

        logger.info("week_pts_map keys=%d (sample: %s)",
            len(week_pts_map),
            list(list(week_pts_map.keys())[:6]))

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

                name_key = f"name:{(getattr(p, 'name', '') or '').lower().strip()}"
                wpts = week_pts_map.get(pid) or week_pts_map.get(name_key) or {}

                # >>> ADDED: log when no weekly points/projections were found for this player
                if not wpts:
                    logger.debug("No week pts for %s (pid=%s, name_key=%s)",
                                 getattr(p, "name", "?"), pid, name_key)

                proj_week = float(wpts.get("proj_week", 0.0) or 0.0)
                pts_week  = float(wpts.get("pts_week", 0.0) or 0.0)

                players_map[pid] = {
                    "name": getattr(p, "name", f"Player {pid}"),
                    "team": team_abbrev,
                    "position": pos,
                    # keep legacy key so existing code keeps working
                    "projections": proj_week,
                    "proj_week": proj_week,
                    "pts_week": pts_week,
                }
                roster_ids.append(pid)

            teams_map[_safe_str(team_id)] = {"name": team_name, "manager": _safe_str(manager), "players": roster_ids}

        # Draft (unchanged)
        draft_list = []
        with contextlib.suppress(Exception):
            if getattr(lg, "draft", None):
                for pick in lg.draft:
                    with contextlib.suppress(Exception):
                        overall = getattr(pick, "overall_pick", None) or getattr(pick, "pick", None)
                        team_name = getattr(getattr(pick, "team", None), "team_name", None) or "Team"
                        pid = _safe_str(getattr(getattr(pick, "player", None), "playerId", None) or getattr(pick, "playerId", None))
                        draft_list.append({
                            "pick": int(overall) if overall else len(draft_list) + 1,
                            "team_name": team_name,
                            "player_id": pid
                        })

        LEAGUE_SNAPSHOT["teams"] = teams_map
        LEAGUE_SNAPSHOT["players"] = players_map
        LEAGUE_SNAPSHOT["draft"] = draft_list
        LEAGUE_SNAPSHOT["meta"]["last_refresh"] = now

        logger.info("League snapshot refreshed (week %s) at %s", current_week, now)

        if INJURY_LIMIT_TO_TOP_N:
            refresh_top_players()

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

# ---------- Simple persistent state for alerts ----------
def _load_state() -> Dict[str, List[str]]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {"scores": [], "injuries": [], "meta": {}}
        data.setdefault("scores", [])
        data.setdefault("injuries", [])
        # new: keep a meta bucket for timestamps, etc.
        meta = data.setdefault("meta", {})
        # ensure inj_last_run exists (ISO string or None)
        if "inj_last_run" not in meta:
            meta["inj_last_run"] = None
        return data
    except Exception:
        return {"scores": [], "injuries": [], "meta": {"inj_last_run": None}}


def _save_state(state: Dict[str, List[str]]) -> None:
    with contextlib.suppress(Exception):
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)

STATE = _load_state()

def _state_has(kind: str, key: str) -> bool:
    return key in STATE.get(kind, [])

def _state_add(kind: str, key: str, cap: int = 400) -> None:
    arr = STATE.setdefault(kind, [])
    if key not in arr:
        arr.append(key)
        if len(arr) > cap:
            del arr[: len(arr) - cap]
        _save_state(STATE)

# ---------- Alerts fetch + post ----------
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

DEFAULT_HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                                               "Chrome/127.0.0.0 Safari/537.36"),
    "Accept-Language": os.getenv("HTTP_ACCEPT_LANGUAGE", "en-US,en;q=0.9"),
}
_ESPN_CONCURRENCY = int(os.getenv("ESPN_CONCURRENCY", "6"))
_ESPN_SEM = asyncio.Semaphore(_ESPN_CONCURRENCY)

# Optional BeautifulSoup (HTML)
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    _BS4 = True
except Exception as e:
    _BS4 = False
    logger.warning("beautifulsoup4 unavailable: %s", e)

NFL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LAR","LAC","MIA","MIN","NE","NO","NYG","NYJ","LV","PHI","PIT","SF","SEA","TB","TEN","WSH"
]
TEAM_NAME = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LAR":"Los Angeles Rams","LAC":"Los Angeles Chargers","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets",
    "LV":"Las Vegas Raiders","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SF":"San Francisco 49ers",
    "SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WSH":"Washington Commanders"
}
ALLOWED_POS = {"QB","RB","WR","TE","K"}
PRIORITY_KEYS = {"questionable","doubtful","out","inactive","probable"}

# --- NEW: injury recency control ---
# Only publish injuries that are newer than the last successful run (with a small buffer).
# If no last-run timestamp is stored yet, only go back this many days on first boot:
INJURY_HTML_MAX_DAYS_BOOTSTRAP = int(os.getenv("INJURY_HTML_MAX_DAYS_BOOTSTRAP", "4"))
# Always allow priority flags (Q/D/OUT/Inactive/Probable) back this many days even if older (keeps big updates):
INJURY_PRIORITY_LOOKBACK_DAYS = int(os.getenv("INJURY_PRIORITY_LOOKBACK_DAYS", "7"))
# Grace buffer before last run to catch items that published right before downtime (minutes):
INJURY_GRACE_MINUTES = int(os.getenv("INJURY_GRACE_MINUTES", "90"))

# Date header patterns like "Sep 4"
_MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
_DATE_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})$")

def _coerce_date_from_header(txt: str) -> Optional[dt.date]:
    """Convert 'Sep 4' into a date in the current year (handles year roll-over in Jan)."""
    m = _DATE_RE.match((txt or "").strip())
    if not m:
        return None
    mon_abbr, day_s = m.group(1), m.group(2)
    month = _MONTHS.get(mon_abbr, 0)
    if not month:
        return None
    today = dt.datetime.now(dt.timezone.utc).date()
    year = today.year
    # Handle possible year rollover if we're in Jan and page shows Dec
    if today.month == 1 and month == 12:
        year -= 1
    try:
        return dt.date(year, month, int(day_s))
    except Exception:
        return None

def _iso_date(d: Optional[dt.date]) -> Optional[str]:
    return d.isoformat() if isinstance(d, dt.date) else None

def _pos_ok(p: str) -> bool:
    return (p or "").upper() in ALLOWED_POS

def _designation_from_text(*parts: str) -> Optional[str]:
    blob = " ".join([p or "" for p in parts]).lower()
    for k in PRIORITY_KEYS:
        if k in blob:
            # Normalize capitalization (Questionable, Doubtful, Out, Inactive, Probable)
            return k.capitalize()
    # Sometimes "Injured reserve", "IR", etc. appear; keep as-is if useful
    # Fall back to first non-empty piece capitalized
    for p in parts:
        if p and p.strip():
            return p.strip().split()[0].capitalize()
    return None

def _extract_text(el) -> str:
    try:
        return " ".join(el.get_text(" ", strip=True).split())
    except Exception:
        return ""

def _find_nearest_date(node) -> Optional[str]:
    # date examples like "Sep 4" in a dotted border div
    try:
        prev = node.find_previous("div", class_=re.compile(r"bb--dotted"))
        text = _extract_text(prev) if prev else ""
        if re.search(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", text):
            return text
    except Exception:
        pass
    return None

def _parse_team_injuries_html(html: str, abbr: str) -> List[dict]:
    """
    Parse ESPN team injuries page (new div-based layout) with fallback to old table layout.
    Returns items:
      { team, abbrev, athlete_id, name, pos, designation, type, detail, when_date (date or None) }
    """
    if not _BS4 or not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: List[dict] = []

    # --- Primary path: div-based layout with date headers and player blocks ---
    # Date headers look like: <div class="pb3 bb bb--dotted brdr-clr-gray-07 n8 fw-medium mb2">Sep 4</div>
    date_headers = []
    for div in soup.find_all("div"):
        txt = _extract_text(div)
        if _DATE_RE.match(txt or "") and "bb--dotted" in " ".join(div.get("class", [])):
            date_headers.append(div)

    if date_headers:
        for header in date_headers:
            date_obj = _coerce_date_from_header(_extract_text(header))
            # Walk following siblings/elements until next date header
            for sib in header.find_all_next():
                if sib is header:
                    continue
                if sib.name == "div":
                    txt = _extract_text(sib)
                    # If we hit another date header, stop this block
                    if _DATE_RE.match(txt or "") and "bb--dotted" in " ".join(sib.get("class", [])):
                        break
                    # Player wrapper pattern you posted:
                    # <div class="Athlete__PlayerWrapper ...">
                    classes = set(sib.get("class", []))
                    if any(cls.startswith("Athlete__PlayerWrapper") for cls in classes):
                        # name + position live in an <h3> with spans
                        h3 = sib.find("h3")
                        if not h3:
                            continue
                        name_span = h3.find("span", class_=lambda c: c and "Athlete__PlayerName" in c)
                        pos_span  = h3.find("span", class_=lambda c: c and "Athlete__NameDetails" in c)
                        player_name = _extract_text(name_span) if name_span else ""
                        pos_text    = _extract_text(pos_span) if pos_span else ""
                        pos = (pos_text.split() or [""])[0].upper().strip(" ,;/")
                        if not player_name or not _pos_ok(pos):
                            continue

                        # status line: <div class="flex n9"><span>Status</span><span class="TextStatus ...">Questionable</span></div>
                        status_div = sib.find("div", class_=lambda c: c and "flex" in c.split())
                        status_txt = ""
                        if status_div and "Status" in _extract_text(status_div):
                            status_val = status_div.find("span", class_=lambda c: c and "TextStatus" in c)
                            status_txt = _extract_text(status_val)

                        designation = _designation_from_text(status_txt)

                        # detail: <div class="pt3 ...">Player (x) was ...</div>
                        detail_div = sib.find("div", class_=lambda c: c and "pt3" in c.split())
                        detail_txt = _extract_text(detail_div)

                        # injury type: try to sniff from detail "(knee)" etc.
                        inj_type = ""
                        m = re.search(r"\(([A-Za-z \-/]+)\)", detail_txt or "")
                        if m:
                            inj_type = m.group(1).strip()

                        items.append({
                            "team": TEAM_NAME.get(abbr, abbr),
                            "abbrev": abbr,
                            "athlete_id": player_name.lower().replace(" ", "-"),
                            "name": player_name,
                            "pos": pos,
                            "designation": designation or "Injury",
                            "type": inj_type,
                            "detail": detail_txt,
                            "when_date": date_obj,  # a datetime.date
                        })
        if items:
            # De-dupe within this page by (name, pos, designation, date)
            seen = set()
            deduped = []
            for it in items:
                k = (it["name"].lower(), it["pos"], it["designation"].lower(), _iso_date(it["when_date"]))
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(it)
            return deduped

    # --- Fallback: old table layout (no per-row dates) ---
    tables = soup.select("table.Table")
    for tbl in tables:
        header_cells = tbl.select("thead tr th")
        if not header_cells:
            continue
        headers = [(_extract_text(th) or "").strip().lower() for th in header_cells]
        def _find(names):
            for nm in names:
                if nm in headers:
                    return headers.index(nm)
            return None
        i_player = _find({"player","player name"})
        i_pos = _find({"pos","position"})
        i_injury = _find({"injury","injuries","report","note","notes"})
        i_status = _find({"status","game status"})
        if i_player is None or i_pos is None:
            continue

        for tr in tbl.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds or len(tds) < max(i_player, i_pos) + 1:
                continue
            player_name = _extract_text(tds[i_player])
            pos = _extract_text(tds[i_pos]).upper()
            if not player_name or not _pos_ok(pos):
                continue
            inj_type = _extract_text(tds[i_injury]) if (i_injury is not None and i_injury < len(tds)) else ""
            status = _extract_text(tds[i_status]) if (i_status is not None and i_status < len(tds)) else ""
            designation = _designation_from_text(status, inj_type) or (status or "Injury")

            items.append({
                "team": TEAM_NAME.get(abbr, abbr),
                "abbrev": abbr,
                "athlete_id": player_name.lower().replace(" ", "-"),
                "name": player_name,
                "pos": pos,
                "designation": designation,
                "type": inj_type or "",
                "detail": "",
                "when_date": None,  # table doesn’t show dates per row
            })
    # dedupe fallback
    seen = set()
    deduped = []
    for it in items:
        k = (it["name"].lower(), it["pos"], it["designation"].lower(), _iso_date(it["when_date"]))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(it)
    return deduped


async def _fetch_text(url: str) -> Optional[str]:
    if not _HTTPX:
        return None
    try:
        async with _ESPN_SEM:
            async with httpx.AsyncClient(timeout=20, headers=DEFAULT_HEADERS) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.text
    except Exception as e:
        logger.info("Fetch text failed %s: %s", url, e)
        return None

async def _scrape_team_injuries(abbr: str) -> List[dict]:
    url = f"https://www.espn.com/nfl/team/injuries/_/name/{abbr.lower()}"
    html = await _fetch_text(url)
    return _parse_team_injuries_html(html or "", abbr)

async def _scrape_all_injuries_html() -> List[dict]:
    # Scrape all teams with limited concurrency
    results = await asyncio.gather(*[_scrape_team_injuries(a) for a in NFL_TEAMS], return_exceptions=True)
    merged: List[dict] = []
    for res in results:
        if isinstance(res, list):
            merged.extend(res)
    return merged

def _fmt_injury_line(team_name: str, player_name: str, pos: str, designation: str, inj_type: str, detail: str, date_iso: Optional[str]) -> str:
    # Human-friendly date tag if we know it (e.g., "Sep 4")
    date_tag = ""
    if date_iso:
        try:
            d = dt.date.fromisoformat(date_iso)
            date_tag = f"[{d.strftime('%b %-d')}] " if hasattr(d, "strftime") else f"[{date_iso}] "
        except Exception:
            date_tag = f"[{date_iso}] "
    extra = f" ({inj_type})" if inj_type and designation.lower() not in (inj_type or "").lower() else ""
    detail_part = f" {detail}" if detail else ""
    return f"{date_tag}🩺 {team_name}: {player_name} ({pos}) — {designation}{extra}.{detail_part}"


def _inj_key(team_abbr: str, athlete_slug: str, designation: str, date_iso: Optional[str]) -> str:
    # Include date in the key when available so historical items don’t collide with fresh ones
    return f"inj:{team_abbr}:{athlete_slug}:{(designation or '').lower()}:{date_iso or 'nodate'}"


async def _already_sent(ch: discord.TextChannel, content: str) -> bool:
    """Prevents duplicate posts by comparing exact content in recent bot messages."""
    try:
        async for m in ch.history(limit=300):
            if m.author.id == (bot.user.id if bot.user else 0):
                if (m.content or "").strip() == content.strip():
                    return True
    except Exception:
        pass
    return False

def _format_score_line(event: dict) -> Optional[str]:
    """Score line without id tag (content-based + state-based de-dupe handles repeats)."""
    try:
        comp = (event.get("competitions") or [])[0]
        status = comp.get("status", {})
        stype = (status.get("type") or {})
        state = stype.get("state")  # "pre", "in", "post"
        clock = status.get("displayClock", "")
        period = status.get("period", 0)

        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[-1])

        hn = (home.get("team") or {}).get("abbreviation") or (home.get("team") or {}).get("shortDisplayName")
        an = (away.get("team") or {}).get("abbreviation") or (away.get("team") or {}).get("shortDisplayName")
        hs = int(home.get("score") or 0)
        as_ = int(away.get("score") or 0)

        if state == "post":
            return f"🏁 Final: {an} {as_} — {hn} {hs}"
        elif state == "in":
            return f"⏱️ Live (Q{period} {clock}): {an} {as_} — {hn} {hs}"
        else:
            return None
    except Exception:
        return None

def _score_key(event: dict) -> Optional[str]:
    eid = event.get("id")
    if eid:
        return f"score:{eid}"
    return None

async def _post_startup_scores(ch: discord.TextChannel):
    if not POST_SCORES or not _HTTPX:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(SCOREBOARD_URL)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.info("Scoreboard fetch failed: %s", e)
        return

    events = data.get("events") or []
    lines: List[Tuple[str, str]] = []
    for ev in events:
        key = _score_key(ev)
        if not key:
            continue
        line = _format_score_line(ev)
        if not line:
            continue
        lines.append((key, line))

    posted = 0
    for key, line in lines[:SCORE_POST_LIMIT]:
        if _state_has("scores", key):
            continue
        if await _already_sent(ch, line):
            _state_add("scores", key)
            continue
        await ch.send(line)
        _state_add("scores", key)
        posted += 1
    if posted:
        logger.info("Posted %s score lines", posted)

def _is_priority(designation: str) -> bool:
    return (designation or "").lower() in PRIORITY_KEYS

async def _post_injuries(ch: discord.TextChannel):
    if not POST_INJURIES:
        return

    # Compute time window
    now = dt.datetime.now(dt.timezone.utc)
    last_run_s = STATE.get("meta", {}).get("inj_last_run")
    if last_run_s:
        try:
            last_run = dt.datetime.fromisoformat(last_run_s.replace("Z", "+00:00"))
        except Exception:
            last_run = None
    else:
        last_run = None

    # Start publishing window:
    if last_run is not None:
        since_dt = last_run - dt.timedelta(minutes=INJURY_GRACE_MINUTES)
    else:
        since_dt = now - dt.timedelta(days=INJURY_HTML_MAX_DAYS_BOOTSTRAP)

    items = await _scrape_all_injuries_html()
    if not items:
        logger.info("No injuries found from ESPN HTML pages.")
        return

    # Normalize + Filter by date
    ready: List[Tuple[str, str, bool]] = []  # (state_key, line, is_priority)
    per_run_seen = set()  # stop duplicates within the same scrape

    for it in items:
        d_iso = _iso_date(it.get("when_date"))
        # Convert date to datetime at UTC midnight for comparison
        if d_iso:
            try:
                when_dt = dt.datetime.fromisoformat(d_iso)  # naive date -> datetime; treat as midnight
                when_dt = when_dt.replace(tzinfo=dt.timezone.utc)
            except Exception:
                when_dt = None
        else:
            when_dt = None

        is_prio = _is_priority(it["designation"])

        # Filter rule:
        # - If we have a date: include if when_dt >= since_dt, OR (is_prio and within priority lookback)
        # - If no date: include only if priority (we can't safely time-bound otherwise)
        include = False
        if when_dt is not None:
            if when_dt >= since_dt:
                include = True
            elif is_prio and when_dt >= now - dt.timedelta(days=INJURY_PRIORITY_LOOKBACK_DAYS):
                include = True
        else:
            include = is_prio  # undated table rows: only priority to avoid old spam

        if not include:
            continue

        # --- Top-N limiter by player projections (from LEAGUE_SNAPSHOT) ---
        # Only allow injuries for players in the top-N list (case-insensitive name match)
        if INJURY_LIMIT_TO_TOP_N:
            nm = (it.get("name") or "").lower().strip()
            if nm not in TOP_PLAYER_NAMES:
                continue

        key = _inj_key(it["abbrev"], it["athlete_id"], it["designation"], d_iso)
        line = _fmt_injury_line(it["team"], it["name"], it["pos"], it["designation"], it["type"], it["detail"], d_iso)

        # Per-run content de-dupe and state de-dupe:
        if (key in per_run_seen) or _state_has("injuries", key) or await _already_sent(ch, line):
            continue
        per_run_seen.add(key)

        ready.append((key, line, is_prio))

    # Sort: priority first, then by line (stable)
    ready.sort(key=lambda t: (not t[2], t[1]))

    posted = 0
    for key, line, _prio in ready:
        # safety cap (use your existing env-driven caps)
        if posted >= INJURY_POST_LIMIT:
            break
        await ch.send(line)
        _state_add("injuries", key)
        posted += 1

    # Update last run timestamp on success (even if 0 posted, we still update to avoid re-scanning far past)
    STATE.setdefault("meta", {})["inj_last_run"] = now.isoformat()
    _save_state(STATE)

    logger.info("Injuries posted: %s (window since %s, priority lookback %sd, grace %smin)",
                posted,
                since_dt.isoformat(),
                INJURY_PRIORITY_LOOKBACK_DAYS,
                INJURY_GRACE_MINUTES)


async def _get_alert_channel() -> Optional[discord.TextChannel]:
    if not ALERT_CHANNEL_ID:
        return None
    ch = bot.get_channel(ALERT_CHANNEL_ID)
    if ch is None:
        with contextlib.suppress(Exception):
            ch = await bot.fetch_channel(ALERT_CHANNEL_ID)
    if isinstance(ch, discord.TextChannel):
        return ch
    return None

async def startup_alerts():
    ch = await _get_alert_channel()
    if not ch:
        if ALERT_CHANNEL_ID:
            logger.warning("ALERT_CHANNEL_ID=%s not found or not a text channel.", ALERT_CHANNEL_ID)
        else:
            logger.info("ALERT_CHANNEL_ID not set; skipping startup alerts.")
        return
    # Rehydrate dedupe from last 300 messages if we can read message content
    if ALLOW_MESSAGE_MEMORY:
        with contextlib.suppress(Exception):
            async for m in ch.history(limit=300):
                if m.author.id != (bot.user.id if bot.user else 0):
                    continue
                if m.content:
                    # legacy inj tags hydration (safe to keep)
                    if "#inj " in m.content:
                        idx = m.content.find("#inj ")
                        if idx != -1:
                            tail = m.content[idx + 5:].split("]")[0].strip()
                            if tail:
                                _state_add("injuries", f"inj:{tail}")
    # Now post new ones
    await _post_startup_scores(ch)
    await _post_injuries(ch)

@tasks.loop(minutes=15)
async def poll_alerts():
    try:
        ch = await _get_alert_channel()
        if not ch:
            return
        await _post_startup_scores(ch)
        await _post_injuries(ch)
    except Exception as e:
        logger.info("poll_alerts error: %s", e)

@poll_alerts.before_loop
async def before_poll_alerts():
    await bot.wait_until_ready()

# ---------- Bot ----------
class FantasyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=INTENTS, help_command=None)
        self.synced = False
        self.assistant = DiscordOpenAIInterface()

    async def setup_hook(self) -> None:
        try:
            commands_list = await self.tree.sync(guild=None)
            self.synced = True
            logger.info(f"✅ Global slash commands synced ({len(commands_list)}): {[c.name for c in commands_list]}")
        except Exception as e:
            logger.exception("❌ Failed to sync global commands: %s", e)

bot = FantasyBot()

# ------ Optional memory over regular messages ------
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
    lineup = ", ".join(f"{k}:{v}" for k, v in LEAGUE_SNAPSHOT.get("lineup", {}).items())
    last = LEAGUE_SNAPSHOT["meta"].get("last_refresh", "never")
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
        await bot.tree.sync(guild=None)
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

    # find matching team
    match_team_id = None
    tlower = (team or "").lower()
    for tid, t in LEAGUE_SNAPSHOT.get("teams", {}).items():
        if tlower in (t.get("name", "")).lower() or tlower in (t.get("manager", "")).lower():
            match_team_id = tid
            break
    if not match_team_id:
        await interaction.followup.send(f"Couldn't find a team matching **{team}**.")
        return

    t = LEAGUE_SNAPSHOT["teams"][match_team_id]
    cur_wk = LEAGUE_SNAPSHOT.get("meta", {}).get("current_week")

    lines = [f"**{t['name']}** (Mgr: {t['manager']})"]
    for pid in t.get("players", []):
        p = LEAGUE_SNAPSHOT["players"].get(pid)
        if not p:
            continue

        name = p.get("name", "?")
        pos  = p.get("position", "?")
        team_abbr = p.get("team", "")
        proj = float(p.get("proj_week", p.get("projections", 0.0)) or 0.0)  # prefer weekly proj
        pts  = p.get("pts_week", None)

        # build the line
        if pts is not None and cur_wk is not None:
            lines.append(f"- {name} ({pos}, {team_abbr}) proj: {proj:.2f} | wk{cur_wk}: {float(pts):.2f}")
        else:
            lines.append(f"- {name} ({pos}, {team_abbr}) proj: {proj:.2f}")

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

def _league_pos_averages() -> Dict[str, float]:
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
    if strong: parts.append("strong at " + "/".join(strong))
    if weak: parts.append("light at " + "/".join(weak))
    if not parts:
        return f"{name}: balanced roster."
    return f"{name}: " + "; ".join(parts) + "."

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

    lines.append(f"\nSnapshot: {LEAGUE_SNAPSHOT['meta'].get('last_refresh','never')}")
    await send_long_followup(interaction, "\n".join(lines))

@bot.tree.command(name="recap", description="Alias of /weekly_report.")
async def recap(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send("Weekly recap will be implemented with box score parsing.")

@bot.tree.command(name="weekly_report", description="Weekly report.")
async def weekly_report(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await recap.callback(interaction)

@bot.tree.command(name="status", description="Show ESPN connection status & snapshot info.")
async def status(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    ready, why = _league_ready()
    lines = [
        f"League ID: {LEAGUE_ID or 'missing'} | Year: {YEAR}",
        f"Snapshot last refresh: {LEAGUE_SNAPSHOT['meta'].get('last_refresh', 'never')}",
        f"Ready: {'yes' if ready else 'no'}{f' — {why}' if not ready else ''}",
        f"Alerts: channel={'set' if ALERT_CHANNEL_ID else 'unset'} scores={'on' if POST_SCORES else 'off'} injuries={'on' if POST_INJURIES else 'off'}",
    ]
    await interaction.followup.send("\n".join(lines))

@bot.tree.command(
    name="injuries",
    description="List ESPN team injury updates for a date range (e.g., 'Sep 1-5', 'Sep 3', or '2025-09-01..2025-09-05')."
)
@app_commands.describe(
    date_range="Examples: 'Sep 1-5', 'Sep 3', '2025-09-01..2025-09-05'"
)
async def injuries(interaction: discord.Interaction, date_range: str):
    await interaction.response.defer(thinking=True)

    # --- tiny helper: parse user date/range into (start_date, end_date) ---
    def _parse_user_date_range(s: str) -> Optional[tuple[dt.date, dt.date]]:
        if not s:
            return None
        s = s.strip()
        # Normalize separators
        s = s.replace("—", "-").replace(" to ", "-").replace("..", "-").replace("—", "-")
        # Current year assumption unless ISO given
        today = dt.datetime.now(dt.timezone.utc).date()
        year = today.year

        # ISO style "YYYY-MM-DD-YYYY-MM-DD"
        m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*[-–]\s*(\d{4})-(\d{2})-(\d{2})\s*$", s)
        if m:
            y1, M1, d1, y2, M2, d2 = map(int, m.groups())
            a = dt.date(y1, M1, d1)
            b = dt.date(y2, M2, d2)
            return (a, b) if a <= b else (b, a)

        # ISO single day "YYYY-MM-DD"
        m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$", s)
        if m:
            y, M, d = map(int, m.groups())
            a = dt.date(y, M, d)
            return (a, a)

        # "Sep 1-5" or "September 1-5"
        m = re.match(r"^\s*([A-Za-z]{3,9})\s+(\d{1,2})\s*[-–]\s*(\d{1,2})\s*$", s)
        if m:
            mon_txt, d1, d2 = m.group(1), int(m.group(2)), int(m.group(3))
            try:
                M = _MONTHS[mon_txt[:3].title()]
                a = dt.date(year, M, min(d1, d2))
                b = dt.date(year, M, max(d1, d2))
                return (a, b)
            except Exception:
                return None

        # "Sep 3" or "September 3"
        m = re.match(r"^\s*([A-Za-z]{3,9})\s+(\d{1,2})\s*$", s)
        if m:
            mon_txt, d = m.group(1), int(m.group(2))
            try:
                M = _MONTHS[mon_txt[:3].title()]
                a = dt.date(year, M, d)
                return (a, a)
            except Exception:
                return None

        return None

    rng = _parse_user_date_range(date_range)
    if not rng:
        await interaction.followup.send(
            "I couldn’t read that date range. Try examples like **Sep 1-5**, **Sep 3**, or **2025-09-01..2025-09-05**."
        )
        return

    start_date, end_date = rng
    # Sanity (inclusive window)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    # Pull fresh HTML per-team and parse
    items = await _scrape_all_injuries_html()
    if not items:
        await interaction.followup.send("No injuries found from ESPN right now.")
        return

    # Filter to items that have an actual date AND fall within the window (inclusive)
    def _within(item) -> bool:
        d = item.get("when_date")
        return isinstance(d, dt.date) and (start_date <= d <= end_date)

    filtered = [it for it in items if _within(it)]

    if not filtered:
        await interaction.followup.send(
            f"No dated injury updates found between **{start_date.strftime('%b %-d')}** and **{end_date.strftime('%b %-d')}**."
        )
        return

    # Sort: date asc, team abbrev, player name
    filtered.sort(key=lambda it: (
        it.get("when_date") or dt.date(1970,1,1),
        (it.get("abbrev") or ""),
        (it.get("name") or "")
    ))

    # Format lines
    lines = [f"**Injury updates {start_date.strftime('%b %-d')} — {end_date.strftime('%b %-d')}**"]
    for it in filtered:
        d = it.get("when_date")
        dtag = d.strftime("%b %-d") if isinstance(d, dt.date) else ""
        team = it.get("team") or it.get("abbrev") or "Team"
        name = it.get("name", "?")
        pos  = it.get("pos", "")
        des  = it.get("designation", "")
        typ  = (f" ({it.get('type')})" if it.get("type") else "")
        detail = (it.get("detail") or "").strip()
        if len(detail) > 140:
            detail = detail[:137] + "…"
        lines.append(f"[{dtag}] {team}: **{name}** ({pos}) — {des}{typ}. {detail}")

    await send_long_followup(interaction, "\n".join(lines), attach_name="injuries.txt")


# -------- Background: live refresh --------
@tasks.loop(minutes=10)
async def auto_refresh_snapshot():
    now = dt.datetime.now(dt.timezone.utc)
    dow = now.weekday()
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
    logger.info("Config check: LEAGUE_ID=%r YEAR=%s ALERT_CHANNEL_ID=%s", LEAGUE_ID, YEAR, ALERT_CHANNEL_ID)

    if not heartbeat.is_running():
        heartbeat.start()
    if not auto_refresh_snapshot.is_running():
        auto_refresh_snapshot.start()
    if not poll_alerts.is_running():
        poll_alerts.start()

    await refresh_snapshot(force=True)
    await startup_alerts()

    with contextlib.suppress(Exception):
        commands_list = await bot.tree.fetch_commands()
        logger.info(f"🔎 Slash commands loaded ({len(commands_list)}): {[c.name for c in commands_list]}")

if __name__ == "__main__":
    if BOT_TOKEN == "REPLACE_ME":
        logger.warning("DISCORD_TOKEN env var not set. Please set it before running.")
    else:
        _server = start_health_server()  # keep Render Web Service alive
        bot.run(BOT_TOKEN)
