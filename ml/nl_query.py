"""Natural-language memory queries (upgrade 5.2).

"poles with corrosion getting worse in the last 3 months"
   → StructuredQuery(asset_types=["utility_pole"], defect_families=["corrosion"], trend="worsening", window_days=90)
   → deterministic filtering and ranking over assets, defects, defect instances and the graph.

Translators (question → StructuredQuery only; they never filter or rank):
  rules  default, deterministic keyword / regex grammar, no network
  llm    a small Claude model with schema-constrained output (`messages.parse` + Pydantic),
         used when nl_query.translator = "llm" and the anthropic SDK + credentials exist;
         any failure falls back to the rules translator
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ml.config import get_nested
from ml.forecast import days_until, family
from ml.revisit_priority import parse_ts
from ml.schemas import utc_now

logger = logging.getLogger(__name__)

ASSET_WORDS = {
    "utility_pole": ["pole", "poles", "insulator", "insulators", "power line", "powerline"],
    "solar_panel": ["panel", "panels", "solar", "pv", "module", "modules"],
    "building": ["wall", "walls", "building", "buildings", "facade", "facades", "concrete"],
    "machinery": ["machine", "machines", "machinery", "pump", "pumps", "pipe", "pipes", "motor"],
}
DEFECT_WORDS = {
    "crack": ["crack", "cracks", "cracked", "cracking", "fracture", "broken", "physical damage"],
    "corrosion": ["corrosion", "corroded", "rust", "rusty", "rusting", "oxidation"],
    "soiling": ["soiling", "dust", "dusty", "dirt", "dirty", "bird", "snow", "soiled"],
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
WORDS_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "twelve": 12}


class StructuredQuery(BaseModel):
    asset_types: List[str] = Field(default_factory=list, description="utility_pole | solar_panel | building | machinery")
    defect_families: List[str] = Field(default_factory=list, description="crack | corrosion | soiling")
    trend: Optional[Literal["worsening", "improving"]] = None
    window_days: Optional[int] = Field(None, description="look-back window for the trend / recent observations")
    has_open_defect: Optional[bool] = None
    min_severity: Optional[float] = None
    health_below: Optional[float] = None
    unseen_days_over: Optional[int] = None
    critical_within_days: Optional[int] = None
    has_open_work_order: Optional[bool] = None
    depends_on_flagged: Optional[bool] = Field(None, description="assets whose priority was raised by a dependency")
    sort: Literal["priority", "severity", "health", "days_to_critical", "unseen"] = "priority"
    limit: Optional[int] = None


def _num(tok: Optional[str]) -> Optional[int]:
    if tok is None:
        return None
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return WORDS_NUM.get(tok)


def _span_days(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern + r"\s*(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|twelve)?\s*(day|week|month|year)s?", text)
    if not m:
        return None
    n = _num(m.group(1)) if m.group(1) else 1
    return int(n * UNIT_DAYS[m.group(2)]) if n else None


def translate_rules(question: str) -> StructuredQuery:
    t = " " + re.sub(r"[^a-z0-9.%\s]", " ", question.lower()) + " "
    q = StructuredQuery()
    for at, words in ASSET_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            q.asset_types.append(at)
    for fam, words in DEFECT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            q.defect_families.append(fam)
    if re.search(r"getting worse|worsen|deteriorat|growing|grew|increasing|degrad|spreading", t):
        q.trend = "worsening"
    elif re.search(r"improv|getting better|shrinking|decreasing", t):
        q.trend = "improving"
    q.window_days = _span_days(t, r"(?:last|past|previous|in the last|over the last|within the last)")
    if q.window_days is None and re.search(r"\bthis (week|month|year)\b", t):
        q.window_days = UNIT_DAYS[re.search(r"\bthis (week|month|year)\b", t).group(1)]
    unseen = re.search(r"(?:not (?:been )?(?:inspected|seen|visited|checked)|unseen|uninspected|haven t been (?:seen|inspected|checked))"
                       r"(?:\s+(?:for|in))?\s+(?:over\s+|more than\s+)?(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*(day|week|month|year)s?", t)
    if unseen:
        q.unseen_days_over = int((_num(unseen.group(1)) or 1) * UNIT_DAYS[unseen.group(2)])
        q.window_days = None
    crit = re.search(r"critical (?:with)?in (?:the next )?(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*(day|week|month|year)s?", t) \
        or re.search(r"(?:become|becomes|go|goes|turn|turns|reach|reaches) critical (?:with)?in (?:the next )?(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|twelve)?\s*(day|week|month|year)s?", t)
    if crit:
        q.critical_within_days = int((_num(crit.group(1)) or 1) * UNIT_DAYS[crit.group(2)])
        if q.window_days == q.critical_within_days:
            q.window_days = None
    h = re.search(r"health (?:index )?(?:below|under|less than|lower than|<)\s*(\d+(?:\.\d+)?)", t)
    if h:
        q.health_below = float(h.group(1))
    s = re.search(r"severity (?:above|over|greater than|more than|>|at least)\s*(\d+(?:\.\d+)?)", t)
    if s:
        q.min_severity = float(s.group(1))
    elif re.search(r"\bcritical (?:assets|defects|ones)\b|\bare critical\b|\bin critical\b", t) and q.critical_within_days is None:
        q.min_severity = 0.8
    if re.search(r"open defect|unresolved|with defects|defective|have defects|has defects|(?:severe|worst|serious) defects", t):
        q.has_open_defect = True
    if re.search(r"work order|ticket|repair order", t):
        q.has_open_work_order = True
    if re.search(r"depend|downstream|upstream|because of|affected by", t):
        q.depends_on_flagged = True
    if re.search(r"worst health|lowest health|least healthy|unhealthiest", t):
        q.sort = "health"
    elif re.search(r"soonest|first .{0,20}to (?:fail|become critical|go critical)|closest to critical", t) or q.critical_within_days is not None:
        q.sort = "days_to_critical"
    elif re.search(r"longest|oldest inspection|least recently", t) or q.unseen_days_over is not None:
        q.sort = "unseen"
    elif re.search(r"most severe|highest severity|worst", t):
        q.sort = "severity"
    lim = re.search(r"\b(?:top|first|worst|best)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", t)
    if lim:
        q.limit = _num(lim.group(1))
    if q.defect_families and q.has_open_defect is None and q.trend is None:
        q.has_open_defect = True
    return q


class LlmTranslator:
    """Question → StructuredQuery with a small Claude model and schema-constrained output."""

    SYSTEM = ("Translate a question about an infrastructure-inspection memory into the given JSON query schema. "
              "Asset types: utility_pole, solar_panel, building, machinery. Defect families: crack, corrosion, soiling. "
              "Convert durations to days (week=7, month=30, year=365). Leave fields null/empty when the question does not "
              "mention them. Output only the query; do not answer the question.")

    def __init__(self, config: Dict[str, Any]):
        import anthropic

        c = get_nested(config, "nl_query", default={}) or {}
        self.model = c.get("llm_model", "claude-haiku-5-5")
        self.client = anthropic.Anthropic()

    def __call__(self, question: str) -> StructuredQuery:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": "low"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": question}],
            output_format=StructuredQuery,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"translation failed (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def translate(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    mode = get_nested(config, "nl_query", "translator", default="rules")
    if mode == "llm":
        try:
            return {"translator": "llm", "query": LlmTranslator(config)(question)}
        except Exception as exc:  # noqa: BLE001 - deterministic fallback
            logger.warning("LLM query translation unavailable (%s); using rules", exc)
    return {"translator": "rules", "query": translate_rules(question)}


# ── deterministic execution ───────────────────────────────────────────────
def _severity_slope(obs: List[Dict[str, Any]]) -> Optional[float]:
    if len(obs) < 2:
        return None
    t0 = parse_ts(obs[0]["observation_ts"])
    xs = [(parse_ts(o["observation_ts"]) - t0).total_seconds() / 86400.0 for o in obs]
    ys = [float(o["severity_score"]) for o in obs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else None


def execute(store, site_id: Optional[str], q: StructuredQuery, now=None) -> List[Dict[str, Any]]:
    now = now or utc_now()
    out = []
    defects = store.list_defects(site_id=site_id)
    instances = store.list_defect_instances(site_id=site_id)
    work = store.list_work_orders(site_id=site_id)
    for a in store.list_assets(site_id):
        aid = a["asset_id"]
        why: List[str] = []
        if q.asset_types and a["asset_type"] not in q.asset_types:
            continue
        open_d = [d for d in defects if d["asset_id"] == aid and d["status"] != "Resolved"]
        inst = [i for i in instances if i["asset_id"] == aid and i["status"] == "active"]
        if q.defect_families:
            open_d = [d for d in open_d if family(d["defect_type"]) in q.defect_families]
            inst = [i for i in inst if family(i["defect_type"]) in q.defect_families]
            if not open_d and not inst:
                continue
            why.append("has " + "/".join(q.defect_families))
        if q.has_open_defect and not open_d:
            continue
        if q.has_open_defect:
            why.append(f"{len(open_d)} open defect(s)")
        if q.min_severity is not None:
            sev = max([d["severity_score"] for d in open_d] + [0.0])
            if sev < q.min_severity:
                continue
            why.append(f"severity {sev:.2f}")
        if q.health_below is not None:
            if a.get("health_index") is None or a["health_index"] >= q.health_below:
                continue
            why.append(f"health {a['health_index']:.0f}")
        days_unseen = (now - parse_ts(a["last_checked_ts"])).total_seconds() / 86400.0
        if q.unseen_days_over is not None:
            if days_unseen <= q.unseen_days_over:
                continue
            why.append(f"unseen {days_unseen:.0f} days")
        dtc = [d for d in (days_until(i.get("predicted_critical_ts"), now) for i in inst) if d is not None]
        soonest = min(dtc) if dtc else None
        if q.critical_within_days is not None:
            if soonest is None or soonest > q.critical_within_days:
                continue
            why.append(f"critical in {max(0.0, soonest):.0f} days")
        if q.has_open_work_order and not any(w["asset_id"] == aid and w["status"] != "Verified" for w in work):
            continue
        if q.depends_on_flagged and not (a.get("revisit_components") or {}).get("graph_boost"):
            continue
        if q.trend is not None:
            obs = store.observations(aid)
            if q.window_days:
                # the window's observations plus the last one before it (the baseline the change is measured from)
                since = now - timedelta(days=q.window_days)
                before = [o for o in obs if parse_ts(o["observation_ts"]) < since]
                obs = before[-1:] + [o for o in obs if parse_ts(o["observation_ts"]) >= since]
            if q.defect_families:
                obs = [o for o in obs if not o["is_defect"] or family(o["defect_type"]) in q.defect_families]
            slope = _severity_slope(obs)
            growing = any(((i.get("forecast") or {}).get("growth_since_last_pct") or 0) > 0 for i in inst)
            if q.trend == "worsening" and not ((slope is not None and slope > 0) or growing):
                continue
            if q.trend == "improving" and not (slope is not None and slope < 0):
                continue
            why.append(f"{q.trend} (slope {slope or 0:+.3f}/day)")
        out.append({"asset_id": aid, "asset_type": a["asset_type"], "revisit_priority": a["revisit_priority_score"],
                    "health_index": a.get("health_index"), "severity": max([d["severity_score"] for d in open_d] + [0.0]),
                    "days_to_critical": soonest, "days_unseen": round(days_unseen, 1), "location": a["location"], "why": why})
    keys = {
        "priority": lambda r: -r["revisit_priority"],
        "severity": lambda r: -r["severity"],
        "health": lambda r: (r["health_index"] if r["health_index"] is not None else 101),
        "days_to_critical": lambda r: (r["days_to_critical"] if r["days_to_critical"] is not None else 1e9),
        "unseen": lambda r: -r["days_unseen"],
    }
    out.sort(key=keys[q.sort])
    return out[: q.limit] if q.limit else out


def ask(store, site_id: Optional[str], question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    tr = translate(question, config)
    results = execute(store, site_id, tr["query"])
    return {"question": question, "translator": tr["translator"], "query": tr["query"].model_dump(), "results": results, "count": len(results)}
