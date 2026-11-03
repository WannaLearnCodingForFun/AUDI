"""5.2 natural-language queries: translation benchmark (exact match) and deterministic execution."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from memory.store import LocalStore
from ml.detector import FullFrameDetector
from ml.nl_query import StructuredQuery, ask, execute, translate, translate_rules
from ml.schemas import utc_now
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import add_crack, make_panel

BENCH = json.loads((Path(__file__).resolve().parent.parent / "evaluation" / "nl_query_benchmark.json").read_text())


def test_benchmark_has_30_questions_and_rules_translate_all_exactly():
    assert len(BENCH) == 30
    hits = [translate_rules(b["q"]).model_dump() == StructuredQuery(**b["expect"]).model_dump() for b in BENCH]
    assert sum(hits) / len(hits) == 1.0, [b["q"] for b, h in zip(BENCH, hits) if not h]


def test_llm_translator_falls_back_to_rules_without_credentials(config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config["nl_query"]["translator"] = "llm"

    class Boom:
        def __init__(self, *_a, **_k):
            raise RuntimeError("no credentials")

    import ml.nl_query as nq
    monkeypatch.setattr(nq, "LlmTranslator", Boom)
    out = translate("poles with rust", config)
    assert out["translator"] == "rules" and out["query"].asset_types == ["utility_pole"]


def test_execution_filters_and_ranks(config, embedder, classifier):
    config["geo"]["identity_mode"] = "appearance"
    store = LocalStore(config)
    store.ensure_site("site_q", "Q")
    e = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    t0 = utc_now() - timedelta(days=40)
    ids = []
    for k, seed in enumerate((151, 152, 153)):
        sid = store.create_session("site_q", "op", "offline", {})["session_id"]
        ids.append(e.observe(make_panel(seed), "solar_panel", sid, timestamp=(t0 + timedelta(minutes=k)).isoformat()).asset_id)
        store.update_session(sid, status="ended", ended_ts=utc_now().isoformat())
    sid = store.create_session("site_q", "op", "offline", {})["session_id"]
    worse = e.observe(add_crack(make_panel(151), 10), "solar_panel", sid, timestamp=(utc_now() - timedelta(days=1)).isoformat())
    assert worse.asset_id == ids[0] and worse.condition["is_defect"]
    store.recompute_revisit("site_q")
    res = ask(store, "site_q", "panels with cracks getting worse in the last month", config)
    assert res["translator"] == "rules" and [r["asset_id"] for r in res["results"]] == [ids[0]]
    stale = execute(store, "site_q", translate_rules("assets not inspected for 30 days"))
    assert {r["asset_id"] for r in stale} == {ids[1], ids[2]}
    assert execute(store, "site_q", translate_rules("top 1 assets with the worst health"))[0]["asset_id"] == ids[0]
    assert execute(store, "site_q", translate_rules("poles with rust")) == []
