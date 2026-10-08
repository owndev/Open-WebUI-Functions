"""
Filters suite: filters/{google_search_tool,vertex_ai_search_tool,time_token_tracker}.py
in front of the probe pipe (tests/e2e/probe/probe_pipe.py), which reports what
reached the pipe, so filter -> pipe coupling is checked without a provider.

Groups (``--only filters.<group>``)
  model     filters attached per model (meta.filterIds): feature -> metadata
            mapping, API path without ``features``, tracker outlet on the API path,
            browser path tracker status, background task (no event emitter)
  global    the same filters switched to global
"""

import json

from harness import Suite, known, short
from harness.config import VERTEX_RAG_STORE

PROBE_FID = "e2e_probe"
PROBE_MODEL = f"{PROBE_FID}.echo"
FILTERS = {
    "google_search_tool": "filters/google_search_tool.py",
    "vertex_ai_search_tool": "filters/vertex_ai_search_tool.py",
    "time_token_tracker": "filters/time_token_tracker.py",
}


def probe_report(text) -> dict:
    """Decode the probe pipe's ``PROBE:{json}`` answer (also inside task JSON)."""
    if not isinstance(text, str):
        return {}
    if text.startswith("{"):
        try:
            text = json.loads(text).get("probe", "")
        except ValueError:
            return {}
    if text.startswith("PROBE:"):
        try:
            return json.loads(text[len("PROBE:") :])
        except ValueError:
            return {}
    return {}


async def run(t: Suite) -> None:
    if not await t.install(PROBE_FID, "probe", "E2E Probe", "load.probe"):
        return
    for fid, path in FILTERS.items():
        if not await t.install(fid, path, fid, f"load.{fid}"):
            return
        info = await t.owui.function(fid) or {}
        t.check(
            f"load.{fid}.type",
            f"{fid} is registered as a filter",
            info.get("type") == "filter",
            f"type={info.get('type')}",
        )
    try:
        if t.selected("model"):
            await per_model(t)
        if t.selected("global"):
            await global_mode(t)
    finally:
        for fid in FILTERS:
            await t.owui.set_global(fid, False)
        await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", [])
    t.scan_log()


async def feature_mapping(t: Suite, tag: str) -> None:
    """features.* in the request -> what the pipe sees in __metadata__."""
    r = await t.owui.chat(PROBE_MODEL, "probe", features={"web_search": True})
    rep = probe_report(r.content)
    features = rep.get("metadata_features") or {}
    t.check(
        f"{tag}.web-search-on",
        "features.web_search=true -> __metadata__.features.google_search_tool",
        r.status == 200
        and features.get("google_search_tool") is True
        and "web_search" not in features,
        f"HTTP {r.status} metadata_features={features}",
    )
    r = await t.owui.chat(PROBE_MODEL, "probe", features={"web_search": False})
    rep = probe_report(r.content)
    t.check(
        f"{tag}.web-search-off",
        "features.web_search=false -> no google_search_tool flag",
        r.status == 200
        and bool(rep)
        and not (rep.get("metadata_features") or {}).get("google_search_tool"),
        f"HTTP {r.status} report={short(rep, 300)}",
    )
    r = await t.owui.chat(
        PROBE_MODEL, "probe", features={"web_search": False, "vertex_ai_search": True}
    )
    rep = probe_report(r.content)
    features = rep.get("metadata_features") or {}
    store = (rep.get("metadata_params") or {}).get("vertex_rag_store")
    t.check(
        f"{tag}.vertex",
        "features.vertex_ai_search -> metadata flag + VERTEX_AI_RAG_STORE env in params",
        r.status == 200
        and features.get("vertex_ai_search") is True
        and store == (VERTEX_RAG_STORE or None),
        f"HTTP {r.status} features={features} vertex_rag_store={store!r}",
    )


async def per_model(t: Suite) -> None:
    filter_ids = list(FILTERS)
    status = await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", filter_ids)
    seen = await t.owui.model_filter_ids(PROBE_MODEL)
    t.check(
        "model.attach",
        "filters attached per model (meta.filterIds, visible after models refresh)",
        status == 200 and seen == filter_ids,
        f"HTTP {status} filterIds={seen}",
    )
    await feature_mapping(t, "model")

    mark = t.mark()
    r = await t.owui.chat(PROBE_MODEL, "probe without features")
    await t.log.settle()
    t.check(
        "model.no-features",
        "API request without a 'features' key passes the filters",
        r.status == 200 and bool(probe_report(r.content)),
        r.brief(),
        known=known.FILTER_SEARCH_KEYERROR,
        since=mark,
    )

    for stream in (False, True):
        mark = t.mark()
        r = await t.owui.chat(
            PROBE_MODEL, "tracker", stream=stream, features={"web_search": False}
        )
        await t.log.settle(1.5)
        errors = t.log.errors(mark)
        t.check(
            f"model.tracker-api.{'stream' if stream else 'nonstream'}",
            f"time_token_tracker outlet on the API path (stream={stream}) runs "
            "without errors",
            r.status == 200 and not errors,
            f"{r.brief()} log_errors={errors[:2]}",
            known=known.FILTER_TRACKER_NO_EMITTER,
            since=mark,
        )

    async with t.browser() as b:
        c = await b.chat(PROBE_MODEL, "tracker in the browser", stream=True)
    rep = probe_report(c.content)
    tracker = [d for d in c.status_descriptions if d and "Req:" in d]
    t.check(
        "model.browser",
        "browser path: pipe has an event emitter and chat/message/session ids",
        c.done
        and rep.get("has_event_emitter") is True
        and all((rep.get("metadata_ids") or {}).values()),
        f"report={short(rep, 300)}",
    )
    t.check(
        "model.tracker-browser",
        "browser path: time_token_tracker status saved in statusHistory",
        bool(tracker),
        f"statusHistory={c.status_descriptions}",
    )

    status, answer, raw = await t.owui.title_task(
        PROBE_MODEL, [{"role": "user", "content": "Hi"}]
    )
    rep = probe_report(answer)
    t.check(
        "model.task",
        "background title task: __task__ set, no __event_emitter__",
        status == 200
        and rep.get("task") == "title_generation"
        and rep.get("has_event_emitter") is False,
        f"HTTP {status} report={short(rep, 300)} raw={short(raw, 200)}",
    )


async def global_mode(t: Suite) -> None:
    await t.owui.upsert_model(PROBE_MODEL, "E2E Probe", [])
    flags = {fid: await t.owui.set_global(fid, True) for fid in FILTERS}
    await t.owui.models(refresh=True)
    t.check(
        "global.switch",
        "filters switched to global (no per-model filterIds)",
        all(flags.values()),
        f"is_global={flags}",
    )
    await feature_mapping(t, "global")
