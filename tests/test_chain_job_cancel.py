"""Cancelling a chain job ends it ``cancelled``, the same as a single job.

Both ``run_job`` and ``run_chain_job`` treat cancel as best-effort: the worker
op is not interrupted, but a cancel requested by the time it finishes turns the
job ``cancelled`` (no result). A cancel that lands while the job is still
queued is caught by ``JobStore.start_job`` and the job never dispatches.
"""

from __future__ import annotations

from api.models import GenerateChainRequest, JobStatus

CHAIN_BASE = {
    "prompt": "a town square", "width": 384, "height": 256, "frame_rate": 24.0,
    "num_inference_steps": 8, "guidance_scale": 1.0, "seed": 1,
    "pipeline": "distilled", "overlap_frames": 2, "overlap_strength": 0.5,
}
CHAIN_CLIPS = [{"num_frames": 25}, {"num_frames": 25}]


def _mark_all_cancel(pm) -> None:
    # What DELETE /jobs/{id} does to a running job: set the best-effort flag.
    for rec in pm.job_store.list():
        rec.cancel_requested = True


def test_chain_cancel_after_dispatch_ends_cancelled(client):
    pm = client.app_context.pipeline_manager
    orig = pm.runner.generate_chain

    def gen(*a, **k):
        _mark_all_cancel(pm)  # DELETE arriving mid-run
        return orig(*a, **k)

    pm.runner.generate_chain = gen
    r = client.post("/api/v1/generate/chain", json={**CHAIN_BASE, "clips": CHAIN_CLIPS})
    assert r.status_code in (200, 202), r.text
    jid = r.json()["job_id"]
    st = client.get(f"/api/v1/jobs/{jid}").json()
    assert st["status"] == "cancelled"
    assert st.get("result") is None
    assert pm.state == pm.STATE_READY


def test_single_cancel_after_dispatch_ends_cancelled(client):
    pm = client.app_context.pipeline_manager
    orig = pm.runner.generate

    def gen(*a, **k):
        _mark_all_cancel(pm)
        return orig(*a, **k)

    pm.runner.generate = gen
    r = client.post(
        "/api/v1/generate",
        json={"prompt": "x", "width": 384, "height": 256, "num_frames": 25},
    )
    assert r.status_code in (200, 202), r.text
    jid = r.json()["job_id"]
    st = client.get(f"/api/v1/jobs/{jid}").json()
    assert st["status"] == "cancelled"
    assert st.get("result") is None


def test_chain_cancel_before_dispatch_never_runs(client):
    # Mirrors tests/test_smoke.py::test_run_job_honours_cancel_before_dispatch.
    ctx = client.app_context
    pm = ctx.pipeline_manager
    calls = []
    pm.runner.generate_chain = lambda *a, **k: calls.append(1)

    rec = ctx.job_store.create_chain_if_idle(
        GenerateChainRequest(**CHAIN_BASE, clips=CHAIN_CLIPS)
    )
    assert rec is not None
    rec.cancel_requested = True

    pm.run_chain_job(rec)

    assert rec.status == JobStatus.cancelled
    assert rec.started_at is None       # never entered the running path
    assert rec.completed_at is not None
    assert rec.result is None
    assert calls == []
