"""A failed job's terminal ``status`` is written LAST, after the clean-up
(台帳 §1-279).

``status`` is both the client's "this job is over" signal and the input of the
admission guards (``create_if_idle`` / ``create_chain_if_idle`` / ``has_active``)
and DELETE's terminal test. So on a failure the order must be: clean-up (the
OOM worker unload, or the READY/UNLOADED state) -> ``error`` ->
``completed_at`` -> ``status = failed``. Before the fix ``status=failed`` came
first, which (a) let a poller see ``failed`` with ``error: null`` for the
length of the unload, and (b) reopened the single-job guard while the worker
was still being torn down, so a second POST was accepted (202).

These tests use the dedicated-thread dispatch (``thread_dispatch_client``,
backend "auto" -> mock at run time) so the job really runs concurrently with
the test's polling. Synchronisation is by ``threading.Event``, not by timing.
"""

from __future__ import annotations

import threading
import time

import pytest

from tests.test_smoke import thread_dispatch_client  # noqa: F401  (pytest fixture)

POLL_INTERVAL_S = 0.01
TIMEOUT_S = 15.0  # upper bound for every wait; the mock finishes in well under it
UNLOAD_SLEEP_S = 1.2

SINGLE = {
    "prompt": "terminal-order", "width": 384, "height": 256, "num_frames": 17,
    "num_inference_steps": 8, "guidance_scale": 1.0, "pipeline": "distilled",
}
CHAIN = {
    "prompt": "terminal-order chain", "width": 384, "height": 256, "frame_rate": 24.0,
    "num_inference_steps": 8, "guidance_scale": 1.0, "seed": 1,
    "pipeline": "distilled", "overlap_frames": 2, "overlap_strength": 0.5,
    "clips": [{"num_frames": 25}, {"num_frames": 25}],
}
OOM_TEXT = "CUDA out of memory. Tried to allocate 2.00 GiB (injected by test)"
TERMINAL = ("completed", "failed", "cancelled")


def _assert_mock_backend(pm) -> None:
    """Sentry, run BEFORE any job is sent: this file must never reach the
    real worker / GPU. "auto" resolves through ``_real_available()`` (a pure
    file probe, no spawn); if it says the real stack is available, stop here.
    If a backend instance already exists, it must be the mock one."""
    from services.engines.ltx.adapter import _MockBackend

    assert pm.runner._real_available() is False, (
        "thread_dispatch_client would auto-select the REAL backend on this "
        "machine; refusing to send any job (no GPU / real worker allowed)"
    )
    backend = getattr(pm.runner, "_backend", None)
    assert backend is None or isinstance(backend, _MockBackend), (
        f"backend is {type(backend).__name__}, not the mock; refusing to continue"
    )


def _inject_once(pm, attr: str, exc: BaseException) -> None:
    """Replace ``pm.runner.<attr>`` so the FIRST call raises ``exc`` and every
    later call delegates to the original (a second job must not fail again)."""
    orig = getattr(pm.runner, attr)
    calls = {"n": 0}
    lock = threading.Lock()

    def fake(*a, **k):
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        if first:
            raise exc
        return orig(*a, **k)

    setattr(pm.runner, attr, fake)


def _slow_unload(pm):
    """Replace ``pm.runner.unload`` with one that signals its start, sleeps,
    runs the original, then signals its end. Returns (started, finished)."""
    orig = pm.runner.unload
    started = threading.Event()
    finished = threading.Event()

    def fake(*a, **k):
        started.set()
        time.sleep(UNLOAD_SLEEP_S)
        try:
            return orig(*a, **k)
        finally:
            finished.set()

    pm.runner.unload = fake  # type: ignore[method-assign]
    return started, finished


class _Poller(threading.Thread):
    """Poll GET /jobs/{id} every 10 ms until the job is terminal with its
    error readable; record every sample showing ``failed`` with ``error``
    None (the §1-279 window)."""

    def __init__(self, client, job_id: str):
        super().__init__(daemon=True)
        self.client = client
        self.job_id = job_id
        self.t0 = time.monotonic()
        self.bad: list[float] = []  # seconds since start, per bad sample
        self.samples = 0
        self.last: dict | None = None
        self.timed_out = False
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            if time.monotonic() - self.t0 > TIMEOUT_S:
                self.timed_out = True
                return
            st = self.client.get(f"/api/v1/jobs/{self.job_id}").json()
            self.samples += 1
            self.last = st
            status = st.get("status")
            if status == "failed" and st.get("error") is None:
                self.bad.append(round(time.monotonic() - self.t0, 3))
            elif status in TERMINAL:
                return
            time.sleep(POLL_INTERVAL_S)


def _wait_terminal(client, job_id: str) -> dict:
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        st = client.get(f"/api/v1/jobs/{job_id}").json()
        if st.get("status") in TERMINAL:
            return st
        if time.monotonic() > deadline:
            pytest.fail(f"job {job_id} not terminal after {TIMEOUT_S}s: {st}")
        time.sleep(POLL_INTERVAL_S)


def _run_oom_case(client, endpoint: str, payload: dict, runner_attr: str) -> None:
    pm = client.app_context.pipeline_manager
    _assert_mock_backend(pm)
    started, finished = _slow_unload(pm)
    _inject_once(pm, runner_attr, RuntimeError(OOM_TEXT))

    r1 = client.post(endpoint, json=payload)
    assert r1.status_code == 202, r1.text
    jid1 = r1.json()["job_id"]
    poller = _Poller(client, jid1)
    poller.start()

    assert started.wait(TIMEOUT_S), "the clean-up unload never started"
    # While the worker is still being unloaded, a second job must be refused.
    r2 = client.post(endpoint, json=payload)
    job2 = None
    if r2.status_code == 202:  # pre-fix: wait for it too before tearing down
        st2 = _wait_terminal(client, r2.json()["job_id"])
        job2 = {"status": st2.get("status"), "error": st2.get("error")}

    assert finished.wait(TIMEOUT_S), "the clean-up unload never finished"
    poller.join(TIMEOUT_S)
    poller.stop.set()
    st1 = _wait_terminal(client, jid1)
    final_state = pm.state

    diag = (
        f"samples={poller.samples}; failed+error-null seen {len(poller.bad)}x "
        f"at t={poller.bad[:10]}...; POST#2 during unload -> {r2.status_code} "
        f"{r2.text[:160]}; job2={job2}; job1={st1.get('status')} "
        f"error={str(st1.get('error'))[:60]!r}; pm.state={final_state}"
    )
    print(diag)
    assert not poller.timed_out, f"poller timed out; last sample {poller.last}"
    assert poller.bad == [], diag
    assert r2.status_code == 409, diag
    assert "JOB_BUSY" in r2.text, diag
    assert st1["status"] == "failed", diag
    assert st1["error"] is not None and st1["error"].startswith("GPU_OOM:"), diag
    assert st1.get("completed_at"), diag
    assert final_state == pm.STATE_UNLOADED, diag


def test_T0_fixture_selects_mock_backend(thread_dispatch_client):
    """T0: the sentry alone (no job is sent). The probe says mock, and an
    explicit load builds the mock backend instance (never a worker)."""
    from services.engines.ltx.adapter import _MockBackend

    pm = thread_dispatch_client.app_context.pipeline_manager
    _assert_mock_backend(pm)
    pm.load()
    assert isinstance(pm.runner._backend, _MockBackend), type(pm.runner._backend)
    assert pm.runner.loaded


def test_single_oom_failed_is_written_after_unload(thread_dispatch_client):
    """T1: single job, OOM. No sample shows failed with error null, a POST
    during the unload is 409 JOB_BUSY, and the job ends failed / GPU_OOM /
    completed_at set / pipeline unloaded."""
    _run_oom_case(thread_dispatch_client, "/api/v1/generate", SINGLE, "generate")


def test_chain_oom_failed_is_written_after_unload(thread_dispatch_client):
    """T2: the same contract for a chain job (run_chain_job)."""
    _run_oom_case(thread_dispatch_client, "/api/v1/generate/chain", CHAIN, "generate_chain")


def test_single_non_oom_failure_has_error_when_failed(thread_dispatch_client):
    """T3: non-OOM failure. When failed is first observed, error is already
    GENERATION_FAILED and the pipeline state is terminal (ready/unloaded)."""
    client = thread_dispatch_client
    pm = client.app_context.pipeline_manager
    _assert_mock_backend(pm)
    _inject_once(pm, "generate", RuntimeError("boom"))

    r = client.post("/api/v1/generate", json=SINGLE)
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        st = client.get(f"/api/v1/jobs/{jid}").json()
        if st.get("status") == "failed":
            state_at_failed = pm.state
            break
        assert st.get("status") not in ("completed", "cancelled"), st
        if time.monotonic() > deadline:
            pytest.fail(f"job {jid} not failed after {TIMEOUT_S}s: {st}")
        time.sleep(POLL_INTERVAL_S)

    assert st["error"] is not None and st["error"].startswith("GENERATION_FAILED:"), st
    assert state_at_failed in (pm.STATE_READY, pm.STATE_UNLOADED), state_at_failed
