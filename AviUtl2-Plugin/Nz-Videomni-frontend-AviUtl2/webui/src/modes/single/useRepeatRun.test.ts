import { act, renderHook } from "@testing-library/react";
import { useCallback } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { JobResponse, JobStatus } from "../../api/types";
import type { SubmitState } from "./useGeneration";
import { __resetRepeatCountForTests, getRepeatCountText } from "./repeatRun";
import { useRepeatRun, useRepeatRunDriver } from "./useRepeatRun";

// §1-80 Repeat count: behaviour of the run state + completion driver, driven
// by `rerender` with hand-made `jobs` / `submitPhase` (no providers needed —
// the driver takes `jobs` as an argument). `noteSubmitted` is called by hand
// where the screen's `onSubmitted` would call it.

interface HarnessProps {
  jobs: readonly JobResponse[];
  submitPhase: SubmitState["phase"];
  canSubmit: boolean;
  fingerprint: string;
  send: () => void;
  onSettingsChanged: () => void;
}

function useHarness(props: HarnessProps) {
  const run = useRepeatRun("single");
  const { fingerprint } = props;
  const getFingerprint = useCallback(() => fingerprint, [fingerprint]);
  useRepeatRunDriver(run, {
    jobs: props.jobs,
    submitPhase: props.submitPhase,
    canSubmit: props.canSubmit,
    getFingerprint,
    send: props.send,
    onSettingsChanged: props.onSettingsChanged,
  });
  return run;
}

function job(id: string, status: JobStatus): JobResponse {
  return { job_id: id, status } as JobResponse;
}

function setup(overrides: Partial<HarnessProps> = {}) {
  const send = vi.fn();
  const onSettingsChanged = vi.fn();
  let props: HarnessProps = {
    jobs: [],
    submitPhase: "idle",
    canSubmit: true,
    fingerprint: "A",
    send,
    onSettingsChanged,
    ...overrides,
  };
  const hook = renderHook((p: HarnessProps) => useHarness(p), { initialProps: props });
  const update = (patch: Partial<HarnessProps>) => {
    props = { ...props, ...patch };
    hook.rerender(props);
  };
  const start = (count: string) => {
    act(() => hook.result.current.setCountText(count));
    act(() => hook.result.current.start({ send, fingerprint: props.fingerprint }));
  };
  return { ...hook, send, onSettingsChanged, update, start };
}

beforeEach(() => {
  __resetRepeatCountForTests();
});
afterEach(() => {
  __resetRepeatCountForTests();
});

describe("useRepeatRun / useRepeatRunDriver", () => {
  it("N=1: sends once, never auto-sends, and never sets a job to wait on", () => {
    const h = setup();
    h.start("1");
    expect(h.send).toHaveBeenCalledTimes(1);
    expect(h.result.current.remaining).toBe(0);
    act(() => h.result.current.noteSubmitted("j1"));
    expect(h.result.current.awaitingJobIdRef.current).toBeNull();
    h.update({ jobs: [job("j1", "running")] });
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(1);
  });

  it("N=3: two completions send three in total; the awaited job is cleared and the same jobs do not resend", () => {
    const h = setup();
    h.start("3");
    expect(h.send).toHaveBeenCalledTimes(1);
    expect(h.result.current.remaining).toBe(2);

    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", "running")] });
    expect(h.send).toHaveBeenCalledTimes(1);
    const completed1 = [job("j1", "completed")];
    h.update({ jobs: completed1 });
    expect(h.send).toHaveBeenCalledTimes(2);
    expect(h.result.current.remaining).toBe(1);
    // The awaited job was cleared, so re-rendering with the same jobs sends nothing more.
    h.update({ jobs: [...completed1] });
    expect(h.send).toHaveBeenCalledTimes(2);

    act(() => h.result.current.noteSubmitted("j2"));
    h.update({ jobs: [job("j1", "completed"), job("j2", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(3);
    expect(h.result.current.remaining).toBe(0);

    // The last run is not awaited.
    act(() => h.result.current.noteSubmitted("j3"));
    expect(h.result.current.awaitingJobIdRef.current).toBeNull();
    h.update({ jobs: [job("j1", "completed"), job("j2", "completed"), job("j3", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(3);
  });

  it("waits while the submit is still 'submitting' and sends once it is 'idle'", () => {
    const h = setup();
    h.start("2");
    h.update({ submitPhase: "submitting" });
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(1);
    h.update({ submitPhase: "idle" });
    expect(h.send).toHaveBeenCalledTimes(2);
  });

  it("does not send after stop()", () => {
    const h = setup();
    h.start("3");
    act(() => h.result.current.noteSubmitted("j1"));
    act(() => h.result.current.stop());
    expect(h.result.current.remaining).toBe(0);
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(1);
  });

  it.each(["failed", "cancelled"] as const)("ends the run when the job is %s", (status) => {
    const h = setup();
    h.start("3");
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", status)] });
    expect(h.send).toHaveBeenCalledTimes(1);
    expect(h.result.current.remaining).toBe(0);
    expect(h.result.current.awaitingJobIdRef.current).toBeNull();
  });

  it("a rejected submit (onFailed → stop) ends the run", () => {
    const h = setup();
    h.start("3");
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(2);
    // The auto-sent submit is rejected: the screen's onFailed calls stop().
    act(() => h.result.current.stop());
    expect(h.result.current.remaining).toBe(0);
    h.update({ jobs: [job("j1", "completed"), job("j2", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(2);
  });

  it("ends the run when the panel cannot submit at completion time", () => {
    const h = setup();
    h.start("3");
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ canSubmit: false });
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(1);
    expect(h.result.current.remaining).toBe(0);
    h.update({ canSubmit: true });
    expect(h.send).toHaveBeenCalledTimes(1);
  });

  it("settings-changed toast: once per gap, again after the next submission, never when unchanged", () => {
    const h = setup();
    h.start("3");
    h.update({ fingerprint: "A" });
    expect(h.onSettingsChanged).not.toHaveBeenCalled();

    h.update({ fingerprint: "B" });
    expect(h.onSettingsChanged).toHaveBeenCalledTimes(1);
    h.update({ fingerprint: "C" });
    expect(h.onSettingsChanged).toHaveBeenCalledTimes(1);

    // Next automatic submission takes "C" as its baseline.
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.send).toHaveBeenCalledTimes(2);
    expect(h.result.current.remaining).toBe(1);
    expect(h.onSettingsChanged).toHaveBeenCalledTimes(1);

    h.update({ fingerprint: "D" });
    expect(h.onSettingsChanged).toHaveBeenCalledTimes(2);
  });

  it("settings-changed toast never fires with nothing remaining", () => {
    const h = setup();
    h.start("2");
    act(() => h.result.current.noteSubmitted("j1"));
    h.update({ jobs: [job("j1", "completed")] });
    expect(h.result.current.remaining).toBe(0);
    h.update({ fingerprint: "Z" });
    expect(h.onSettingsChanged).not.toHaveBeenCalled();

    // N=1 never fires either.
    const h1 = setup();
    h1.start("1");
    h1.update({ fingerprint: "Z" });
    expect(h1.onSettingsChanged).not.toHaveBeenCalled();
  });

  it("start() clamps the field text and writes the clamped value back", () => {
    const h = setup();
    h.start("150");
    expect(h.result.current.remaining).toBe(98);
    expect(h.result.current.countText).toBe("99");
    act(() => h.result.current.stop());
    h.start("");
    expect(h.result.current.remaining).toBe(0);
    expect(h.result.current.countText).toBe("1");
  });

  it("commitCountText (blur) clamps the field text", () => {
    const h = setup();
    act(() => h.result.current.setCountText("0"));
    expect(h.result.current.countText).toBe("0");
    act(() => h.result.current.commitCountText());
    expect(h.result.current.countText).toBe("1");
    act(() => h.result.current.setCountText("4.7"));
    act(() => h.result.current.commitCountText());
    expect(h.result.current.countText).toBe("4");
  });

  it("a remount ends the run but keeps the field text", () => {
    const h = setup();
    h.start("5");
    expect(h.result.current.remaining).toBe(4);
    act(() => h.result.current.noteSubmitted("j1"));
    h.unmount();

    const again = setup();
    expect(again.result.current.remaining).toBe(0);
    expect(again.result.current.awaitingJobIdRef.current).toBeNull();
    expect(again.result.current.countText).toBe("5");
    expect(getRepeatCountText("single")).toBe("5");
    again.update({ jobs: [job("j1", "completed")] });
    expect(again.send).not.toHaveBeenCalled();
  });
});
