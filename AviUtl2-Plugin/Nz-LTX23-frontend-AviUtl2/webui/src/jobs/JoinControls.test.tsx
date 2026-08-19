import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { createApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import type { JobResponse } from "../api/types";
import { LanguageProvider } from "../i18n/LanguageContext";
import { JoinControls } from "./JoinControls";

/**
 * I6: `JoinControls` is now control-only — the trim/crossfade inputs, the
 * Join/Unjoin buttons, the fps-mismatch note and the error state. The joined
 * `<video>` preview and the 🎞 insert of the joined clip live in `JobCard`
 * (see `JobCard.test.tsx`), which drives them off `onJoinedStateChange` /
 * `unjoinToken` (asserted here).
 */
function renderWithLanguage(ui: ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

/** Drives the mock bridge through a real V2V chain job to completion, so
 * these tests exercise `JoinControls` against a realistic `JobResponse`
 * (echoed `source_video`, `status: "completed"`) rather than a hand-rolled
 * fixture that might drift from what `mockBridge.ts` actually produces. */
async function completedV2VJob(
  mockBridge: ReturnType<typeof createMockBridge>,
  frameRate = 24,
): Promise<JobResponse> {
  const apiClient = createApiClient(mockBridge);
  const accepted = await apiClient.generateChain({
    prompt: "continue this video",
    width: 512,
    height: 320,
    frame_rate: frameRate,
    seed: -1,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [{ num_frames: 49 }],
    source_video: { video_id: "mock-video-1", context_frames: 25 },
  });

  let job: JobResponse = await apiClient.getJob(accepted.job_id);
  for (let i = 0; i < 20 && job.status !== "completed"; i += 1) {
    job = await apiClient.getJob(accepted.job_id);
  }
  if (job.status !== "completed") throw new Error("mock V2V job never completed");
  return job;
}

describe("JoinControls", () => {
  it("renders nothing for a non-V2V completed job", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const apiClient = createApiClient(mockBridge);
    const accepted = await apiClient.generate({
      prompt: "a cat",
      width: 384,
      height: 256,
      num_frames: 17,
      frame_rate: 24,
      seed: 1,
    });
    let job = await apiClient.getJob(accepted.job_id);
    for (let i = 0; i < 10 && job.status !== "completed"; i += 1) job = await apiClient.getJob(accepted.job_id);

    const { container } = renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for a non-completed V2V job", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    const apiClient = createApiClient(mockBridge);
    const accepted = await apiClient.generateChain({
      prompt: "continue this video",
      width: 512,
      height: 320,
      frame_rate: 24,
      seed: -1,
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: [{ num_frames: 49 }],
      source_video: { video_id: "mock-video-1", context_frames: 25 },
    });
    const job = await apiClient.getJob(accepted.job_id); // still queued/running

    const { container } = renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("idle -> joining -> joined: the Join button gives way to Unjoin", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);
    const user = userEvent.setup();

    renderWithLanguage(<JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />);

    await user.click(screen.getByRole("button", { name: /join with source/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^unjoin$/i })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^join with source$/i })).not.toBeInTheDocument();
    // The joined preview + 🎞 insert are JobCard's job now — not rendered here.
    expect(document.querySelector("video")).toBeNull();
    expect(screen.queryByRole("button", { name: /insert joined video/i })).not.toBeInTheDocument();
  });

  it("onJoinedStateChange reports (true, trim info) after a successful join and (false, null) on Unjoin", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);
    const user = userEvent.setup();
    const onJoinedStateChange = vi.fn();

    renderWithLanguage(
      <JoinControls
        job={job}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
        onJoinedStateChange={onJoinedStateChange}
      />,
    );

    // Default trim 120f @24fps = 5.0s -> mock trims (12 - 5) = 7.0s from the head.
    await user.click(screen.getByRole("button", { name: /join with source/i }));
    await waitFor(() => {
      expect(onJoinedStateChange).toHaveBeenCalledWith(true, { trimmedSourceSeconds: 7.0 });
    });

    onJoinedStateChange.mockClear();
    await user.click(screen.getByRole("button", { name: /^unjoin$/i }));
    expect(onJoinedStateChange).toHaveBeenCalledWith(false, null);
  });

  it("a join failure surfaces an error with a retry button", async () => {
    // No fixture in mockBridge.ts actually fails POST /jobs/{id}/join (it's a
    // fixed 200 for any known job per the task brief) — simulate the
    // transport-level failure path instead via an apiClient whose bridge
    // rejects the call, proving JoinControls' error branch renders and offers
    // a retry.
    const workingBridge = createMockBridge({ delayMs: 0 });
    const job = await completedV2VJob(workingBridge);

    const failingBridge = createMockBridge({ delayMs: 0, backendUnreachable: true });
    const failingApiClient = createApiClient(failingBridge);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={failingApiClient} nativeBridge={failingBridge} />,
    );

    await user.click(screen.getByRole("button", { name: /join with source/i }));

    await waitFor(() => {
      expect(screen.getByText(/BACKEND_UNREACHABLE/)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /retry join/i })).toBeInTheDocument();
  });

  it("I5 ①: mounts straight into the joined view when job.joined is true (Unjoin shown, trim unknown)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);
    const onJoinedStateChange = vi.fn();

    renderWithLanguage(
      <JoinControls
        job={{ ...job, joined: true }}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
        onJoinedStateChange={onJoinedStateChange}
      />,
    );

    // No click needed — the joined UI is restored from the server flag.
    expect(screen.getByRole("button", { name: /^unjoin$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^join with source$/i })).not.toBeInTheDocument();
    // A mount-restored join has no trim info (no join was run this session), so
    // the joined insert will degrade to position-less.
    expect(onJoinedStateChange).toHaveBeenCalledWith(true, null);
  });

  it("I5 ②: Unjoin returns to idle without calling the server", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const joinSpy = vi.spyOn(apiClient, "joinJob");
    const job = await completedV2VJob(mockBridge);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls
        job={{ ...job, joined: true }}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^unjoin$/i }));

    expect(screen.getByRole("button", { name: /^join with source$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^unjoin$/i })).not.toBeInTheDocument();
    expect(joinSpy).not.toHaveBeenCalled(); // Unjoin is a local toggle only
  });

  it("I5 ③: a joined:true poll after Unjoin does not rewind back to joined", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);
    const user = userEvent.setup();

    const { rerender } = renderWithLanguage(
      <JoinControls
        job={{ ...job, joined: true }}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^unjoin$/i }));
    expect(screen.getByRole("button", { name: /^join with source$/i })).toBeInTheDocument();

    // Polling delivers a fresh JobResponse still reporting joined:true — the
    // one-way follow must NOT re-assert the joined view over the local Unjoin.
    rerender(
      <LanguageProvider>
        <JoinControls
          job={{ ...job, joined: true }}
          baseUrl="http://127.0.0.1:18620"
          apiClient={apiClient}
          nativeBridge={mockBridge}
        />
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: /^join with source$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^unjoin$/i })).not.toBeInTheDocument();
  });

  it("I5 ④: a joined true->false poll collapses the joined view to idle", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);

    const { rerender } = renderWithLanguage(
      <JoinControls
        job={{ ...job, joined: true }}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
      />,
    );
    expect(screen.getByRole("button", { name: /^unjoin$/i })).toBeInTheDocument();

    // joined.mp4 vanished (deleted in Explorer) — polling now reports false.
    rerender(
      <LanguageProvider>
        <JoinControls
          job={{ ...job, joined: false }}
          baseUrl="http://127.0.0.1:18620"
          apiClient={apiClient}
          nativeBridge={mockBridge}
        />
      </LanguageProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^join with source$/i })).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^unjoin$/i })).not.toBeInTheDocument();
  });

  it("I6: bumping unjoinToken collapses a joined view back to idle", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);

    const { rerender } = renderWithLanguage(
      <JoinControls
        job={{ ...job, joined: true }}
        baseUrl="http://127.0.0.1:18620"
        apiClient={apiClient}
        nativeBridge={mockBridge}
        unjoinToken={0}
      />,
    );
    expect(screen.getByRole("button", { name: /^unjoin$/i })).toBeInTheDocument();

    // JobCard bumps the token (its joined-preview error fallback spent its retry).
    rerender(
      <LanguageProvider>
        <JoinControls
          job={{ ...job, joined: true }}
          baseUrl="http://127.0.0.1:18620"
          apiClient={apiClient}
          nativeBridge={mockBridge}
          unjoinToken={1}
        />
      </LanguageProvider>,
    );

    expect(screen.getByRole("button", { name: /^join with source$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^unjoin$/i })).not.toBeInTheDocument();
  });

  it("I5 ⑤: the trim-length radio sets source_tail_seconds at the job's fps", async () => {
    // frame_rate 24: default 120f -> 5.0s, 240f -> 10.0s.
    {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const apiClient = createApiClient(mockBridge);
      const spy = vi.spyOn(apiClient, "joinJob");
      const job = await completedV2VJob(mockBridge, 24);
      const user = userEvent.setup();
      renderWithLanguage(
        <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
      );
      await user.click(screen.getByRole("button", { name: /join with source/i }));
      await waitFor(() => expect(spy).toHaveBeenCalled());
      expect(spy).toHaveBeenCalledWith(job.job_id, expect.objectContaining({ source_tail_seconds: 5.0 }));
    }
    {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const apiClient = createApiClient(mockBridge);
      const spy = vi.spyOn(apiClient, "joinJob");
      const job = await completedV2VJob(mockBridge, 24);
      const user = userEvent.setup();
      renderWithLanguage(
        <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
      );
      await user.click(screen.getByRole("radio", { name: /^240 frames/i }));
      await user.click(screen.getByRole("button", { name: /join with source/i }));
      await waitFor(() => expect(spy).toHaveBeenCalled());
      expect(spy).toHaveBeenCalledWith(job.job_id, expect.objectContaining({ source_tail_seconds: 10.0 }));
    }
    // frame_rate 30: 120f -> 4.0s, 240f -> 8.0s.
    {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const apiClient = createApiClient(mockBridge);
      const spy = vi.spyOn(apiClient, "joinJob");
      const job = await completedV2VJob(mockBridge, 30);
      const user = userEvent.setup();
      renderWithLanguage(
        <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
      );
      await user.click(screen.getByRole("radio", { name: /^240 frames/i }));
      await user.click(screen.getByRole("button", { name: /join with source/i }));
      await waitFor(() => expect(spy).toHaveBeenCalled());
      expect(spy).toHaveBeenCalledWith(job.job_id, expect.objectContaining({ source_tail_seconds: 8.0 }));
    }
  });

  it("I5 ⑥: fps match but the source was re-encoded -> normalized note + project-fps advice", async () => {
    // Default mock: join returns source_normalized:true / source_fps:24, and
    // getEditInfo reports rate 30 / scale 1 -> project fps 30. The job ran at
    // 24fps, so the source fps (24) matches the generated video (24) — no
    // "did not match" body — but the re-encode (normalized) note shows, plus
    // the project(30)/video(24) mismatch advice.
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 24);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    await user.click(screen.getByRole("button", { name: /join with source/i }));

    const note = await screen.findByText(/re-encoded/i);
    expect(screen.queryByText(/did not match/i)).toBeNull();
    expect(note).toHaveTextContent("project 30 / video 24");
  });

  it("V2V Join note: source fps differs -> 'did not match' body carries both fps, no advice when project==video", async () => {
    // Job at 30fps (videoFps 30) vs. the mock source_fps 24 -> the source-fps
    // conversion body shows both numbers. Project fps 30 == video 30, so no
    // advice line.
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 30);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    await user.click(screen.getByRole("button", { name: /join with source/i }));

    const note = await screen.findByText(/did not match/i);
    expect(note).toHaveTextContent("24");
    expect(note).toHaveTextContent("30");
    expect(screen.queryByText(/recommended/i)).toBeNull();
  });

  it("V2V Join note: project fps matches video fps -> normalized note without advice", async () => {
    // Project fps 24 (rate 24 / scale 1) == video 24, so only the re-encode
    // (normalized) body shows; no advice line.
    const mockBridge = createMockBridge({ delayMs: 0, editInfo: { rate: 24, scale: 1 } });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 24);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    await user.click(screen.getByRole("button", { name: /join with source/i }));

    expect(await screen.findByText(/re-encoded/i)).toBeInTheDocument();
    expect(screen.queryByText(/recommended/i)).toBeNull();
  });

  it("V2V Join note: no normalization and every fps matches -> no note at all", async () => {
    // Source needed no re-encode (joinSourceNormalized:false), source fps 24 ==
    // video 24, project fps 24 == video 24 -> no body and no advice, so the note
    // element is absent entirely.
    const mockBridge = createMockBridge({ delayMs: 0, joinSourceNormalized: false, editInfo: { rate: 24, scale: 1 } });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 24);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    await user.click(screen.getByRole("button", { name: /join with source/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^unjoin$/i })).toBeInTheDocument();
    });
    expect(document.querySelector(".warning-banner")).toBeNull();
  });

  it("V2V Join note: no open project (getEditInfo fails) -> body shows, advice suppressed", async () => {
    // getEditInfo rejects -> projectFps null. The re-encode body still shows
    // (it describes the source, not the project), but with no project fps to
    // compare there's no advice line.
    const mockBridge = createMockBridge({ delayMs: 0, failEditInfo: true });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 24);
    const user = userEvent.setup();

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );
    await user.click(screen.getByRole("button", { name: /join with source/i }));

    expect(await screen.findByText(/re-encoded/i)).toBeInTheDocument();
    expect(screen.queryByText(/recommended/i)).toBeNull();
  });

  it("two JoinControls for the same job name their trim radios per instance (no cross-instance selection)", async () => {
    // The real bug: three panels are always mounted, so the same job can render
    // up to three `JoinControls`. A job-scoped radio `name` merged them into one
    // document-wide group; `useId` keeps each instance's group independent.
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge, 24);
    const user = userEvent.setup();

    renderWithLanguage(
      <>
        <div data-testid="inst-a">
          <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />
        </div>
        <div data-testid="inst-b">
          <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />
        </div>
      </>,
    );

    const radios = screen.getAllByRole("radio") as HTMLInputElement[];
    expect(radios).toHaveLength(4);
    // Two distinct group names, one per instance (a job-scoped name would
    // collapse all four inputs into a single document-wide radio group).
    expect(new Set(radios.map((r) => r.name)).size).toBe(2);

    const a240 = within(screen.getByTestId("inst-a")).getByRole("radio", { name: /^240 frames/i }) as HTMLInputElement;
    const b240 = within(screen.getByTestId("inst-b")).getByRole("radio", { name: /^240 frames/i }) as HTMLInputElement;
    await user.click(a240);
    expect(a240.checked).toBe(true);
    expect(b240.checked).toBe(false); // the sibling instance's group is untouched
  });

  it("I5 ⑦: the crossfade label reads 'Audio crossfade'", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const job = await completedV2VJob(mockBridge);

    renderWithLanguage(
      <JoinControls job={job} baseUrl="http://127.0.0.1:18620" apiClient={apiClient} nativeBridge={mockBridge} />,
    );

    expect(screen.getByText("Audio crossfade")).toBeInTheDocument();
  });
});
