import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import { createMockBridge } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { MockBridgeOptions, ResultOf } from "./bridge";
import { AppShell } from "./shell/AppShell";
import {
  PREFILL_SIZE_POLICY_STORAGE_KEY,
  PREFILL_FPS_POLICY_STORAGE_KEY,
} from "./shell/PrefillPolicyContext";
import { resetProvisionalReservation } from "./timeline/provisionalReservation";
import { resetCreateLiveCommands } from "./timeline/createLiveCommands";
import { ACCELERATION_STORAGE_KEY } from "./shell/accelerationSettings";

// W1 integration coverage: the right-click prefill SIZE and FPS policies are now
// two independent axes (Settings → PrefillPolicyContext), each persisted under
// its own localStorage key that `PrefillPolicyProvider` reads on mount — so each
// test writes the key(s) it needs BEFORE rendering `AppShell`. Only the
// right-click prefill's initial values change; the routing/reservation machinery
// is unchanged (covered by App.prefill.test.tsx).
//
// §3-13 (2026-09-01): the fps axis's `material` is live again and now reads the
// selected object's OWN framerate (contract v11 `mediaFps`), so the cases that
// used to assert X1's coercion to `project` assert the material's snapped fps
// instead — the mock-side verification of the whole native→bridge→prefill path.

function selectionWith(
  mediaWidth: number,
  mediaHeight: number,
  rate = 30,
  scale = 1,
  effectName = "動画ファイル",
  filePath = "C:\\v\\a.mp4",
  // §3-13 (contract v11): the object's OWN probed framerate, RAW as native
  // reports it. Omitted by default so every pre-existing case keeps the
  // "native told us nothing" shape it was written against.
  mediaFps?: number,
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 120,
    selected: [
      {
        layer: 1,
        frameStart: 0,
        frameEnd: 120,
        effectName,
        filePath,
        objectName: "a",
        textContent: null,
        mediaWidth,
        mediaHeight,
        mediaDurationSec: 0,
        ...(mediaFps !== undefined ? { mediaFps } : {}),
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate,
    scale,
    sampleRate: 44100,
  };
}

function imageSel(
  mediaWidth: number,
  mediaHeight: number,
  rate = 30,
  scale = 1,
  mediaFps?: number,
): ResultOf<"timeline.getSelection"> {
  return selectionWith(mediaWidth, mediaHeight, rate, scale, "画像ファイル", "C:\\i\\a.png", mediaFps);
}

const panel = () => screen.getByRole("tabpanel");
const fpsInput = () => within(panel()).getByRole("spinbutton", { name: /fps/i }) as HTMLInputElement;

type Policy = "defaults" | "project" | "material";
function setSizePolicy(policy: Policy): void {
  window.localStorage.setItem(PREFILL_SIZE_POLICY_STORAGE_KEY, policy);
}
function setFpsPolicy(policy: Policy): void {
  window.localStorage.setItem(PREFILL_FPS_POLICY_STORAGE_KEY, policy);
}

async function renderReady(options: MockBridgeOptions) {
  const bridge = createMockBridge(options);
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  return bridge;
}

describe("App / right-click prefill size & fps policies (W1)", () => {
  beforeEach(() => {
    resetProvisionalReservation();
    resetCreateLiveCommands();
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  // --- Create ---------------------------------------------------------------

  it(
    "default (nothing stored): size from the material, fps from the PROJECT (getEditInfo)",
    async () => {
      // No keys written -> size defaults to material, fps defaults to project.
      // editInfo fps (60) is distinct from the selection's own rate (30), so a
      // resulting fps of 60 proves the fps came from the project, not the material.
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      // size from the material (800x600 -> 832x640); fps overwritten from the project (60).
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("640").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("60");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "§3-13: size=material × fps=material — both axes come from the MATERIAL, the fps snapped from 29.97 to 30",
    async () => {
      // §3-13's mock-side main verification: the fps axis's "material" reads the
      // object's OWN probed framerate (contract v11 `mediaFps`) and snaps it to
      // an integer. 30 is distinguishable from every other candidate in play —
      // the selection's project rate (25), getEditInfo's project fps (60) and the
      // config default (24) — so only the material path can produce it.
      setSizePolicy("material");
      setFpsPolicy("material");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      // media 800x600 (64 grid -> 832x640) and 29.97 fps, from the material.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSel(800, 600, 25, 1, 29.97),
        });
      });

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("640").length).toBeGreaterThan(0);
          // 29.97 -> 30: neither the selection's 25, nor getEditInfo's 60, nor 24.
          expect(fpsInput().value).toBe("30");
        },
        { timeout: 5_000 },
      );
      // size stays the material's 832x640, never the project's 1024x512.
      expect(within(panel()).queryByDisplayValue("1024")).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "§3-13: fps=material falls back to the selection's project rate when the material has no readable fps",
    async () => {
      // The mkv/webm (and old-native-build) case: no `mediaFps` on the selection
      // at all -> tier 2, the selection's own rate/scale (25). getEditInfo is NOT
      // consulted, since neither axis is `project`.
      setSizePolicy("material");
      setFpsPolicy("material");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 25, 1) });
      });

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("25");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "① size=defaults × fps=defaults: config generation_defaults width/height/fps, material ignored",
    async () => {
      setSizePolicy("defaults");
      setFpsPolicy("defaults");
      const bridge = await renderReady({ delayMs: 0 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      // Config defaults (mock): 1280x768, fps 24 — the material 832x640/30 is ignored.
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1280").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("768").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );
      expect(within(panel()).queryByDisplayValue("832")).not.toBeInTheDocument();
      expect(fpsInput().value).toBe("24");
    },
    15_000,
  );

  it(
    "② size=project × fps=project: overwrites width/height/fps from getEditInfo exactly once",
    async () => {
      setSizePolicy("project");
      setFpsPolicy("project");
      // Project resolution/fps distinct from both the material (832x640/30) and
      // the config defaults (1280x768/24).
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("512").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("60");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "② size=project with acceleration all-on: DURATION is RE-derived at the project's own resolution using the SMART ceiling (1280x768 -> 361, not the legacy 273)",
    async () => {
      // Post-review fix A-1 (2026-08-31): `SingleScreen.tsx`'s size=project
      // mount effect must recompute the smart ceiling at the OVERWRITTEN
      // (project) width/height, not reuse whatever `resolvePrefillSeed`
      // resolved at the material's geometry — see the comment above
      // `smartCeiling` in that file. All five Acceleration toggles on selects
      // the served `ltx` row.
      setSizePolicy("project");
      setFpsPolicy("project");
      window.localStorage.setItem(
        ACCELERATION_STORAGE_KEY,
        JSON.stringify({
          attentionBackend: "sage",
          blockSwapPrefetch: true,
          keepResident: true,
          fusedGgufDequantKernel: true,
          vaeMode: "prune_vaed",
        }),
      );
      // The project's resolution (1280x768) differs from the material
      // (832x640) so the overwrite is observable, and is itself the
      // `spill_free_frames` legacy table's key with the widest gap to the
      // smart value (273 vs. 361).
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1280, height: 768, rate: 24, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1280").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("768").length).toBeGreaterThan(0);
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("361");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "② project falls back to the material seed when getEditInfo fails",
    async () => {
      setSizePolicy("project");
      setFpsPolicy("project");
      const bridge = await renderReady({ delayMs: 0, failEditInfo: true });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      // getEditInfo rejects -> the material-derived seed stays (832x640, fps 30).
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("640").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );
      expect(fpsInput().value).toBe("30");
    },
    15_000,
  );

  // --- W1 axis crossings ----------------------------------------------------

  it(
    "crossing size=material × fps=project: material size kept, only the fps taken from getEditInfo",
    async () => {
      setSizePolicy("material");
      setFpsPolicy("project");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      // size stays the material's 832x640 (NOT the project's 1024x512); fps -> 60.
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("60");
        },
        { timeout: 5_000 },
      );
      expect(within(panel()).queryByDisplayValue("1024")).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "crossing size=project × fps=material: size overwritten from getEditInfo, fps left at the material's own",
    async () => {
      setSizePolicy("project");
      setFpsPolicy("material");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSel(800, 600, 25, 1, 29.97),
        });
      });

      // The axes split cleanly: size 1024x512 from getEditInfo, while the fps
      // stays the material's own 29.97 -> 30 — the `project` mount effect must
      // apply only ITS axis and leave the fps alone (not the project's 60).
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("512").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("30");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  // --- Chain ----------------------------------------------------------------

  it(
    "§3-13 (Chain): size=material × fps=material — common size AND fps both from the material",
    async () => {
      setSizePolicy("material");
      setFpsPolicy("material");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "extendVideo",
          selection: selectionWith(800, 600, 25, 1, "動画ファイル", "C:\\v\\a.mp4", 29.97),
        });
      });

      await waitFor(
        () => {
          expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
        },
        { timeout: 5_000 },
      );
      await waitFor(
        () => {
          // common size stays the material's 832x640; the common fps is the
          // material's 29.97 -> 30 (not the selection's 25, nor getEditInfo's 60).
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("640").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("30");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  // --- 台帳§3-71/§3-72: the fps-snap toast -----------------------------------
  //
  // Whole frame rates only, at every entry point. The PREFILL route is the one
  // that says so out loud (owner ruling 2026-09-02 ②): the value the user gets
  // is not the value their project or their material has. Manual typing and the
  // `project` mount overwrite stay silent — the field shows the rounded number
  // the same instant, so a toast would only repeat it.

  it(
    "project 29.97: the field lands on 30 AND one toast names the original rate",
    async () => {
      setSizePolicy("material");
      setFpsPolicy("project");
      // Both the selection's rate/scale (the seed's tier 2) and getEditInfo (the
      // mount overwrite) are the NTSC 30000/1001, so the two agree on 30 and the
      // only thing under test is the toast.
      const bridge = await renderReady({
        delayMs: 0,
        editInfo: { width: 1024, height: 512, rate: 30000, scale: 1001 },
      });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSel(800, 600, 30000, 1001),
        });
      });

      await waitFor(() => expect(fpsInput().value).toBe("30"), { timeout: 5_000 });
      const toasts = await screen.findAllByText(/rounded to 30 fps/i, {}, { timeout: 5_000 });
      // Exactly one — the seed reports it, the `project` overwrite does not.
      expect(toasts).toHaveLength(1);
      expect(toasts[0]).toHaveTextContent("29.97");
    },
    15_000,
  );

  it(
    "a project already on a whole frame rate produces NO toast (no false positives)",
    async () => {
      setSizePolicy("material");
      setFpsPolicy("project");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 30, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSel(800, 600, 30, 1) });
      });

      await waitFor(() => expect(fpsInput().value).toBe("30"), { timeout: 5_000 });
      expect(screen.queryByText(/rounded to/i)).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "material 23.976: the toast names the material's own rate, formatted to 23.98",
    async () => {
      setSizePolicy("material");
      setFpsPolicy("material");
      const bridge = await renderReady({ delayMs: 0 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSel(800, 600, 30, 1, 23.976),
        });
      });

      await waitFor(() => expect(fpsInput().value).toBe("24"), { timeout: 5_000 });
      // `jobs/fpsConvert.ts`'s `formatFps` caps the display at two decimals.
      const toast = await screen.findByText(/rounded to 24 fps/i, {}, { timeout: 5_000 });
      expect(toast).toHaveTextContent("23.98");
    },
    15_000,
  );

  it(
    "② project (Chain): overwrites the common width/height/fps from getEditInfo",
    async () => {
      setSizePolicy("project");
      setFpsPolicy("project");
      const bridge = await renderReady({ delayMs: 0, editInfo: { width: 1024, height: 512, rate: 60, scale: 1 } });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "extendVideo", selection: selectionWith(800, 600, 30, 1) });
      });

      await waitFor(
        () => {
          expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
        },
        { timeout: 5_000 },
      );
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("512").length).toBeGreaterThan(0);
          expect(fpsInput().value).toBe("60");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );
});
