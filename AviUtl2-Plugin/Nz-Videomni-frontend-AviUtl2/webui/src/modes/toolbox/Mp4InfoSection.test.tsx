import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { MOCK_MP4_INFO_COMMENT, createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { Mp4InfoSection } from "./Mp4InfoSection";

// §3-164 (2026-09-24): Toolbox › mp4 info. A dropped or picked video's path
// goes to `POST /utils/mp4-info`; the `comment` string is shown verbatim.
// The mock bridge's rule: a path containing "nometa" has no comment, anything
// else answers with `MOCK_MP4_INFO_COMMENT`.

const t = en.toolbox.mp4info;
const MP4_INFO_PATH = "/api/v1/utils/mp4-info";

function isMp4InfoCall(call: unknown[]): boolean {
  return call[0] === "backend.request" && (call[1] as { path?: string } | undefined)?.path === MP4_INFO_PATH;
}

function setup(options: MockBridgeOptions = {}, wrap?: (bridge: NativeBridge) => NativeBridge) {
  const mock = createMockBridge({ delayMs: 0, ...options });
  const requests = vi.spyOn(mock, "request");
  const nativeBridge = wrap ? wrap(mock) : mock;
  const view = render(
    <LanguageProvider>
      <Mp4InfoSection nativeBridge={nativeBridge} />
    </LanguageProvider>,
  );
  return { ...view, requests };
}

/** Wraps the mock so `POST /utils/mp4-info` answers with the backend's error
 * envelope; every other call goes to the mock as usual. */
function answerMp4InfoWith(status: number, code: string, message: string) {
  return (mock: NativeBridge): NativeBridge => {
    const request = (method: Parameters<NativeBridge["request"]>[0], params: unknown): Promise<unknown> => {
      if (method === "backend.request" && (params as { path?: string }).path === MP4_INFO_PATH) {
        return Promise.resolve({ status, body: { error: { code, message } } });
      }
      return mock.request(method, params as never);
    };
    return { ...mock, request: request as NativeBridge["request"] };
  };
}

function drop(container: HTMLElement, fileName: string) {
  fireEvent.drop(container.querySelector(".mp4info-drop")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function textarea(): HTMLTextAreaElement {
  return screen.getByRole("textbox", { name: t.resultLabel }) as HTMLTextAreaElement;
}

describe("Mp4InfoSection", () => {
  it("starts idle: an empty read-only textarea and the idle message", () => {
    setup();
    expect(textarea()).toHaveAttribute("readonly");
    expect(textarea().value).toBe("");
    expect(screen.getByRole("status")).toHaveTextContent(t.idle);
  });

  it("a dropped video shows its comment verbatim and names the file", async () => {
    const { container, requests } = setup();

    drop(container, "clip.mp4");

    await waitFor(() => expect(textarea().value).toBe(MOCK_MP4_INFO_COMMENT));
    expect(screen.getByText(`${t.fileLabel}: clip.mp4`)).toBeInTheDocument();
    expect(screen.queryByRole("status")).toBeNull();
    // The request carries the resolved local path, nothing else.
    const call = requests.mock.calls.find(isMp4InfoCall);
    expect(call?.[1]).toMatchObject({ method: "POST", body: { path: "C:\\Users\\mock\\Downloads\\clip.mp4" } });
  });

  it("a video with no comment tag says so", async () => {
    const { container } = setup();

    drop(container, "clip-nometa.mp4");

    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(t.notFound));
    expect(textarea().value).toBe("");
  });

  it("the pick button reads the chosen video the same way", async () => {
    setup({ pickFileName: "picked.mov" });
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: t.pickButton }));

    await waitFor(() => expect(textarea().value).toBe(MOCK_MP4_INFO_COMMENT));
    expect(screen.getByText(`${t.fileLabel}: picked.mov`)).toBeInTheDocument();
  });

  it("a cancelled pick changes nothing", async () => {
    const { requests } = setup({ failPickFile: "CANCELLED" });
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: t.pickButton }));

    expect(screen.getByRole("status")).toHaveTextContent(t.idle);
    expect(requests.mock.calls.some(isMp4InfoCall)).toBe(false);
  });

  it("MEDIA_UNREADABLE (422) shows the translated unreadable sentence, not the server's text", async () => {
    const { container } = setup({}, answerMp4InfoWith(422, "MEDIA_UNREADABLE", "ffprobe could not read the file"));

    drop(container, "broken.mp4");

    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(t.errorUnreadable));
    expect(screen.queryByText(/ffprobe could not read the file/)).toBeNull();
    expect(textarea().value).toBe("");
  });

  it("MEDIA_NOT_FOUND (404) shows the translated not-found sentence", async () => {
    const { container } = setup({}, answerMp4InfoWith(404, "MEDIA_NOT_FOUND", "file not found"));

    drop(container, "gone.mp4");

    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(t.errorNotFound));
    expect(textarea().value).toBe("");
  });

  it("any other error code keeps the server's own message", async () => {
    const { container } = setup({}, answerMp4InfoWith(403, "LOCAL_ONLY", "loopback only"));

    drop(container, "clip.mp4");

    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(t.error("loopback only")));
  });

  it("an unsupported extension is refused before any request", async () => {
    const { container, requests } = setup();

    drop(container, "notes.txt");

    await waitFor(() => expect(screen.getByText(en.dnd.unsupportedVideo)).toBeInTheDocument());
    expect(requests.mock.calls.some(isMp4InfoCall)).toBe(false);
  });
});
