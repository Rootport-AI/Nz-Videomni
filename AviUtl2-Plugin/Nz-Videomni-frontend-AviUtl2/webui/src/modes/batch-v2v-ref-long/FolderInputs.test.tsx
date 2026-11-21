import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { FolderInputs } from "./FolderInputs";
import { createDefaultFrame, withAddedField } from "./folderRules";
import { RULES, makeForm } from "./fakeForm.testSupport";
import { createInitialV2vRefLongState } from "./store";
import type { FolderFrame, UseV2vRefLongFormResult, V2vRefLongInputs } from "./types";

function inputsWith(patch: Partial<V2vRefLongInputs>): V2vRefLongInputs {
  return { ...createInitialV2vRefLongState().inputs, ...patch };
}

function renderInputs(form: UseV2vRefLongFormResult) {
  return render(
    <LanguageProvider>
      <FolderInputs form={form} />
    </LanguageProvider>,
  );
}

function frameWith(id: number, paths: Record<number, string>): FolderFrame {
  const frame = createDefaultFrame(id);
  return { ...frame, fields: frame.fields.map((f) => ({ ...f, path: paths[f.id] ?? "" })) };
}

describe("FolderInputs", () => {
  it("draws the default four fields of clip 1 with their tags and counts out of the rules", () => {
    renderInputs(makeForm());
    const frame = screen.getByTestId("v2v-frame-0");
    expect(within(frame).getByText("Clip 1")).toBeInTheDocument();
    expect(within(frame).getByText("(0/12)")).toBeInTheDocument();
    expect(within(frame).getByRole("textbox", { name: "Clip 1 <Picture 1>" })).toBeInTheDocument();
    expect(within(frame).getByRole("textbox", { name: "Clip 1 <Picture 2>" })).toBeInTheDocument();
    expect(within(frame).getByRole("textbox", { name: "Clip 1 <Video 1>" })).toBeInTheDocument();
    expect(within(frame).getByRole("textbox", { name: "Clip 1 <Audio 1>" })).toBeInTheDocument();
    expect(within(frame).getAllByRole("textbox")).toHaveLength(4);
    expect(within(frame).getByText("Image folders (0/9)")).toBeInTheDocument();
    expect(within(frame).getByText("Video folders (0/3)")).toBeInTheDocument();
    expect(within(frame).getByText("Audio folders (0/3)")).toBeInTheDocument();
  });

  it("counts only the fields with a path, against the limits of the rules", () => {
    const rules = { ...RULES, maxImages: 4, maxTotal: 6 };
    const state = { ...createInitialV2vRefLongState() };
    state.inputs = inputsWith({ frames: [frameWith(1, { 1: "C:\\p", 3: "C:\\v" })] });
    renderInputs(makeForm({ state, rules, filledFolderCount: 2 }));
    const frame = screen.getByTestId("v2v-frame-0");
    expect(within(frame).getByText("(2/6)")).toBeInTheDocument();
    expect(within(frame).getByText("Image folders (1/4)")).toBeInTheDocument();
    expect(within(frame).getByText("Video folders (1/3)")).toBeInTheDocument();
    expect(screen.getByText("Reference folders (2)")).toBeInTheDocument();
  });

  it("adds a field through the form and disables the button at the kind's limit", () => {
    let frame = createDefaultFrame(1);
    frame = withAddedField(frame, "video");
    frame = withAddedField(frame, "video");
    const state = { ...createInitialV2vRefLongState(), inputs: inputsWith({ frames: [frame] }) };
    const form = makeForm({ state });
    renderInputs(form);
    const box = screen.getByTestId("v2v-frame-0");
    expect(within(box).getByRole("button", { name: "Add video folder" })).toBeDisabled();
    expect(within(box).getByRole("textbox", { name: "Clip 1 <Video 3>" })).toBeInTheDocument();
    fireEvent.click(within(box).getByRole("button", { name: "Add image folder" }));
    expect(form.addField).toHaveBeenCalledWith(0, "picture");
  });

  it("disables every add button at the total limit", () => {
    const rules = { ...RULES, maxTotal: 4 };
    renderInputs(makeForm({ rules }));
    const box = screen.getByTestId("v2v-frame-0");
    for (const name of ["Add image folder", "Add video folder", "Add audio folder"]) {
      expect(within(box).getByRole("button", { name })).toBeDisabled();
    }
  });

  it("commits a typed path on blur and picks through 📁", () => {
    const form = makeForm();
    renderInputs(form);
    const box = screen.getByTestId("v2v-frame-0");
    const input = within(box).getByRole("textbox", { name: "Clip 1 <Video 1>" });
    fireEvent.change(input, { target: { value: "D:\\videos" } });
    expect(form.setFieldPath).not.toHaveBeenCalled();
    fireEvent.blur(input);
    expect(form.setFieldPath).toHaveBeenCalledWith(0, 3, "D:\\videos");
    fireEvent.click(within(box).getByRole("button", { name: "Clip 1 <Video 1> Choose folder…" }));
    expect(form.pickFieldDir).toHaveBeenCalledWith(0, 3, "Choose folder…");
  });

  it("the frame's ❌ only on the last frame and only with two or more frames", () => {
    const one = renderInputs(makeForm());
    expect(screen.queryByRole("button", { name: "Remove this clip's folders" })).toBeNull();
    one.unmount();

    const state = {
      ...createInitialV2vRefLongState(),
      inputs: inputsWith({ frames: [createDefaultFrame(1), createDefaultFrame(2), createDefaultFrame(3)] }),
    };
    const form = makeForm({ state });
    renderInputs(form);
    expect(screen.getAllByRole("button", { name: "Remove this clip's folders" })).toHaveLength(1);
    expect(within(screen.getByTestId("v2v-frame-0")).queryByRole("button", { name: "Remove this clip's folders" })).toBeNull();
    expect(within(screen.getByTestId("v2v-frame-1")).queryByRole("button", { name: "Remove this clip's folders" })).toBeNull();
    fireEvent.click(within(screen.getByTestId("v2v-frame-2")).getByRole("button", { name: "Remove this clip's folders" }));
    expect(form.removeLastFrame).toHaveBeenCalled();
    expect(within(screen.getByTestId("v2v-frame-2")).getByText("Clip 3")).toBeInTheDocument();
  });

  it("the fill-following checkbox sits on the last frame only and toggles the form value", () => {
    const state = {
      ...createInitialV2vRefLongState(),
      inputs: inputsWith({ frames: [createDefaultFrame(1), createDefaultFrame(2)], fillFollowing: true }),
    };
    const form = makeForm({ state });
    renderInputs(form);
    const name = "Use the same reference files for the following clips";
    expect(screen.getAllByRole("checkbox", { name })).toHaveLength(1);
    const box = within(screen.getByTestId("v2v-frame-1")).getByRole("checkbox", { name });
    expect(box).toBeChecked();
    fireEvent.click(box);
    expect(form.setFillFollowing).toHaveBeenCalledWith(false);
  });

  it("adds a clip frame", () => {
    const form = makeForm();
    renderInputs(form);
    fireEvent.click(screen.getByRole("button", { name: "Add clip" }));
    expect(form.addFrame).toHaveBeenCalled();
  });

  it("notes the frames past the clip count with the display number, outside the accordion", () => {
    const quiet = renderInputs(makeForm());
    expect(screen.queryByText(/are not used/)).toBeNull();
    quiet.unmount();
    const { container } = renderInputs(makeForm({ unusedFrameStart: 2 }));
    const note = screen.getByText(
      "The folders for clip 3 and later are not used (they still count toward the number of rows).",
    );
    const accordion = container.querySelector("details.v2v-ref-folders") as HTMLElement;
    expect(accordion.contains(note)).toBe(false);
    expect(accordion.nextElementSibling).toBe(note);
  });

  it("names each field and its 📁 with the clip number; the visible label stays the tag", () => {
    const state = {
      ...createInitialV2vRefLongState(),
      inputs: inputsWith({ frames: [createDefaultFrame(1), createDefaultFrame(2)] }),
    };
    const form = makeForm({ state });
    renderInputs(form);
    const second = within(screen.getByTestId("v2v-frame-1"));
    expect(second.getByRole("textbox", { name: "Clip 2 <Picture 1>" })).toBeInTheDocument();
    expect(second.getByText("<Picture 1>")).toBeInTheDocument();
    expect(screen.getAllByRole("textbox", { name: "Clip 1 <Picture 1>" })).toHaveLength(1);
    fireEvent.click(second.getByRole("button", { name: "Clip 2 <Audio 1> Choose folder…" }));
    expect(form.pickFieldDir).toHaveBeenCalledWith(1, 4, "Choose folder…");
  });

  it("Joined is inside the opening-video box, on by default, and goes through the form", () => {
    const form = makeForm();
    const { container } = renderInputs(form);
    const joined = screen.getByRole("checkbox", { name: "Joined" });
    expect(joined).toBeChecked();
    expect(joined.closest(".source-section")).toBe(container.querySelector(".source-section"));
    expect(within(joined.closest(".source-section") as HTMLElement).getByRole("textbox", { name: "Video folder (opening)" })).toBeInTheDocument();
    fireEvent.click(joined);
    expect(form.setJoined).toHaveBeenCalledWith(false);
    expect(screen.getByText("No folder of opening videos selected.")).toBeInTheDocument();
  });

  it("the output folder shows the automatic path, or the not-set line (a blank value too)", () => {
    const unset = renderInputs(makeForm());
    expect(screen.getByText(en.batchV2vRefLong.outDir.none)).toBeInTheDocument();
    unset.unmount();
    const blank = renderInputs(
      makeForm({ state: { ...createInitialV2vRefLongState(), inputs: inputsWith({ outDir: "   ", outDirIsAuto: true }) } }),
    );
    expect(screen.getByText(en.batchV2vRefLong.outDir.none)).toBeInTheDocument();
    expect(screen.queryByText(/^Auto:/)).toBeNull();
    blank.unmount();
    const state = {
      ...createInitialV2vRefLongState(),
      inputs: inputsWith({ outDir: "C:\\clips_out", outDirIsAuto: true }),
    };
    renderInputs(makeForm({ state }));
    expect(screen.getByText("Auto: C:\\clips_out")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Output folder" })).toHaveValue("C:\\clips_out");
  });

  it("inputsLocked disables every input and button", () => {
    const state = {
      ...createInitialV2vRefLongState(),
      inputs: inputsWith({ frames: [createDefaultFrame(1), withAddedField(createDefaultFrame(2), "picture")] }),
    };
    const { container } = renderInputs(makeForm({ state, inputsLocked: true }));
    const controls = container.querySelectorAll("input, button");
    expect(controls.length).toBeGreaterThan(10);
    controls.forEach((el) => expect(el).toBeDisabled());
    // The fields' ❌ too.
    const t = en.batchV2vRefLong;
    expect(screen.getByRole("button", { name: t.refFolders.clearField("Clip 1 <Picture 1>") })).toBeDisabled();
    expect(screen.getByRole("button", { name: t.refFolders.removeField("Clip 2 <Picture 3>") })).toBeDisabled();
  });

  it("❌ after each field's 📁: a default field's clears the path, an added field's removes it", () => {
    let frame = withAddedField(withAddedField(frameWith(1, { 1: "C:\\p1" }), "picture"), "video");
    frame = withAddedField(frame, "audio");
    const state = { ...createInitialV2vRefLongState(), inputs: inputsWith({ frames: [frame] }) };
    const form = makeForm({ state });
    renderInputs(form);
    const t = en.batchV2vRefLong.refFolders;
    const box = within(screen.getByTestId("v2v-frame-0"));

    // Right after 📁, on the same row.
    const pick = box.getByRole("button", { name: "Clip 1 <Picture 1> Choose folder…" });
    const clear = box.getByRole("button", { name: t.clearField("Clip 1 <Picture 1>") });
    expect(pick.nextElementSibling).toBe(clear);
    expect(clear).toHaveAttribute("title", t.clearField("Clip 1 <Picture 1>"));
    expect(clear.textContent).toBe("❌");

    for (const tag of ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"]) {
      expect(box.getByRole("button", { name: t.clearField(`Clip 1 ${tag}`) })).toBeEnabled();
      expect(box.queryByRole("button", { name: t.removeField(`Clip 1 ${tag}`) })).toBeNull();
    }
    for (const tag of ["<Picture 3>", "<Video 2>", "<Audio 2>"]) {
      expect(box.getByRole("button", { name: t.removeField(`Clip 1 ${tag}`) })).toBeEnabled();
      expect(box.queryByRole("button", { name: t.clearField(`Clip 1 ${tag}`) })).toBeNull();
    }
    expect(t.clearField("x")).not.toBe(t.removeFrame);
    expect(t.removeField("x")).not.toBe(t.removeFrame);

    fireEvent.click(box.getByRole("button", { name: t.clearField("Clip 1 <Picture 1>") }));
    expect(form.removeField).toHaveBeenLastCalledWith(0, 1);
    fireEvent.click(box.getByRole("button", { name: t.removeField("Clip 1 <Video 2>") }));
    expect(form.removeField).toHaveBeenLastCalledWith(0, 6);
  });
});
