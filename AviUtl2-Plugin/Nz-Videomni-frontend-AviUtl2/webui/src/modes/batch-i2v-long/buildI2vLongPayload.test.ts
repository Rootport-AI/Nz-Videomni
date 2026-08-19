import { describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import type { UseChainFormResult } from "../chained/useChainForm";
import type { ChainSnapshotSource } from "./chainSnapshot";
import { buildI2vLongTemplate, buildRowPayload, composeBatchPrompt } from "./buildI2vLongPayload";

/**
 * Compile-time pin for the ONLY coupling this feature has to the Chain screen:
 * the real `UseChainFormResult` must stay structurally assignable to
 * `ChainSnapshotSource`. A Chain refactor that renames/retypes any of the
 * seven fields fails `npm run typecheck` here instead of silently breaking the
 * batch at runtime. Type-only import — no `modes/chained/` code is pulled in.
 */
type ChainFormSatisfiesSnapshot = UseChainFormResult extends ChainSnapshotSource ? true : never;
// `never` にはどんな値も代入できないので、代入不能になった時点で型検査が落ちる。
const chainFormSatisfiesSnapshot: ChainFormSatisfiesSnapshot = true;
void chainFormSatisfiesSnapshot;

/** A "everything the Chain form can possibly emit" request — every optional
 * field of `GenerateChainRequest` present — so the key-set pin below actually
 * has something to detect a regression against. Key order matches
 * `chainUtils.buildChainRequest`'s emission order. */
function fullBase(): GenerateChainRequest {
  return {
    prompt: "base prompt",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: 42,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [
      { num_frames: 121, conditioning_images: [{ image_id: "img-chain-start", frame_idx: 0, strength: 1.0 }] },
      { num_frames: 121, prompt: "clip 1 override" },
      { num_frames: 97 },
    ],
    chunked_upsample: true,
    negative_prompt: "worst quality",
    nag_enabled: true,
    nag_scale: 5,
    nag_tau: 2.5,
    nag_alpha: 0.25,
    neg_method: "vsf",
    vsf_scale: 1.5,
    // Acceleration (2026-07-31): the Chain form emits this right after the
    // NAG/VSF block, and this batch must pass it through verbatim — it has
    // NO source changes of its own for this feature.
    attention_backend: "sage",
    crop_output: { width: 704, height: 480 },
    source_video: { video_id: "vid-1", context_frames: 41 },
    // §1-16 長尺A2V: `buildChainRequest` emits this right after `source_video`.
    // Both slots filled at once is not a state the Chain form can reach (they are
    // mutually exclusive), but the template must strip BOTH regardless.
    source_audio: { audio_id: "aud-1" },
    loras: [{ name: "style", strength: 1.0 }],
    reference_video_id: "ref-1",
    conditioning_attention_strength: 0.8,
    reference_video_strength: 0.9,
    // 素材（末尾）v2: `buildChainRequest` emits this right after
    // `source_audio`. Like the two slots above it is mutually exclusive with
    // them on the real form, but the template must strip it regardless.
    end_source: { video_id: "end-1", context_frames: 72, strength: 1.0 },
  };
}

/** The minimal, realistic scratch-mode base (no source video, no start frame,
 * no LoRAs) — the shape the batch actually sees most of the time. */
function scratchBase(): GenerateChainRequest {
  return {
    prompt: "a cat",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: -1,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [{ num_frames: 121 }, { num_frames: 121 }],
    chunked_upsample: true,
  };
}

describe("composeBatchPrompt", () => {
  it("行プロンプトが空(空白のみ含む)なら常に共通プロンプトのみ（modeを問わない）", () => {
    expect(composeBatchPrompt("base", "", "add")).toBe("base");
    expect(composeBatchPrompt("base", "   ", "replace")).toBe("base");
  });

  it("add: 共通プロンプトの末尾に行プロンプトを追記する", () => {
    expect(composeBatchPrompt("base", "extra", "add")).toBe("base extra");
    expect(composeBatchPrompt("", "extra", "add")).toBe("extra");
  });

  it("replace: 行プロンプトのみを使う（共通プロンプトを丸ごと置換）", () => {
    expect(composeBatchPrompt("base", "extra", "replace")).toBe("extra");
  });
});

describe("buildI2vLongTemplate", () => {
  // 2026-07-30 行ごとプロンプト再設計: テンプレートはプロンプトに一切触らない。
  // 合成は `buildRowPayload`（＝行ごと）の仕事になった。
  it("promptはChain画面の共通プロンプトのまま（合成しない・clip個別も無改変＝案A）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect(template.prompt).toBe("base prompt");
    expect(template.clips[1]?.prompt).toBe("clip 1 override");
    expect(template.clips[2]?.prompt).toBeUndefined();
  });

  it("source_video キーそのものが消える（多重防御）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect("source_video" in template).toBe(false);
  });

  it("source_video が元から無い場合も壊れない", () => {
    const template = buildI2vLongTemplate({ base: scratchBase() });
    expect("source_video" in template).toBe(false);
    expect(template.clips).toHaveLength(2);
  });

  // §1-16 長尺A2V: 音声も同じ多重防御。Startは`sourceAudioAttached`で既に
  // 止まるが、テンプレートに残っていると全行が同じ音声で駆動されてしまう。
  it("source_audio キーそのものが消える（多重防御）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect("source_audio" in template).toBe(false);
  });

  it("source_audio が元から無い場合も壊れない（同一オブジェクトの素通し経路）", () => {
    const base = scratchBase();
    const template = buildI2vLongTemplate({ base });
    expect("source_audio" in template).toBe(false);
    expect(template.clips).toHaveLength(2);
    // 剥がすキーが1つも無いときでも `base` 自体は書き換えない。
    expect(template).not.toBe(base);
  });

  it("音声だけが付いている（動画なし＝A2Vチェーン）テンプレートからも音声が消える", () => {
    const base: GenerateChainRequest = { ...scratchBase(), source_audio: { audio_id: "aud-9" } };
    const template = buildI2vLongTemplate({ base });
    expect("source_audio" in template).toBe(false);
    // 入力は不変（コピーオンライト）。
    expect(base.source_audio).toEqual({ audio_id: "aud-9" });
  });

  // §1-15 参照動画: 参照も同じ多重防御。Startは`referenceVideoAttached`で既に
  // 止まるが、テンプレートに残っていると全行が同じ参照動画に従ってしまう。
  it("reference_video_id と強度2種がまとめて消える（多重防御）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect("reference_video_id" in template).toBe(false);
    expect("conditioning_attention_strength" in template).toBe(false);
    expect("reference_video_strength" in template).toBe(false);
  });

  it("参照だけが付いている（元動画・音声なし）テンプレートからも参照が消える", () => {
    const base: GenerateChainRequest = {
      ...scratchBase(),
      loras: [{ name: "canny-control", strength: 1.0 }],
      reference_video_id: "ref-9",
      conditioning_attention_strength: 0.4,
      reference_video_strength: 0.6,
    };
    const template = buildI2vLongTemplate({ base });
    expect("reference_video_id" in template).toBe(false);
    expect("conditioning_attention_strength" in template).toBe(false);
    expect("reference_video_strength" in template).toBe(false);
    // LoRAは参照とは独立の設定なので残る（スタイル指定として有効）。
    expect(template.loras).toEqual([{ name: "canny-control", strength: 1.0 }]);
    // 入力は不変（コピーオンライト）。
    expect(base.reference_video_id).toBe("ref-9");
    expect(base.conditioning_attention_strength).toBe(0.4);
  });

  it("参照が元から無い場合も壊れない（同一オブジェクトの素通し経路）", () => {
    const base = scratchBase();
    const template = buildI2vLongTemplate({ base });
    expect("reference_video_id" in template).toBe(false);
    expect(template.clips).toHaveLength(2);
  });

  // 素材（末尾）v2: 四つ目の多重防御。v2 では帯の分だけ出力が伸びるので、
  // 残っていると全行が同じ素材で終わるだけでなく、全行の尺もパネルの表示とずれる。
  it("end_source が消える（多重防御）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect("end_source" in template).toBe(false);
  });

  it("素材（末尾）だけが付いているテンプレートからも消える（画像版も同じ）", () => {
    const withVideo: GenerateChainRequest = {
      ...scratchBase(),
      end_source: { video_id: "end-9", context_frames: 136, strength: 1.0 },
    };
    expect("end_source" in buildI2vLongTemplate({ base: withVideo })).toBe(false);
    // 入力は不変（コピーオンライト）。
    expect(withVideo.end_source).toEqual({ video_id: "end-9", context_frames: 136, strength: 1.0 });

    const withImage: GenerateChainRequest = {
      ...scratchBase(),
      end_source: { image_id: "end-img-9", context_frames: 8, strength: 1.0 },
    };
    expect("end_source" in buildI2vLongTemplate({ base: withImage })).toBe(false);
  });

  it("素材（末尾）が元から無い場合も壊れない（同一オブジェクトの素通し経路）", () => {
    const base = scratchBase();
    const template = buildI2vLongTemplate({ base });
    expect("end_source" in template).toBe(false);
    expect(template.clips).toHaveLength(2);
  });

  it("clips[0].conditioning_images キーが消える（Chain画面の開始フレーム画像は使わない）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect("conditioning_images" in template.clips[0]!).toBe(false);
    expect(template.clips[0]?.num_frames).toBe(121);
  });

  it("clips[1..] は同一参照でそのまま素通しする", () => {
    const base = fullBase();
    const template = buildI2vLongTemplate({ base });
    expect(template.clips[1]).toBe(base.clips[1]);
    expect(template.clips[2]).toBe(base.clips[2]);
    expect(template.clips).toHaveLength(3);
  });

  it("prompt/source_video/clips[0]以外のフィールドは値も参照もそのまま素通しする", () => {
    const base = fullBase();
    const template = buildI2vLongTemplate({ base });
    expect(template.width).toBe(768);
    expect(template.height).toBe(512);
    expect(template.frame_rate).toBe(24);
    expect(template.seed).toBe(42);
    expect(template.overlap_frames).toBe(3);
    expect(template.overlap_strength).toBe(0.5);
    expect(template.chunked_upsample).toBe(true);
    expect(template.neg_method).toBe("vsf");
    expect(template.vsf_scale).toBe(1.5);
    expect(template.nag_enabled).toBe(true);
    expect(template.negative_prompt).toBe("worst quality");
    expect(template.attention_backend).toBe("sage");
    // 共有参照（絶対にmutateしない対象）
    expect(template.loras).toBe(base.loras);
    expect(template.crop_output).toBe(base.crop_output);
  });

  it("base を一切破壊しない（deep比較で元の姿のまま）", () => {
    const base = fullBase();
    const pristine = structuredClone(base);
    const template = buildI2vLongTemplate({ base });
    expect(base).toEqual(pristine);
    // 新しいオブジェクト・新しい clips 配列であること
    expect(template).not.toBe(base);
    expect(template.clips).not.toBe(base.clips);
    expect(template.clips[0]).not.toBe(base.clips[0]);
  });

  it("回帰ピン: template のキー集合そのもの（未知キーの出現・既知キーの消失を検知する）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    // `GenerateChainRequest` に新しいフィールドが生えて Chain 画面が送るように
    // なったとき、それをバッチが素通ししている（＝ここに現れる）ことを目視で
    // 確認するためのピン。落ちたら「意図した追加か」を判断して更新する。
    expect(Object.keys(template)).toEqual([
      "prompt",
      "width",
      "height",
      "frame_rate",
      "seed",
      "overlap_frames",
      "overlap_strength",
      "clips",
      "chunked_upsample",
      "negative_prompt",
      "nag_enabled",
      "nag_scale",
      "nag_tau",
      "nag_alpha",
      "neg_method",
      "vsf_scale",
      "attention_backend",
      "crop_output",
      "loras",
      // §1-15 参照動画: `reference_video_id` とその強度2種はここに現れない
      // （テンプレート化の時点で剥がすため）。素材（末尾）の `end_source`
      // も同じ理由で現れない。
    ]);
  });

  it("回帰ピン: clip のキー集合（clip0は画像なし・他はそのまま）", () => {
    const template = buildI2vLongTemplate({ base: fullBase() });
    expect(Object.keys(template.clips[0]!)).toEqual(["num_frames"]);
    expect(Object.keys(template.clips[1]!)).toEqual(["num_frames", "prompt"]);
  });
});

describe("buildRowPayload", () => {
  const template = buildI2vLongTemplate({ base: scratchBase() });
  /** The common "no row prompt" call — image only, add mode. */
  const withImage = (imageId: string) => buildRowPayload(template, { imageId, rowPrompt: "", promptMode: "add" });

  it("clip0にのみ frame_idx=0 / strength=1.0 の画像を1点載せる", () => {
    const request = withImage("img-row-1");
    expect(request.clips[0]?.conditioning_images).toEqual([{ image_id: "img-row-1", frame_idx: 0, strength: 1.0 }]);
    expect(request.clips[1]?.conditioning_images).toBeUndefined();
    expect("conditioning_images" in request.clips[1]!).toBe(false);
  });

  it("行プロンプトをmodeに従って共通プロンプトへ合成する（3分岐）", () => {
    expect(buildRowPayload(template, { imageId: "i", rowPrompt: "in the rain", promptMode: "add" }).prompt).toBe(
      "a cat in the rain",
    );
    expect(buildRowPayload(template, { imageId: "i", rowPrompt: "a dog", promptMode: "replace" }).prompt).toBe("a dog");
    // 行プロンプトが空ならmodeを問わず共通プロンプトそのまま。
    expect(buildRowPayload(template, { imageId: "i", rowPrompt: "   ", promptMode: "replace" }).prompt).toBe("a cat");
  });

  // Acceleration (2026-07-31): this batch has ZERO source changes for the
  // feature — the Chain form's `attention_backend` must survive both the
  // template step (pinned above) and the per-row stamp, unexamined.
  it("attention_backend は行ペイロードまで逐語で素通しする（本機能のソース改修はゼロ）", () => {
    const accelTemplate = buildI2vLongTemplate({ base: fullBase() });
    const request = buildRowPayload(accelTemplate, { imageId: "img-row-1", rowPrompt: "", promptMode: "add" });
    expect(request.attention_backend).toBe("sage");
    // sdpa（既定）を送らないChain画面の出力では、キー自体が現れない。
    const plain = buildRowPayload(buildI2vLongTemplate({ base: scratchBase() }), {
      imageId: "img-row-1",
      rowPrompt: "",
      promptMode: "add",
    });
    expect(plain).not.toHaveProperty("attention_backend");
  });

  it("行プロンプトはclip個別プロンプトを書き換えない（案A）", () => {
    const withClipPrompt = buildI2vLongTemplate({ base: fullBase() });
    const request = buildRowPayload(withClipPrompt, { imageId: "i", rowPrompt: "a dog", promptMode: "replace" });
    expect(request.prompt).toBe("a dog");
    expect(request.clips[1]?.prompt).toBe("clip 1 override");
  });

  it("template を一切変更しない（コピーオンライト）", () => {
    const pristine = structuredClone(template);
    buildRowPayload(template, { imageId: "img-row-1", rowPrompt: "in the rain", promptMode: "add" });
    expect(template).toEqual(pristine);
    expect("conditioning_images" in template.clips[0]!).toBe(false);
    expect(template.prompt).toBe("a cat");
  });

  it("2回呼び出した結果は互いに独立している（同じtemplateを使い回せる）", () => {
    const first = buildRowPayload(template, { imageId: "img-row-1", rowPrompt: "one", promptMode: "add" });
    const second = buildRowPayload(template, { imageId: "img-row-2", rowPrompt: "two", promptMode: "add" });
    expect(first.clips[0]?.conditioning_images?.[0]?.image_id).toBe("img-row-1");
    expect(second.clips[0]?.conditioning_images?.[0]?.image_id).toBe("img-row-2");
    expect(first.prompt).toBe("a cat one");
    expect(second.prompt).toBe("a cat two");
    expect(first.clips[0]).not.toBe(second.clips[0]);
    expect(first).not.toBe(second);
  });

  it("clips[1..] は同一参照のまま共有される", () => {
    const request = withImage("img-row-1");
    expect(request.clips[1]).toBe(template.clips[1]);
  });

  it("clip0のキー順は num_frames …(既存)… conditioning_images の末尾追加になる", () => {
    const request = withImage("img-row-1");
    expect(Object.keys(request.clips[0]!)).toEqual(["num_frames", "conditioning_images"]);
  });

  it("prompt など他のフィールドはそのまま・キー順も変わらない", () => {
    const request = withImage("img-row-1");
    expect(request.prompt).toBe("a cat");
    expect(request.seed).toBe(-1);
    expect(Object.keys(request)).toEqual(Object.keys(template));
  });
});
