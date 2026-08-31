import type { AppConfig } from "../../api/types";

/**
 * Built-in fallback used when `GET /config` cannot be reached at all (e.g.
 * the backend is offline). Values transcribed from
 * Docs/API_REFERENCE.md §3.2 (`config.yaml` defaults) so the form still
 * behaves sanely — the UI must show a warning whenever this is in use,
 * since it may drift from the real server's configuration over time.
 */
export const FALLBACK_APP_CONFIG: AppConfig = {
  generation_presets: {
    smoke_test: { width: 384, height: 256, crop_output: null, num_frames: 17 },
    minimal: { width: 512, height: 320, crop_output: null, num_frames: 481 },
    small: {
      width: 960,
      height: 576,
      crop_output: { width: 960, height: 540 },
      num_frames: 481,
    },
    standard_720p: {
      width: 1280,
      height: 768,
      crop_output: { width: 1280, height: 720 },
      num_frames: 361,
    },
    FHD_1080p: {
      width: 1920,
      height: 1088,
      crop_output: { width: 1920, height: 1080 },
      num_frames: 169,
    },
    WQHD_1440p: {
      width: 2560,
      height: 1472,
      crop_output: { width: 2560, height: 1440 },
      num_frames: 89,
    },
  },
  generation_defaults: {
    width: 1280,
    height: 768,
    crop_output: null,
    num_frames: 361,
    frame_rate: 24.0,
    seed: -1,
  },
  limits: {
    max_width: 4096,
    max_height: 4096,
    max_num_frames: 481,
    max_conditioning_images: 5,
    conditioning_frame_idx_multiple: 8,
    conditioning_keyframe_grid_offset: 1,
    phase1_max_concurrent_jobs: 1,
    // 2026-08-31 再測定・判定規則v3・LTX 2.3 既定構成（正本は
    // `Nz-Videomni/config.yaml` の `limits.spill_free_frames`）。
    spill_free_frames: {
      "512x320": 481,
      "960x576": 481,
      "1280x768": 273,
      "1920x1088": 161,
      "2560x1472": 81,
    },
    v2v_context_frames_default: 73,
    v2v_context_frames_min: 25,
    v2v_context_frames_max: 145,
    retake_window_min_frames: 73,
    retake_window_max_frames: 169,
    // 素材（末尾）。8の倍数・[8,136]・既定72（24fpsで3秒）。冒頭の
    // `v2v_context_frames_*` が 8n+1 なのと非対称なのは因果VAEの末尾グリッドの
    // ため（`api/types.ts` の `EndSourceSpec` 参照）。
    end_context_frames_default: 72,
    end_context_frames_min: 8,
    end_context_frames_max: 136,
    chain_comfort_token_budget: 40000,
    single_comfort_token_budget: 44880,
    // 快適上限マーカーの配信テーブル（2026-08-31）。正本は
    // `Nz-Videomni/config.py` の `_default_comfort_budgets()`；ここはオフライン
    // フォールバック用のミラーで、`bridge/mockBridge.ts` の
    // `MOCK_CONFIG_BODY.limits.comfort_budgets` と同内容でなければならない。
    //
    // ⚠ `ltx` に `requires: {}` の行が無いのは意図。LTX 2.3 の既定構成は
    // 快適境界がトークン数に対して単調でなく（境界がデコードのチャンク数増分
    // 7→8／4→5／2→3 と一致）、1本のトークン線で表せないので、そこは上の
    // `spill_free_frames` が正である。「既定行を足せば全構成で賢くなる」は誤り。
    comfort_budgets: {
      ltx: {
        spatial_factor: 32,
        temporal_factor: 8,
        rows: [
          {
            requires: {
              attention_backend: "sage",
              block_swap_prefetch: true,
              keep_resident: true,
              fused_gguf_dequant_kernel: true,
              vae_mode: "prune_vaed",
            },
            single_budget: 44880,
            chain_budget: 40000,
          },
        ],
      },
      ltx25: {
        spatial_factor: 32,
        temporal_factor: 8,
        rows: [{ requires: {}, single_budget: 44880, chain_budget: 44880 }],
      },
    },
  },
  upload: {
    max_image_size_mb: 20,
    allowed_image_extensions: [".png", ".jpg", ".jpeg", ".webp"],
    max_video_size_mb: 200,
    allowed_video_extensions: [".mp4", ".mov", ".webm", ".mkv"],
    max_audio_size_mb: 50,
    allowed_audio_extensions: [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"],
  },
};

export const MIN_WIDTH = 256;
export const MIN_HEIGHT = 128;
export const MIN_NUM_FRAMES = 9;
