/**
 * レガシー／フォールバックの正 —— サーバ配信の解像度別の実測表
 * `AppConfig.limits.spill_free_frames` から「この解像度なら何フレームまでが
 * 快適か」を引く、面積最近傍の粗い探索。
 *
 * 2026-08-31 以降、賢い（トークン式の）マーカーは
 * `shell/comfortTable.ts` の配信テーブルが担当し、このモジュールは
 * **一致する行が無かったときに必ず戻ってくる場所**になった。とくに LTX 2.3 の
 * 既定構成は意図的に賢い行を持たない（快適境界がデコードのチャンク数増分と
 * 一致し、トークン1本線で表せないため）ので、そこではここが唯一の正である。
 * その 2026-08-31 の再測定は3解像度（1280×768・1920×1088・2560×1472）が対象で、
 * うち2つが動いた——512×320・960×576 は API 上限 481 で頭打ちのため測定対象外
 * （据え置き）。
 * 賢い側の関数（`singleComfortFrames`／`SINGLE_COMFORT_TOKEN_BUDGET`／
 * `resolveSingleComfortBudget`）はこのファイルから
 * `shell/comfortTable.ts` へ移設済み。
 */

/**
 * Resolves the "comfortable" `num_frames` ceiling for a given resolution
 * from `AppConfig.limits.spill_free_frames` (Docs/API_REFERENCE.md §3.2/§8):
 * a map of `"WIDTHxHEIGHT" -> frame count` beyond which generation slows
 * 2-4x (without OOMing). Looks up an exact "WxH" match first; if the
 * current resolution isn't a key (the width/height sliders move in
 * 64px steps, independent of the handful of resolutions the backend
 * happens to have measured), falls back to whichever key's pixel area is
 * closest to the current resolution's area, since the VRAM-spill slowdown
 * is driven by total pixel count rather than the exact aspect ratio.
 *
 * Returns `null` if the map is empty (or has no parseable keys at all),
 * so callers can skip the warning/tick-mark entirely rather than showing a
 * bogus threshold.
 */
export function resolveSpillFreeFrames(
  spillFreeFrames: Record<string, number>,
  width: number,
  height: number,
): number | null {
  const exactKey = `${width}x${height}`;
  const exactValue = spillFreeFrames[exactKey];
  if (exactValue !== undefined) return exactValue;

  const targetArea = width * height;
  let bestKey: string | null = null;
  let bestDiff = Infinity;

  for (const key of Object.keys(spillFreeFrames)) {
    const area = parseAreaFromKey(key);
    if (area === null) continue;
    const diff = Math.abs(area - targetArea);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestKey = key;
    }
  }

  if (bestKey === null) return null;
  return spillFreeFrames[bestKey] ?? null;
}

function parseAreaFromKey(key: string): number | null {
  const match = /^(\d+)x(\d+)$/.exec(key);
  if (!match) return null;
  return Number(match[1]) * Number(match[2]);
}
