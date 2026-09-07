# §3-90 start+end bridge — エンジン側テストの実行対象と実測（文書整備用の材料）

作成: 2026-09-07／実装者A（バックエンド）／ブランチ `feature/start-end-bridge`

本書は `Docs/VERIFICATION_LOG.md` の新節を書くための**素材**である。ここに書いてあるのは
実際に走らせたコマンドと実測値だけで、推定値は含まない。

---

## 0. 前置き — なぜ「自分で定義した集合」なのか

`Docs/VERIFICATION_LOG.md` §101.10 は、エンジン側の基準を

> LTX 2.3エンジン側（`.venv-engine`・`--noconftest`）**69 passed**、
> LTX 2.5エンジン側（`.venv-engine-ltx25`・同）**235 passed**。

と記録しているが、**その 69／235 を生んだ「実行対象ファイルの一覧」はどこにも書かれていない**。
仮想環境名と `--noconftest` しか記載がないため、同じ数字を再現する手段がない（詳細は §3）。

そこで本改修では、**「エンジン仮想環境で収集エラーも環境起因の失敗も出さずに走りきるエンジン側テストファイル」**
という基準で対象集合を自分で定義し、その全緑を確認した。以下の一覧はその定義の実体である。

---

## 1. 実行対象ファイル一覧

### 1.1 LTX 2.3（`.venv-engine`）— 14ファイル

- `tests/test_block_swap_release.py`
- `tests/test_chain_reference_engine.py`
- `tests/test_gemma_keep_resident_move.py`
- `tests/test_ic_lora_engine_conditioning.py`
- `tests/test_ic_lora_forward.py`
- `tests/test_outpaint_canvas.py`
- `tests/test_outpaint_pyramid_blend.py`
- `tests/test_pipeline_vae_mode_swap.py`
- `tests/test_pruned_video_decoder.py`
- `tests/test_registry_swap.py`
- `tests/test_retake_math.py`
- `tests/test_worker_fused_dequant_resolve.py`
- `tests/test_worker_keep_resident_resolve.py`
- `tests/test_worker_vae_mode_resolve.py`

**この集合から意図的に外した2ファイル**（いずれも本改修とは無関係の、環境起因の除外）:

| ファイル | 外した理由（実測） |
|---|---|
| `tests/test_chain_lora.py` | 収集時に `ModuleNotFoundError: No module named 'fastapi'`。エンジン仮想環境にアプリ依存が無いため、この1ファイルだけで収集が中断する（`1 error during collection` → `Interrupted`） |
| `tests/test_stage2_window.py` | `3 failed, 334 passed, 3 errors`。失敗3件は `test_gradio_chain_precheck_accepts_and_resolves_the_window` 等の Gradio 依存、エラー3件は `test_mock_backed_chain_*` 等の FastAPI/モックバックエンド依存。torch も使うがアプリ側の資産を併用するテストで、エンジン仮想環境では成立しない |

**対象の選び方**: `tests/*.py` のうち `import torch` ないし `pytest.importorskip("torch")` /
`importorskip("ltx_core")` を持つファイル（= エンジン側テスト）を起点にし、
そこから上表の2ファイルを除いたもの。

### 1.2 LTX 2.5（`.venv-engine-ltx25`）— 7ファイル

- `tests/test_ltx25_band.py` ← **本改修で1件追加**（後述）
- `tests/test_ltx25_outpaint.py`
- `tests/test_ltx25_reference_encode.py`
- `tests/test_ltx25_keep_resident_registry.py`
- `tests/test_retake_math.py`
- `tests/test_outpaint_canvas.py`
- `tests/test_outpaint_pyramid_blend.py`

後半3ファイルは 2.3 側と重複しているが、これは意図的である。`chain_math` と outpaint の幾何は
両エンジンが**同じ純Pythonモジュールを共有**しており、両方の仮想環境で同じ値が出ることを
確認するのがこれらの役割だから（`test_retake_math.py` を両側で走らせる作法は
VERIFICATION_LOG の既存記録 §4497 行目付近と同じ）。

---

## 2. 実行コマンド全文

作業ディレクトリはいずれも `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`。
環境変数は 2.3／2.5 とも**不要**（アプリ全体走行のみ1つ必要。§4）。

### 2.1 LTX 2.3（`.venv-engine`）

この仮想環境には `pytest` が入っている（`.venv-engine\Scripts\pytest.exe` が実在）ため、
ランナーを噛ませる必要はない。`--noconftest` は必須（アプリ側の `tests/conftest.py` が
FastAPI アプリを組み立てるが、この仮想環境には FastAPI が無い）。

PowerShell（1行。実際は改行なしで実行した）:

```powershell
.\.venv-engine\Scripts\python.exe -m pytest --noconftest -p no:cacheprovider tests\test_block_swap_release.py tests\test_chain_reference_engine.py tests\test_gemma_keep_resident_move.py tests\test_ic_lora_engine_conditioning.py tests\test_ic_lora_forward.py tests\test_outpaint_canvas.py tests\test_outpaint_pyramid_blend.py tests\test_pipeline_vae_mode_swap.py tests\test_pruned_video_decoder.py tests\test_registry_swap.py tests\test_retake_math.py tests\test_worker_fused_dequant_resolve.py tests\test_worker_keep_resident_resolve.py tests\test_worker_vae_mode_resolve.py
```

`-p no:cacheprovider` は `.pytest_cache` を作らせないためで、結果には影響しない。

### 2.2 LTX 2.5（`.venv-engine-ltx25`）— sys.path 追加ランナー経由

**この仮想環境には `pytest` が入っていない**（`.venv-engine-ltx25\Scripts\pytest.exe` は存在せず、
`python -c "import pytest"` は `ModuleNotFoundError: No module named 'pytest'`）。
したがって既存の作法どおり、**アプリ側仮想環境の `site-packages` を `sys.path` の末尾に足す**
ランナーを経由する。**末尾に足す**のは、`torch` と `ltx_core` の解決をエンジン側に勝たせるため
（`Docs/VERIFICATION_LOG.md` §76.4 末尾・§78・§10855行目付近の申し送りと同一手順）。

既存のランナー `outputs/ltx25-sage-gate/run_ltx25_pytest.py` は対象ファイルがハードコードされて
いる（`test_ltx25_keep_resident_registry.py` 固定）ため、**対象を `argv` から受け取るだけ**の
同型ランナーを scratchpad に置いて使った。中身は以下のとおり（既存ランナーとの差は
最終行の `args` の扱いのみ）。

置き場所:
`C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\fbc2fa77-1012-4a5e-9383-9efc3f240f9b\scratchpad\implA\run_ltx25_pytest.py`

```python
"""§3-90 implementer A: run tests inside .venv-engine-ltx25.

Same shape as outputs/ltx25-sage-gate/run_ltx25_pytest.py (the documented
runner): that venv has no pytest, so the app venv's site-packages is APPENDED
to sys.path (appended, never prepended: torch / ltx_core must keep resolving to
the engine venv). --noconftest because the app conftest builds a FastAPI app
this venv cannot import. Targets come from argv.
"""
import sys
from pathlib import Path

ROOT = Path("S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni")
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv" / "Lib" / "site-packages"))

import pytest  # noqa: E402

args = sys.argv[1:] or [str(ROOT / "tests")]
raise SystemExit(pytest.main(args + ["--noconftest", "-p", "no:cacheprovider"]))
```

実行（PowerShell。1行）:

```powershell
.\.venv-engine-ltx25\Scripts\python.exe "C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\fbc2fa77-1012-4a5e-9383-9efc3f240f9b\scratchpad\implA\run_ltx25_pytest.py" tests\test_ltx25_band.py tests\test_ltx25_outpaint.py tests\test_ltx25_reference_encode.py tests\test_ltx25_keep_resident_registry.py tests\test_retake_math.py tests\test_outpaint_canvas.py tests\test_outpaint_pyramid_blend.py
```

**scratchpad はセッション限りの領域なので、再走するならランナーを恒久的な場所へ移すこと。**
既存の `outputs/ltx25-sage-gate/run_ltx25_pytest.py` の隣（例: `outputs/start-end-bridge-2026-09-07/`）
に置くのが、既存の作法と揃う。

---

## 3. 実測結果と、基準「69／235」を再現できなかった理由

### 3.1 実測

| 対象 | 実測 |
|---|---|
| LTX 2.3（`.venv-engine`・14ファイル） | **247 passed / 0 failed**（約6〜7秒） |
| LTX 2.5（`.venv-engine-ltx25`・7ファイル） | **172 passed / 0 failed**（約5〜6秒） |

いずれも `chain_math.py` のコメント再整形を含む最終状態で再走し、同じ結果を確認している。

本改修で追加したエンジン側テストは1件:
`tests/test_ltx25_band.py::test_a_marker_clear_with_both_a_head_and_a_tail_band_builds_all_three`
（`bridge` の最終区画が「キーフレーム標識クリア＋頭の凍結＋尾の凍結」を1回の
`_band_conditionings` に同時に載せる初の出荷経路であることを固定する）。
**アプリ側仮想環境では `test_ltx25_band.py` はモジュールごと skip される**（`importorskip("ltx_core")`）
ため、この1件はアプリ全体の件数には現れず、LTX 2.5 仮想環境でのみ実走・PASS を確認した。

### 3.2 §101.10 の「69 passed／235 passed」が再現できなかった理由（事実のみ）

1. **§101.10 に実行対象ファイルの記載がない。** 記載は仮想環境名と `--noconftest` のみで、
   どのファイルを渡したのかが書かれていない。
2. **`tests` ディレクトリ全体を渡す形ではない。** `.venv-engine` で
   `python -m pytest tests --noconftest -p no:cacheprovider` を実行すると、
   **収集エラー34件で中断する**（`!!! Interrupted: 34 errors during collection !!!` / `34 errors in 4.40s`）。
   エラーの内訳はアプリ依存を要求するテスト群（`test_mcp_*.py`・`test_outpaint_api.py`・
   `test_runtime_state.py`・`test_smoke.py`・`test_models_endpoint_compat.py` など）。
3. **収集エラーを許容しても一致しない。** 同じコマンドに `--continue-on-collection-errors` を足すと
   **`20 failed, 1267 passed, 223 errors`** となり、69 とも 235 とも桁が違う。
4. したがって 69／235 は**より狭い、明示されていないファイル集合**の値である。
   **推測でファイル一覧をでっち上げて数字を合わせにいくことはせず**、§0 の基準で
   自分の集合を定義し直して全緑を確認する方針を採った（本書 §1・§2 がその定義）。

**文書化時の申し送り**: 新節を書く際は、§101.10 の 69／235 の由来一覧を掘り起こすか、
あるいは本書 §1 の集合を**今後の基準として明記し直す**かのどちらかを選ぶ必要がある。
前者ができないなら後者を採り、以後は「247／172」を基準値として引き継ぐのが実務的である。

---

## 4. アプリ側仮想環境（`.venv`）の全体走行

環境変数 `LTX_MCP_BASE_URL` を実バックエンドとは別のポートへ向けるのは、
`test_mcp_registration` の「バックエンド到達不可」テストを、実バックエンド（ポート18620）
稼働中でも成立させるため。

PowerShell:

```powershell
$env:LTX_MCP_BASE_URL = "http://127.0.0.1:18999"
.\.venv\Scripts\python.exe -m pytest --junitxml="C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\fbc2fa77-1012-4a5e-9383-9efc3f240f9b\scratchpad\implA\pytest_after.xml" -p no:cacheprovider
```

**結果: 2,235 passed / 25 skipped / 0 failed**（216秒）。

- 基準は 2026-09-05 時点の **2,202 passed / 25 skipped**（§101.10）。差分は **+33 passed・skip増減ゼロ・失敗ゼロ**。
- +33 はすべて本改修の新規テストで、内訳は次のとおり（合計が33で完全一致することを確認済み）:
  - `test_bridge_audio_boundaries_hold_across_the_grid` … **24**（6クリップ集合 × 4帯幅）
  - `test_the_schedule_tables_are_inert_outside_reverse` の `bridge` 行追加 … **3**（3クリップ集合）
  - `test_new_frames_px_identity_holds_everywhere` の `bridge` 行追加 … **2**
  - 新設の単体4件（`test_forcing_reverse_with_a_start_source_trips_the_assert` /
    `test_the_same_last_clip_boundary_holds_in_bridge` /
    `test_bridge_accepts_a_one_latent_overlap_like_reverse` /
    `test_the_bridge_worked_example_is_pinned`）… **4**
- 既存テストの削除・skip化はゼロ。反転（意味を逆にした）テストは3件あるが、いずれも
  件数は増減しない（`test_a_start_source_with_two_or_more_clips_selects_bridge` /
  `test_the_clip_count_and_the_start_source_pick_the_mode_and_the_segment_list` /
  `test_start_and_end_source_together_bridge_two_or_more_clips`）。
- `--junitxml` の出力は scratchpad の `pytest_after.xml`（291KB）。**scratchpad はセッション限りなので、
  残す必要があれば `outputs/` 配下へ複製すること。**

**注意（既知の落とし穴）**: 上記コマンドにさらに `-q` を足すと、`pyproject.toml` の
`addopts` 側の `-q` と二重になって**集計行そのものが消える**。件数を読みたいときは `-q` を付けないこと。

---

## 5. 参考 — `bridge` の assert 到達性の網羅掃引（GPU不要・数秒）

`compute_chain_layout` の assert は発火すると 500 になるため、「422 で止まるべきものが
素通りしていないか」を掃引で確認した。スクリプトは
`…\scratchpad\implA\bridge_sweep.py`（同じく scratchpad）。

- 掃引範囲: 14クリップ集合 × 9 fps × `kv` 4種 × stage-2窓 2種 × `source_context_px` 6種
  × `end_context_px` 7種 = **41,580通り**
- 結果: **assert 発火 0件**／受理 18,876／`ValueError`（422）拒否 22,704
- 拒否の内訳はすべて既存の検査（`source_context_px` の上限・最終クリップ空き検査・
  `degenerate audio overlap`・既知の分数fps音声タイリング癖）で、`bridge` が新たに
  作った拒否は無い
- 実測の最小余裕: 音声 **4潜在**（`n_end_a + ka_list[-1]` 対 `seg_audio[-1]`）／
  開始素材側 **11潜在**（`n_ctx_a + n_end_a` 対 `a_total`）。
  計画が引用していた「2」は単一クリップを含む全モード横断の値で、`bridge` 単独ではより広い。
  この2つの実測値は `chain_math.py` のコメントと新設 sweep テストに反映済み。
