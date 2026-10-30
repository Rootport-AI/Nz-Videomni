# §1-106 MiniMax H3 の Chained（連結）の GPU の門 G-C0〜G-C2（2026-10-29）

§1-106 の GPU の門の台本です。**実行済み。結果は記録簿 `Docs/VERIFICATION_LOG.md` §184.6。** 製品コードは変更しません。製品の入口 `POST /generate/chain` を HTTP で呼び、製品が `metadata.json` に残す証拠で合否を決めます。数値の正本は記録簿です（ここには結果を書きません）。

## 中身

| ファイル | 内容 |
|---|---|
| `g_chain.py` | 門の台本（HTTP クライアント。製品を import しない）。門の名前 `G-C0`／`G-C1`／`G-C2`、後片付けの `cleanup`、CPU だけの `--selfcheck` を受ける |
| `run_gates.ps1` | 全体の流れ。前処理の確認 → 自己検査 → 計測台本 → バックエンドの起動 → 3 つの門 → 後片付け |
| `run_common.ps1` | 起動の共通部。`h3-chain-2026-10-29\run_common.ps1` の写し（1 条件＝1 プロセス・時間切れでプロセスの木ごと停止・コンソールを `<名前>.console.log` へ） |
| `README.md` | このファイル |

実行後に増えるもの: `g_selfcheck.*`、`g_c0_harness.*`（計測台本の出力）、`g_c0.*`・`g_c1.*`・`g_c2.*`（各門の JSON とコンソール）、`g_cleanup.*`、`backend.console.log`、`run_gates.runner.log`。動画と `metadata.json` は `outputs\<ジョブ番号>\`（git 管理外）に残ります。

## 走らせ方

リポジトリ直下で（GPU を使う。AviUtl2 を閉じ、ほかに GPU を使うプロセスが無い状態で）:

```
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-chain-gates-2026-10-29\run_gates.ps1
```

流れ（`run_gates.ps1` が順に行う）:

1. 前処理の確認: `nvidia-smi` の表示、AviUtl2 の起動検査（起動していれば止まる）、H3 の重み・Turbo LoRA・2 つの仮想環境の存在、GPU の使用量が 2,000MiB 以下、コミット上限（物理メモリ＋ページファイル）が 45GB 以上、指定のポートが空いていること。
2. `g_chain.py --selfcheck`（CPU だけ）。
3. 計測台本 `c_chain.py` を G-C0 と同じ条件で実行（`--name g_c0_harness --width 640 --height 384 --inject clean --attention sdpa --strength 1.0 --overlap 22 --decode once --save-latents --prompts prompts_c.json`）。**バックエンドが H3 を読み込んでいる間は GPU が足りないので、必ずバックエンドの起動より前に行う。**
4. バックエンドを別のポート（既定 18629）で起動（`.venv\Scripts\python.exe main.py --port 18629`。`state.json` は先に控えを取り、最後に戻す）。
5. G-C0 → G-C1 → G-C2。1 つの門が 0 以外で終わったら止まる（`-ContinueOnMismatch` は終了コード 2〔基準の不一致〕のときだけ先へ進む。1〔失敗〕と時間切れでは進まない）。
6. 後片付け: `cleanup`（`POST /pipeline/unload`）→ バックエンドの停止 → `state.json` を元に戻す（記録簿 §180 の G-R5・G-R4 と同じ作法）。途中で止まったときもここは必ず通る。

主な引数: `-SkipPreflight`（AviUtl2・GPU 使用量・コミット上限の検査を飛ばす）、`-Port`、`-Gates G-C0,G-C1`（門の絞り込み）、`-SkipHarness`（`g_c0_harness.json` が ok のものを使い回す）、`-UseRunningBackend`（起動済みのバックエンドを使う。`state.json` には触れない）。

| 門 | 時間切れ（走らせる側） | 台本の待ち時間 |
|---|---|---|
| 計測台本 | 1,500 秒 | – |
| G-C0 | 1,500 秒 | 1,450 秒 |
| G-C1 | 2,400 秒 | 2,350 秒 |
| G-C2 | 11,000 秒 | 10,900 秒 |

## 注意

- **途中で取り消せません。** 連結のジョブには協調的な停止が無く（`op` は止められず、終わった時点で `cancelled` になるだけ）、G-C2 は一度始めたら終わるまで待つか、バックエンドごと止めるしかありません。
- G-C2 は主記憶（コミット）が 37GB 前後になる見込みです。ページファイルが小さいと失敗します（前処理の確認で物理メモリ＋ページファイルが 45GB 未満なら止まる）。
- 台本は転送の失敗に強くしてあります（`GET /jobs/{id}` の一時的な失敗は 12 回連続まで許す）が、バックエンドが落ちたら門は失敗として終わります。
- `POST /pipeline/load` は時間切れなし（`timeout=None`。別のベースモデルからの切り替えは数分かかる）。ほかの呼び出しは 30 秒。
- `state.json` の MiniMaxH3 の transformer は Ref2VA の重みが選択されていることがあるため、台本は **FL2VA の重みを登録名 `default`（`MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`）で明示して読み込みます**（計測台本と同じ重み）。
- `weight_identity.json`（重みの種別の判定の保存先）は、初回の読み込みで増えることがあります。戻しません。
- `backend.console.log` は大きくなりえます。保管庫に入れる前にサイズを確認してください。

## 門の条件（台本の `GATES` と `build_request`）

| 門 | 画角 | クリップ | 注意の計算 | ステップ／shift | VRAM 16GB mode | LoRA | そのほか |
|---|---|---|---|---|---|---|---|
| G-C0 | 640×384 | 124×3 | sdpa | 20／12・3 | 既定（送らない。H3 の既定はオン） | なし | seed 42・`crop_output` なし・`embed_mp4_metadata=false` |
| G-C1 | 1344×768 | 124×3 | sage | 8／6・3 | 明示オン | Turbo（`minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16`・強度 0.8） | seed 42 |
| G-C2 | 1344×768 | 124×13 | sage | 20／12・3 | 既定（オン） | なし | seed 42 |

プロンプト: 共通の文は `h3-chain-2026-10-29\prompts_c.json` の 1 つ目。クリップ 1・2 の追記は 2・3 つ目から共通の文を引いた残り（台本が前方一致で切り出し、再合成して元の文字列と一致することを確かめる）。13 クリップでは追記の 2 文を交互に使う。クリップ 0 は共通の文だけ。合成の式は製品と同じ（両端を取り除き、空でないものを半角空白 1 つでつなぐ）。製品の `clips[i].prompt_sha256` は合成後の文の UTF-8 の sha256。

## JSON の欄（`g_c0.json` など）

| 欄 | 意味 |
|---|---|
| `gate`・`conditions`・`request_body` | 門の名前・条件・送った本文 |
| `expected_prompts`・`expected_prompt_sha256` | 合成後のプロンプトと sha256 |
| `status_before`・`models_before`・`load_response`・`load_seconds`・`status_after_load`・`models_after_load` | 開始前の状態、読み込みの応答と秒数、読み込み後の状態（`acceleration.sage_available` が偽ならモックとみなして止まる） |
| `job_id`・`submit_response`・`job` | ジョブ番号、受付の応答、最後の `GET /jobs/{id}` |
| `poll` | `samples`（`progress`・`stage`・`clip`／`clip_count`・`current_step` が変わった時点と 60 秒おきの点）、`stage_log`（段の移り変わり）、`monotonic_ok`／`monotonic_problems`（全ポーリングでの単調性の検査）、`max_progress`、`polls`、`seconds` |
| `metadata_path`・`mp4` | `outputs\<ジョブ番号>\metadata.json` と `output.mp4` のパス、サイズ、sha256、フレーム数（PyAV で全フレームを復号して数える。PyAV が無ければ ffprobe） |
| `metadata_h3` | `metadata["h3"]` そのまま（`clips`〔`prompt_sha256`・`seconds`・`latents_sha256`・`head_equals_prev_tail`〕・`seeds`・`phases`・`loras`・`rss_peak_gib`・`commit_peak_gib` など） |
| `metadata_summary`・`metadata_request` | `backend`・`attention_used`・`vram16_mode_used`・`generation_time_seconds`・`models`・`chain`・`output`、保存された request |
| `mp4_psnr_vs_script_once` | G-C0 のみ。mp4 の sha256 が台本の `once.mp4` と違うときの、フレームごとの PSNR（最小・平均・一致フレーム数）。記録のみ |
| `memory_sampler` | バックエンドのプロセスの木（ワーカーを含む）の作業セットとコミットの最大値と、10 秒おきの推移（最大約 400 点）。`ctypes` で読む（仮想環境に `psutil` が無いため）。`--backend-pid` を渡されたときだけ |
| `verdict` | `pass`・`failed`（落ちた基準の名前）・`checks`（基準ごとの `ok`・`detail`・`kind`＝`criterion`〔合否に使う〕／`record`〔記録のみ〕）・`records`（所要・クリップごとの秒・復号の秒・VRAM の山・主記憶） |
| `ok`・`error`・`traceback`・`seconds_total`・`cleanup` | 全体の成否と例外 |

終了コード: 0＝すべての基準に合格、1＝走行の失敗（例外・ジョブの失敗・時間切れ・モックのバックエンド）、2＝完走したが基準に落ちた。

## 合否の基準

全門に共通（落ちたら 2）: ジョブが `completed`／`metadata.backend` が `minimax-h3`／進捗とクリップ番号が単調で最後が 1.0（`clip_count` は変わらない）／`h3.clips` がクリップ数ぶん／2 本目以降の `head_equals_prev_tail` がすべて真／mp4 が全フレーム再生でき、フレーム数が期待どおり／`seeds` が 42＋i／`attention_used`・`vram16_mode_used`・ステップと shift が条件どおり／合成後のプロンプトが期待と一致（保存された request を合成し直したものと `prompt_sha256` で確かめる）。

- **G-C0**（フレーム数 328）: 追加の基準 **第 1 基準＝各クリップの `latents_sha256`（映像と音声）が、台本の JSON（`--compare`）の `clips[i].latents.v_sha256`／`a_sha256` と一致**。種が台本と一致、`embed_mp4_metadata` が偽。**第 2（記録のみ）**＝mp4 の sha256 を台本の `decode.once.sha256` と比べる。不一致なら PSNR を `mp4_psnr_vs_script_once` に記録する（製品は `deterministic=True`、台本は偽なので、復号の cuDNN の選択が違いうる既知の差）。`--compare` が無ければ判定できないので不合格。
- **G-C1**（328）: 追加の基準 `h3.loras` に Turbo が 1 件載り、`pairs_applied` が 1 以上。記録: 所要、VRAM の山（クリップごとの段と復号の段の最大）。
- **G-C2**（1,348）: 追加の基準 `h3.phases.decode` に秒数と VRAM がある、主記憶が記録されている（ワーカー自身の `commit_peak_gib` か台本の標本）。記録: 復号の秒数と VRAM の山、クリップごとの秒、主記憶の作業セットとコミットの最大。11,544 フレームまでは測らない（実測は §184 に残し、上限の見直しは §1-95）。
- **G-C3**（操作パネル）はオーナーの目視で、この台本の範囲外。

## 自己検査

```
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-chain-gates-2026-10-29\g_chain.py --selfcheck
```

CPU だけ・HTTP なし。プロンプトの合成と sha256 の式（製品の `compose_clip_prompt` と同じ結果であることも、製品を import できる環境では確かめる）、3 つの門の送信本文、単調性・潜在の sha256 の比較・フレーム数の判定、各門の合否の判定を小さなデータで検査する。最後の行が `SELFCHECK PASS`。
