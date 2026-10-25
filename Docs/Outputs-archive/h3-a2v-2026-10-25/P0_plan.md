# MiniMax H3 の A2V（音声から映像）実装計画 — 単発 `/generate` の `source_audio` と、行ごとの連続値マスク

- 台帳: §1-103。日付: 2026-10-25。起草: 設計担当（Opus）、査読と裁定: 監督。敵対的レビュー（Opus・R3）: 指摘 16 件（Critical 0・Major 5・Minor 11）を反映済み（§9）。
- 調査の正本（セッション限り。完了時に `Docs/Outputs-archive/h3-a2v-2026-10-25/` へ写す）: `scratchpad\h3a2v\R1_audio_encoder_retake.md`（音声符号化器・Retake・連続値マスクの仕様）、`R2_chain_api_entry.md`（受け口の棚卸し）、R3（計画のレビュー。`~/.claude/plans/cosmic-giggling-bubble-agent-a4d34dc7f18179840.md`）。
- パス: バックエンドは `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\`（以下 BE）。操作パネルのコードは `BE\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\`（以下 FE）。操作パネルの文書は `BE\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\Docs\`（以下 FE-Docs）。

## 0. 結論

H3〔MiniMax H3〕の A2V〔音声から映像〕は、単発の `POST /api/v1/generate` に任意の項目 `source_audio: {audio_id, strength}` を足して受ける。エンジンには、映像と音声の両方に使える「行ごとの連続値マスク」を足す（マスクの値 m は 0〜1。m=0 は元の潜在に固定、m=1 は通常どおり作り直す）。V1 で使うのは音声のマスクだけで、値は全行で一つ、m = 1 − Strength。

- Strength 1.0 のときは音声 VAE〔音声の圧縮器〕の復号を飛ばし、元の波形を mux〔映像と音声を 1 ファイルにまとめる〕する。
- Strength が 1.0 未満のときは、生成された音声潜在を今日と同じ経路で復号する（モデルの解釈した音声を聞かせる）。
- 操作パネルの A2V カードは見た目を変えない。サーバーが公開する規約 `constraints.source_audio` があるエンジンのときだけ、送り先を `/generate` に切り替える。
- `/generate/chain` とそのスキーマ、`engine/`、`engine25/` には触らない。H3 は `chain`（複数クリップ）を引き続き断る。

### オーナー裁定（2026-10-25・確定）
1. 受け口は単発 `/generate` ＋ `source_audio`。カードは残し、H3 のときだけ送信先を切り替える。
2. エンジンの部品は行ごとの連続値マスク（映像・音声の両方。V2A〔映像から音声〕は部品だけで、製品には出さない）。
3. Strength は既定 1.0、1.0 未満は ComfyUI 式。
4. 出力の音声は Strength 1.0 のときだけ元波形を重ねる。
5. 音声の長さの規則は LTX と同じ（映像の尺が主・音声は尺以上が必須・足りなければ 422・切り詰めのみ・トリム指定は出さない）。
6. キーフレーム（先頭画像）との併用は許可。

### 監督の裁定
- **LTX 側の `source_audio` の拒否**: LTX 2.3／2.5 の `REJECT_TABLE` にデータ行を 1 行ずつ足し、文言は共有の `H3_ONLY_REFUSALS`（`BE/services/engines/ltx/adapter.py:148-150`）に置く（VRAM 16GB mode の前例と同じ形。無いと「もう一方の LTX を選んで」という誤った案内になる）。LTX の公開する `unsupported_features` に `source_audio` が増えるので、完全一致を見ている試験の期待値を更新する（§3 の T1・T2・F3）。
- **式**: §2.1 の一般形。実装は整理した形を使わず `(x0_b − x)/σ` をそのまま計算する（R3 #1）。
- **幾何の置き場**: 新モジュールは作らない。`RequestRules` の任意の項目 1 つで済ませる（§2.5。R3 #16。前例は `prompt_token_limit`）。

## 1. 文脈（なぜ・何を）

- 使い方の中心は「立ち絵＋音声で喋らせる」（先頭フレームのキーフレーム＋音声）。
- 今は使えない理由: H3 の音声 VAE は復号器しか持たない（`BE/engine_h3/audio_vae.py:32-36`、59-67 行で符号化器の重みを読み捨てている）。ワーカーのプロトコルに音声の入力が無い（`engine_h3/worker.py:41-56`、165-170）。`/generate/chain` はスキーマの検査が LTX の規則（8n+1、64 の倍数、25 潜在/秒、distilled は 8 ステップ、キーフレームに strength 0.8 を補完）を直書きしているので H3 では使えない（R2）。操作パネルの A2V カードは H3 が `chain` を公開しているために灰色（FE `shell/featureScope.ts:151`）。
- 仕組みの根拠: DiffSynth の Retake は残す音声の行を毎ステップ元の潜在に差し替え、時刻を 1.0 にする（最終値は元の潜在そのもの。R1 §3.1）。ComfyUI は 0〜1 の連続値のマスクを受け、値 m の行を「σ が m·σ の行」として扱う（R1 §3.2・§4.1）。ComfyUI は GPL-3.0 なので式だけを使い、コードは写さない。
- 今回の部品は §1-106（H3 の Chained）でも使い回す。映像の行のマスクはエンジンに置くが API には出さない。

## 2. 設計

### 2.1 エンジン: 行ごとの連続値マスクの式（正本は完了後 MINIMAX_H3_DESIGN.md §3.1 へ）

記号: 流れ（映像か音声）ごとの σ=σ_i、σ'=σ_{i+1}（評価する σ_i は常に正。`scheduler.py:39-41`）。状態 x は float32 で、今日と同じく純ノイズから始める。clean は元の潜在、inject は DiT に入れる固定値（音声は clean、映像は `0.999·clean + 0.001·noise`。noise の出どころは §2.2）。m は行ごとの値。DiT の出力 v は「データ方向の速度」で `x0 = x + σ·v` と読む（`engine_h3/scheduler.py:16-20`）。

各ステップ:
1. **DiT への入力**: `x̃ = m·x + (1−m)·inject`
2. **行の時刻**: m<1 の行は `t = min(1 − m·σ, 上限)`。上限は映像 0.999、音声 1.0。m=1 の行は流れの t（今日と同じ）。テキスト行は映像の t、条件行は今日どおり `max(t_v, 0.999)`。
3. **推定した元の潜在**: `x0̂ = x̃ + σ·(m·v)`（ComfyUI が速度に m を掛け、標本器が流れの σ で割り戻すのと同じ形）
4. **元の潜在との混合**: `x0_b = m·x0̂ + (1−m)·clean`
5. **状態の更新**: 状態の速度 **`v_s = (x0_b − x)/σ`** をそのまま計算し、既存の `euler_step(x, v_s, σ, σ')` に渡す。展開すると `v_s = m²·v + (1−m²)·(clean − x)/σ + m(1−m)·(inject − clean)/σ` で、音声（inject = clean）では最後の項が消える。実装は展開形を使わない。

ループの後に、m=0 の行だけを `torch.where(m==0, clean, x)` で clean に置き換える。

性質:
- **m=1**: v_s = v で今日と同じ更新。全行が 1 のマスクは組み立ての時点で None にし、None の経路は今日のコードを一字も通らない形にする（ビット一致は作りで保証）。
- **m=0**: DiT が受け取るのは毎回 inject（音声は clean）、時刻は上限（音声 1.0）。更新は `clean + (σ'/σ)(x − clean)` で DiffSynth の `return_to_timestep` と同じ式。最後の置き換えで clean にビット一致。
- **0<m<1**: ComfyUI の仕様と代数的に同じ。最終値は `m·x0̂ + (1−m)·clean` で clean に戻り切らない。
- RoPE〔位置の回転埋め込み〕の座標は変えない。行の数は増えないので、快適上限（§1-95）の系列長も変わらない。

**粒度**: V1 は映像は潜在フレームごと（`[T']`）、音声は潜在ごと（`[A]`、左右で共通）。より細かくするにはマスクを `[T', lh, lw]` で持ち、2×2 パッチ単位で最大値に寄せて行に並べる処理とトークン単位の時刻が要る（式は同じ）。

### 2.2 エンジン: 実装の場所

**新規 `engine_h3/denoise_mask.py`**（torch だけに依存。CPU で試験できる）

```python
@dataclass(frozen=True)
class StreamMask:
    m: Tensor      # f32、状態に掛けられる形（音声 [1,1,A]・映像 [1,1,T',1,1]）
    rows: Tensor   # f32 [行数]。系列の並び順（音声 m.repeat(2)、映像 repeat_interleave(gh*gw)）
    clean: Tensor
    inject: Tensor
    zero: Tensor   # bool。m==0
def audio_mask(clean, per_latent) -> StreamMask | None                       # 全部 1 なら None
def video_mask(clean, per_frame, *, grid: int, noise: Tensor) -> StreamMask | None
def dit_input(x, mask) -> Tensor                                 # None ならそのまま x
def state_velocity(v, x, mask, sigma: float) -> Tensor           # (x0_b − x)/σ をそのまま
def finish(x, mask) -> Tensor
def masked_row_times(t_stream: float, sigma: float, rows, cap: float) -> Tensor
```

**映像の inject の noise（R3 #11・R1 で確認済み）**: その行の**初期雑音（997-1003 で引いた映像の乱数）を使い回す**。ComfyUI は `KSamplerX0Inpaint` に渡す `noise` を初期雑音そのものにし（`comfy/samplers.py:990-991`。`inpaint_options` で `random` を指定したときだけ別の雑音）、毎ステップ同じ雑音で `0.999·clean + 0.001·noise` を作る（`model_base.py:2303-2311`。音声は clean のまま）。DiffSynth の映像 Retake は clean そのもの（雑音増しなし）。本製品は ComfyUI の 0.999 を採る（§6 の 4）。新しい乱数は引かないので乱数の引き順は今日と同じまま。そのため `_denoise` は「行ごとの値と clean」を受け、マスクの組み立ては `_denoise` の中（乱数を引いた後）で行う。

**`engine_h3/pipeline_h3.py`**
- `time_rows`（257-270）にキーワード引数 `audio_t: Tensor|None=None, video_t: Tensor|None=None`。None のときは今日のコードそのもの。値があるときは `t[layout.audio]=audio_t`、`t[layout.video]=video_t` を代入する。条件行の `t[layout.cond] = max(...)` は今日どおり**最後**に置く（267-268 の順を保つ）。
- `H3GenerateRequest`（157-179）に `source_audio: H3AudioInput | None = None`（`H3AudioInput(path: str, strength: float = 1.0)`）。
- `validate_request`（186-233）: パスが空でない、ファイルがある、強度が有限で 0〜1。
- `generate`（567〜）: ①画像を読んだ直後（597 の後、テキストエンコーダより前）に `source_audio.load_source_waveform` で波形を読む（短ければここで止める）。②`image_encode` の後に新しい段 `audio_encode` を置き `self._encode_audio(wave, a_lat, entry)`（strength 0 は m=1 で効き目が無いので飛ばす）。③`_denoise(..., audio_fix=(clean_a, per_latent_m))`。④`_decode(..., source_wave=wave if strength==1.0 else None)`。⑤戻り値の `h3` に `a2v` を A2V のときだけ足す（633-635 の LoRA と同じ作法）。
- `_encode`（684）: 進捗の総数を `3 + 画像の有無 + 音声の符号化の有無`。`_encode_image` の `("encode", 4, 4)`（732）は添字を引数で受ける。アダプタの `_progress_frac` は `encode` を常に 0.06 に写すので（`ltx/adapter.py:523-524`）、A2V でないジョブの進捗は変わらない。
- **`_denoise`（915〜）**: 引数 `audio_fix=None, video_fix=None`（clean と行ごとの値の組）。

| 場所（行） | 変更 |
|---|---|
| 997-1003（乱数） | 引き順は変えない。乱数を引いた後に `am = audio_mask(...)`、`vm = video_mask(..., noise=video の初期雑音)` を組む（全部 1 なら None） |
| 1016（時刻） | `time_rows(..., audio_t=None if am is None else masked_row_times(t_a, sig_a[i], am.rows, 1.0), video_t=None if vm is None else masked_row_times(t_v, sig_v[i], vm.rows, 0.999))` |
| 1020（入力） | `embed_rows(text_refined, audio_to_rows(dit_input(audio, am)), patchify_video(dit_input(video, vm)), cond_rows=cond)` |
| 1038-1041（更新） | `v_audio = state_velocity(v_audio, audio, am, sig_a[i])`。映像も同じ。`euler_step` は今日のまま |
| 1046 の前 | `audio = finish(audio, am)`、`video = finish(video, vm)`。`entry["mask"]` はマスクがあるときだけ書く（LoRA の前例 634-635。A2V でないジョブの metadata を変えない） |

- `_decode`（1086-1141）: `source_wave` が渡されたら 1114-1129 の音声 VAE を組まずに（常駐している物は触らずに）`wav = source_wave`。`entry["audio_output"] = "original"`、`audio_vae_source = "skipped"` を記録。`write_mp4` の `audio_keep_samples` による `round(F·32000/24)` への切り詰めは今日のまま（`engine_h3/media_out.py:154-156`）。

**`engine_h3/dit.py`**: 説明文だけ（28 行の「M は 1〜3」を「重複を除いた時刻の数。マスクがあると 3 を超えうる」）。確認済み: `H3AdaLNProj.forward` は `m = x.shape[0]`（283-284）、`adaln_embed` は t=1.0 でも `i0` を G−2 に抑える（529-532）、`lowvram` も M に依存しない（`lowvram.py:131`、196）。

### 2.3 エンジン: 音声の符号化器（DiffSynth から移植。Apache-2.0）

**`engine_h3/audio_vae.py` に足す**: `Snake`（α は生の値。exp を取らない）、`ResidualUnit`、`EncoderBlock`、`DACEncoder`（ストライド 2・4・4・5・5）、`GeGluMlp`、`CausalAttention`（`q_bias`・`v_bias` は Parameter、`zero_k_bias` は**永続の** buffer〔`persistent=True`。非永続だと `load_state_dict(strict=False)` が捨てて meta のまま残る。`sft_load.py:90-97`、105〕。ファイルの名前のまま受ける。出力はヘッド方向に平均し `adaptive_avg_pool1d` で 32 に）、`AttnProjection`、**`H3AudioEncoder`**（`encoder`・`pre_block`・`mean_proj`）。
- `from_file(path, *, device)`: `sft_load.load_streaming`。key map は新しい `audio_encoder_key_map`（`encoder.`・`pre_block.`・`mean_proj.` を残し、`logs_proj` と復号器と統計量のテンソルは捨てる）。統計量 `latents_mean`／`latents_std` は復号器と同じく `__metadata__` から読む（`audio_vae.py:27-30`、221-226）。2 つの key map は 1 つの接頭辞の表から作る。
- `encode(wave: [2, L] f32) -> [2, 32, ceil(L/800)]`: 右側を 0 で埋めて 800 の倍数にし、平均（mean）を使い、`(z−mean)/std` で正規化。すべて fp32。
- 復号器の `H3AudioVAE` と `audio_vae_key_map` は変えない（T2V の読み込みと常駐を変えないため）。

**新規 `engine_h3/source_audio.py`**: `load_source_waveform(path, *, keep_samples, read_samples) -> (Tensor[2,N] f32, info)`
- PyAV〔Python から ffmpeg を使うライブラリ。18.1.0〕で復号。`AudioResampler(format="fltp", layout=None, rate=32000)` でチャンネルの並びは変えず 32 kHz に。**読み終えたら必ず `resample(None)` で末尾を出し切り、返り値の `None` の要素は捨てる**（入力がすでに 32 kHz・fltp のときは素通しになり `[None]` が返る。`av/audio/resampler.py:55-79`）。途中で形式が変わる入力の `ValueError` は日本語の理由をつけて包む。1 チャンネルなら 2 つに複製、3 チャンネル以上なら先頭 2 つ。ピーク正規化はしない。`read_samples` に達したら読み込みを止める。
- **長さの歯止め**: `N < keep_samples − 32`（1 ミリ秒）なら日本語の理由つき `ValueError`。32 サンプル以内の不足は 0 で埋めて `keep_samples` に揃える（事前検査の許容と一致させる。R3 #6）。
- **符号化する長さ（決定）**: `min(手に入る長さ, A·800 + 32000)` サンプル（約 1 秒の余裕）を符号化し、潜在を先頭から A 個に切り詰める。理由: pre_block の注意は因果（後ろを見ない）なので余裕が効くのは畳み込みの受容野の末尾だけ。長い音源を全部符号化する無駄も避ける。例: オーナーの wav（3.886 秒）と 90 フレームでは A=150、映像の長さは 120,000 サンプル、手元 124,352 サンプルを 124,800 に 0 埋めして符号化すると 156 潜在、先頭 150 を使う。
- `_encode_audio` は `_encode_image`（706-733）と同じ作法: そのジョブのためだけに組み、fp32 で 1 回符号化し、DiT を組む前に `del`＋`_free_cuda()`。常駐させない。

### 2.4 ワーカーのプロトコル（`engine_h3/worker.py`）
- `generate` に任意キー `source_audio: {"path": str, "strength": float}`。`_GENERATE_FIELDS`（165-170）に加え、`_SOURCE_AUDIO_FIELDS = {"path","strength"}`。知らないキーは `images` と同じくログに 1 回出して無視。
- 新関数 `_source_audio(msg)` を `_generate_request`（316-350）に渡す。`done` の `h3.a2v` は A2V のときだけ。説明文（41-56、68-75）とセルフテスト（`--source-audio PATH --audio-strength S`。79-86）。

### 2.5 アプリ: API の形

```json
POST /api/v1/generate
{ "prompt": "...", "width": 896, "height": 1152, "num_frames": 90, "frame_rate": 24,
  "conditioning_images": [{"image_id": "img-…", "frame_idx": 0}],
  "source_audio": {"audio_id": "aud-…", "strength": 1.0} }
```

**`api/models.py`**: 新モデル `SingleSourceAudioSpec(audio_id: str = Field(..., min_length=1), strength: float = Field(1.0, ge=0.0, le=1.0))`。チェーン用 `SourceAudioSpec`（887-905）は変えない。`GenerateRequest` に `source_audio: SingleSourceAudioSpec | None = None`（追加だけ）。`generation_mode`（786-788）は t2v/i2v のまま（A2V かどうかは metadata の `a2v` ブロックで表す。mock とアダプタの `mode == "i2v"` 分岐を壊さない）。`to_clip_request`（1718〜）は今日どおり音声を写さない（説明文を直す）。

**系統ごとの受理と拒否**
- H3: `services/engines/h3/adapter.py` の `HONOURED_FIELDS`（225-245）に `source_audio`。ほかの拒否（`REJECT_TABLE` 202-213、`UNSUPPORTED_FEATURES` 279-281 の `chain` 含む）は変えない。キーフレームとの併用は通す。キーフレームの強度は今日どおり断る。
- LTX 2.3／2.5: `REJECT_TABLE` に `("source_audio", "source_audio", lambda r: r.source_audio is not None)` の行。文言は `H3_ONLY_REFUSALS["source_audio"] = "単発の A2V は MiniMax H3 専用です。LTX の A2V は Chained の音声で使えます。"`（2.5 も同じ表を共有。`ltx25/adapter.py:529-535`）。

**規約の公開（新モジュールは作らない。R3 #16）**
- `services/engines/ltx/request_rules.py` の `RequestRules` に `source_audio_strength: SamplingRange | None = None` を足す。`constraints()` は値があるときだけ `"source_audio": {"strength": {...}}` を出す（前例 `prompt_token_limit` 115・136）。LTX の constraints は 1 バイトも変わらない。
- H3 の `REQUEST_RULES`（`h3/adapter.py:130`）に `source_audio_strength=SamplingRange(0.0, 1.0, 0.05, 1.0)`。
- 音声の幾何（40 潜在/秒・800 ホップ）はアプリ側で使わない。事前検査は秒で比べ、エンジンは `engine_h3/constants.py` の値を使う。

**事前検査**: `api/generate.py` でキーフレームの存在を確かめた後（100-102 の後）に、①`audio_upload_store.path_for` で 404、②新しい `PipelineManager.preflight_single_source_audio(spec, num_frames, frame_rate)`（ffprobe で音声の流れの有無〔前例 `pipeline_manager.py:1265-1268` の `has_audio_stream`〕と長さを調べ、`duration + 0.001 < F/fps` なら 422 `SOURCE_AUDIO_TOO_SHORT`。測れないときはエンジンの歯止めに任せる）。既存の `preflight_source_audio`（1243-1283。チェーン用）は変えない。

### 2.6 アプリ: 実行と metadata
- `services/pipeline_manager.py` の `run_job`（958-969）: `source_audio_path` を解決し、値があるときだけ `**{"source_audio_path": p}` で `runner.generate` に渡す（LTX には前段で 422 になるので届かない）。アプリ側の出所（`source_audio_id`・`strength`・ffprobe の事実）は `a2v_provenance=` として `_finalize`（1802〜）経由で `_write_metadata` に渡す（inpaint の前例と同じ）。
- `services/engines/h3/adapter.py`: `H3Runner.generate` を上書きし `source_audio_path` を追加で受けて backend に渡す（`LTXRunner.generate`（806-845）は名前つき引数なので変えない）。`_RealBackendH3.generate` のペイロード（511-533）に `source_audio` を値があるときだけ足す。置く位置は `output_path` の前（`keep_resident` が末尾であることを見る試験 `test_engine_h3_adapter.py:608`）。強度は `request.source_audio.strength` から読む（「読まれている」試験 105-107 の字面に合う）。パス未解決なら `RuntimeError`。
- **mock**: `_MockBackendH3(_MockBackend)` を作り `_MOCK_BACKEND_CLS`（595）を差し替える（`is_mock` の `isinstance` は保たれる）。`generate(..., source_audio_path=None)` は `super().generate` で mp4 を作り、A2V のときだけ `dataclasses.replace(outcome, h3={"a2v": {...}})`（ffprobe の事実、`audio_output`、`"mock": True`、`muxed_audio_samples: None`。mock は音声を mux しないので mux したとは書かない）。
- **metadata の `a2v` ブロック**（`_write_metadata`。1959-1960 の `h3` の扱いを直す）: `h3` の写しから `a2v` を取り出し、残りが空でなければ `metadata["h3"]`。`metadata["a2v"] = {**エンジン側, **アプリ側}`（1969-1973 の inpaint の前例）。アプリ側は `source_audio_id`・`strength`・`source{duration_sec, sample_rate, channels}`、エンジン側は `mask_value`・`audio_latents`・`required_samples`・`encoded_samples`・`channels_used`・`resampler`・`encoder_tensors`・`audio_output`（original／decoded）・`muxed_audio_samples`。同じキーを 2 か所から書かないことを試験で確かめる。

### 2.7 操作パネル
- **featureScope**: `FEATURE_UI`（150-173）は変えない。`createSection.a2v` の意味を「チェーン経由の A2V が閉じている」と読み替える。新関数 `a2vRouteFor(unsupported, singleA2vAvailable): "generate" | "chain" | null`（`singleA2vAvailable` が真なら `"generate"`、そうでなく `createSection.a2v` が閉じていなければ `"chain"`、どちらでもなければ `null`）。`singleSectionsDisabledFor`（398-405）は 2 つ目の引数を受け、`a2v` を「経路が null か」で決める。`singleA2vAvailable` の正体は規約に `source_audio` があるかどうか（エンジン名は読まない。古いサーバーはチェーン経由のまま）。呼び出し元 `AppShell.tsx` 1540・1706（確認済み）。規約は `GET /models` → `useBaseModels` → `activeConstraints` → `SingleScreen` の `constraints` で届く（`AppShell.tsx:1705`）。
- **規約と要求の型**: `api/types.ts` の `BaseModelConstraints`（1245〜）に任意の `source_audio?: {strength: SamplingParamConstraint} | null`、`GenerateRequest`（170）に `source_audio?: {audio_id: string; strength: number}`。`shell/generationConstraints.ts` の `GenerationConstraints`（33-63）に `singleSourceAudio`（無ければ null。`resolveGenerationConstraints` に 1 行）。
- **送り先の切り替え**: `SingleScreen.tsx` の `buildSubmission`（1029-1043）で `isA2v && route==="generate"` なら `/generate` の本文（`omitStrength` 済みの画像、LoRA の音声の強度を外したもの）に `source_audio: {audio_id, strength}` を足す。`"chain"` なら今日どおり。
- **Strength のスライダー**: `GenerationForm.tsx` の `SourceAudioSection`（306-319）の中に `singleSourceAudio` があるときだけ出す。見た目は `KeyframeCard.tsx:249-263` と同じ素の `<input type="range">` に `toFixed(2)` の表示。min・max・step・既定はサーバーの `strength`（0／1／0.05／1.0）。状態は `useGenerationForm.ts` の `sourceAudioStrength`（既定 1.0）、同じ設定の判定（fingerprint）にも入れる。文言で「0.0 は音声を使わないのと同じ」と案内する。`singleClipNote`（`i18n/strings.ts:1688`・3310「長い音声は Chained の long a2v が使えます」）は経路が `generate` のときは出さない。
- **音声に合わせたフレーム数**: 新規 `modes/single/audioFit.ts`（`suggestFramesForSingleAudio(sec, fps, grid, min, max)`: `floor(sec·fps)` 以下で格子に乗る最大のフレーム数。オーナーの wav では 93→90。`singleAudioPrecheck(sec, F, fps)`: `F/fps ≤ sec`）。`useGenerationForm.ts` 1043-1095 と `audioPrecheck` は規約があるときこれを使う。`chainUtils.ts` の 25 潜在/秒（285）と `suggestFramesForAudio`（389-400）は LTX のチェーン専用として残す。
- **模擬サーバー `bridge/mockBridge.ts`**（R3 #5）: H3 の constraints（362-377）に `source_audio` を足す。LTX 2.3・2.5 の `unsupported_features` の写しに `source_audio` を足す（379-391 の「サーバーの一覧をそのまま写す」規則）。LTX が有効なとき `source_audio` 付きの `/generate` を 422 FEATURE_UNSUPPORTED にする。`handleGenerate`（1671-1759）は本文を受ける。
- **Batch A2V**: `chain` が閉じるので H3 では今日どおり閉じたまま（変更しない）。

### 2.8 MCP と Gradio
- **MCP**: `mcp_server/tools/generate.py` の `submit_generate`（151〜）に `source_audio_id: str | None = None` と `source_audio_strength: float | None = None`（`source_audio` に入れ子。強度だけの指定は POST 前に `ToolError`）。説明文「H3 専用。LTX は `submit_chain` の `source_audio_id`」。`server.py` の説明も直す。`Docs/MCP_SERVER_DESIGN.md` に D30。`plan_a2v_batch` は LTX 専用のまま。
- **Gradio**: 範囲外（§3-243 で後回し）。Gradio は知らない機能名も知らない constraints のキーも捨てる（`gradio_ui/feature_scope.py:77-86`、`gradio_ui/constraints.py:67-90`）ので影響なし。

## 3. 変更するファイル（弾ごと・担当・1 ファイル 1 担当）

全員に「設計がおかしいと思ったら止めて上に上げる（根拠＝実測値・エラー全文）。自分の判断で設計を変えない」を渡す。語調は中立。

| 弾 | 担当（モデル） | ファイル |
|---|---|---|
| 1a | E1（Opus） | `engine_h3/audio_vae.py`・`engine_h3/source_audio.py`（新）・`tests/test_h3_audio_encoder.py`（新）・`scripts/h3_a2v_g0_roundtrip.py`（新） |
| 1b | E2（Opus） | `engine_h3/denoise_mask.py`（新）・`tests/test_h3_denoise_mask.py`（新） |
| 2 | E2（Opus） | `engine_h3/pipeline_h3.py`・`engine_h3/dit.py`（説明文だけ）・`tests/test_h3_engine.py`（`time_rows` の拡張。ワーカーの既存試験もこのファイルにあるが、新しい試験は W1 の別ファイルへ） |
| 2 | W1（Sonnet） | `engine_h3/worker.py`・`tests/test_h3_worker_a2v.py`（新） |
| 3 | A1（Opus） | `api/models.py`・`api/generate.py`・`services/engines/ltx/request_rules.py`（`source_audio_strength`）・`services/engines/ltx/adapter.py`（拒否表の 1 行＋`H3_ONLY_REFUSALS`）・`services/engines/ltx25/adapter.py`（拒否表の 1 行） |
| 3 | A2（Opus） | `services/engines/h3/adapter.py`・`services/pipeline_manager.py` |
| 3 | T1（Sonnet） | `tests/test_engine_h3_adapter.py`（73-81 は `HONOURED_FIELDS` で通る。271 は `issubclass`。300-304 の constraints の完全一致）・`tests/test_ltx_adapter_audit.py`（129-130・160）・`tests/test_ltx25_adapter.py`（513）・`tests/test_ltx25_api_guard.py`（512・1620-1624）・`tests/test_engine_dispatch.py`（224-228）・`tests/test_h3_a2v_api.py`（新） |
| 4 | F1（Opus） | FE `shell/featureScope.ts`・`shell/generationConstraints.ts`・`api/types.ts`・`shell/AppShell.tsx`・`modes/single/SingleScreen.tsx`・`modes/single/useGenerationForm.ts`・`modes/single/audioFit.ts`（新） |
| 4 | F2（Sonnet） | FE `modes/single/GenerationForm.tsx`・`i18n/strings.ts`（`sourceAudio` の文言と `singleClipNote` の出し分け）・`bridge/mockBridge.ts`（§2.7 の 3 点） |
| 4 | F3（Sonnet） | FE `shell/featureScope.test.ts`・`shell/generationConstraints.test.ts`・`shell/useBaseModels.test.ts`（245・262・398）・`bridge/mockBridge.test.ts`（1001）・`modes/single/useGenerationForm.test.ts`・`modes/single/GenerationForm.sourceAudio.test.tsx`・`modes/single/audioFit.test.ts`（新） |
| 5 | M1（Sonnet） | `mcp_server/tools/generate.py`・`mcp_server/server.py`・`tests/test_mcp_tools_generate.py`・`tests/test_mcp_tools_system.py`（334-338）・`Docs/MCP_SERVER_DESIGN.md` |
| 6 | D1（Sonnet） | 文書（§5） |

**順序**: 1a と 1b は並行。G0 と G1 を確かめてから 2。2 の後に G1 の残り、3・4・5 を並行。G2 の後に G3（GPU・了承）、G4（オーナー）、6。実装差分のレビュー（Opus・読み取りのみ）を弾 2 と弾 3＋4 の後に 1 本ずつ。コミットは監督が行う（dev → 検証後 `--no-ff` で main）。

## 4. 門と合格基準

| 門 | 内容 | 合格基準 | GPU |
|---|---|---|---|
| **G0** | CPU で、オーナーの wav を符号化し、既存の復号器で戻して元と比べる（H3 の venv。`scripts/h3_a2v_g0_roundtrip.py`） | ①重みの網羅: `LoadReport.loaded == 134`（`encoder.*` 110＋`pre_block.*` 22＋`mean_proj.*` 2）、`skipped` が「`logs_proj` 2 本＋復号器＋統計量 2 本」と一致、meta のまま残る重みが無い。②判定は 1 行: **RMS〔音量の実効値〕の比が ±1.5 dB 以内 かつ（波形の相関 r ≥ 0.8 または STFT〔短時間の周波数分解〕の振幅の相関 ≥ 0.95）**。相関は ±800 サンプルの範囲で遅延を探して最大値を取る。数値と復元 wav を記録 | なし |
| **G1** | 小さなテンソルで CPU で式を試験 | ①全部 1 のマスクは None になり、`_denoise` 全体（既存の小さな実物の DiT の試験台 `tests/test_h3_engine.py:1936-1957`）がマスク無しとビット一致。②部品ごとの試験: `dit_input`（m=1 の行は x そのまま、m=0 の行は inject）、`state_velocity`、`masked_row_times` が各行で定義どおり（注意で行が混ざるので、混在マスクの `_denoise` 全体での行ごとの一致は求めない）。③m=0 の行の最終値が clean とビット一致し、DiT への入力も毎ステップ inject と一致。④時刻の集合: 音声 m=0 で 1.0、映像 m=0 で 0.999、m=0.5 で `1−0.5σ`。上限が効く。M>3 でも AdaLN と lowvram が通る。⑤`state_velocity` が `(x0_b−x)/σ` と許容誤差内で一致（**映像と音声の両方**。映像は inject≠clean）。⑥映像のマスクの行の並び（t→h→w）。⑦`time_rows` を引数なしで呼んだ結果が今日と同じ。⑧`load_source_waveform`: 44.1 kHz の wav で出し切りの有無でサンプル数が変わること、32 kHz・fltp の入力（素通し）で落ちないこと、1 ミリ秒以内の不足は 0 埋め、それ以上は `ValueError` | なし |
| **G2** | pytest（app の venv 全体）・`scripts/run_h3_pytest.py`・`npm run typecheck`・`npx vitest run --exclude "**/backend.integration.test.ts"`・MCP の試験 | 失敗 0。H3 の mock で A2V を 1 本流し、mp4 と `a2v` ブロックが出て `h3` キーが空のまま出ない。LTX の `/generate`＋`source_audio` が 422 FEATURE_UNSUPPORTED で文言が H3 専用の案内。短い音声が 422 `SOURCE_AUDIO_TOO_SHORT`。操作パネルの模擬サーバーで H3 の A2V カードが `/generate` に送る | なし |
| **G3-0** | 【GPU・了承】変更前（main）と変更後（dev）で、同じ種の I2V（`source_audio` 無し・小さい画角・4 ステップ）を 1 本ずつ流す | 復号した映像と音声の PCM〔音声の生の値〕がビット一致（マスク None の経路が今日と同じことの実機での確認） | あり |
| **G3** | 【GPU・了承】オーナーの素材で A2V。896×1152（重ければ 32 の倍数の小さい画角）・90 フレーム・先頭キーフレーム・Turbo LoRA 可。strength 1.0 と 0.5 の 2 本（順に） | 完走。VRAM の山を記録。1.0: 音声のサンプル数 120,000、出力音声の復号と入力の 32 kHz 版の相関 r ≥ 0.99（AAC の先頭の遅延を探してから比べる）、`audio_output: "original"`。0.5: `audio_output: "decoded"`、`m_per_step` が 3 以下 | あり |
| **G4** | オーナーの目視と試聴 | 口の動きと音声の一致。0.5 の聞こえ方。go／no-go はオーナー | — |

GPU の門は同時に 1 体だけ。記録は VERIFICATION_LOG §178。

## 5. 文書の更新（事実は 1 か所・他は § で参照）
- `Docs/MINIMAX_H3_DESIGN.md`: §3 の後に「§3.1 行ごとの連続値マスクと A2V」を新設（一般形の式・粒度・inject の雑音の出どころ・符号化器・出力音声の分岐。式の正本）。§3「番号は最大 3 つ」、§4 の段の表（`audio_encode`）、§8（`source_audio`、`encode` の総数、`done` の `h3.a2v`）、§11、§12（V2A を出さない）、§13（0<m<1 の品質は目視のみ・V2A は部品だけ）。
- `Videomni_Backend_Specification.md`: §6.2（`source_audio`、422）と GET /models の `constraints.source_audio`（節番号は実装時に確認）。API の契約の正本。
- FE-Docs `API_REFERENCE.md`: §3.12、§5.1、§3.5（仕様書を参照する形）。FE-Docs `DEVLOG.md` §149（直前は §148）。
- `Docs/VERIFICATION_LOG.md` §178（§176 と同じ構成）。
- 台帳: §1-103 → `PENDING_TASKS_CLOSED.md` 3-257。§1-106 の前提を「行ごとの連続値マスク（MINIMAX_H3_DESIGN §3.1、CLOSED 3-257。映像のマスクはエンジンにあり API には未公開）」に。§3-256 はそのまま。
- `Docs/MCP_SERVER_DESIGN.md` D30。
- 調査 R1／R2／R3 を `Docs/Outputs-archive/h3-a2v-2026-10-25/` へ写す。

## 6. リスクと未確認事項
1. `dit.py` の時刻の数 M が 3 を超える場合は G1 ④で確かめる（V1 は音声の値が一つなので M ≤ 3）。
2. t=1.0 の AdaLN の表は範囲外を読まない（529-532）。値 1.0 は DiffSynth の Retake と同じで学習時の分布の内側と見る。
3. 精度: 状態とマスクの計算は fp32。DiT への入力は今日と同じく射影の後で bf16。符号化器は fp32 固定。
4. 上限 0.999（映像）／1.0（音声）の根拠は ComfyUI の仕様。DiffSynth は映像も 1.0。V1 は映像のマスクを出さないので製品への影響なし。
5. 映像の inject の雑音は初期雑音の使い回しで確定（§2.2）。映像の固定行に雑音増し 0.999 を掛ける（ComfyUI）か clean のまま（DiffSynth）かは V1 では製品に出ない。§1-106 の計画で目視により決める。
6. PyAV の変換: swresample と torchaudio の値の差は未確認。G0 で確かめる。
7. 0<m<1 の品質は一次資料に評価が無く、G4 だけで決める。
8. mock は音声を mux せず、経路と metadata の形だけを保証する。
9. mp4 のレシピを読み戻すとき `source_audio` の ID が消えている場合（404）を FE が穏当に扱うかは未確認。
10. G3 の費用: Turbo の推奨ステップ数は §176 を参照。

## 7. 範囲外
V2A の API／UI への露出・音声の切り出しの引数・Gradio の H3 の整理（§3-243）・複数クリップの連結（§1-106）・Batch A2V と `plan_a2v_batch` の H3 対応・映像のマスクのパッチ単位化。

## 8. 進め方の規律
- 実装は本計画の承認後。各弾の着手前に監督が指示文を書く（禁止事項: バックエンドの他エンジン・担当外ファイルの変更）。
- GPU の門は弾ごとにオーナーの了承を取る。

## 9. 敵対的レビュー（R3・Opus）の採否
16 件すべてを採用した。過剰設計の棄却（#16: 計画自身の新モジュール `source_audio_rules.py` を捨て、`RequestRules` の項目 1 つに）と独立の裏取り（#1 の式の展開、#2 の `H3_ONLY_REFUSALS`、#4 の PyAV の素通し、#16 の前例は監督が確認）を通している。

| # | 要旨 | 反映先 |
|---|---|---|
| 1 Major | 映像では整理した式に項が足りない | §2.1（一般形・実装は `(x0_b−x)/σ` を直接）、G1 ⑤ |
| 2 Major | `H3_ONLY_REFUSALS` への追加が無い | §0・§2.5 |
| 3 Major | 完全一致の試験 7 か所が担当表に無い | §3（T1・M1・F3） |
| 4 Major | PyAV の出し切りと素通し | §2.3、G1 ⑧ |
| 5 Major | 模擬サーバーが規約を公開しない | §2.7（F2） |
| 6 | 許容の食い違い | §2.3（エンジンも 1 ミリ秒の許容＋0 埋め） |
| 7 | 数値例の誤り | §2.3 |
| 8 | 1016 行の None の扱い・条件行の順 | §2.2 |
| 9 | `entry["mask"]` の無条件書き込み | §2.2 |
| 10 | G1 ② の前提 | §4 |
| 11 | 映像マスクの雑音と組み立ての場所 | §2.2（R1 に確認中） |
| 12 | G3-0 の価値 | §4（main と dev の比較に置き換え） |
| 13 | G0 の判定の書き方 | §4 |
| 14 | 担当の漏れと重なり | §3・§2.6（`_finalize`）・§2.7（型） |
| 15 | 細かい抜け 6 点 | §2.3・§2.5・§2.6・§2.7・§5 |
| 16 | 過剰設計 | §2.5 |
