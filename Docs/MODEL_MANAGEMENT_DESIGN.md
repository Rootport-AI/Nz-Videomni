# モデル管理ドロップダウン機能 — S0 設計書

- 作成: 2026-07-05（子B・branch `feature/model-management-dropdown`）
- 正本の親: [`MODEL_MANAGEMENT_FUTURE_WORKORDER.md`](MODEL_MANAGEMENT_FUTURE_WORKORDER.md) 冒頭★注記2つ
- スコープ: **既存レイアウトのまま**「追加ファインチューニングモデルをドロップダウンで選択してロード」。ディレクトリ再編・移行スクリプト・インストーラ書き換えは**やらない**。API は加算のみ。凍結契約（`GET /status` の `vram_optimization`・`metadata.json` のキー集合）は不変。
- 本書は **S0（設計のみ）**。実装は親レビュー後の S1 以降。

---

## 0. 事前調査で確定した結合点（コード根拠つき）

| 要素 | 場所 | 事実 |
|---|---|---|
| モデルパスの定義 | `config.py:44-101 ModelConfig` | 全アーティファクトが単一固定パス文字列（Pydantic デフォルト）。`config.yaml` の `model:` で上書き可。 |
| worker への伝搬 | `services/ltx_runner.py:840-855`（`_RealBackend.load` 内） | `{"op":"load", ...}` JSON ペイロードを `config.model.*` から直接組み立てて送る。 |
| ペイロード受領 | `engine/worker.py:149-192 _do_load` | `_PIPE` を**1回だけ**構築。`_PIPE is not None` なら再構築せず `ready` を返す（**in-place 再ロード op は無い**）。 |
| 切替の握りつぶし① | `ltx_runner.py:126-131 LTXRunner.load` | `self._backend.loaded` なら早期 return。 |
| 切替の握りつぶし② | `ltx_runner.py:721-723 _RealBackend.load` | `self.loaded` なら早期 return。 |
| unload（worker kill） | `ltx_runner.py:876-902` | subprocess を落とす。`POST /pipeline/unload`→`pm.unload()`→`runner.unload()`。 |
| busy ガード前例 | `api/pipeline.py:23` | `context.job_store.has_active()` → `job_busy()`（409）。 |
| 列挙 UI 前例 | `services/lora_registry.py:25-74` + `gradio_ui/adapters.py:26-41` | name→path レジストリ＋`GET /config` の `model.ic_loras`→Dropdown。**今回の踏襲テンプレ**。 |
| GET /config | `api/status.py:35-38` | `context.config.model_dump()` をそのまま返す。 |
| 凍結地雷 | `services/low_vram.py:21-29 _STATUS_KEYS` | ここにモデル情報を混ぜない。 |

### コンポーネント（VAE/音声）の実配線 — カテゴリ設計の根拠

`engine/pipeline/fast_video_pipeline.py:292-376 _install_component_sources` を精読した結果:

- **video_vae** = `component_video_vae_path`（1ファイル `LTX23_video_vae_bf16.safetensors`）→ `vae_decoder_builder` + `vae_encoder_builder` の両方を再ソース。
- **audio 系** = `component_audio_vae_path`（1ファイル `LTX23_audio_vae_bf16.safetensors`）→ `audio_decoder_builder` + `audio_encoder_builder` + **`vocoder_builder`** の**3つすべて**が同一ファイルから来る。

→ **確定: 「音声 VAE + vocoder」は単一ファイル＝単一エントリ**として1カテゴリ `audio` に括る。ワークオーダーが言う「音声 VAE＋vocoder（別重み）」は物理的に1ファイルなのでドロップダウン1個で正しい。

- コンポーネント経路は `vram.use_component_files` で gate。`config.yaml:82` で `true`（=comp=1 既定）なので **video_vae/audio の差し替えは既定で有効**。
- text projection（`component_text_projection_path`）は Phase 2 の connector で、transformer GGUF と結合（`fast_video_pipeline.py:225-238`）。**独立に差し替えられない**ため今回のドロップダウン対象外（§9 決定事項3）。

### 4カテゴリと payload フィールドの対応（確定）

| カテゴリ | payload キー（load） | 既定ファイル | 拡張子 | 走査ディレクトリ |
|---|---|---|---|---|
| `transformer` | `gguf_transformer_path` | `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | `.gguf` | `models/ltx-2.3-gguf/`（**再帰**：既定は `LTX-2.3-distilled-1.1/` 配下） |
| `text_encoder` | `gguf_gemma_path` | `gemma-3-12b-it-Q4_K_M.gguf` | `.gguf` | `models/gemma-3-12b-it-gguf/`（非再帰） |
| `video_vae` | `component_video_vae_path` | `LTX23_video_vae_bf16.safetensors` | `.safetensors` | `models/ltx-2.3-components/vae/`（非再帰・下記の video/audio 判別あり） |
| `audio` | `component_audio_vae_path` | `LTX23_audio_vae_bf16.safetensors` | `.safetensors` | `models/ltx-2.3-components/vae/`（同上） |

対象外（v1）: `gemma_root`（tokenizer・40MB 固定）、`spatial_upsampler_path`、`checkpoint_path`（参照専用）、`component_text_projection_path`（transformer と結合）、`ic_loras`（既に専用 Dropdown あり）。

---

## 1. カテゴリ別レジストリ設計

### 1.1 config 拡張（ic_loras の型を踏襲・加算のみ）

`ModelConfig` に**カテゴリごとに** `dict[str, str]`（name→project-relative path）を追加。`ic_loras`（`config.py:101`）と同じ `default_factory=dict` で、**未記述なら空**＝現行 `config.yaml` は無改変で従来通り動く。

```python
# config.py ModelConfig に追加（すべて additive・default 空）
class ModelEntry(BaseModel):        # 将来の拡張余地（今は path のみ）。IcLoraEntry と同型思想
    path: str

# name -> path(str) | ModelEntry。空 = 既定のみ（自動登録）
transformers:   dict[str, str | ModelEntry] = Field(default_factory=dict)
text_encoders:  dict[str, str | ModelEntry] = Field(default_factory=dict)
video_vaes:     dict[str, str | ModelEntry] = Field(default_factory=dict)
audio_models:   dict[str, str | ModelEntry] = Field(default_factory=dict)
```

> v1 は `str`（path）だけで足りるが、`str | ModelEntry` を許すことで IC-LoRA と同じ「後で属性を足せる」拡張性を確保（ここは親判断ポイント §9-4：単純に `dict[str,str]` でよいか）。

### 1.2 既定エントリの自動登録（byte 同一の要）

新サービス `services/model_registry.py`（`LoraRegistry` を鏡像）を作る。構築時に**各カテゴリへ必ず `"default"` エントリを注入**し、その path は**現行 `ModelConfig` の既定フィールド値そのもの**。

```python
DEFAULT_NAME = "default"
_CATEGORY_DEFAULT_FIELD = {
    "transformer":  "gguf_transformer_path",
    "text_encoder": "gguf_gemma_path",
    "video_vae":    "component_video_vae_path",
    "audio":        "component_audio_vae_path",
}
# registry[cat] = { "default": <config.model.<field>>, **config.model.<cat_dict>, **scanned }
```

→ 呼び出し側が category を省略 = `"default"` = 既定フィールド値。**解決される絶対パスは現行と完全同一**（§4 の byte-match の根拠）。

`ModelRegistry` の公開契約（`LoraRegistry` に倣う）:

```python
class ModelRegistry:
    def __init__(self, config: AppConfig): ...
    def categories(self) -> list[str]                       # 固定4カテゴリ
    def names(self, category: str) -> list[str]             # default 先頭 + sorted
    def entries(self, category: str) -> list[EntryInfo]     # GET /models 用（下記）
    def resolve(self, category: str, name: str) -> Path     # name -> 絶対path。未知/欠損は raise
    def default_name(self, category: str) -> str            # 常に "default"
    def rescan(self) -> None                                # ディレクトリ再走査（§2）
```

`resolve` の失敗は `LoraRegistry.resolve`（`lora_registry.py:58-74`）を鏡像:
- path 風 name（`/ \ ..` 含む）→ `model_not_found`（§5）
- 未知 name → `model_not_found`（既知一覧を detail に）
- 登録済みだがファイル欠損 → `model_file_missing`

---

## 2. 走査方式（既存レイアウトのまま発見）

### 2.1 規則

- 各カテゴリの **既定ファイルの親ディレクトリ**（§0 表）を走査対象にする（`config._abs` で解決）。
- 拡張子フィルタ: transformer/text_encoder=`.gguf`、video_vae/audio=`.safetensors`。
- **除外**: `.cache/`（HF ダウンロードキャッシュ・実在する）、ドットディレクトリ、既定パスそのもの、config で明示登録済みのパス、`INSTALLED_PATHS.txt` 等の非モデル。
- transformer のみ**再帰**（既定が `LTX-2.3-distilled-1.1/` サブ配下のため）。他は非再帰。
- 発見ファイルの自動 name = **ファイル名 stem**（重複時は親フォルダ名を prefix）。
- **video_vae / audio の同居問題**: 両者は同じ `.../vae/` に居るため拡張子だけでは判別不能。判別ヒューリスティック = ファイル名に `audio` を含めば `audio`、`video` を含めば `video_vae`、**どちらでもなければスキップ**（ログのみ）。判別不能なファイルはユーザーが config へ明示登録すれば確実に載る（config が権威・走査は補助）。→ §9 決定事項2。

### 2.2 タイミング

- **構築時**（app 起動時 = `AppContext.__post_init__` で `ModelRegistry(config)`）に1回走査。
- **`GET /models` の各呼び出しで再走査**（`rescan()`）。`os.scandir` 1枚のディレクトリ列挙は安価なので、ユーザーが新モデルを DL 後に**サーバー再起動なしで**ドロップダウンへ反映される。キャッシュ不要。

---

## 3. `GET /models`（新規・加算エンドポイント）

`api/models_registry.py`（新 router）に実装。`GET /api/v1/models`。

レスポンス形状（カテゴリ→エントリ一覧＋active＋default マーク）:

```json
{
  "categories": {
    "transformer": {
      "default": "default",
      "active": "default",
      "entries": [
        {"name": "default", "path": "models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
         "is_default": true, "exists": true, "source": "config"},
        {"name": "distilled-Q6_K", "path": "models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/...-Q6_K.gguf",
         "is_default": false, "exists": true, "source": "scan"}
      ]
    },
    "text_encoder": { "...": "..." },
    "video_vae":    { "...": "..." },
    "audio":        { "...": "..." }
  }
}
```

- `active` = **直近の成功 load で使われた選択**（`PipelineManager` が保持。初回ブート時/未ロード時は `default`）。切替後は新選択を反映。
- `path` は project-relative 表示（絶対パスは漏らさない）。
- `exists` は走査時の実在チェック（config 登録だが未 DL のエントリを UI が薄字表示できる）。
- `source` = `"config"` | `"scan"`。
- `GET /config` は**無改変**（`model_dump` の形は変えない）。モデル一覧は新エンドポイント専任。

---

## 4. `POST /pipeline/load` 拡張

### 4.1 リクエスト形状（optional・省略で完全後方互換）

```jsonc
// body なし / {} → 現行と完全に同一挙動（既定一式をロード）
// 部分指定可。省略カテゴリ = 現在の active（未ロードなら default）を維持
{ "models": { "transformer": "distilled-Q6_K", "video_vae": "default" } }
```

FastAPI 側は `LoadPipelineRequest`（全フィールド optional・`models: dict[str,str] | None = None`）を**optional body** で受ける（`body: LoadPipelineRequest | None = Body(None)`）。body 無し＝現行の引数なし呼び出しと同じ。

### 4.2 省略時の byte 同一保証（スナップショットテスト具体案）

**リファクタ前提**: `_RealBackend.load()`（`ltx_runner.py:840-855`）のペイロード辞書組み立てを**純関数へ抽出**する:

```python
def _build_load_payload(self, selection: dict[str, str] | None = None) -> dict:
    # selection: category -> 絶対path（解決済み）。None のカテゴリは config 既定を使う。
    # 現状のフィールド構築ロジックをそのまま移し、gguf_transformer_path 等だけ
    # selection があれば差し替える。**キー順・キー集合は不変**。
```

スナップショットテスト（`tests/` に追加、mock で GPU 不要）:

```python
def test_load_payload_byte_identical_without_override(tmp_config):
    be = _RealBackend(tmp_config, low_vram)
    p_none  = be._build_load_payload(None)     # 省略
    p_empty = be._build_load_payload({})       # 空 dict
    p_dflt  = be._build_load_payload({c: reg.resolve(c, "default") for c in CATS})
    assert p_none == p_empty == p_dflt
    assert p_none == GOLDEN_LOAD_PAYLOAD        # 現行コードから採取した固定 golden
```

`GOLDEN_LOAD_PAYLOAD` は**現行実装（override 導入前）が実際に送る dict** を1度採取して固定（`_send` を monkeypatch して捕捉、または既存の worker 起動テストから採る）。これで「省略時＝現行 byte 同一」を回帰で守る。

### 4.3 差し替え経路（loaded 早期 return の回避＝unload→load）

**worker 再起動必須**（§0：`_PIPE` は1回構築・再ロード op 無し）。`loaded` 早期 return を回避する自然な方法は **先に unload して loaded を False にしてから load** すること（force フラグの発明は不要）:

```
POST /pipeline/load {models}
 └ api層: busy ガード → job_store.has_active() なら job_busy(409)
 └ 選択を解決: ModelRegistry.resolve(cat, name) → 絶対path（未知/欠損は §5 の 404/422 で即停止）
 └ front 互換検査（§5 の軽い検査：magic bytes 等）→ 失敗は model_incompatible(422)
 └ 選択 == 現在 active かつ loaded？ → no-op（loaded 状態を返す・再起動しない）
 └ それ以外:
     pm.reload(selection):
       with pm._lock:
         runner.unload()                 # subprocess kill（loaded=False へ）
         state = LOADING
       runner.load(selection=resolved)   # 早期 return は loaded=False なので発火しない
                                          #   → _RealBackend.load(selection) が
                                          #     _build_load_payload(selection) で新 worker 起動
       成功: pm._active_selection = selection; state = READY
       失敗: § 下記フォールバック
```

**シグネチャ加算**（すべて optional・default None で現行呼び出し不変）:
- `PipelineManager.load(self, selection=None)` / 新 `reload(selection)`
- `LTXRunner.load(self, selection=None)`（→ `backend.load(selection)`）
- `_RealBackend.load(self, selection=None)` / `_MockBackend.load(self, selection=None)`（mock は無視でよい）

**失敗時フォールバック（§9 決定事項1・親判断）**: 推奨は**自動フォールバックしない**（未ロードのまま `state=error`・`PIPELINE_LOAD_FAILED(503)` を detail つきで返す）。理由: 自動で既定へ戻すと本当の非互換を隠し、ロード時間が倍になる。ただし config トグル `model.reload_fallback_to_default`（既定 False）で「失敗時は直前 active へ戻す」を選べる余地を残す設計を提案。→ 親に諮る。

`busy` ガードは load 側にも追加（現行の unload だけでなく）。切替中はさらに `state=LOADING` で `_lock` を保持し、二重ロードを直列化。

---

## 5. 互換検査とエラー（native crash させない）

2層で守る:

### 5.1 前段（app 側・軽量・worker 起動前）
1. **name 解決**: `ModelRegistry.resolve` で未知 name / path 風 name を弾く。
2. **実在**: ファイルが無ければ弾く。
3. **拡張子/カテゴリ整合**: transformer/text_encoder は `.gguf`、vae/audio は `.safetensors` 必須。
4. **magic bytes 検査（native crash 予防の要）**: 先頭数バイトのみ読む。
   - `.gguf` → 先頭4バイトが `GGUF`。
   - `.safetensors` → 先頭8バイト（little-endian u64 = header長）を読み、続く JSON ヘッダが parse 可能かだけ確認（**重みはロードしない**）。
   - 余裕があれば safetensors ヘッダのテンソル名/shape を覗いて VAE の期待キー有無を軽くチェック（任意・S2 では magic までで可）。
   - これにより「拡張子だけ合った別物ファイル」が native GGUF/safetensors ローダに渡って segfault するのを防ぐ。

### 5.2 engine 側（fail-fast・権威）
- worker の builder はロード時にキー/shape を検証し、不整合は `_do_load` 内で例外→ `{"event":"error", detail}`（`worker.py:420-423`）→ runner が `RuntimeError`→ `pm` が `PIPELINE_LOAD_FAILED(503)` へ翻訳（`pipeline_manager.py:99-106`）。既存経路をそのまま使う。

### 5.3 エラーコード（`api/errors.py` に加算・既存体系準拠）

```python
def model_not_found(category, name, detail=None):      # 404（lora_not_found 鏡像）
    "MODEL_NOT_FOUND", f"unknown model name '{name}' in category '{category}'"
def model_file_missing(category, name, detail=None):    # 422
    "MODEL_FILE_MISSING", "registered model file is missing on disk"
def model_incompatible(category, name, detail=None):    # 422（前段 magic/shape 検査失敗）
    "MODEL_INCOMPATIBLE", "selected model failed the compatibility precheck"
```
- engine 側失敗は既存 `PIPELINE_LOAD_FAILED(503)`、busy は既存 `JOB_BUSY(409)` を再利用。

---

## 6. GUI（Settings タブ内「モデル」セクション）

**触ってよいファイルのみ**: `gradio_ui/adapters.py`・`api_client.py`・`i18n.py`・`ui.py` の Settings タブ部分・`handlers.py` の**新規関数**。Clip Chain/Generate タブと `_poll_job_until_done`・トップバー共通 `load_model`/`unload_model` クロージャは**不可触**（子Aの領域）。

### 6.1 部品案（`ui.py` の Settings タブ・`h_server` の直後あたりに新セクション）

```
### モデル (model_section_title)
  Row:
    Dropdown model_dd_transformer   (label=model_cat_transformer)
    Dropdown model_dd_text_encoder  (label=model_cat_text_encoder)
  Row:
    Dropdown model_dd_video_vae     (label=model_cat_video_vae)
    Dropdown model_dd_audio         (label=model_cat_audio)
  Button  model_load_btn (model_btn_load_selected)   # variant=primary
  Markdown model_load_status  (ロード中/成功/失敗の1行表示)
  Markdown model_hint (note: 切替は数分・worker 再起動を伴う旨)
```

- choices は **`GET /models`** から `adapters.py` の新 `build_model_choices(models_json, category)` で構築（`build_adapter_choices` の型踏襲・(label, value)=(name, name)、既定は `"default"` を先頭・欠損エントリは薄字ラベル）。
- ページロード / トップバー Refresh 時に `GET /models` を引いて4つの Dropdown choices を更新（`on_page_load`/`on_refresh_config` に**出力を加算**、または独立の `gr.Timer`/ボタンで。子Aの outputs を壊さぬよう新規 state/outputs を足す形にする）。
- `model_load_btn.click` → **新規ハンドラ** `handlers.load_selected_models(api, tr, te, vv, au, lang)`:
  - 4 Dropdown の値から `{"models": {...}}` を組み、`api_client.load_pipeline_models(body)`（新メソッド）を POST（`load_pipeline` は 600s timeout 既存流用）。
  - 成功→ `GET /status` で `pipeline_loaded` を確認し成功メッセージ。409/422/404/503 は本文の error code を i18n で表示。
  - ロード中はボタン `interactive=False` + 「ロード中…」表示（gradio ハンドラのブロッキングで代替。進捗ポーリングは S3 では最小化＝トップバー Refresh に委ねる）。
- **`_poll_job_until_done` は使わない**（あれは生成ジョブ用・子A所有）。モデルロードは同期 POST（202 ではなく完了応答）。

### 6.2 api_client 新メソッド

```python
def get_models(self) -> dict:                      # GET /api/v1/models
def load_pipeline_models(self, models: dict) -> dict:  # POST /pipeline/load {"models": models} (timeout=600)
```

### 6.3 i18n（`model_` 接頭辞・en/ja 両方）

`model_section_title` / `model_cat_transformer` / `model_cat_text_encoder` / `model_cat_video_vae` / `model_cat_audio` / `model_btn_load_selected` / `model_loading` / `model_load_ok` / `model_load_failed` / `model_hint` / エラー表示 `model_err_not_found` / `model_err_incompatible` / `model_err_busy`。

---

## 7. 検証計画

### 7.1 mock 経路 pytest（今スライスで実施可・GPU 不要）
- `ModelRegistry`: 既定自動登録・走査発見（tmp ディレクトリに擬似ファイル）・`.cache` 除外・video/audio 判別・未知 name/欠損ファイルの raise。
- `GET /models`: 形状（categories/entries/active/default/is_default/exists/source）・`GET /config` が無改変であること。
- `POST /pipeline/load`: body 無し=現行挙動、部分 override（mock backend で選択が伝搬）、busy 時 409、未知 name 404、欠損 422、magic 検査失敗 422。
- **byte-match スナップショット**（§4.2）: `_build_load_payload(None)==({})==golden`。
- GUI ハンドラ（`httpx.MockTransport`）: `get_models`→Dropdown choices、`load_selected_models` が正しい body を POST、エラー分岐の i18n。
- 凍結回帰: `GET /status` の `vram_optimization` キー集合・`metadata.json` キー集合が不変（既存テストが緑のまま）。

### 7.2 実機の別名二重登録切替（後続スライス S4・親が GPU 所有・本セッションでは実施しない）
- 既存 `Q4_K_M.gguf` を **config で第2の name（例 `default-alias`）として同一パスに二重登録**（実代替モデルの DL はしない・ユーザー決定）。
- `default` で生成 → `default-alias` へ切替（unload→load）→ 同 seed で再生成 → **出力 mp4 が byte 一致**（同一ファイル＝同一重み＝同一出力）を確認。
- 既定構成の byte-match 回帰・16GB fit 不変も同時確認（`ltx_worker.log` の `peak_vram_mb`）。
- 検証後、一時登録・一時成果物を**削除**（ストレージ配慮・ユーザー指示）。

---

## 8. スライス分割案とゲート

| スライス | 内容 | 完了ゲート |
|---|---|---|
| **S1 列挙** | config 拡張（4カテゴリ空 dict + ModelEntry）・`services/model_registry.py`・`GET /models` router・走査 | registry/endpoint の pytest 緑・`GET /config` 無改変・load 挙動は無変更（byte-match 既存回帰緑） |
| **S2 load 拡張** | `_build_load_payload` 抽出・`POST /pipeline/load` override・selection 伝搬（pm→runner→backend・全 optional）・unload→load 強制再構築・busy ガード・§5 エラー＆magic 検査 | **byte-match スナップショット緑**・override 切替が mock で成立・409/404/422/503 経路・`/status`・`/metadata` 契約 drift なし |
| **S3 GUI** | Settings「モデル」セクション・`build_model_choices`・api_client 2メソッド・`handlers.load_selected_models`・i18n `model_*` | ハンドラ単体（MockTransport）緑・mock UI スモーク（GPU 無し）・子A領域（Generate/Chain/`_poll_job_until_done`/トップバー load）無改変 |
| **S4 実機** | 別名二重登録切替 byte-match・既定 byte-match 回帰・16GB fit（**親が GPU で・別セッション**） | 切替後 byte 一致・既定 byte 一致・peak_vram 不変・一時成果物削除 |

各スライス末にコミット。push/マージはしない（親承認）。

---

## 9. 親に諮る判断ポイント

1. **swap-load 失敗時ポリシー**: 未ロードのまま error を返す（推奨）か、直前 active/既定へ自動フォールバックか。トグル `model.reload_fallback_to_default`（既定 False）を用意するか。
2. **走査の video/audio 自動判別**: 同居 `vae/` ディレクトリはファイル名 `video`/`audio` ヒューリスティックで振り分け・判別不能はスキップ（config 明示が権威）。この妥協で良いか。
3. **対象カテゴリを 4 に限定**（transformer/text_encoder/video_vae/audio）。tokenizer_root・spatial_upsampler・text_projection connector・ic_loras は対象外。text_projection は transformer GGUF と結合のため独立差し替え不可＝除外で良いか。
4. **config の値型**: `dict[str, str | ModelEntry]`（ic_loras 踏襲・将来拡張余地）か、単純に `dict[str, str]` か。
5. **selection 伝搬の実装**: `pm.load/runner.load/backend.load` に optional `selection=None` を加算（推奨・明示的）で良いか。あるいは PipelineManager に active_selection を持たせ backend が読む方式か。
6. **`GET /models` の active の定義**: 「直近成功 load の選択」（未ロード時 default）で良いか。worker が落ちている間の active 表示の扱い。
