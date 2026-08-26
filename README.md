# Nz-Videomni

LTX 2.3 / LTX 2.5 の動画生成モデルを **VRAM 16GB** のコンシューマーGPUで動かし、REST API として公開するバックエンドサーバー。検証用に Gradio UI(`/ui`) を同梱します。
AviUtl2 用の拡張フロントエンド（`.aux2` プラグイン）も同じリポジトリに入っており、`AviUtl2-Plugin/` 以下がそれです。API は汎用設計なので、DaVinci Resolve など他のフロントエンドからも使えます。

> **リポジトリの構成（モノレポ）**
>
> | 場所 | 中身 |
> |------|------|
> | リポジトリ直下 | バックエンド（`main.py` / `api/` / `services/` / `engine/` / `engine25/` / `gradio_ui/` / `mcp_server/`） |
> | `AviUtl2-Plugin/NzVideomni.aux2` | ビルド済みの AviUtl2 プラグイン（配布物。利用者はこれを AviUtl2 へドラッグ＆ドロップします） |
> | `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/` | そのプラグインのソース（C++ の `native/` ＋ React/TypeScript の `webui/`） |
> | `Docs/` | プロジェクト全体の文書と課題台帳 |
>
> **Nz-Videomni は製品の名前、LTX 2.3 はモデルの名前**です。**LTX 2.5 には 2026-08-22 に対応しました**（画面上部のドロップダウンで切り替えます。対応範囲は基本生成〔テキストから動画・画像から動画〕と、2026-08-23 に加わったクリップ連結〔Chained〕・V2V〔動画の続きを作る〕・A2V〔音声から動画〕、および 2026-08-24 に加わったスタイル LoRA・IC-LoRA〔参照動画による制御。長尺 IC-LoRA を含む〕、2026-08-26 に加わった撮り直し〔Retake〕と素材（末尾）〔End source〕です。**まだ使えない機能の一覧は §7.1 にあります**）。さらに Wan 2.x など別のモデルも載せられる基盤を目指しているため、製品名にモデル名を含めていません。

**Phase 1**（T2V + 最小I2V を同一MVP）の凍結 API を土台に、その後キーフレーム誘導・クリップ連結（`POST /generate/chain`）・
V2V（元動画からの継続生成）・end source（素材（末尾）＝添付した画像・動画へ**繋がる**動画の生成——§7参照）・A2V（音声から動画生成）・IC-LoRA／スタイルLoRA
などを加算的に拡張しています。
API 契約・スキーマの詳細仕様は [`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) を参照してください。

> 実エンジンは **first-party の `engine/` パッケージ**（GGUF 量子化トランスフォーマー + block-swap +
> GGUF Gemma 逐次オフロード + DiT CPU 構築 + VAE タイリング）です。**VRAM 16GB の実機で 720p 級（1280×768）の生成に対応**します。
> GPU / モデルウェイトが無い環境では自動的に **モック backend**（合成クリップ）へフォールバックし、
> API・ジョブ管理・Gradio・テストまで完全に疎通します。
>
> 公式 `ltx_pipelines` の safetensors ローダは本機(16GB/Windows)で native crash するため**不採用**で、GGUF + component-file
> 経路にしています。詳しい設計判断・実測は `Docs/VERIFICATION_LOG.md` と `engine/VENDOR_NOTICE.md` が一次情報です。

---

## 0. 環境分離ポリシー（最重要・最初に読む）

**このプロジェクトは PC のシステム Python 環境を一切汚しません。** Python 本体を含め、必要なものはすべて
プロジェクトディレクトリ配下（`.venv/`, `.venv-engine/`, `.venv-engine-ltx25/`, `.python/`, `.uv_cache/`, `tools/`）に
閉じ込めます（仕様書 2.5）。

- グローバル/システムの `pip install` は **禁止**。必ず `uv` + プロジェクトローカル venv。
- 環境変数（`PYTORCH_CUDA_ALLOC_CONF`, `UV_PYTHON_INSTALL_DIR`）は **そのプロセス内のみ**。永続化しない。
- 前提ツール（`uv` / `ffmpeg` / `ffprobe`）も `tools/` に取り込み、`PATH` への追加は **そのプロセス内のみ**。
  Windows の環境変数設定は書き換えません。
- パッケージのダウンロードキャッシュも `.uv_cache/` としてプロジェクト内に置きます（システムのユーザープロファイル配下は使いません）。
  3つの venv は、実体をここに置いてハードリンク（同じ実体を指す別名）で共有します。
- 後片付けはこのディレクトリ（`.venv` / `.venv-engine` / `.venv-engine-ltx25` / `.python` / `.uv_cache` / `tools` 含む）を
  削除するだけで完全に元に戻ります。

### venv の構成（重要）

このバックエンドは **アプリ1プロセス＋エンジン系統ごとのワーカー**という構成です。いずれも別インタプリタで、共存させません。

| venv | 役割 | 主要依存 |
|------|------|----------|
| `./.venv` | FastAPI アプリ（`main.py`・API・ジョブ・Gradio・モック backend） | FastAPI / Pydantic / Pillow / ffmpeg 呼び出し。**torch は入れない** |
| `./.venv-engine` | LTX 2.3 用のエンジン worker（`engine/worker.py`） | **torch 2.9.1+cu128** + LTX 推論スタック（`ltx_core`/`ltx_pipelines`@`00dc53d` + `gguf`） |
| `./.venv-engine-ltx25` | LTX 2.5 用のエンジン worker（`engine25/worker.py`） | **torch 2.9.1+cu128** + 公式 v1.2.0 の推論スタック + `transformers` 5.x |

エンジン系統ごとに venv を分けているのは、LTX 2.3 と LTX 2.5 が要求するパッケージのバージョンが同居できないためです。**worker は同時に1つだけ動き**、モデルを切り替えると古い worker を終了させてから新しい worker を起こします。

アプリ(`./.venv`)は torch も LTX も import しません。実生成は、選ばれているモデルに対応する python で
worker を **subprocess** として起動し、JSON-lines プロトコルで駆動します（下記アーキテクチャ参照）。
`.venv-engine` の依存スナップショットは [`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt) に凍結してあります
（`uv.lock` は fork 撤去時に失われたため）。

---

## 1. セットアップ

### 必要なもの

- Windows 10/11 x64
- **git** — [公式サイト](https://git-scm.com/download/win)からインストーラを入手してください。画面上のボタンを押していくだけで導入が終わり、コマンドの入力は必要ありません。
- 実モデルで生成する場合のみ: NVIDIA CUDA GPU と十分なメインメモリ（下記「ハードウェア要件」）

**あらかじめ用意しておくものは git だけです。** パッケージ管理ツールの
[`uv`](https://docs.astral.sh/uv/) と、動画の変換に使う `ffmpeg` / `ffprobe` は、`setup.bat` が
このプロジェクトの中（`tools/` フォルダ）へ自動的に取り込みます。Windows 側の設定（PATH など）は
一切書き換えないので、後片付けはこのフォルダを削除するだけで済みます。

### かんたんインストール（`setup.bat` → `run.bat`）

コマンドを打つ必要はありません。次の順に進めてください。

1. **`setup.bat` をダブルクリックする。**
   黒い画面が開き、道具の取り込み（`uv` / `ffmpeg`）→ 専用の Python 環境の作成 → モデルのダウンロード
   （約 33GB）が順に進みます。所要時間の目安は、光回線（下り 90〜100Mbps）でおよそ 50 分、
   30Mbps 程度の回線ではおよそ 2 時間半です。
   **途中でこの画面を閉じても構いません。** もう一度 `setup.bat` を実行すれば続きから再開します。
   ただし途中でパソコンがスリープすると通信が止まるので、長時間そのままにする場合は、電源の設定で
   スリープを「なし」にしておいてください。
2. **`run.bat` をダブルクリックする。**
   サーバーが起動し、黒い画面に `http://127.0.0.1:18620/ui` のようなアドレスが表示されます。
   起動が終わると、このアドレスが囲み枠つきの案内としてもう一度表示されるので、それをブラウザの
   アドレス欄に入力してください。
3. **表示されたアドレスをブラウザで開く。** これで Web の操作画面（Gradio UI）が使えます。
4. **AviUtl2 から使う場合**は、このリポジトリの **`AviUtl2-Plugin\NzVideomni.aux2`** を、
   **AviUtl2 のプレビュー画面へドラッグ＆ドロップ**してください。AviUtl2 公式のプラグイン導入方法です
   （本体添付の `aviutl2.txt` に記載があります）。
5. **AviUtl2 を再起動する。** 上部メニューから Nz-Videomni を開けるようになります。

#### サーバーの止め方

`run.bat` で開いた黒い画面を、**右上の × ボタンで閉じてください**。これでサーバーが止まります。

> **この案内に `Ctrl+C` を書き足さないでください（意図的な省略です）。** `Ctrl+C` で止めると、`cmd` が
> `Terminate batch job (Y/N)?` という英語のプロンプトを返すことがあり、本プロジェクトの想定利用者
> （PowerShell を自分で開けないリテラシー）はここで手が止まります。× で閉じる 1 通りだけを案内する、
> というのが決定事項です（`Docs/PENDING_TASKS_CLOSED.md` §3-37）。

#### AviUtl2 から使うときの注意（重要）

**先に `run.bat` を起動して、その黒い画面を開いたままにしておいてください。**
AviUtl2 のプラグインは、バックエンドのサーバーを自分で起動しません（プラグイン側にサーバーを
立ち上げる仕組みは入っていません）。サーバーが動いていないと、プラグインの画面には
**「サーバー未起動」**というバッジが出るだけで、生成はできません。黒い画面を閉じるとサーバーも
止まるので、AviUtl2 から使っている間は閉じないでください。

#### 更新のしかた

1. VSCode の画面から `git pull`（同期）を実行して、最新のコードを取り込む。
2. **そのあと、`setup.bat` をもう一度実行する。**

**2 を省かないでください。** 依存パッケージの内容が変わっていた場合、`run.bat` は起動こそするものの
中身が古いままで正しく動かない、という分かりにくい状態になります。`setup.bat` は 2 回目以降、
すでに揃っているものを飛ばすので、変更が無ければ短時間で終わります。

自動更新の仕組みはあえて用意していません（更新確認の通信がウイルス対策ソフトに誤検知される
リスクを避けるためです）。

#### 設定ファイル `config.yaml` の扱い

`config.yaml` は **git の管理対象外**です。配布されるのはひな型の `config.yaml.example` で、
`setup.bat`（および `run.bat`）が、`config.yaml` がまだ無いときにひな型から複製する作りになっています。
利用者が自分のマシンに合わせて書き換えた設定が、`git pull` のたびに衝突しないようにするための
作りです。設定を変えたい方は `config.yaml` のほうを編集してください（ひな型を編集しても、
動作中の設定には反映されません）。

> **この複製経路は、2026-07-27 のサブマシン検証で初めて実際に走り、正しく動くことを確認しました。**
> それまでは開発機に以前から `config.yaml` が存在したため、`setup.bat` の実行ログでも「既にあります」の
> 分岐しか通っていませんでした。`config.yaml` が無い状態（＝clone 直後）の新規環境で `setup.bat` を
> 実行したところ、ひな型からの複製が行われ、実モデル（real backend）での生成に成功しています。
> もし複製されずに起動した場合、設定は既定値のままになり、生成が「お試し表示」（モック backend）へ
> 切り替わってしまうので、**実生成が通ったこと自体が複製成功の裏付け**になります。

> **すでに `config.yaml` を編集して使っていた方への注意（移行時に一度だけ）**
>
> この変更を取り込む `git pull` は、**「リポジトリ側での削除」と「手元での変更」がぶつかって途中で
> 止まります**。`git pull` する前に、`config.yaml` を `config.yaml.bak` などの別名でコピーして
> 退避しておいてください。pull のあと、必要な設定を新しい `config.yaml` へ書き戻せます。
> （編集していなかった場合は何も起きません。静かに消え、次の起動時にひな型から復元されます。）

### ハードウェア要件（実モデルで生成する場合）

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA 製・**VRAM 16GB 以上**。対応世代は Turing（GeForce RTX 20系）／Ampere（同 30系）／Ada Lovelace（同 40系）／Hopper／Blackwell（同 50系） |
| GPU ドライバ | **R570 以上を推奨**（Blackwell では必須）。CUDA 12.x のマイナーバージョン互換だけを見れば Windows では 525 以上が下限ですが、本プロジェクトは cu128 ビルドの torch を使うため R570 以上を勧めます |
| メインメモリ | **32GB 以上、かつページファイルを有効にしておくこと**（下の「メインメモリとページファイル」が最重要）。**モデル骨格の常駐（`keep_resident`）を使う場合は 64GB 以上を推奨**します（LTX 2.3 では約 20GB、LTX 2.5 では約 7.7GiB を常時占有するため。既定は off なので、使わないかぎりこの要件は増えません。§5「モデル骨格の常駐（`keep_resident`）」）。**LTX 2.5 を使う場合も 64GB 以上を推奨**します——2.5 のワーカーは仕上げ工程のために重みをメインメモリへ持ち続ける設計（`cache_weights`、既定 on）で、**生成中のメインメモリの山が実測で約 26GiB** あるためです。この既定を off にすれば常駐は減りますが、そのぶん仕上げ工程の作り直しに時間がかかります。**LTX 2.5 でモデル骨格の常駐も同時に on にする場合は、安全側の見積りとして合計 約 34GiB を見ておいてください**（内訳は連結生成の OFF ピーク 26.33GiB ＋ 常駐増分 7.68GiB ≒ 34.0GiB。[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §69.19・§76.2(3)・**§76.8(3)**〔§76.2(3) の「約 33.6GiB」はこの節で 約 34.0GiB へ訂正した〕） |
| ストレージ | **このフォルダを置くドライブに約 40〜41GB**（モデル 約 32.51GiB ＋ Python 環境 7〜8GiB ＋ `tools/` 約 0.4GiB）。**これとは別に**、ページファイルを置いたドライブに 60GB 以上の空き（下の「必要な空き容量の内訳」参照） |
| attention（注意機構の計算方法） | 既定は全世代で **SDPA**（PyTorch 標準の実装）。**2026-07-31 から、生成のたびに SageAttention へ切り替えられます**（§5「生成の高速化（Acceleration）」）。xformers・flash-attn は引き続き導入も使用もしません |

#### 対応する GPU 世代

エンジン venv の torch は **2.9.1+cu128** に固定してあり、この配布ビルドが同梱しているコンパイル済みカーネルの一覧
（`torch.cuda.get_arch_list()` の実測値）は次のとおりです。

```text
['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

`sm_XX` は GPU の compute capability（世代を表す番号）です。Ampere（RTX 30系＝`sm_86`）・Hopper（`sm_90`）・
Blackwell（RTX 50系＝`sm_120`、データセンター向け B200＝`sm_100`）は、この一覧にそのまま含まれています。
PyTorch v2.9.1 のビルド設定（[`build_cuda.sh`](https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh)）も
`TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0"` で、実測値と一致します。
なお **Ampere（RTX 30系＝`sm_86`）は、2026-07-27 に RTX 3080 mobile 16GB で実機動作を確認済み**です
（導入から 768p・257 フレームの IC-LoRA 制御生成まで。§7「制限事項」）。残る世代は理論上の互換のみです。

> **`sm_89`（Ada Lovelace＝RTX 40系）が一覧に無いのはなぜか**: CUDA のバイナリ互換性により、`sm_86` 向けに
> コンパイルされたカーネル（cubin）はそのまま Ada の GPU 上で動作するためです。NVIDIA の Ada 互換性ガイドが
> 「compute capability 8.6 向けに生成された cubin は compute capability 8.9 の GPU で動作することがサポートされる」と
> 名指しの例として明記しています（[Ada Compatibility Guide](https://docs.nvidia.com/cuda/archive/12.8.0/ada-compatibility-guide/index.html)）。
> 逆方向（8.9 向けの cubin を 8.6 の GPU で動かすこと）はできません。本プロジェクトの開発機は RTX 4070 Ti SUPER（Ada）で、
> 実際にこの経路で動いています。

attention は全世代で PyTorch の SDPA を既定にしており、xformers や flash-attn はインストールもしなければコードからも
呼びません（16GB 環境で速度を決めているのは attention ではなく重みの転送であるため。仕様書 §5.4）。

**2026-07-31 から、SDPA に加えて SageAttention（量子化を使って注意機構の計算そのものを速くする外部カーネル）を
生成のたびに選べるようになりました。** 必要なパッケージは `setup.bat` が標準で入れるため、追加の作業は要りません
（`sageattention` 2.2.0 と `triton-windows` 3.5.1.post24。後者は必要なコンパイラを同梱しているため、**Visual Studio の
導入は不要**です）。既定は従来どおり SDPA で、選ばないかぎり生成結果はこれまでと変わりません。使い方と注意点は
§5「生成の高速化（Acceleration）」を参照してください。

#### 必要な空き容量の内訳

このフォルダの中に入るものは、実測で次のとおりです。合計 **約 40〜41GB** を見てください。
この「約 40〜41GB」が本プロジェクトで統一している必要容量の数字で、`setup.bat`（`scripts/setup.ps1`）が
起動時に出す空き容量の案内・失敗時の案内も同じ数字を使います（判定のしきい値はここに余裕を足した 45GB）。

| 中身 | 実測サイズ | 備考 |
|------|-----------|------|
| `models/`（モデル一式） | 約 32.51 GiB | `models/LTX23/` に GGUF transformer（`Weights/`）＋ GGUF Gemma と tokenizer（`TextEncoder/`）＋ VAE 一式（`VAE/`、枝刈りデコーダ PrunaVAED を含め計 2.34 GiB）＋ アップサンプラ（`Upscaler/`）＋ IC-LoRA 2点（1.22 GiB）と Deblur 1点（0.91 GiB）と In-Outpainting 1点（1.22 GiB）（`IC-LoRA/`）、`models/Preprocessors/` に DWPose 前処理器 2点（0.33 GiB）と Video-Depth-Anything 2点（0.12 GiB）。フォルダの意味は下の「models フォルダの構成」を参照 |
| Python 環境（`.uv_cache/` ＋ `.venv/` ＋ `.venv-engine/` ＋ `.venv-engine-ltx25/` ＋ `.python/`） | 約 7〜8 GiB | 実体はほぼ `.uv_cache/` にあり、3つの venv はそこへのハードリンク（同じ実体を指す別名）で共有するため、単純な足し算にはなりません |
| `tools/`（`uv` ＋ `ffmpeg`） | 約 0.4 GiB（実測 378 MB） | `setup.bat` が取り込む前提ツール。ffmpeg のダウンロードは約 104 MB だが、展開後はこの大きさになる |

> **この表の数値は、LTX 2.5 用の3つ目の仮想環境（`.venv-engine-ltx25`）が加わったあとの再実測がまだ済んでいません。**
> 上の「約 7〜8 GiB」「合計 約 40〜41GB」は 2 つの仮想環境だった時期の実測値です。3 つ目のぶんは共有キャッシュ
> （`.uv_cache/`）にも積み上がるため、実際にはこれより大きくなる可能性があります。**クリーンな環境で測り直すまで
> 数値は動かさない**方針で、測り直しは課題として起票済みです（[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-107）。

> **ページファイル用の 60GB は、この 40〜41GB の代わりにはなりません。** ページファイルは
> 別のドライブに置いていても構わない性質のもので（Windows の既定では C ドライブ）、
> 用途もまったく別です。**両方**必要だと考えてください。たとえばこのフォルダを D ドライブへ
> 置き、ページファイルが C ドライブにあるなら、D に 40〜41GB・C に 60GB の空きが要ります。

#### メインメモリとページファイル（最重要）

**実際に生死を分けるのはここです。** LTX 2.3 の推論では、Windows の「コミット」（プロセスが OS に確保を約束させた
仮想メモリの総量。物理 RAM とページファイルの合計で裏打ちされる）が、物理メモリをはるかに超える量まで伸びます。
開発機（メインメモリ 63.8GB ＋ ページファイル 48GB）で採取した実測値は次のとおりです。

| 時点 | コミット総量 | アイドル比 |
|------|--------------|-----------|
| アイドル（生成前） | 22.3GB | — |
| 生成ジョブ1のピーク | **70.1GB** | **+48GB** |
| 続けてジョブ2のピーク | 90.2GB | +68GB |
| 続けてジョブ3のピーク | 101.6GB | +79GB |

1回の生成でアイドル比 **+48GB**、さらに**連続して生成するとジョブごとに +12〜15GB ずつせり上がっていきます**
（アプリを再起動すれば元に戻ります）。つまり **メインメモリ 32GB では物理メモリだけでは全く足りず、
ページファイル（物理メモリが足りないときに中身をディスクへ退避させるための、ドライブ上の領域）が必須**です。
設定は次のようにしてください。

- Windows のページファイル設定は **「システム管理サイズ」のままにする**（Windows の既定値。手動で固定サイズにしない）。
- ページファイルを置いているドライブに **60GB 以上の空き容量**を確保しておく。OS は足りなくなるとページファイルを
  自動的に広げますが、広げる先の空きが無いと失敗します。
- **ページファイルを無効化しない・小さな固定サイズにしない**。「SSD の寿命が心配」「メインメモリが潤沢だから要らない」
  といった理由で無効化・固定小サイズにしている環境では、**生成の途中でワーカープロセスがエラーメッセージを一切出さずに
  落ちます**。Python の例外もログも残らない（OS レベルのアクセス違反で即死するため）ので、この状態は原因究明が
  非常に困難です。「生成が途中で止まるのにログには何も出ていない」場合は、まずページファイルの設定を疑ってください。
  実測記録は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §8.8 にあります。

### アプリ venv（`./.venv`, torch 無し）

> ここから先は**中で何が起きているかの説明**です。`setup.bat` が `scripts/install_ltx.ps1` を通じて
> ほぼ同じことを自動で行うので、通常の利用では手で打つ必要はありません。
> **2026-07-28更新**: インストーラは常に `uv sync --extra dev`（開発用の追加依存＝pytest ほかを含む）で
> 同期するようになりました。以前は `-RunSmoke` 指定時だけ `dev` extra が入る仕組みでしたが、
> 「セットアップし直すと pytest が消える」罠を仕組みで塞ぐため、常時同梱に変更しています
> （詳細は[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §40）。テストの動かし方は §6 を参照してください。

```powershell
# Python 本体もプロジェクト内に固定する（システムを汚さない）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"

uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev   # テストも動かす場合。実行だけなら素の uv sync でよい
```

これで `.venv/`（アプリ仮想環境）と `.python/`（uv管理のPython 3.12）がプロジェクト内に作成されます。
`pyproject.toml` / [`requirements.txt`](requirements.txt) には FastAPI 側の依存のみ定義しています（`torch` は含みません）。
これだけで **モック backend** で API/UI/テストが動きます（GPU 不要）。

### エンジン venv（torch+cu128）と実モデル

実生成には、アプリ venv とは別に**エンジン系統ごとの venv**と GGUF/component モデル群が必要です。

| venv | 対象 | 依存の宣言（再構築に使うファイル） |
|------|------|------------------------------------|
| `./.venv-engine` | LTX 2.3（`engine/worker.py`） | [`engine/engine-venv-pyproject.toml`](engine/engine-venv-pyproject.toml)（`[tool.uv.sources]` に torch cu128 index と `ltx-core`/`ltx-pipelines`/`diffusers` の git rev を記載）＋ [`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt)（`name==version` の完全スナップショット） |
| `./.venv-engine-ltx25` | LTX 2.5（`engine25/worker.py`） | [`engine25/engine25-venv-pyproject.toml`](engine25/engine25-venv-pyproject.toml) ＋ [`engine25/venv-engine-ltx25.freeze.txt`](engine25/venv-engine-ltx25.freeze.txt)（同じ作法。公式 LTX-2 v1.2.0 ＋ `transformers` 5.x） |

**2つに分かれているのは、LTX 2.3 と LTX 2.5 が要求するパッケージのバージョンが同居できないためです**（`transformers` 4.57 と
5.x）。どちらも `setup.bat` が自動で作るので、通常は意識する必要はありません。
`.venv-engine` の provenance と再現手順の詳細は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md) を参照してください。

> **`setup.bat` を再実行したときのふるまい**: エンジン venv は「すでに存在するから飛ばす」のでは
> なく、**ピン留めされた依存の内容が前回と変わっていないときだけ飛ばします**。freeze ファイルの中身と、
> `install_ltx.ps1` が持つ git リビジョンの指定をまとめてハッシュにし、
> venv 内の `.nz-engine-state` に記録した前回の値と突き合わせる方式です（**2つのエンジン venv は
> それぞれ独立に判定されます**）。`git pull` で依存が
> 変わっていれば自動で貼り直され、記録が無い場合（前回の導入が途中で中断した場合や、この仕組みが
> できる前に作られた環境）も貼り直しになります。「`git pull` のあとに `setup.bat` を再実行する」
> という更新手順は、この仕組みで成り立っています。

必要なモデル（`setup.bat` / `install_ltx.ps1` が自動でダウンロードします。**どのファイルがどこに要るかを宣言しているのは
`config.yaml` ではなく「ベースモデル記述子」**＝`scripts/manifests/*.json` で、そこに書かれた相対パスは
`config.yaml` の `model.models_dir`（既定 `./models`）を起点に解決されます。下から3行目までのうち IC-LoRA 系の 3 行は
`config.yaml` の `model.ic_loras:` が登録名と結びつけ、
DWPose 前処理器と VDA 深度前処理器は `engine/preprocess/dwpose.py`・`engine/preprocess/depth.py` がそれぞれ固定パスで読みます）:

| 要素 | 既定パス | 概算 | 取得元リポジトリ | 役割 |
|------|----------|------|------------------|------|
| GGUF transformer (Q4_K_M) | `models/LTX23/Weights/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | ~17GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | 本番トランスフォーマー |
| GGUF Gemma (Q4_K_M) | `models/LTX23/TextEncoder/gemma-3-12b-it-Q4_K_M.gguf` | ~7.3GB | [`Rootport/Nz-Gemma3-12B`](https://huggingface.co/Rootport/Nz-Gemma3-12B) | text encoder（GPU 推論・逐次オフロード） |
| component VAE / audio / text-projection | `models/LTX23/VAE/*.safetensors`（映像・音声 VAE と `prunavaed/` の枝刈りデコーダ）＋ `models/LTX23/TextEncoder/ltx-2.3_text_projection_bf16.safetensors` | ~3.9GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | 46GB モノリスを置換する小単体ファイル |
| spatial upsampler | `models/LTX23/Upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~0.95GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | 2段生成の x2 アップサンプラ |
| Gemma tokenizer dir (`gemma_root`) | `models/LTX23/TextEncoder/tokenizer/` | ~40MB | [`Rootport/Nz-Gemma3-12B`](https://huggingface.co/Rootport/Nz-Gemma3-12B) | tokenizer/preprocessor のみ（`tokenizer.model` 等）。**重みは含まない**（text encoder は上の GGUF Gemma が供給） |
| IC-LoRA 2点 | `models/LTX23/IC-LoRA/{pixel-spatial-upscaler,union-control}/*.safetensors` | ~1.2GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | `config.yaml` の `ic_loras:` が登録するアダプタの実体（`pixel-spatial-upscaler-x2` と、同一の union-control ファイルを3つの名前で公開した `canny-control` / `pose-control` / `depth-control`） |
| DWPose 前処理器 2点 | `models/Preprocessors/DWPose/{yolox_l,dw-ll_ucoco_384_bs5}.torchscript.pt` | ~0.34GB | [`Rootport/Nz-DWPose`](https://huggingface.co/Rootport/Nz-DWPose) | `pose-control` アダプタが参照動画から骨格を起こすときに使う姿勢推定モデル（`engine/preprocess/dwpose.py` が絶対パスで読む） |
| IC-LoRA Deblur 1点 | `models/LTX23/IC-LoRA/deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | ~0.91GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | ピンぼけした動画をくっきりさせる `deblur` アダプタの実体。前処理を必要とせず、ぼけた参照動画をそのまま渡す（2026-08-03 追加） |
| VDA 深度前処理器 2点 | `models/Preprocessors/VDA/video_depth_anything_vits.pth` ＋ `LICENSE` | ~0.12GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | `depth-control` アダプタが参照動画から深度マップ（手前と奥の距離を明暗で表した白黒映像）を起こすときに使う Video-Depth-Anything Small（`engine/preprocess/depth.py` が絶対パスで読む）。同梱の `LICENSE` は Apache-2.0 の全文で、この重みだけライセンスが異なるため必ず一緒に置かれる（2026-08-03 追加） |
| IC-LoRA In-Outpainting 1点 | `models/LTX23/IC-LoRA/in-outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` | ~1.22GB | [`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights) | 動画のキャンバス拡張（Outpainting）で使う `in-outpainting` アダプタの実体（2026-08-08 追加） |

上記10要素はすべて、本プロジェクトが再ホストした **3つの公開リポジトリ**（`Rootport/Nz-LTX23-weights`・
`Rootport/Nz-Gemma3-12B`・`Rootport/Nz-DWPose`）から取得します。いずれも Public かつ非 Gated（ライセンス承諾の壁が無い）ため、
**HuggingFace のアカウントもアクセストークンも一切必要ありません**。`setup.bat`（内部で
`scripts/install_ltx.ps1` を呼びます）を実行すれば、6回のダウンロードで全部揃います。

取得の手順は、どのファイルをどこへ置くかを宣言した**manifest**（`scripts/manifests/*.json`）が決めています。
リポジトリの中身はまず一時置き場（`models/.dl/`）へ落とし、そのあと manifest の対応表にしたがって
上表の場所へ移動します（3リポジトリの内部構造は旧フォルダ構成の名残なので、そのまま展開すると
新しい構成にはなりません）。一時置き場は成功するまで消さないので、途中で失敗しても再実行すれば
ダウンロード済みの分は再取得されません。

**IC-LoRA・DWPose 前処理器・VDA 深度前処理器も `install_ltx.ps1` が自動で取得します（手動配置は不要です）。** インストールの最後に出る
検証テーブル（**15 項目**）は、これらも含めて 1 ファイルずつ PASS/MISSING を表示します（表の行は manifest の
期待ファイル定義から作られるので、ダウンロードを守るサイズ判定と必ず同じ内容になります。**LTX 2.5 の記述子は
期待ファイルを 1 件も宣言していない**ため、この 15 行はすべて LTX 2.3 と共用前処理器のものです。LTX 2.5 を
`setup.bat` の対象外にしてあるのは設計上の意図で、下の「models フォルダの構成」を参照してください）。ここが MISSING のまま気づかないと、
UI にはアダプタ名（`pixel-spatial-upscaler-x2` / `canny-control` / `pose-control` / `depth-control` / `deblur`）が出るのに、
選んだ瞬間に 404 になる——という分かりにくい壊れ方をするため、あえて検証の対象に含めてあります。

> 生成の中核として実際にロードされるモデルは合計 **~28GB**（GGUF transformer + GGUF Gemma + components + upscaler + tokenizer dir＝28.15GiB）で、
> ComfyUI の GGUF 16GB レシピと同等のフットプリントです。`install_ltx.ps1` はこれに IC-LoRA 2点（1.22GiB）・DWPose 前処理器 2点（0.33GiB）・Deblur 1点（0.91GiB）・
> Video-Depth-Anything 2点（0.12GiB）・PrunaVAED 枝刈りデコーダ 1点（0.64GiB、`vae_mode=prune_vaed` 選択時のみ読み込み）・In-Outpainting 1点（1.22GiB）を加えた
> **約 33GB（32.51GiB）** をダウンロードします。IC-LoRA・DWPose・Deblur・VDA・In-Outpainting は無くても T2V/I2V の生成自体は成立しますが、`config.yaml` が IC-LoRA を
> 無条件に登録するため、欠けていると UI から選んだときに 404 になります。PrunaVAED は `vae_mode=prune_vaed` を選ばない限り読み込まれないため影響しません
> （上の検証テーブルの説明を参照）。
>
> この構成に至るまでに物理削除した重量物（モノリス safetensors・QAT Gemma dir）の経緯は
> [`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) §5.2 と
> [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §14・§40 にあります。

backend の選択は `config.model.backend`（`auto`/`mock`/`real`）で行います。既定 `auto` は「そのエンジンの python・
worker スクリプト・記述子が宣言するロード対象ファイルが全て存在」すれば **real**、無ければ **mock** です
（`_real_available`）。**判定材料は選んでいるベースモデルのエンジン系統ごとに違います**。

| | LTX 2.3（エンジン系統 `ltx`） | LTX 2.5（エンジン系統 `ltx25`） |
|---|---|---|
| python | `config.yaml` の `model.engine_python`（既定 `./.venv-engine/Scripts/python.exe`） | `config.yaml` の `model.engine_python_ltx25`（既定 `./.venv-engine-ltx25/Scripts/python.exe`） |
| worker | `model.engine_dir`（既定 `./engine`）の `worker.py` | `./engine25/worker.py`（**設定項目は無く固定**。`engine25/` は本リポジトリ同梱のため置き場所の選択肢がありません） |
| 重み | 記述子 `scripts/manifests/10-ltx23.json` の 4 カテゴリ＋固定ファイル 3 点 | 記述子 `scripts/manifests/20-ltx25.json` の 4 カテゴリ＋固定ファイル 1 点（空間アップスケーラのみ） |
| 実装 | [`services/engines/ltx/adapter.py`](services/engines/ltx/adapter.py)（旧パス `services/ltx_runner.py` は再エクスポート用の薄い層として残っています） | [`services/engines/ltx25/adapter.py`](services/engines/ltx25/adapter.py) |

### 追加の transformer GGUF / LoRA を配置する

**transformer GGUF**: `models/LTX23/Weights/`（LTX 2.5 なら `models/LTX25/Weights/`）**直下**に `.gguf` を置くだけで、
ファイル名から自動認識され UI/API のドロップダウンに列挙されます。サブフォルダに入れても再帰スキャンで拾われます。
**どこを・どの拡張子で・再帰するかを決めているのはベースモデル記述子**（`scripts/manifests/*.json` の
`categories.transformer` の `scan` / `extensions` / `recursive`）で、スキャンを実行するのが
[`services/model_registry.py`](services/model_registry.py) です。登録名はファイル名（拡張子除く）で、既定の登録名と
衝突する場合は親フォルダ名が `親フォルダ名__ファイル名` の形で前置されます。`config.yaml` の編集は不要です
（`model.transformers` への明示登録は、スキャンでは拾えないファイルを公開するための上書き用の代替手段です）。

> **モデルの取得元**は上の表のとおり、本プロジェクトが再ホストした3つの公開リポジトリです。
> 再ホストの経緯は [`Docs/NEXT_SESSION_HANDOFF.md`](Docs/NEXT_SESSION_HANDOFF.md)「2026-07-26 α版インストール導線の整備」
> ブロックにあります（当時の記述は旧フォルダ構成を前提にしています。旧構成のまま残っている環境は
> `setup.bat` の再実行で自動的に新しい構成へ移ります。下の「models フォルダの構成」を参照）。

選択は UI の「Models」設定タブのドロップダウン、または API `GET /models`（登録名の一覧確認）→
`POST /pipeline/load`（body `{"models": {"transformer": "<登録名>"}}`）で行います。選択が現在ロード中のものと
異なる場合のみワーカーが再構築されます。

GGUF の要件: (1) KVメタデータに `config`（モデル設定のJSON文字列）が埋め込まれていること、(2) テンソル名が
LTXネイティブの生キーであること、(3) `embeddings_connector` 層が非量子化（F32/BF16）であること。これらを
満たさない外部配布 GGUF はロードに失敗します（条件を満たすのは QuantStack 製、および自家製変換ツール
`Nz-GGUF-Converter-LTX23` の出力）。量子化タイプは既定の Q4_K_M に加え Q6_K / Q8_0 等にも対応します。

**LoRA**: ここで言う LoRA は、利用者が自分で用意する**画風・キャラクター系（スタイル LoRA）**のことです。
`config.yaml` の `ic_loras:` に登録済みの IC-LoRA（`pixel-spatial-upscaler-x2` / `canny-control` / `pose-control` / `depth-control` / `deblur`）は
`install_ltx.ps1` が自動取得するので、下記の手動配置の対象ではありません。

**LoRA の置き場所はベースモデルで分かれていません。** LTX 2.5 を選んでいるときも、下記の `models/LTX23/StyleLoRA/`
と `config.yaml` の `ic_loras:` に登録した同じファイルがそのまま使われます（**LTX 2.5 での LoRA・IC-LoRA 対応は
2026-08-24 から**。効き方の違いは §7.1 の LoRA の強さの項）。

`models/LTX23/StyleLoRA/` に `.safetensors` を置くと自動認識されます（`GET /loras` で一覧確認、
`POST /loras/reload` で明示再スキャン）。生成時は API の `loras: [{"name": ..., "strength": ...}]`、または
Gradio UI のプロンプト内 `<lora:名前:強度>` 記法で適用します（強度は 0.05〜2、0は不可）。**音声側の適用強度だけを映像側と別に指定したい場合は `<lora:名前:映像の強度:音声の強度>` の第3引数（0〜2。音声側だけ0=適用しないを指定できる）を使います。省略時は音声側も映像側の値に追従します（後方互換）。詳細は `Docs/LORA_AUDIO_STRENGTH_WORKORDER.md`。** ComfyUI 形式
（`diffusion_model.` プレフィックス＋ `lora_A`/`lora_B`）に対応し、量子化 GGUF モデルにもそのまま適用できます
（実行時加算方式のため、モデル側の量子化と衝突しません）。

### models フォルダの構成

`models/` は「どのベースモデルのものか」を最上位で分ける構成になっています。置き場所がそのまま
「このファイルは LTX 2.3 用です」という宣言になるため、ベースモデルが増えても、
ファイルの中身を見分ける仕組みを足さずに並べていけます。**LTX 2.5 への対応で、実際に `LTX23/` と `LTX25/` の2つが並んでいます。**

```
models/
├─ Preprocessors/            ベースモデルに依存しない前処理器（共用）
│   ├─ DWPose/               yolox_l.torchscript.pt, dw-ll_ucoco_384_bs5.torchscript.pt
│   └─ VDA/                  video_depth_anything_vits.pth（＋ LICENSE）
├─ LTX23/                    LTX 2.3 のためのファイル一式
    ├─ Weights/              transformer の GGUF（公式・自家変換とも。サブフォルダも再帰的に認識）
    ├─ TextEncoder/          gemma-3-12b-it-Q4_K_M.gguf ＋ ltx-2.3_text_projection_bf16.safetensors
    │   └─ tokenizer/        tokenizer 一式（重みは含まない）
    ├─ VAE/                  映像 VAE・音声 VAE
    │   └─ prunavaed/        枝刈りデコーダ（PrunaVAED）
    ├─ Upscaler/             ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    ├─ StyleLoRA/            利用者が用意する画風・キャラクター系 LoRA
    └─ IC-LoRA/              pixel-spatial-upscaler / union-control / deblur / in-outpainting
└─ LTX25/                    LTX 2.5 のためのファイル一式
    ├─ Weights/              transformer の GGUF
    ├─ TextEncoder/          Gemma 4 の GGUF（tokenizer は GGUF の中に入っています）
    ├─ VAE/                  映像 VAE（畳み込みデコーダ版）・音声 VAE
    │   └─ diffvae/          拡散デコーダ版の映像 VAE（現在は使いません・退避先）
    └─ Upscaler/             空間アップスケーラ
```

**LTX 2.5 の重みは、設計上 `setup.bat` の取得対象ではありません**（§1 の約33GBには含まれません）。`setup.bat` は Nz-Videomni 本体のインストーラで、
「最初にお試しいただくAI」として LTX 2.3 だけを一緒に導入します。**別のベースモデルは、そのモデル専用のバッチファイル
（`install-LTX25.bat` のようなもの）をダブルクリックして導入する**——という導線を予定しています。そのバッチ群はまだ用意できていないため、
**当面は下記の2つの公開リポジトリから手でダウンロードし、`models/LTX25/<カテゴリ>/` へ置いてください**（ファイル名は下表の期待名のままにします）。
置けばドロップダウンで「導入済み」になり、そのまま選べます（バッチ群の整備は[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-111）。

**LTX 2.5 が探すファイルは 5 本です**（記述子 `scripts/manifests/20-ltx25.json` が宣言している既定のファイル名。
すべて `models/` からの相対パスです）。

| 役割 | 期待するパスとファイル名 | 取得元リポジトリ |
|------|--------------------------|------------------|
| transformer（本体） | `LTX25/Weights/LTX-2.5-22B-distilled-transformer.gguf` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| テキストエンコーダ（Gemma 4） | `LTX25/TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf` | [`Rootport/Nz-Gemma4-12B-LTX25`](https://huggingface.co/Rootport/Nz-Gemma4-12B-LTX25) |
| 映像 VAE（畳み込みデコーダ版） | `LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| 音声 VAE | `LTX25/VAE/ltx-2.5-audio-vae-bf16.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| 空間アップスケーラ | `LTX25/Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |

**どちらのリポジトリも Public かつ非 Gated** なので、HuggingFace のアカウントもアクセストークンも要りません。
リポジトリ内のフォルダ名（`Weights/` ・ `TextEncoder/` ・ `VAE/` ・ `Upscaler/`）は上表のフォルダ名とそのまま対応しているので、
ダウンロードしたファイルを同じ名前のフォルダへ入れるだけで済みます。テキストエンコーダだけリポジトリが分かれているのは、
基になったモデル（Gemma 4）のライセンスが異なるためです（LTX 2.3 で `Rootport/Nz-DWPose` を分けているのと同じ理由）。

> **自動で取得する仕組みはまだありません（意図的にそうしてあります）。** 記述子のダウンロード定義は空のままで、
> `setup.bat` は LTX 2.5 のぶんを単に飛ばします（インストール最後の検証テーブルにも LTX 2.5 の行は出ません）。
> これは「本体のインストーラは LTX 2.3 だけを導入し、別のベースモデルは専用のバッチで足す」という設計によるもので、
> 重みが手に入らないからではありません——上表のとおり、重みは HuggingFace で公開済みです。
> **LTX 2.3 だけを使うぶんには何の影響もありません**（LTX 2.5 は、重みを置くまで「選べるが未導入」として画面に出ます）。
>
> テキストエンコーダのトークナイザは GGUF の中に入っているので別途置く必要はありません（隣に自動生成される
> `*.assets.safetensors` はその展開結果で、消しても次回に作り直されます）。

各フォルダには `put_〇〇_here.txt` という案内ファイルが1つ入っています。ファイル名がそのまま
「ここに置ける形式」の掲示になっているので、手に入れたファイルの置き場所に迷ったときの目印にしてください
（この案内ファイル自体はモデルの読み込み対象にはなりません）。

**すでに古い構成で使っている場合**、`setup.bat` を再実行すると自動で新しい構成へ移ります。
中身を**移動するだけ**（同じドライブ内なので数秒で終わり、再ダウンロードは発生しません）で、
移動先に同名のファイルがある場合は上書きせずその場で止まります。自分で置いた LoRA・自家変換 GGUF・
x4 アップスケーラーなども、対応表に無いものはそのまま持ち上がります。実行のたびに、何をどこへ移すかの
一覧が画面に出て、1ファイル1行の記録が `logs\model_migration_日付_時刻.log` に残ります。

> **元に戻したいとき**: 上の移行ログの `MOVE` 行は「移動元」「移動先」の順に並んだ表になっているので、
> ログを下から順に読み、各行の移動先を移動元へ戻す（PowerShell なら
> `Move-Item -LiteralPath models\<移動先> -Destination models\<移動元>`）だけで、移行前の配置に戻せます。
> `CONFIG` 行がある場合は `config.yaml.bak` を `config.yaml` へ戻してください。ただし、この作業は
> **古いコードへ戻す場合にだけ意味があります**（現在のコードは新しい構成のパスを見にいくため）。

> **`config.yaml` を自分で編集している場合の注意**: 移行時に、インストーラが `config.yaml` へ加える変更は
> **次の2種類だけ**です（どちらも実行前に `config.yaml.bak` を書き出すので、必要ならそこから戻せます。
> 何行書き換え・何行削除するかは実行前に画面へ出ます）。
>
> 1. **`model:` 配下のモデルパスの書き換え**——旧構成のパスを新構成へ直します。これを行わないと
>    `backend: "auto"` がモデルを見つけられず、エラーも出さないまま mock（実際には生成しない模擬動作）へ
>    降格してしまうため、安全のために自動化しています。
> 2. **廃止済みキーの行の削除**——モデルの既定パスを指定していた 8 つのキー
>    （`gguf_transformer_path` / `gguf_gemma_path` / `component_video_vae_path` / `component_audio_vae_path` /
>    `component_text_projection_path` / `component_video_vae_pruned_path` / `spatial_upsampler_path` / `gemma_root`）は、
>    ベースモデル記述子へ移った結果**読まれなくなりました**。残っていても動作は壊れませんが、起動のたびに
>    1キーにつき1本の警告が出続けるので、その行だけを消します（前後のコメント行や、`model:` の外にある
>    同名キーには触れません）。
>
> **これら以外の設定値・コメント・空行には一切触れません。**

---

## 2. 起動

**通常は `run.bat` をダブルクリックしてください。** 起動が終わると、黒い画面に囲み枠つきで
`http://127.0.0.1:18620/ui` のようなアドレスが表示されるので、それをブラウザで開きます
（このアドレスを表示するのは `main.py` の起動バナーで、そこが表示の正本です）。止めるときは、
その画面を × ボタンで閉じてください。

`run.bat` は、リポジトリ直下の `run.ps1` を実行ポリシーの制約を受けない形で呼び出すだけの
薄いラッパーです。`run.ps1` は残っており、PowerShell から直接実行することもできます
（引数はどちらからでも同じように渡せます）。

```powershell
# 起動スクリプト（環境変数設定 → tools/ を PATH の先頭へ → アプリ venv → main.py 起動 を一括）
./run.ps1

# もしくは直接（環境変数と PATH は自分で用意することになります）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe main.py
```

> **`PYTORCH_CUDA_ALLOC_CONF` の行は、Windows では効果が無いことが確認済みです。** PyTorch は
> 「`expandable_segments not supported on this platform`」という警告を出して従来型のメモリ管理のまま動きます
> （実測。[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75.7(2)）。さらに torch 2.9 では変数名そのものが
> `PYTORCH_ALLOC_CONF` へ移っています。**現状は害の無い置物**なので上の例からは消していませんが、
> 扱いは台帳[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-112 で整理する予定です。

`run.ps1` は **アプリ**（`./.venv` の `main.py`）を起動します。real backend が選ばれると、アプリが
`./.venv-engine\Scripts\python.exe -m engine.worker` を subprocess として自動 spawn します（手動起動は不要）。

`run.ps1` は起動を軽く保つため、**依存の再同期（`uv sync` など）は行いません**。`git pull` のあとは
`setup.bat` を実行してください（§1「更新のしかた」）。なお `.venv` がまだ無い場合は、その旨を表示して
終了します（自動では作りません。`.venv` だけ作ってもモデルもエンジン環境も無く、実生成はできないためです）。

起動後:
- UI: <http://127.0.0.1:18620/ui>
- API docs (Swagger): <http://127.0.0.1:18620/docs>
- Status: <http://127.0.0.1:18620/api/v1/status>

### CLI オプション

| オプション | 説明 |
|-----------|------|
| `--listen` | `0.0.0.0` にバインド（家庭内LAN公開。インターネット公開は非対応） |
| `--port <n>` | ポート変更（デフォルト 18620） |
| `--api-key <key>` | `Authorization: Bearer <key>` を要求 |
| `--allow-all-cors` | CORS 全許可（既定は localhost のみ） |
| `--config <path>` | `config.yaml` のパス指定 |
| `--te-offload` / `--no-te-offload` | Gemma text-encode の逐次 per-layer CPU オフロード（既定 ON） |
| `--dit-cpu-load` / `--no-dit-cpu-load` | DiT(transformer) を CPU 構築しブロックを GPU へストリーム（既定 ON。無効化するとロード時 ~16.9GB GPU スパイクが復活） |

```powershell
./run.ps1 --listen --port 19000
```

`run.bat` に渡した引数もそのまま `run.ps1` へ届きます（`run.bat --port 19000` のように使えます）。

---

## 3. アーキテクチャ

**2プロセス分離**（凍結 API 層 + 実エンジン worker）です。**worker は選んでいるベースモデルのエンジン系統によって
中身が入れ替わります**（同時に生きているのは常に1つ）。

```text
[Gradio /ui]──HTTP──┐
                     ▼
┌──────────────────────────────────────────────┐
│  FastAPI アプリ  (./.venv, torch 無し)          │
│   api/ (router, models, status, generate, ...) │
│   services/ (job_store, upload_store,          │
│              pipeline_manager, video_io,       │
│              low_vram, gpu_info)               │
│   services/engines/<系統>/adapter.py ── 唯一の  │
│                              推論エンジン接点   │
│        ├─ _MockBackend  (合成クリップ・GPU不要) │
│        └─ _RealBackend  ── subprocess.Popen ──┐ │
└───────────────────────────────────────────────┼─┘
                                                 │  JSON-lines (@@LTX@@ frames)
                                                 ▼  stdin/stdout
   ┌─────────────────────────────────────────────┬──────────────────────────────────────┐
   │ LTX 2.3 の worker（系統 `ltx`）              │ LTX 2.5 の worker（系統 `ltx25`）     │
   │  ./.venv-engine, torch+cu128                │  ./.venv-engine-ltx25, torch+cu128    │
   │  python -m engine.worker                    │  python -m engine25.worker            │
   │  engine/pipeline/fast_video_pipeline.py     │  engine25/pipeline25.py               │
   │  engine/gguf/  (GGUF dequant/loader)        │  engine25/gguf_transformer.py         │
   │  engine/gemma/ (GGUF Gemma + 層オフロード)   │  engine25/gguf_gemma4.py（Gemma 4）    │
   │  engine/transformer/ (block-swap, dit-cpu)  │  公式 LTX-2 v1.2.0 の推論スタック      │
   │  engine/preprocess/ (canny/dwpose/depth)    │  （前処理器は持たない＝v1 の範囲外）    │
   │  ログ: logs/ltx_worker.log                   │  ログ: logs/ltx25_worker.log          │
   │  → output.mp4 を共有 output dir に直接書く   │  → 同じ（書き出し方は共通）            │
   └─────────────────────────────────────────────┴──────────────────────────────────────┘
```

- **`engine/`（LTX 2.3）と `engine25/`（LTX 2.5）はどちらも first-party**（project root 直下・git 追跡）。
  旧・同梱フォーク `vendor/LTX-Desktop-LOW-VRAM` は
  refactor（Stage 2b）で完全削除済み。`engine/` の由来と provenance は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)。
  上流参照用の `vendor/LTX-2` は温存しています。**LTX 2.5 対応で `engine/` と `./.venv-engine` は 1 バイトも変更していません。**
- **プロトコル**: アプリは real backend でも torch/LTX を import しません。`_RealBackend` が
  `subprocess.Popen([<そのエンジンの python>, "-u", "-m", "<engine.worker または engine25.worker>"], cwd=<root>, env["PYTHONPATH"]=<root>)`
  で worker を常駐起動
  → `{"op":"load",...}` → `@@LTX@@{"event":"ready"}` → `{"op":"generate",...}` → `@@LTX@@{"event":"done",...}`。
  worker がモデルを **1度だけ**構築してジョブを使い回し、mp4 は worker が直接ディスクへ書きます（制御 JSON のみパイプを渡る）。
  **この受け答えの作法は2系統で共通**で、違うのは起動する python とモジュール、ロードペイロードの項目名、
  そして書き出すワーカーログの名前だけです。
- **16GB 技術**（すべて `engine/` に実装・実測済み）: GGUF Q4_K_M transformer + block-swap（GPU 常駐 8 ブロック）+
  GGUF Gemma の逐次 per-layer CPU オフロード（`--te-offload`）+ DiT の CPU 構築（`--dit-cpu-load`）+ VAE タイリング +
  component-file 経路。512×320 で peak_vram ~9.2GB、720p（1280×768→crop）実証済み。
- **モック backend** は `./.venv` のみで動く合成クリップ生成で、テストと GPU 無し開発に使います（`GenerationOutcome.backend`
  だけが real と異なり、API/スキーマ/出力構造は同一）。

---

## 4. API 概要（`/api/v1`）

> **この表は主要なものだけを載せた抜粋です。** 全ルートの一覧と各フィールドの詳細は、同じリポジトリ内のフロントエンドにある
> `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md` が正本です。稼働中のサーバーであれば
> `http://127.0.0.1:18620/docs`（Swagger UI）でも全ルートを確認できます。

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/status` | サーバー/GPU/低VRAM/キュー状態（`vram_optimization` ブロックを含む） |
| GET | `/config` | 現在の設定 |
| POST | `/pipeline/load` / `/pipeline/unload` | パイプラインの明示ロード/アンロード |
| POST | `/upload/image` | 最小I2V用画像をアップロードし `image_id` を返す |
| POST | `/generate` | 生成ジョブ開始（即 `job_id` を返す, 202） |
| GET | `/jobs` / `/jobs/{id}` | ジョブ一覧 / 状態 |
| GET | `/jobs/{id}/video` | 完了動画(mp4)を取得 |
| DELETE | `/jobs/{id}` | 実行中ジョブのキャンセル(best-effort) / 完了ジョブの削除 |

### 凍結 API 契約（付録A・変更しない）
- 幅・高さは **64の倍数**（two-stage distilled は stage1 を半解像度で生成し x2 アップサンプルするため ÷64。
  検証は [`api/models.py`](api/models.py) の `GenerateRequest` validator）。
- フレーム数は **8n+1**（9, 17, 25, … 121）。尺 cap は **20s（481f=8×60+1）@24fps** まで許容
  （旧 257f/10.67s から緩和）。溢れ/低速/非実用は**クライアント UI 警告に委ねる**方針で、
  解像度別 spill-free フレーム数を `GET /api/v1/config` の `limits.spill_free_frames`
  （720p:257 / 1080p:153 / 1440p:81）に露出する。これを超えると shared へ溢れ ~2-4x 低速化
  （OOM せず）。1080p の長尺は非実用（~40分・commit リスク）のため **720p 生成＋外部 upscale** 推奨。
  閾値の正本は [`Docs/RESOLUTION_DURATION_CAPABILITY.md`](Docs/RESOLUTION_DURATION_CAPABILITY.md) §8.4/§8.6。
- Distilled は **8 steps / CFG=1.0** 固定。
- I2V のキーフレーム画像は **最大5枚**（画像なし=T2V、1枚以上=I2V）。`frame_idx` は `0`（開始フレーム）か 8n+1 で、
  それ以外の値を送っても 422 にはならず、8n+1 グリッドへ丸めて `[1, num_frames-8]` の範囲へ収められます。
  「1枚・`frame_idx=0` 固定」は Phase 1 当時の制約で、Phase 3 で解除済みです。
- `crop_output` を指定すると、任意の非64サイズ（例 960×540, 1280×720）を中央クロップで得ます。

---

## 5. 16GB 向け生成テスト

> **本節は LTX 2.3 を選んでいるときの説明です。LTX 2.5 を選んでいるときは一部が使えません。**
> 生成の高速化（Acceleration）の5項目のうち、LTX 2.5 でまだ使えないのは PrunaVAED（`vae_mode`）の**1つだけ**です。
> これは **LTX 2.5 では既定値（Default）しか受け付けず**、それ以外を指定すると 422 で断られます（降格はしません）。
> 残る4項目は LTX 2.5 でも実際に効きます——先読み block swap と GGUF逆量子化の1カーネル化は **2026-08-24 から**（既定は on。
> 2つ併用で生成時間が 61.8%短縮され、出来上がる動画は変わりません）、モデル骨格の常駐（`keep_resident`）と
> `sage attention` は **2026-08-25 から**です（前者は既定 off のままで2本目以降が約25%短縮、後者は既定 `sdpa` のままで
> 選ぶと 1280×768 の連結生成が約1.10倍。**ただし `sage` は選ぶと絵の細部が変わります**）。
> **撮り直し（Retake）と素材（末尾）は 2026-08-26 から LTX 2.5 でも使えます**（§7「制限事項」の LTX 2.5 の項。
> **ただしオーナーの目視確認がまだ済んでいません**）。**LTX 2.5 でまだ使えないのは、キャンバス拡張（Outpainting）・
> NAG／VSF・PrunaVAED・非蒸留モデル用の高品質パイプラインの4つ**です。
> **クリップ連結（Chained）・V2V・A2V（長尺 A2V・バッチ A2V を含む）は 2026-08-23 に対応済み**で、LTX 2.5 でも
> そのまま使えます（操作パネルからの実機確認済み）。**スタイル LoRA と IC-LoRA（参照動画による制御。長尺を含む）も
> 2026-08-24 に対応済み**です（操作パネルからの目視確認も 2026-08-25 に合格しています）。

出力は `outputs/{job_id}/output.mp4` と `outputs/{job_id}/metadata.json` に保存されます。
`peak_vram_mb` は `metadata.json` またはワーカーログの `GENERATED_OK peak_vram_mb=` から取得できます
（jobs API 応答には含まれません）。**ワーカーログはエンジン系統ごとに別ファイル**で、LTX 2.3 は
`logs/ltx_worker.log`、LTX 2.5 は `logs/ltx25_worker.log` です（切り替えても両方の記録が残るようにしてあります）。

### smoke_test (T2V, 384x256 / 17 frames)

```bash
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A red ball rolling on a white floor","width":384,"height":256,"num_frames":17,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[]}'
# -> {"job_id":"...","status":"queued",...}

curl http://127.0.0.1:18620/api/v1/jobs/<job_id>
curl http://127.0.0.1:18620/api/v1/jobs/<job_id>/video --output out.mp4
```

### minimal (T2V, 512x320 / 49 frames)

```powershell
# seed 固定で決定的に検証（別プロセスでもバイト一致することを実測済み）
$body = '{"prompt":"a calm ocean wave rolling onto a sandy beach at sunset, cinematic","width":512,"height":320,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":12345,"conditioning_images":[]}'
# POST /api/v1/generate に body を送り、GET /api/v1/jobs/{id} で完了確認
# ログは LTX 2.3 なら ltx_worker.log、LTX 2.5 なら ltx25_worker.log
Select-String -Path logs/ltx_worker.log -Pattern "GENERATED_OK|LOAD_FAILED|GENERATE_FAILED"
```

### 720p (1280x768 生成 → 任意で 1280x720 にクロップ)

```powershell
# crop_output を付けると worker が full-size を書き、ffmpeg 中央クロップで 1280x720 配信
$body = '{"prompt":"...","width":1280,"height":768,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":12345,"crop_output":{"width":1280,"height":720},"conditioning_images":[]}'
```

### 最小I2V (512x320 / 49 frames + 画像1枚)

```bash
# 1) 画像アップロード
curl -X POST http://127.0.0.1:18620/api/v1/upload/image -F "file=@first_frame.png"
# -> {"image_id":"...",...}

# 2) 生成（conditioning_images に image_id を1件だけ・frame_idx=0）
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The scene slowly comes alive","width":512,"height":320,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[{"image_id":"<id>","frame_idx":0,"strength":0.8}]}'
```

### worker 単体スモーク（engine を直接叩く場合）

```powershell
$env:PYTHONPATH = (Get-Location).Path
& ".\.venv-engine\Scripts\python.exe" -m engine.worker   # LTX 2.3。{"op":"load",...} を stdin へ → @@LTX@@{"event":"ready"}

# LTX 2.5 は別の仮想環境・別のモジュール（受け答えの作法は同じ）
& ".\.venv-engine-ltx25\Scripts\python.exe" -m engine25.worker
```

### Gradio UI
`/ui` を開き、画像なしで「生成」→ T2V、画像1枚を指定して「生成」→ 最小I2V。

**A2V（音声から動画生成）**: Generate タブの「A2V（音声から動画生成）」アコーディオンに音声ファイルを添付する。`.wav` 推奨（添付すると音声長に収まる最大フレーム数を自動でFramesへ入力してくれる。他形式は自動調整の対象外でサーバー側チェックに委ねる）。画像でキャラクター等を固定したい場合は「キーフレーム画像」アコーディオンを使う（5スロットとも A2V と併用可）。**スタイルLoRA（画風・キャラクター系の軽量アダプタ。`<lora:...>` 記法）とは併用できる**（2026-07-11 に解禁。それ以前は排他だった）。**参照動画を必要とする control 系 IC-LoRA（参照動画から輪郭線 canny・骨格 pose 等を読み取って条件付けするアダプタ）も、A2V との併用が α版として解禁された**（2026-07-11。上のアダプタ欄で control 系を選び、参照動画をアップロードする）。Clip Chain タブでの複数クリップ連結における control 系アダプタの併用は、**長い参照動画を1本だけ添付すると各クリップが担当する区間をサーバーが自動で切り出してstage-1にのみ注入する「長尺IC-LoRA」として2026-08-11に解禁された**（参照が生成の尺より短ければ、足りない分は参照なしで生成される）。ただし `depth-control` のみ、前処理（Video-Depth-Anything）が全編一括設計でメモリに載らないため2クリップ以上のチェーンでは引き続き非対応で、指定すると `422 LORA_DEPTH_CHAIN_UNSUPPORTED` で拒否される（クリップ1本のチェーンと単発生成は従来どおり使える）。

**Batch A2V（就寝中の一括生成・2026-07-12 実装・GPU実機ゲート合格済み）**: A2V アコーディオンの下にある「Batch A2V」アコーディオンの Enable チェックをオンにすると、ゆっくり実況・VOICEROID実況（音声合成ソフトによるキャラクター実況動画）向けに、音声フォルダの中身をまとめて一括生成できる。音声フォルダ・画像フォルダ・出力先を入力し、「Set audios」を押すと、音声ファイルごとの行（プロンプト・使用画像・状態など）を持つ表が作られる。表の上でプロンプトや使用画像を行ごとに編集し、「Start a2v batch」で開始すると、Generate タブの現在の設定（解像度・fps・シード・LoRA・共通プロンプト・共通キーフレーム画像など）をその時点でのスナップショットとして全行に適用しながら、1件ずつ順番に生成する。バッチ実行中は誤操作による二重投入を防ぐため、Generate ボタンが「Batching a2v...」表示になり押下できなくなる（完了で「Start a2v batch」に復元）。

このバッチの実処理はブラウザの画面ではなく**サーバー側のバックグラウンドスレッド**で回るため、ブラウザを閉じてもタブがスリープしても生成は止まらない。進行状況は音声フォルダ直下の CSV マニフェスト（`batch_a2v_manifest.csv`）に逐次記録され、翌朝ブラウザを開き直すか、電源断・アプリ再起動のあとに「Set audios」→「Start a2v batch」をやり直すことで、続きから再開できる（Done/Skip 済みの行は再実行されない）。出力動画は音声フォルダの隣に自動生成される `{フォルダ名}_a2v_out` フォルダ（または指定したカスタムフォルダ）に、音声と同じファイル名の mp4 として保存され、同名ファイルが既にある場合は連番を振って上書きしない（ガチャ＝同じセリフの撮り直し運用にも対応）。

サーバー側の `outputs`／`uploads` フォルダは自動掃除されない（ユーザーによる手動削除での運用を想定した設計判断）。行の個別プロンプトへの `<lora:...>` トークン指定は現状非対応（LoRA は共通プロンプト側／バッチ開始時の設定で全行共通）。

**開始前チェック（フールプルーフ、2026-07-12 実装）**: 「Start a2v batch」を押すと、処理対象の行（Waiting／Failed／クラッシュ遺残）に対して入力忘れがないか検査し、問題があれば理由を表示してバッチ自体を開始しない。プロンプトは、共通プロンプトが入力済みなら開始可、空でも Replace モードかつ処理対象行のプロンプトが全部埋まっていれば開始可（Add モードで共通プロンプトが空の場合や、Replace モードで空行が残っている場合は開始不可）。画像も同様に、共通キーフレーム画像の**1枚目**（Keyframe images のスロット1＝先頭フレーム。Use チェックがオンで画像も入っている必要がある）が設定済みか、処理対象行が全部個別画像指定になっていれば開始可で、1枚目が未設定のまま "Shared" の行が残っていると開始不可になる（スロット2以降だけ設定されていても1枚目扱いにはならない）。一部の行だけ入力忘れのまま走って Failed が量産される事態を防ぐための仕組みで、全行に個別入力を済ませた使い方はこれまでどおり問題なく開始できる。

また、開始時には処理対象の全行についてフレーム数と Skip 判定をその時点のフレームレートで再計算してから CSV へ書き込みバッチを開始する。手動で Waiting に戻した上限超過行も確実に Skip へ戻り、「Set audios」後に fps や解像度を変えてもフレーム数は開始時の設定で計算し直される（行の frames 値を手で編集していても、開始時にこの再計算で上書きされる点に注意）。

詳しい仕様・設計上の決定事項は [`Docs/BATCH_A2V_WORKORDER.md`](Docs/BATCH_A2V_WORKORDER.md) を、CSV マニフェストの列定義・文字コードなど相互運用のための共通仕様は [`Docs/BATCH_A2V_CSV_SPEC.md`](Docs/BATCH_A2V_CSV_SPEC.md) を参照。

### 非CFGネガティブプロンプト（NAG）

蒸留版 LTX 2.3 は CFG（Classifier-Free Guidance。正負2パスの denoise でネガティブプロンプトを効かせる従来手法）が `guidance_scale=1.0` に凍結されているため、従来型のネガティブプロンプトはこれまで何も効かない no-op だった。**NAG（Normalized Attention Guidance）** は、cross-attention（テキストと映像/音声の対応を取る注意機構）の出力レベルで正プロンプト出力と負プロンプト出力を外挿・正規化・ブレンドすることで、CFG の2パス化なしに1パスのままネガティブプロンプトを効かせる非CFG手法。単発 Generate（`/generate`）・Clip Chain（`/generate/chain`）・バッチA2V（内部的に行ごとに `/generate/chain` を叩く）の**全経路で使える**。

**API**（`GenerateRequest` / `GenerateChainRequest` 共通）:

| フィールド | 型 | デフォルト | 範囲 | 説明 |
|-----------|----|-----------|------|------|
| `nag_enabled` | bool | `false` | — | ONにするとNAGが有効になる。ONで`negative_prompt`が空だと422 |
| `nag_scale` | float | `11.0` | `1.0`〜`20.0` | 負プロンプトをどれだけ強く外挿するか |
| `nag_tau` | float | `2.5` | `1.0`〜`10.0` | ノルムの頭打ち上限（暴れ防止） |
| `nag_alpha` | float | `0.25` | `0.0`〜`1.0` | 正出力とのブレンド比率 |

`negative_prompt` にも `max_length=2000`（`prompt` と同じ上限）が付いている。既定値 11.0/2.5/0.25 は、先行実装 [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) の `LTX2_NAG` に準拠したもの。

**GUI**: 共有プロンプト欄の直下（Generate/Clip Chain 両タブの外）にある「Negative Prompt」アコーディオンから使う。テキスト欄は既定では入力不可（グレーアウト）で、「non-CFG Negative」チェックボックスをONにすると編集可能になり、下の3スライダー（scale/tau/alpha）も効くようになる。「NAG / Other」ラジオは将来の拡張用の枠で、Other を選ぶと即座に NAG へ戻るフォールバック動作になる（現状は NAG のみ実装済み）。

**コスト**: cross-attention の計算が正負2回に増える（自己注意は増えないため全体では数%〜15%程度の増加見込み）。VRAM は negative context とゲート用の中間テンソル分だけ増える（768p 帯で数百MB程度の見込み）。詳しい設計判断・非対称設計（AdaLN変調の扱い）の根拠は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §38 を参照。

### 生成の高速化（Acceleration）

生成そのものを速くするための切替を、設定画面の「Acceleration（生成の高速化）」という区画にまとめました
（AviUtl2 の操作パネルなら Settings、Gradio UI なら Settings タブ）。項目は5つあり、**LTX 2.3 では5つとも実際に効きます**
（2026-08-05 に最後の1つ「VAE」が実装され、将来の実装枠として場所だけ確保してあったグレーアウトの項目は
なくなりました）。**LTX 2.5 では、このうち4つが効きます**——GGUF逆量子化の1カーネル化と先読み block swap が 2026-08-24 から、
モデル骨格の常駐（`keep_resident`）と SageAttention（Attention の `sage`）が 2026-08-25 からです。

| 項目 | 選択肢 | LTX 2.3 での状態 | LTX 2.5 での扱い |
|------|--------|------------------|------------------|
| Fused GGUF Dequantization Kernel（GGUF逆量子化の1カーネル化） | On / Off | **実装済み**。既定は on（2026-08-04） | **実装済み**。既定は on（2026-08-24） |
| Attention（注意機構の実装） | `sdpa` / `sage attention` | **実装済み**。既定は `sdpa` | **実装済み**。既定は `sdpa`（2026-08-25） |
| Block-swap prefetch（先読みblock swap） | On / Off | **実装済み**。既定は on | **実装済み**。既定は on（2026-08-24） |
| モデル骨格の常駐（keep_resident） | On / Off | **実装済み**。既定は off | **実装済み**。既定は off（2026-08-25） |
| VAE（映像の復元処理） | Default / PrunaVAED | **実装済み**。既定は Default（＝off。恒久的に off のままです）（2026-08-05） | Default のみ。PrunaVAED は **422** |

**GGUF逆量子化の1カーネル化と先読み block swap の2つは、2026-08-24 に LTX 2.5 でも使えるようになりました**（それまでは「指定しても無視して生成を続ける」扱いでした）。
**2つを併用したときの生成時間は、実測で 102.86 秒 → 39.28 秒＝61.8%短縮です。**
出力そのものは変わらず、**固定の検証用ジョブ17本すべてで動画のファイルが1バイトも変わっていない**ことを確認しています。
**モデル骨格の常駐（`keep_resident`）も 2026-08-25 に LTX 2.5 で使えるようになりました**（既定は off のままです。
2本目以降の生成が実測で 27.6秒 → 20.5秒＝約25%短縮になります。下の「モデル骨格の常駐」の節を参照してください）。
**同じ 2026-08-25、Attention の `sage`（SageAttention）も LTX 2.5 で使えるようになりました**（既定は `sdpa` のままです。
実測は 1280×768・2クリップ×121フレームの連結生成で **1.1017倍**）。**ただしこれは、上の3つと違って
「選ぶと出来上がる絵の細部が変わる」種類の高速化です**——下の注意書きを必ず読んでください。
**LTX 2.5 での `sage` は、開発者による検証に加えて、2026-08-25 にオーナーによる目視確認にも合格しています。**
残る1つ（PrunaVAED）は LTX 2.5 では **422** で断られます——利用者が意図して on にしたときだけ
付くものなので、効かないまま黙って通すより断ったほうが親切だという判断です。詳しくは §7.1 と
[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75・§76・§77 を参照してください。

**選び方**: `sdpa` は PyTorch 標準の実装で、これまでどおりの結果が出ます。`sage` は
[SageAttention 2.2.0](https://github.com/thu-ml/SageAttention)（量子化を使って注意機構の計算そのものを速くする外部
カーネル）を使い、**生成が速くなります**。切替はラジオボタンを押すだけで、サーバーの再起動もモデルの読み込み直しも
要りません。選んだ内容はブラウザ側に保存され、次に開いたときも保たれます。

**どれくらい速いか**: 開発機（RTX 4070 Ti SUPER 16GB）での実測は次のとおりです。

| ベースモデル | 条件 | `sdpa` | `sage` | 倍率 |
|---|------|--------|--------|------|
| LTX 2.3 | 720p（1280×768）・257 フレーム | 261.4 秒 | 224.3 秒 | **1.17 倍** |
| LTX 2.3 | 同上・二段目（stage 2）だけを見た場合 | 32.76 秒/ステップ | 20.95 秒/ステップ | **1.56 倍** |
| LTX 2.3 | i2v（画像からの動画生成）＋NAG・1344×1728・153 フレーム | 460.63 秒 | 366.85 秒 | **1.26 倍** |
| LTX 2.3 | 512×320・121 フレーム | 80.8〜83.1 秒 | 77.8〜87.5 秒 | **0.95〜1.04 倍**（効かない） |
| **LTX 2.5** | 1280×768・2 クリップ×121 フレーム（クリップ連結） | 119.5 秒 | 108.5 秒 | **1.10 倍** |
| **LTX 2.5** | 1280×768・121 フレーム・二段目（stage 2）だけを見た場合 | 27.74 秒 | 22.71 秒 | **1.22 倍** |

VRAM の使用量は実測で変わりません（LTX 2.3 でピークの差は 0.08% 以内、LTX 2.5 では 0.31% 以内）。

> **⚠ 効き目は動画の大きさに強く左右されます。** `sage` が速くするのは注意機構の計算だけなので、
> **その計算量が大きいほど効きます**——大きい画面・長い尺ほど効き、**小さい画面（512×320 級）では
> ほとんど効かないか、かえって遅くなることがあります**（上の表の 0.95 倍の行がそれです）。
> また、`sage` が触れない区間（文章の読み取りと動画への書き出し）が全体の4分の1ほどあるため、
> 二段目だけを見た倍率がそのまま全体の倍率になることはありません。
> **「on にすれば必ず速くなる」ものではない**とお考えください。

> <a id="sage-seed-note"></a>**⚠ `sage` を選ぶと、同じシードを指定しても生成結果の細部が変わります。** 計算に使う数値の精度が違うためで、
> 不具合ではありません。構図や被写体といった大枠は同じままで、質感やノイズの出方といった細かいところが変わります
> （実測での差は PSNR〔ピーク信号対雑音比。元の絵との違いを 1 つの数値にしたもので、単位は dB。数値が大きいほど元の絵に近い〕で 27〜28dB 程度）。**以前つくった動画とまったく同じものを作り直したい場合は、`sdpa` を
> 選んでください。** 既定を `sdpa` のままにしてあるのは、アップデートによって利用者の生成結果が黙って変わることを
> 避けるためです。**この注意書きが `sage` についての正本**で、他の箇所（§8 の MCP 注意事項など）はここを参照します。
>
> **LTX 2.5 でも 2026-08-25 から `sage` が選べます**（それまでは 422 で断っていました）。**同じシードでも細部が変わる
> という上の注意は、LTX 2.5 でもそのまま当てはまります**——むしろ、そのことを確かめるために LTX 2.3 の実績値と
> 見比べる検証を行ったうえで開通させています（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §77）。
> 既定は LTX 2.5 でも `sdpa` のままです。

**入っていない環境ではどうなるか**: `sageattention` は `setup.bat` が標準で入れますが、何らかの事情で入っていない
環境（古い手順で作った仮想環境など）では、`sage` を選んでも**エラーにはならず、自動的に `sdpa` に切り替わって
最後まで生成されます**。**この降格のふるまいは LTX 2.3 でも LTX 2.5 でもまったく同じです**（2026-08-25 から）。利用可否は `GET /status` の `acceleration.sage_available` で確認でき、操作パネル側は
使えないときはボタン自体が選べなくなります。実際にどちらで生成されたかは、生成後に
`outputs/{ジョブID}/metadata.json` の `attention_used`（`"sdpa"` / `"sage"` / `"sage->sdpa"`）で確認できます。

> **最初の1本だけ遅く感じることがあります。** `sage` は内部で triton という仕組みを使い、その初回にだけ
> 「JIT コンパイル」（実行時にカーネルを組み立てる処理）が走るためです。2本目以降は本来の速さになります。
> 速度を測って比べたい場合は、この点と、**worker プロセスの最初のジョブだけ約 8% 速い**という癖の両方を踏まえて、
> `sdpa` と `sage` を交互に流して隣り合う組で比べてください（詳細は
> [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §43.6）。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `attention_backend`
（`"sdpa"` または `"sage"`、既定 `"sdpa"`）を指定できます。詳しい設計判断・検証結果は
[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §43 を参照してください。

### 先読みblock swap（`block_swap_prefetch`）

block swap（VRAM を節約するために transformer のブロックを CPU と GPU のあいだで出し入れする既定の仕組み）の転送を、
計算とは別の CUDA stream で先回りさせて待ち時間を隠す機能です。Settings の Acceleration 区画にある「Block-swap
prefetch」トグルで、Attention と同じくジョブ単位で切り替えられます。**既定は on**（2026-08-02、実機ゲート全項目
合格を受けて反転）。

`sage` と違い、**転送方式だけを変えるので出力は変わりません**——同じシードなら off/on でビット単位で完全に同じ動画
になります。実測（768p/257フレーム、交互対比較3組）で平均約14.7%短縮、VRAM の増加はほぼ0です。off にすると従来の
同期スワップへ戻ります（block swap 自体が無効な設定では、on/off にかかわらず何も起きません）。詳しい設計判断・
マイクロベンチ・実機ゲートの実測値は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §44 を参照してください。

**LTX 2.5 でも 2026-08-24 から使えます**（既定は on）。単独で使ったときの実測は**平均 25.3%短縮**、
GGUF逆量子化の1カーネル化と併用すると**61.8%短縮**です。ここでも**出力は変わりません**——固定の検証用ジョブ17本すべてで
動画のファイルが1バイトも変わらないことを確認しています（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75）。

> **メインメモリを約416MB（207.9MB × 2枠）、サーバーを起動しているあいだずっと使います。** 重みを CPU から GPU へ
> 速く渡すために、OS が動かせない形で確保した受け渡し用の枠（ページ固定メモリ）を2つ持ち続けるためです。
> **この枠はジョブが終わっても返しません。off にしても返りません。解放されるのはサーバーを再起動したときだけです**
> ——枠を手放す処理は「block swap の仕組みそのものを取り外す」経路にしかなく、本番の運転ではその経路を通らないためです
> （枠を作り直すには 100ms 前後かかるうえ、長く動かしたプロセスほど確保に失敗しやすくなるので、
> 作っては返すのではなく持ち続ける設計にしてあります）。枠の大きさはモデルの最も大きなブロックに合わせて決まるので、
> LTX 2.3 では 253.8MB × 2枠（約508MB）になります。**LoRA を使うと最も大きなブロックがそのぶん育つので、枠も一緒に
> 大きくなります**（LTX 2.3 の実測で 260.8MB／266.8MB／277.9MB × 2枠。使う LoRA の大きさによって変わります）。

### GGUF逆量子化の1カーネル化（`fused_gguf_dequant_kernel`）

GGUF ファイルの中で圧縮された形で持っている重みを計算に使える形へ展開する処理（逆量子化）を、**これまでの18〜33個の
細かい GPU 処理から、量子化形式ごとに1個の GPU 処理へまとめた**機能です。Settings の Acceleration 区画にある
「Fused GGUF Dequantization Kernel」トグルで、他の項目と同じくジョブ単位で切り替えられます。**既定は on**
（2026-08-04、実機ゲート全項目合格を受けて反転）。

先読みblock swap と同じく、**展開の手順しか変えないので出力は変わりません**——同じシードなら off/on でビット単位で
完全に同じ動画になります。実測（768p/257フレーム、交互対比較3組）で**約17.5%短縮**（144.5秒 → 119.2秒）。この環境で
動かせない場合（Triton が入っていない等）は自動的に従来の方法へ戻り、生成そのものは止まりません。実際に効いたかどうかは
生成後のメタデータの `fused_gguf_dequant_kernel_used`（`"off"` / `"on"` / `"on->off"`）で確認できます。詳しい設計判断・
自己検証・実機ゲートの実測値は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §51 を参照してください。

**LTX 2.5 でも 2026-08-24 から使えます**（既定は on）。単独で使ったときの実測は**生成時間の中央値が 99.95秒 → 65.72秒**で、
**映像を作る本体だけでなく、文章を読み取る部分（テキストエンコーダ）にも効いている**ことを実機で確認しました。
ここでも出力は変わりません（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75）。

### モデル骨格の常駐（`keep_resident`）

生成のたびに作り直している**モデルの「骨格」**（GGUF ファイルから組み上げた重みの一覧とモジュールの構造。DiT 約
16.5GB ＋ Gemma 約 8〜9GB）を、次のジョブでもそのまま使い回す機能です。Settings の Acceleration 区画にある
「モデル骨格の常駐（ジョブ間キャッシュ）」トグルで切り替えます。**既定は off** です。

効くのは毎回の生成の先頭に乗っている待ち時間（前処理）だけで、開発機での実測は **68.6〜75.0 秒 → 5.32 秒**でした。
2本目以降の生成から効くので、シードを変えて何本も試すときやバッチを流すときに差が出ます。**生成結果は変わりません**
——同じシードなら on/off でビット単位まで同じ動画になります。GPU 側には何も置かないため、VRAM の使用量も変わりません。

> **⚠ 使うときはメインメモリ 64GB 以上を推奨します。** 骨格を持ち続けるぶん、**約 20GB のメインメモリを常時占有**
> します（実測 19.4GB）。既定を off にしてあるのはこのためで、**off に戻せば解放されます**（ただし OS がメモリを完全に
> 返すまでには少し間があります。詳しくは [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §48.8）。32GB 級のマシンで
> on にすると、ページファイルへの追い出しが起きて逆に遅くなることがあります。

一部の設定とは併用できません。`dit_cpu_load` か先読み block swap を off にしている場合は、警告を出したうえで
**そのジョブだけ自動的に off** になって最後まで生成されます（メモリが二重に必要になるため）。GGUF の逐次量子化を
off にしている場合だけは、生成結果が壊れる（LoRA を融合した重みがキャッシュに焼き付く）ので**エラーで止まります**。
実際にどう扱われたかは `outputs/{ジョブID}/metadata.json` の `keep_resident_used`（`"off"` / `"on"` /
`"on->off"`）で確認できます。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `keep_resident`（真偽値・既定
`false`）を指定できます。以前あった環境変数 `LTX_KEEP_RESIDENT` は**撤去済み**で、設定しても効きません。詳しい
設計判断・検証結果は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §48 を参照してください。

#### LTX 2.5 での常駐（2026-08-25 から）

**LTX 2.5 でもこのトグルが効くようになりました。既定は off のままです。** ただし**同じ名前でも中身は別物**です——
LTX 2.3 が抱え込むのは全部品の骨格（約 20GB）ですが、**LTX 2.5 が抱えるのは文章を読み取る部分（Gemma 4 テキスト
エンコーダ）の重み1つだけ**で、**常駐に使うメインメモリは実測 7.7GiB** です。

- **効くのは2本目以降です。** 1本目は重みを組み立てるので、これまでどおりの時間がかかります。2本目からは
  その組み立て（6〜8秒）が **0.4秒**になり、**生成そのものが実測で 27.6秒 → 20.5秒＝約25%短縮**されます
  （連結生成では約5秒の短縮です）。**出来上がる動画は1バイトも変わりません。**
- **VRAM は増えません。** メインメモリと引き換えの機能なので、GPU 側には何も置きません（実測でも VRAM の
  ピークは on/off で完全に同じ値でした）。
- **off に戻すと、そのジョブの先頭で解放されます**（ログに「released 7.68 GiB」と出ます）。もう一度 on にすると
  組み立てを1回だけ払い直します（実測 4.4秒）。
- **LTX 2.5 には併用の制限も自動 off もありません。** `metadata.json` の `keep_resident_used` に出る値は
  **`"on"` か `"off"` の2つだけ**で、LTX 2.3 で出ることのある `"on->off"`（自動的に off へ落ちた）は
  **LTX 2.5 では出ません**。
- **メインメモリの目安**: LTX 2.5 は常駐 off のときでも生成中に約 26GiB まで使います。常駐を on にする場合は、
  安全側に見積もって **合計 約 34GiB** を見ておいてください（§1「ハードウェア要件」）。
- **これまであった落とし穴が1つ消えました。** 以前は、ベースモデルに LTX 2.5 を選んだまま Settings でこの常駐を
  on にすると、**以後のすべての生成が 422 で断られました**。画面上は「LTX 2.5 が壊れた」ようにしか見えず、しかも
  原因は別のタブの設定にありました。2026-08-25 からはそのまま生成できます。

詳しい実測は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §76 を参照してください。

### 枝刈り版の映像VAEデコーダ（PrunaVAED・`vae_mode`）

映像の復元処理（生成の最後に、内部表現から実際の映像フレームを作る処理）を、**枝刈り**（pruning＝寄与の小さい部分を
削ること）と**蒸留**（distillation＝軽くしたモデルに元の出力を教え込むこと）を施した軽量なデコーダへ差し替える機能です。
上流は [Pruna AI](https://huggingface.co/PrunaAI/PrunaVAED) が公開している LTX-2.3 専用のデコーダで、これを当方で
LTX-2.3 のエンジンが直接読める形（約690MB の単体ファイル）へ変換し、`setup.bat` が他のモデルと一緒に取得します。
Settings の Acceleration 区画にある「VAE」の Default / PrunaVAED で、他の項目と同じくジョブ単位で切り替えられます。
**既定は Default（off）** で、**今後も既定を PrunaVAED へ変えることはありません**（理由は次の注意書きのとおりです）。

> <a id="prunavaed-quality-note"></a>**⚠ PrunaVAED を選ぶと、出力品質がわずかに低下する可能性があります。** 別のデコーダで映像を作るので、`sage` と
> 同じく「絵が変わる」種類の切替です。オーナーによる同一シードの見比べでは「劣化は肉眼ではほとんど分からない」水準
> でしたが（客観指標では PSNR 36.06dB・SSIM〔構造類似度。2 つの映像がどれだけ同じ構造をしているかを 0〜1 で表す指標で、1 に近いほど同じ〕0.9854〔輝度〕）、**以前つくった動画とまったく同じものを作り直したい
> 場合は Default を選んでください**。既定を Default のままにしてあるのは、アップデートによって利用者の生成結果が
> 黙って変わることを避けるためです。**この注意書きが PrunaVAED についての正本**で、他の箇所（§7・§8）はここを参照します。
>
> **LTX 2.5 を選んでいるときは PrunaVAED そのものが使えません**——重みファイルの有無にかかわらず 422 で断られます
> （枝刈りデコーダは LTX 2.3 用のもので、LTX 2.5 には対応物がありません。§7「制限事項」）。

**どれくらい速いか**: 720p（1280×768）・257 フレームの実測（交互対比較4組）で、**1本あたり平均 12.5 秒短縮**
（119.2 秒 → 106.7 秒＝約10.5%）。映像の復元処理そのものは 32.5 秒 → 20.2 秒（**1.61 倍**）で、短縮のほぼ全部が
この区間で説明できます。**VRAM の予約量は約 2.6GB 減ります**（13,911MB → 11,276MB）。動画が長いほど・解像度が
高いほど、この区間の比重が大きくなるぶん効きやすくなります。

**ファイルが無いときはどうなるか**: 枝刈りデコーダのファイルが見つからない場合（古い手順で作った環境や、利用者が
削除した場合）は、**エラーにはならず、通常のデコーダで最後まで生成されます**。ファイルを戻せば次のジョブから
また使われます（サーバーの再起動は要りません）。実際にどちらで生成されたかは、生成後に
`outputs/{ジョブID}/metadata.json` の `vae_mode_used`（`"off"` / `"on"` / `"on->off"`）で確認できます。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `vae_mode`（`"default"` または
`"prune_vaed"`、既定 `"default"`）を指定できます。**フィールドの値が `"prune_vaed"` と綴られているのは、この機能を
用意した当時の名前の誤記がそのまま外部仕様になったためです**（正しい名前は PrunaVAED）。値を変えると既存の利用者を
壊すので、表示名だけを直してあります。なお、VRAM 節約のための `vram_optimization.vae_tiling`（タイル分割）とは
**まったく別の設定**です。詳しい設計判断・重みの変換手順・実機ゲートの実測値は
[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §52 を参照してください。

---

## 6. テスト

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

> **2026-07-28更新**: `setup.bat`（`scripts/install_ltx.ps1` 経由）で作った環境には、pytest など `dev` extra の
> 依存が最初から入っています（インストーラが常に `uv sync --extra dev` で同期するようになったため）。
> ただし、素の `uv sync` を単体で実行すると `dev` extra（pytest / iniconfig / pluggy）はいまも
> 取り除かれます（`uv sync` は明示したextraだけを「入っているべきもの」とみなし、それ以外を削除する
> 仕様のため）。取り除かれてしまった場合は次のコマンドで入れ直してください。
>
> ```powershell
> uv sync --extra dev
> ```

pytest は **アプリ venv（`./.venv`, torch 無し）** で動きます。`tests/conftest.py` が `model.backend="mock"` を強制するため、
GPU/モデル無しで T2V/I2V のバリデーション（64倍数・8n+1・キーフレーム画像の上限5枚・`frame_idx` の丸め）と
モックランナーによる生成疎通、`GET /status` の `vram_optimization` 契約を検証します。

---

## 7. 制限事項（2026-08-26 現在）

**この節は「できないこと・気をつけること」だけを集めた場所です。** 長いので、先に中身を並べておきます。

| 節 | 内容 |
|----|------|
| [7.1](#limit-ltx25) | **LTX 2.5 を選んでいるときの制限**（使えない機能・無視される設定） |
| [7.2](#limit-onejob) | **同時に走る生成は1本だけ**（409 の理由・キャンセルの効き方） |
| [7.3](#limit-hardware) | 動作確認済みのハードウェア |
| [7.4](#limit-server) | サーバーとファイルの扱い（設定・ジョブ履歴・保存領域・ログ） |
| [7.5](#limit-generation) | 生成そのものの限界 |
| [7.6](#limit-legacy) | 旧「未実装（Phase 2以降）」一覧の現状 |

<a id="limit-ltx25"></a>
### 7.1 LTX 2.5 を選んでいるときの制限

**LTX 2.5 で使えるのは、基本生成（テキストから動画・画像から動画）と、クリップ連結（Chained）と、
V2V（素材（冒頭）に動画を使って続きを作る）と、A2V（音声から動画）と、
撮り直し（Retake）と、素材（末尾）（End source）と、
スタイル LoRA・IC-LoRA（参照動画による制御）と、高速化4つ（GGUF逆量子化の1カーネル化・先読み block swap・
モデル骨格の常駐・SageAttention）です。**
**このうち撮り直しと素材（末尾）には 2026-08-26 に対応しました。**
**このうちクリップ連結・V2V・A2V には 2026-08-23 に、スタイル LoRA と IC-LoRA には 2026-08-24 に対応しました**
——それまではクリップ連結そのものが丸ごと断られていましたが、いまは複数のクリップを1本に繋ぐ生成も、
動画の続きを作る生成も、音声に合わせた生成も、絵柄を寄せる LoRA も、参照動画で構図を制御する IC-LoRA も、
そのまま動きます（基本生成はそれ以前から使えます）。
**A2V は Single 画面の音声アコーディオン・複数クリップにまたがる長尺 A2V・バッチ A2V のどれも使えます。**
**IC-LoRA は 1 本の参照動画をクリップごとに配る長尺 IC-LoRA も使えます。**
**さらに 2026-08-24、高速化2つ（GGUF逆量子化の1カーネル化・先読み block swap）が使えるようになりました**（いずれも既定 on）。
**クリップ連結・V2V・A2V は AviUtl2 の操作パネルからの実機確認で動作を確認済みです**（クリップ連結は 2026-08-23、
V2V と A2V は 2026-08-24）。**スタイル LoRA と IC-LoRA は、機械での検証・実 GPU での検証に加えて、
2026-08-25 に操作パネルからの目視確認にも合格しました。**
**高速化2つには目視確認がありません**——出力が1バイトも変わらないことを機械で証明できているためです
（固定の検証用ジョブ17本すべてで動画のファイルの SHA-256 が従来と一致。
[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75.12）。
下記は要求すると **422 で断られます**
（AviUtl2 の画面側でも該当するタブとパネルが灰色になるので、通常の操作でここへ到達することはありません）。

| 断られる機能 | 対応するリクエスト項目 |
|--------------|------------------------|
| キャンバス拡張（Outpainting） | `outpaint`（単発生成のみ。連結生成の項目にはありません） |
| 非CFGネガティブプロンプト（NAG・VSF） | `nag_enabled` |
| 枝刈り版の映像VAEデコーダ（PrunaVAED） | `vae_mode` |
| 高品質パイプライン（`two_stage_hq`） | `pipeline` |

**断り方は「入口ごとまとめて」ではなく項目ごとです。** 連結生成（`POST /generate/chain`）を通る機能——素のクリップ連結・
V2V・A2V・LoRA・参照動画・撮り直し・素材（末尾）——は**いずれもそのまま走ります**。**2026-08-26 に撮り直しと
素材（末尾）が加わったことで、連結生成で断られるモードは1つも無くなりました**（上の表に残る4つは、いずれも
連結生成のモードではなく、エンジンが持っていない機能を指す項目です）。画面でも Chained タブと Edit タブは開き、
灰色のまま残るのは Edit タブの中の「キャンバス拡張（Outpainting）」サブタブだけです。

**LoRA の強さについて1つ知っておくと便利です。** LTX 2.5 では、LTX 2.3 と同じ LoRA ファイルでも**効き方がやや穏やか**
になる傾向が実測されています（ぼけ取りの IC-LoRA で、強度 1.0 のときの復元が 46%、1.3 まで上げると 136%）。
**既定の強度は LTX 2.3 と同じ 1.0 のまま**にしてあるので、効きが弱いと感じたときは
プロンプト内の LoRA タグで `1.3` のように強度を指定してください。**スタイル LoRA は、その LoRA の「トリガー語」
（学習時に使われた合言葉）をプロンプトに入れないと効きが穏やかになります。**
**参照動画で輪郭線制御（canny）を使うときは、ぼけていない鮮明な素材を渡してください**——ぼけた素材からは
輪郭がほとんど検出されず、制御が効かないように見えます。

**高速化のうち2つは、2026-08-24 に LTX 2.5 でも使えるようになりました。** 先読み block swap（`block_swap_prefetch`）と
GGUF逆量子化の1カーネル化（`fused_gguf_dequant_kernel`）です。それまでは「指定しても無視して生成を続ける」扱いでしたが、
いまは**既定 on で実際に効きます**——**2つ併用で生成時間が 61.8%短縮**（実測 102.86秒 → 39.28秒）で、しかも
**出来上がる動画は1バイトも変わりません**（§5「生成の高速化」と [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75）。
**画面の見た目は何も変わりません**——もともと灰色になっていた項目ではないためです。

**さらに 2026-08-25、モデル骨格の常駐（`keep_resident`）も LTX 2.5 で使えるようになりました。** こちらは上の2つと違って
**もともと 422 で断っていた機能**なので、**上の表から1行消えて、画面の灰色が1つ解けます**（高速化3つのうち、
この表を短くしたのはこれが初めてです——2026-08-24 のスタイル LoRA・参照動画の対応でも、この表から1行消えています）。**既定は off のまま**で、on にすると2本目以降の生成が **約25%短縮**されます（実測 27.6秒 → 20.5秒）。
こちらも**出来上がる動画は変わりません**。詳しくは §5「モデル骨格の常駐」と
[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §76 を参照してください。

**同じ 2026-08-25、SageAttention（設定画面の Attention の `sage`）も LTX 2.5 で使えるようになりました。** これも
もともと 422 で断っていた機能なので、**上の表からもう1行消えて、画面の灰色がもう1つ解けます**。**既定は `sdpa` のまま**で、
`sage` を選んだときだけ効きます（実測 1280×768・2クリップ×121フレームの連結生成で **1.1017倍**）。
**ただし、ここまでの3つと決定的に違う点が1つあります**——**`sage` を選ぶと、同じシードを指定しても出来上がる動画の細部が
変わります**（構図や被写体といった大枠は同じままで、質感やノイズの出方が変わります）。**以前つくった動画と
まったく同じものを作り直したい場合は `sdpa` を選んでください。** また **効き目は動画の大きさに強く左右され、
512×320 級ではほとんど効かないか、かえって遅くなることがあります**。詳しくは §5「生成の高速化（Acceleration）」の
[注意書き](#sage-seed-note)と [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §77 を参照してください。
**この機能は 2026-08-25 にオーナーによる目視確認にも合格しています**（絵が変わる種類の高速化なので、機械の検証だけでは
採否を決められず、`sdpa` と `sage` を対にした素材を実際に見比べてもらいました）。

**2026-08-26、撮り直し（Retake）と素材（末尾）（End source）が LTX 2.5 でも使えるようになりました。** 撮り直しは
Edit タブの「撮り直し」サブタブ（とタイムラインの右クリックからそこへ入る導線）、素材（末尾）は Chained タブの
「素材（末尾）」カードです。**どちらも上の表から1行ずつ消えて、画面の灰色が2つ解けます。**
**これで連結生成で断られるモードは1つも無くなりました。**
**ただし、この2つにはオーナーによる目視確認がまだ済んでいません**——出来上がる動画そのものが増える改修なので、
機械の検証だけでは採否を決められないためです（[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §2、
実測は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §78）。

**使うときに知っておいてほしいことが3つあります。**

1. **撮り直しは、窓（作り直す範囲）を 121 フレーム以上にしてください。** 73 フレームの窓では、前後の「のりしろ」
   （25＋24フレーム）を除くと**自由に作り直せるのは 24 フレームしか残りません**。動きはしますが、出来上がりが
   元とほとんど変わりません。**これは LTX 2.3 のときからそうで、LTX 2.5 で悪くなったわけではありません。**
2. **素材（末尾）は、クリップ1本で使うことを推奨します。** これは 2026-08-18 のオーナー裁定で、
   **クリップ2本以上（逆順Chained）は受理されますが推奨外**という位置づけです（クリップの境目・末尾に品質劣化が
   出ることがあり、仕様として許容しています）。**この裁定は LTX 2.5 でもそのまま引き継がれます。**
3. **その推奨外の使い方は、LTX 2.5 では音声の継ぎ目がさらに悪くなります。** まったく同じ依頼・同じ素材で比べると、
   LTX 2.3 は継ぎ目6か所すべてが「連続」判定だったのに対し、**LTX 2.5 は 4 か所が「不連続」判定**になりました
   （音量が段差状に落ちる・音楽の質感が切り替わるといった聞こえ方になります）。**映像側は LTX 2.5 でも 6 か所
   すべて連続で、問題ありません。** **推奨のクリップ1本には影響しません。** 品質を重視して複数クリップを終端付きで
   繋ぎたい場合は、クリップ1本の使い方を1本ずつ重ね、生成物を次の素材にして過去へ遡って生成する**手動リレー**
   （AviUtl2 のタイムラインで組み合わせる）が実用的な回避策です。

**LTX 2.5 でまだ使えないのは4つです**——**キャンバス拡張（Outpainting）**・**NAG／VSF（非CFGネガティブプロンプト）**・
**PrunaVAED（枝刈り版の映像VAEデコーダ）**・**非蒸留モデル用の高品質パイプライン（`two_stage_hq`）**。
このうち Outpainting は、**LTX 2.3 の作り方をそのまま移植することが成り立たない**ことが分かっており、別の設計で
作り直すことになります。PrunaVAED は **LTX 2.5 版の枝刈り済み重みが世の中に無い**ため先送りです
（[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-102）。

**指定しても断らず、黙って無視して生成を続ける項目もあります**——ネガティブプロンプト系
（`negative_prompt` / `guidance_scale` / `num_inference_steps` / `neg_method` / `vsf_scale`）の5つです。
蒸留版の LTX 2.5 に CFG（プロンプトへの従い具合の制御）そのものが無いためです。
**無視したことは `logs/server.log` に1行残ります。**

**LTX 2.3 を選んでいるあいだは、これらはすべて従来どおり使えます。** LTX 2.5 での対応は今後の課題です
（[`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-102。**クリップ連結・V2V・A2V は 2026-08-23 に、スタイル LoRA と IC-LoRA と高速化2つは 2026-08-24 に、モデル骨格の常駐と SageAttention は 2026-08-25 に、撮り直しと素材（末尾）は 2026-08-26 に対応済みです。高速化で残っているのは PrunaVAED だけで、これは LTX 2.5 版の枝刈り済み重みが存在しないため先送りです**）。**なお LTX 2.5 の重みは HuggingFace で公開済みですが、`setup.bat` の取得対象ではありません**——
[`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) と
[`Rootport/Nz-Gemma4-12B-LTX25`](https://huggingface.co/Rootport/Nz-Gemma4-12B-LTX25) からダウンロードし、
`models/LTX25/<カテゴリ>/` へ置くと認識されます（§1「models フォルダの構成」の LTX 2.5 の項）。

<a id="limit-onejob"></a>
### 7.2 同時に走る生成は1本だけ

- **同時実行は 1 ジョブのみ**。実行中に新しい `POST /generate`（および `POST /generate/chain`）を投げると
  **409 Conflict**（`JOB_BUSY`）が返ります。単一ユーザー向けのローカルツールという前提で、本格的なジョブキューは
  作らない方針です（仕様書 §13.5 でスコープ削除）。
- **実行中ジョブのキャンセルは best-effort**。PyTorch 推論を安全に中断できないため、`running` のジョブは推論完了後に
  `cancelled` へ遷移します。まだ実行に移っていない `queued` のジョブは、`DELETE /jobs/{id}` で**即座に** `cancelled` に
  なり単一ジョブガードが解放されます（2026-07-11 改修）。
- **MCP サーバー（§8）経由でも同じ制約です。** 複数エージェント・複数セッションからの並行操作は非対応です。
  MCP サーバー自体は実機検証済みで（2026-08-04・2026-08-06）、稼働中のバックエンドに対して22ツールを叩き、
  生成（T2V・I2V・V2V・A2V）・ジョブ操作・出力保存・`join`・パイプラインの読み込み/解放・Bearer 認証まで
  全項目が通ることを確認しています（Claude Code 本体の画面からの操作も含む。詳細は
  [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §39.6）。

<a id="limit-hardware"></a>
### 7.3 動作確認済みのハードウェア

**動作確認済みのハードウェアは 2 構成です。**

- **開発機**: RTX 4070 Ti SUPER 16GB（Ada Lovelace）／メインメモリ 64GB／ページファイル 48GB。日常的な開発と検証はすべてこの 1 台で行っています。
- **サブマシン**: RTX 3080 mobile 16GB（Ampere）／メインメモリ 32GB。**AviUtl2 を導入していない新規環境**で、2026-07-27 に一通りの導入から生成までを実測し、全項目に成功しました（`setup.bat` での導入 → `run.bat` での起動 → ブラウザで WebUI を開く → `smoke_test` サイズの生成 → **IC-LoRA の DWPose（pose-control）と canny をそれぞれ 768p・257 フレームで制御生成** → `NzVideomni.aux2` を AviUtl2 のプレビュー画面へドラッグ＆ドロップして導入 → 再起動後に操作パネルを表示 → タイムラインからの生成と、生成済み動画の右クリックからのタイムライン配置）。このとき AviUtl2 は **2026-07-25 更新の公開最新版**（開発機で使っている v2.0.54 より新しい版）を新規に導入しており、最新版との互換もあわせて確認できています。
- 残る GPU 世代（Turing・Hopper・Blackwell）は、torch 2.9.1+cu128 が同梱するカーネルの一覧と CUDA のバイナリ互換性から**理論上は動作するはずですが、実機では未検証**です。

**メインメモリ 32GB では、上記サブマシン 1 台での実測合格があります。** §1 に載せたコミットの実測値（アイドル比 +48GB、連続実行でジョブごとに +12〜15GB）は 64GB の開発機で採取したものですが、32GB の環境でも、最小構成の生成だけでなく **768p・257 フレームの IC-LoRA 制御生成まで実際に通りました**（2026-07-27）。ただしこれは**この 1 台での実測結果**であり、あらゆる 32GB 環境での動作を保証するものではありません。§1 の「メインメモリとページファイル」の案内は引き続き必ず守ってください。他の 32GB 環境で試された結果を共有していただけると助かります。**LTX 2.5 を使う場合は 64GB 以上を推奨**します（§1 のハードウェア要件）。

<a id="limit-server"></a>
### 7.4 サーバーとファイルの扱い

- **`low_vram_mode=true` がデフォルト**。16GB 環境前提。`low_vram_mode=false` は高VRAM/クラウド用の任意検証で、16GB成功は保証しません。
- ジョブ履歴は in-memory（再起動で消える）。`outputs/{job_id}/metadata.json` はディスクに残ります。
- **`uploads/`（アップロードした画像・動画・音声）に自動削除はありません**。ディスクに残るファイルの区別と整理のしかたは [`Docs/STORAGE_POLICY.md`](Docs/STORAGE_POLICY.md) にまとめてあります（`outputs/` は成果物、`uploads/` は入力素材のキャッシュ）。
- **ワーカーのログはエンジン系統ごとに別ファイル**です——LTX 2.3 は `logs/ltx_worker.log`、LTX 2.5 は `logs/ltx25_worker.log`。
  アプリ側のログは `logs/server.log` で、LTX 2.5 が「この設定は無視した」と書くのもこちらです。切り替えても両方の記録が
  残るように分けてあります。

<a id="limit-generation"></a>
### 7.5 生成そのものの限界

- transformer は Q4_K_M 量子化のため、フル bf16 公式とビット一致ではありません（聴感・視感は良好）。
- **キーフレーム画像で「最終フレームちょうど」を条件にすることはできません**（`frame_idx` は `num_frames-8` にクランプされます）。
  末尾を指定したいときは end source（素材（末尾））を使ってください。
- **end source の既知の限界**: 素材が本体のシーンと意味論的に遠いと、クロスフェードやカットで繋がります（モデルの限界で、
  素材の選び方で回避します）。また**推奨はクリップ1本**で、クリップ2本以上は受理されますが継ぎ目や末尾に品質劣化が
  出ることがあります（§7.6 と、正本の [`Docs/CHAIN_STAGE2_RESEARCH_NOTES.md`](Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11）。
- **「生成の高速化（Acceleration）」のうち2項目は、選ぶと絵が変わります**——`sage attention` と PrunaVAED です。
  詳しくは §5 の[`sage` の注意書き](#sage-seed-note)と[PrunaVAED の注意書き](#prunavaed-quality-note)を参照してください
  （`sage attention` は LTX 2.3・LTX 2.5 のどちらでも使えます。PrunaVAED は LTX 2.5 では 422 になります）。

<a id="limit-legacy"></a>
### 7.6 旧「未実装（Phase 2以降）」一覧の現状

Phase 1 当時の「未実装」一覧は、その後の拡張で大半が実装済みになりました。

- **実装済み**: 複数キーフレームI2V（キーフレーム画像・最大5枚・任意 `frame_idx`）／V2V（元動画からの継続生成）／A2V（音声から動画生成）／クリップ連結（`POST /generate/chain`）／IC-LoRA・スタイルLoRA（`<lora:...>` 記法含む）／1080p の直接生成（`FHD_1080p` プリセット。なお「1080p アップスケール機能」としての提供は仕様書 §13.5 でスコープ削除）／簡易認証（`--api-key` 指定時の Bearer 認証・任意）。
- **終了フレーム指定 → end source（素材（末尾））として実現済み**（2026-08-16 実装、2026-08-17 窓内モード、2026-08-18 逆順Chained）。**添付した画像・動画へ繋がる動画**を作る機能で、**クリップの本数だけで挙動が決まり、どちらの場合も出力の尺は伸びません**（クリップの合計そのもの）。**推奨はクリップ1本**（2026-08-18 のオーナー裁定）。契約の詳細は仕様書 [`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) §6.2 の `end_source` 補足とフロントエンド `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md` §5.2・§5.4、実装と実機実験は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §61・§63〜§65、設計面の考察は [`Docs/CHAIN_STAGE2_RESEARCH_NOTES.md`](Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11 が正本です。
- **複数クリップを品質重視で繋ぐ運用手順が2つあります**（どちらもコード変更不要・出荷済み機能の組み合わせ）。①**手動リレー**——end source をクリップ1本ずつ使い、生成物の冒頭を次の素材にして過去へ遡る。②**正順Chained＋補間仕上げ**（**未実機検証**）——本体は通常の正順Chainedで生成し、最終クリップだけを end source で仕上げる。②は生成回数が終端の1回で済むため①の上位互換になる可能性がありますが、採用前に実機での通し確認をおすすめします。**手順の正本は [`Docs/CHAIN_STAGE2_RESEARCH_NOTES.md`](Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11**（音声錨の幅と `context_frames` の関係もそちらにあります）、A/B目視の根拠は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §64.7・§65.8 です。
- **「生成の高速化（Acceleration）」の5項目は、2026-08-05 に全項目が実装済みになりました**。**LTX 2.3 を選んでいるあいだは、グレーアウトしていて選べない項目はもうありません**（§5「生成の高速化（Acceleration）」）。**LTX 2.5 を選んでいるときは §7.1 のとおり一部が使えず、画面でも灰色になります**——これは実装が無いからではなく、LTX 2.5 側にその機能がまだ無いためです。
  - **PrunaVAED**（枝刈りを施した VAE デコーダ＝映像の復元処理の軽量版）: 2026-08-05 に実装（`vae_mode`）。**既定は off で、今後も既定を変えることはありません**。効果と注意点は §5 の[注意書き](#prunavaed-quality-note)が正本です。**旧称「PruneVAED」は上流の正式名称の誤記**で、表示名は PrunaVAED へ訂正しました（API の値 `"prune_vaed"` は外部仕様なので据え置きです）。
  - **fused GGUF dequant + GEMM**（GGUF の逆量子化と行列積を1つの計算に融合する案）は**採否検討の結果 no-go** です。ただしその手前にある「逆量子化そのものの1カーネル化」は実装され、既定 on になりました（§5 の1項目目）。受理するだけだった旧フィールド `fused_gguf_dequant_gemm` は 2026-08-04 に撤去済みです。
- **引き続き未実装**: 本格的なジョブキュー（§7.2 のとおり「1ジョブ＋busy 409」を正式仕様としてスコープ削除）。なお AviUtl2 拡張フロントエンドは実装済みで、`AviUtl2-Plugin/` 以下に同居しています（バックエンドは引き続き汎用 REST API のままです）。

---

## 8. AIエージェント連携（MCPサーバー）

Claude Code などの **MCP（Model Context Protocol。AIエージェントが外部ツールを呼び出すための標準規格）クライアント**から、このバックエンドを直接操作できます。`mcp_server/` パッケージが、Web の操作パネルと同等の**22個のツール**（状態確認・アップロード・生成・ジョブ管理・出力取得・バッチ計画）を公開する MCP サーバーです。**AviUtl2 のタイムラインへの配置・編集はこのツール群の対象外**です（あくまでバックエンド単体の操作。タイムライン連携はフロントエンド側の拡張機能です）。

### 前提

`setup.bat` を一度実行していれば、セットアップの最後にリポジトリ直下へ **`.mcp.json` が自動生成**されます（このマシンの `.venv\Scripts\python.exe` への絶対パス入り）。手で書く必要はありません。実際に操作するには、生成ジョブを受け付ける **`run.bat` でバックエンドを起動しておく**必要があります（MCP サーバー自身はバックエンドを起動しません。`backend_status` ツールで疎通を確認できます）。

### Claude Code での使い方

1. このリポジトリのフォルダを Claude Code で開く。
2. 初回はワークスペースの信頼確認と、プロジェクトスコープの MCP サーバー登録に対する承認（**⏸ Pending approval**）が表示されるので、内容を確認して承認する。
3. `/mcp` コマンドで `nz-videomni` サーバーと 22 個のツールが一覧に出れば成功。
4. 承認をやり直したい場合（設定を変えた・一度拒否してしまった等）は `claude mcp reset-project-choices` を実行すると、次回起動時に承認確認からやり直せる。

### 他のMCPクライアント向け設定

Claude Code 以外の MCP クライアントでは、`.mcp.json` と同じ内容を各クライアントの設定に絶対パスで書きます（`<repo>` はこのリポジトリの絶対パスに置き換えてください）。

```json
{
  "mcpServers": {
    "nz-videomni": {
      "command": "<repo>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

### ツール一覧（22個）

| ツール | 説明 |
|---|---|
| `backend_status` | バックエンドの疎通状況を確認する |
| `get_config` | バックエンドの実効設定を取得する（`GET /config`） |
| `list_models` | 選択可能なモデル（transformer / text_encoder / video_vae / audio）を一覧する。**ベースモデル（LTX 2.3 / LTX 2.5）の一覧・導入状況・いま選ばれているもの**もここで分かる |
| `load_pipeline` | パイプライン（推論モデル一式）を読み込む（`POST /pipeline/load`）。`base_model` に `"LTX23"` / `"LTX25"` を渡すと**ベースモデルを切り替える** |
| `unload_pipeline` | パイプラインをメモリから解放する |
| `list_loras` | 選択可能な IC-LoRA アダプタを一覧する（`GET /loras`） |
| `upload_image` | ローカルの画像ファイルをアップロードする（I2V・キーフレーム用） |
| `upload_video` | ローカルの動画ファイルをアップロードする（V2V・参照動画用） |
| `upload_audio` | ローカルの音声ファイルをアップロードする（A2V用） |
| `submit_generate` | 単発の動画生成ジョブを登録する（T2V/I2V、`POST /generate`）。`attention_backend` ほか生成の高速化4項目を指定できる |
| `submit_chain` | クリップチェーン生成ジョブを登録する（V2V/A2V/連結、`POST /generate/chain`）。同じく `attention_backend` ほか生成の高速化4項目を指定できる |
| `job_status` | 1件のジョブの詳細を取得する（全文） |
| `list_jobs` | 全ジョブの一覧を要約付きで取得する |
| `wait_for_job` | ジョブが終端状態になるまで待つ（最大45秒でタイムアウト） |
| `cancel_job` | 未終了のジョブをキャンセルする |
| `delete_job` | 終了済みジョブの記録と出力フォルダを削除する |
| `purge_terminal_jobs` | 終了済みジョブをまとめて削除する（`dry_run` あり） |
| `join_job` | V2V継続ジョブの音声を元動画に繋ぎ直す |
| `get_job_video_path` | ジョブの出力動画（`output.mp4`）のローカル絶対パスを返す |
| `get_joined_video_path` | join済み動画（`joined.mp4`）のローカル絶対パスを返す |
| `save_job_video` | 出力動画をローカルの任意フォルダへコピーする |
| `plan_a2v_batch` | A2Vバッチの実行計画を立てる（音声フォルダを走査するだけ、HTTP不使用） |

### 典型ワークフロー

**T2V（テキストから動画）**:
1. `submit_generate` でプロンプト等を指定してジョブを登録する（`job_id` が返る）。
2. `wait_for_job` を繰り返し呼ぶ（1回で終わらなければ `timed_out: true` が返るので、終端状態になるまで呼び直す）。
3. `get_job_video_path` で出力動画のローカル絶対パスを取得する。

**I2V（画像から動画）**:
1. `upload_image` でローカルの画像をアップロードし `image_id` を得る。
2. `submit_generate` の `conditioning_images` にその `image_id` を指定してジョブを登録する。
3. 以降は T2V と同じ（`wait_for_job` → `get_job_video_path`）。

**LTX 2.5 へ切り替えて T2V**:
1. `list_models` で `base_models` を見て、切り替えたいベースモデルが `installed: true` であることを確認する（いま選ばれているものは `active_base_model`）。
2. `load_pipeline(base_model="LTX25")` を呼ぶ。ワーカーの載せ替えが起きるので数秒〜十数秒かかります。
3. `backend_status` の `status.state` が `ready`・`status.base_model` が `LTX25` になったことを確認する。
4. 以降は T2V と同じ（`submit_generate` → `wait_for_job` → `get_job_video_path`）。**切り替え直後の1本目だけは通常の約2倍**かかるので、`wait_for_job` を多めに呼び直してください。
5. LTX 2.3 へ戻すときは `load_pipeline(base_model="LTX23")`。
6. **クリップ連結（`submit_chain`）も LTX 2.5 で使えます**（2026-08-23 から）。**`source_video_id`（V2V）と
   `source_audio_id`（A2V。長尺 A2V を含む）も同日から使えます。** さらに **`loras`（スタイル LoRA）と
   `reference_video_id`（IC-LoRA。長尺 IC-LoRA を含む）、およびそれに従う 2 つの強度
   （`conditioning_attention_strength`・`reference_video_strength`）は 2026-08-24 から動作します。**
   **`keep_resident`（モデル骨格の常駐）と `attention_backend`（SageAttention）も 2026-08-25 から動作します**
   （既定はそれぞれ off ／ `"sdpa"` のままです。**`attention_backend="sage"` は選ぶと絵の細部が変わります**）。
   **LTX 2.5 での `sage` は、開発者による検証に加えて、2026-08-25 にオーナーによる目視確認にも
   合格しています**（§5「生成の高速化（Acceleration）」と同じ記述です）。
   **`end_source_video_id` / `end_source_image_id`（素材（末尾））は 2026-08-26 から LTX 2.5 で使えます**
   ——この日に撮り直し（Retake）と素材（末尾）が開通し、**`submit_chain` が投げられるモードは LTX 2.5 でも
   全部通るようになりました**（ただしオーナーの目視確認はまだ済んでいません。§7.1）。
   **なお撮り直しは、そもそも `submit_chain` に引数がありません**——**LTX 2.3 でも同じ**で、LTX 2.5 で
   失われた機能ではありません（引数を足すかどうかは
   [`Docs/PENDING_TASKS.md`](Docs/PENDING_TASKS.md) §3-115 で検討中です）。
   いま 422 になるのは、キャンバス拡張（`outpaint`。単発生成のみ）と、NAG／VSF（`nag_enabled`）・
   PrunaVAED（`vae_mode`）・非蒸留パイプライン（`pipeline`）です（§7.1）。

**A2Vバッチ（音声フォルダの一括生成）**:
1. `plan_a2v_batch` で音声フォルダを走査し、行ごとの計画（音声パス・提案フレーム数・同stem画像等）を得る。
2. 各行について **順番に**（同時1ジョブ制約のため直列で）: `upload_audio` → `submit_chain` → `wait_for_job` を繰り返し呼ぶ → `save_job_video` で任意の出力フォルダへ保存する。

### 注意事項

- **同時実行は1ジョブまで**: ジョブ実行中に新しい `submit_generate` / `submit_chain` を呼ぶと **409 JOB_BUSY** になります。
- **`wait_for_job` は最大45秒でタイムアウト**します。エラーにはならず `timed_out: true` とその時点の進捗を返すので、終端状態になるまで繰り返し呼んでください。
- **ベースモデルの切り替えは `load_pipeline` の `base_model` 引数で行います**（ツールは増えていません。22個のままです）。LTX 2.5 を選んでいる間に使えない機能は §7.1「[LTX 2.5 を選んでいるときの制限](#limit-ltx25)」を参照してください（ツール側からは `list_models` の `base_models[].unsupported_features` でも確認できます）。切り替えは**ワーカーの載せ替え**を伴い、ジョブ実行中は **409 JOB_BUSY** で断られます。
- **`config.yaml` を変更した場合は MCPサーバーの再起動が必要**です（設定は起動時に1回だけ読み込みます）。MCPサーバーは Claude Code のプロセス内で管理されるサブプロセスなので、**Claude Code 自体を再起動**すれば再読み込みされます。
- 生成された動画は base64 等で埋め込まれず、**常にローカルの絶対パス**で返されます（`save_job_video` で任意のフォルダへコピーも可能）。パスは MCP サーバーを動かしているマシン上のものです。
- **`attention_backend="sage"` の注意**: 生成結果が同じシードでも変わります。詳しくは §5「生成の高速化（Acceleration）」の[注意書き](#sage-seed-note)を参照してください（**LTX 2.5 でも 2026-08-25 から選べるようになった**点もそこに書いてあります）。利用可否は `backend_status` の `acceleration.sage_available` で確認でき、`sageattention` が入っていない環境ならエラーにならず `"sdpa"` へ降格して完走します（**この降格の規律は LTX 2.3・LTX 2.5 のどちらでも逐語で同じ**です）。実際に使われた方式はメタデータの `attention_used` に記録されます。
- **「生成の高速化（Acceleration）」の5項目は、すべて MCP のツールに公開しています**（`attention_backend` / `block_swap_prefetch` / `keep_resident` / `fused_gguf_dequant_kernel` / `vae_mode`。最後に残っていた `vae_mode` は 2026-08-05 に公開しました。ツールの本数は22個のまま変わっていません）。**LTX 2.5 を選んでいるときにこの5項目のうち 422 になるのは、`vae_mode`（PrunaVAED）の1つだけです**——`keep_resident` と `attention_backend` は 2026-08-25 から、残る2つは 2026-08-24 から LTX 2.5 でも動作します。
- **`vae_mode="prune_vaed"`（PrunaVAED）の注意**: こちらも生成結果が変わります。詳しくは §5「枝刈り版の映像VAEデコーダ」の[注意書き](#prunavaed-quality-note)を参照してください（**LTX 2.5 では 422 になる**点もそこに書いてあります）。`sage` と違って LTX 2.3 での降格の可否は環境ではなく**枝刈りデコーダのファイルの有無**で決まり、無ければエラーにならず通常のデコーダで完走します。実際にどちらで生成されたかはメタデータの `vae_mode_used`（`"off"` / `"on"` / `"on->off"`）に記録されます。**指定しなければ従来とまったく同じ**です（既定は `"default"` で、省略したときはこの項目自体がバックエンドへ送られません）。

---

## ライセンス
Apache-2.0 を推奨（上流コード利用時の整合性を優先）。`engine/` は LTX-2 / LTX-Desktop 由来の派生コードを含むため、
再配布時は上流の帰属表示を保持してください（[`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)）。モデルウェイトは
LTX-2 Community License に従ってください。
