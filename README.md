![Nz-Videomni-logo](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/Videomni_logo.jpg)

---  
# Nz-Videomni   
AviUtl2から動画生成AIを動かせるバックエンドPythonサーバーです。フロントエンドとしてAviUtl2 用`.aux2` プラグインを同梱。さらに、検証用の簡易なGradio UIを同梱しています。API は汎用設計なので、DaVinci Resolve など他の動画編集ソフトからも（フロントエンド用プラグインを開発すれば）使用できる見込みです。  

| 必須スペック | 要件 |  
|------|------|  
| GPU | **VRAM 16GB 以上** NVIDIA製GPU。動作検証には RTX 4070 ti SUPER および RTX 3080 mobile を使用 |  
| メインメモリ | 64GB （※ 32GB でもWindowsの`ページファイル`を有効にすれば動きます ） |  
| ストレージ | 50GB （ページファイル利用の場合は＋60GB）|  
| ストレージ（モデル追加時） | LTX 2.5の追加時：＋30GB |  
| その他 | [AviUtl2](https://spring-fragrance.mints.ne.jp/aviutl/)をインストール済みであること |

# インストール方法
### 概要    
1. [Git](https://git-scm.com/install/windows)をインストールする  
2. このアプリのデータを`git clone`でダウンロードする（後述）  
3. 同梱のsetup.batを実行してインストール  
4. 同梱のrun.batを実行してアプリを起動  
5. AviUtl2を起動し、` \AviUtl2-Plugin\NzVideomni.aux2`をプレビュー画面にドラッグ＆ドロップ   
※ 最初のAIとして`LTX 2.3`がインストールされます。  
  
> - 動画生成AIを追加したいとき：`install-〇〇.bat`を実行してください。  
> ※例）`install-LTX25.bat`を実行すると`LTX 2.5`がインストールされます。  
> 
> - アップデートするとき：`git pull`を実行し、その後、`Setup.bat`を再度実行する（後述）  
 
  
### 1) gitをインストールする  
[![git-installer](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git_installer.jpg)](https://git-scm.com/install/windows)   
  
gitのインストールページ（ https://git-scm.com/install/windows ） から、あなたのマシンに合わせたインストーラーをダウンロードして実行し、gitをインストールしてください。  
> **Gitとは？**  
> プログラムの更新履歴を管理するアプリです。エンジニアの間では広く使われています。WEB上で公開されているアプリのダウンロードやアップデートにも利用できます。
  
  
### 2) このアプリのデータを`git clone`でダウンロードする  
**2-1. Windowsで、このアプリをインストールしたいフォルダを開く。**  
![任意のフォルダを開いた図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git-clone-001.jpg)  
※重要：OneDrive配下のフォルダは避けてください。数十GB～数百GB級のデータがダウンロードされるため。  
※パソコン初心者には、新しいフォルダを作ることを推奨します。  
  
**2-2. アドレスの入力欄をクリックし、`CMD`と打ちこみ、エンターキーを押す。**  
![アドレス入力欄の図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git-clone-002.jpg)  
![アドレス入力欄に「CMD」と打ち込んだ図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git-clone-003.jpg)  
　※この状態でエンターキーを押す。  
  
**2-3. 黒い画面（コマンドプロンプト）が開くので、以下のコマンドを打ちこむ。**  
```  
git clone https://github.com/Rootport-AI/Nz-Videomni.git  
```  
![コマンドプロンプトの図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git-clone-005.jpg)  
　※この状態でエンターキーを押す。  
  
**2-4. インストールに必要なファイル群がダウンロードされる。**  
![コマンドプロンプトにダウンロード経過が表示された図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/git-clone-006.jpg)  
※`Updating files: 100% (*** / ***), done.`と表示されたら、この黒い画面は閉じていい。  
  
### 3) 同梱のsetup.batを実行してインストール  
**3-1. ダウンロードされたファイル群の中から`setup.bat`を探し、ダブルクリックする。**  
![Setup.batを探そうの図](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/setup.bat.jpg)  

**3-2. 40～60分間ほど待つ。黒い画面は閉じない。**  
![インストール完了画面](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/setup.bat_002.jpg)
`続行するには何かキーを押してください．．．`という文章が出たらインストール完了。黒い画面を閉じていい。      

### 4) 同梱のrun.batを実行してアプリを起動  
![run.batの位置](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/run.bat.jpg)
![run.batのコンソール](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/run.bat_002.jpg)
  ※アプリの利用中はこの黒い画面を閉じてはならない。（※逆に、アプリ終了時にはこの画面を閉じるだけでいい。）  
  ※`Uvicorn running on http://127.0.0.1:18620 (Press CTRL+C to quit)`と表示されたら起動完了。  
  
  
### 5) `NzVideomni.aux2`をAviUtl2のプレビュー画面にドラッグ＆ドロップ  
ファイルの場所："\Nz-Videomni\AviUtl2-Plugin\NzVideomni.aux2"  
![aux2ファイルの位置](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_001.jpg)  
![aux2のドラッグ＆ドロップ時](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_002.jpg)
※`このプラグイン・スクリプトを信用して使用する`をクリックする。  
  
**5-2. メニューの`表示` → `Nz-Videomni`で、Videomniの操作パネルが表示されたらインストール成功。**  
![表示メニュー](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_003.jpg)　　
![Videomniの操作パネル](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_004.jpg) 
※操作パネルが表示されたらインストール成功です。  

  
---  
## 動作テストa： 簡単な動画を生成する。      
### 6. 操作パネルはウィンドウ分離すると使いやすい。  
![ウィンドウ分離](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_005.jpg)   
※操作パネルの`Nz-Videomni`という名前欄を右クリック → ウィンドウ配置 → ウィンドウ分離  
  
### 7. 動画を生成する。  
ベースモデルが「LTX 2.3」になっていることを確認する。プロンプト入力欄に適当な文章を入力し、「生成 / Generate」ボタンを押す。（※ここでは`a girl is walking in the forest`というプロンプトで動画生成した）  
  
**7-1. 基本的な操作方法**  
![画面の基本的な説明](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_006.jpg)  
  
**7-2. Generateボタンを押すと生成が始まる。**  
![生成中の画面](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_007.jpg)  
  
**7-3. 生成完了後、「🎞」ボタンを押すと、AviUtl2のタイムラインに動画が挿入される。**  
![挿入ボタン](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_008.jpg)  
![胴が挿入済みのタイムライン](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/aux2_009.jpg)  


---  
# 動作テストb： 動画の続きを生成する  
### 8. タイムライン上の動画オブジェクトを右クリック  
![右クリックメニュー](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/v2v_001.jpg)
※右クリック → プラグイン → `🎬Video: この動画の続きを生成(v2v)`を選択する  

### 9. タイムラインに仮オブジェクトが挿入される。  
![仮オブジェクト](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/v2v_002-1.jpg)  
※仮オブジェクト＝「生成中の動画が将来この場所に挿入されること」を示すだけのテキストオブジェクト  
    
### 10. 操作パネルに、動画ファイルが送られる。  
![動画ファイル入力後の操作パネル](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/v2v_002-2.jpg)
※右クリック → プラグイン → `🎬Video: この動画の続きを生成(v2v)`を選択すると…  
- 画面上部のタブが`Chained`に切り替わる。  
- 動画ファイルが入力されたことを示すメッセージが出る。
- `START SOURCE`のカードに、動画ファイルがセットされる。
- `CLIP`カードのスライダーで、追加したい動画の長さ（フレーム数）を指定できる。
- `Generate`ボタン（`生成`ボタン）を押すと、生成が始まる。  

### 11. 生成された動画ファイルを、`🎞`ボタンでタイムラインに送る。  
![挿入ボタンの位置](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/v2v_003.jpg)  
![挿入後のタイムライン](https://github.com/Rootport-AI/Nz-Videomni/blob/main/images/v2v_004.jpg)  
※生成後に`🎞`ボタンを押すと、タイムライン上の仮オブジェクトがその動画ファイルに置き換わる。  

---
# Nz-Videomniの詳細　　

> **リポジトリの構成（モノレポ）**
>
> | 場所 | 中身 |
> |------|------|
> | リポジトリ直下 | バックエンド（`main.py` / `api/` / `services/` / `engine/` / `engine25/` / `gradio_ui/` / `mcp_server/`） |
> | `AviUtl2-Plugin/NzVideomni.aux2` | ビルド済みの AviUtl2 プラグイン（配布物。AviUtl2 へドラッグ＆ドロップしてインストールします） |
> | `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/` | そのプラグインのソース（C++ の `native/` ＋ React/TypeScript の `webui/`） |
> | `Docs/` | プロジェクト全体の文書 |
> | `images/` | 本 README へ貼るための画像素材 |

**Nz-Videomni は製品の名前、LTX 2.3 はベースモデルの名前**です。API 契約・スキーマの詳細仕様は [`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) を参照してください。

実エンジンは **first-party の `engine/` パッケージ**（GGUF 量子化トランスフォーマー + block-swap + GGUF Gemma 逐次オフロード + DiT CPU 構築 + VAE タイリング）で、**VRAM 16GB の実機で 720p 級（1280×768）の生成に対応**します。GPU / モデルウェイトが無い環境では自動的に **モック backend**（合成クリップ）へフォールバックし、API・ジョブ管理・Gradio・テストまで完全に疎通します。公式 `ltx_pipelines` の safetensors ローダは 16GB / Windows で native crash するため不採用で、GGUF + component-file 経路にしています（設計判断と provenance の一次情報は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)）。

---

## 0. 環境分離ポリシー（最重要）

**このプロジェクトは PC のシステム Python 環境を一切汚しません。** Python 本体を含め、必要なものはすべて
プロジェクトディレクトリ配下（`.venv/`, `.venv-engine/`, `.venv-engine-ltx25/`, `.python/`, `.uv_cache/`, `tools/`）に閉じ込めます（仕様書 2.5）。

- グローバル/システムの `pip install` は **禁止**。必ず `uv` + プロジェクトローカル venv。
- 環境変数（`UV_PYTHON_INSTALL_DIR` など）は **そのプロセス内のみ**。永続化しない。
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

アプリ（`./.venv`）は torch も LTX も import しません。実生成は、選ばれているモデルに対応する python で worker を **subprocess** として起動し、JSON-lines プロトコルで駆動します（§3「アーキテクチャ」）。`.venv-engine` の依存スナップショットは [`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt) に凍結してあります。

---

## 1. セットアップの詳細　　

### 必要なもの

- Windows 10/11 x64
- **git** — [公式サイト](https://git-scm.com/download/win)からインストーラを入手してください。画面上のボタンを押していくだけで導入が終わり、コマンドの入力は必要ありません。
- 実モデルで生成する場合のみ: NVIDIA CUDA GPU と十分なメインメモリ（下記「ハードウェア要件」）

**あらかじめ用意しておくものは git だけです。** パッケージ管理ツールの [`uv`](https://docs.astral.sh/uv/) と、動画の変換に使う `ffmpeg` / `ffprobe` は、`setup.bat` がこのプロジェクトの中（`tools/` フォルダ）へ自動的に取り込みます。Windows 側の設定（PATH など）は一切書き換えないので、後片付けはこのフォルダを削除するだけで済みます。

### かんたんインストール（`setup.bat` → `run.bat`）

コマンドを打つ必要はありません。次の順に進めてください。

1. **`setup.bat` をダブルクリックする。** 黒い画面が開き、道具の取り込み（`uv` / `ffmpeg`）→ 専用の Python 環境の作成 → モデルのダウンロード（約 33GB）が順に進みます。所要時間の目安は、光回線（下り 90〜100Mbps）でおよそ 50 分、30Mbps 程度の回線ではおよそ 2 時間半です。**途中でこの画面を閉じても構いません。** もう一度実行すれば続きから再開し、取得済みのファイルは取り直しません。通信が一時的に途切れたときは、その場で何度かやり直します。ただしパソコンがスリープすると通信が止まるので、長時間そのままにする場合は、電源の設定でスリープを「なし」にしておいてください。
2. **`run.bat` をダブルクリックする。** サーバーが起動し、黒い画面に `http://127.0.0.1:18620/ui` のようなアドレスが囲み枠つきで表示されます。
3. **表示されたアドレスをブラウザで開く。** これで Web の操作画面（Gradio UI）が使えます。
4. **AviUtl2 から使う場合**は、このリポジトリの **`AviUtl2-Plugin\NzVideomni.aux2`** を、**AviUtl2 のプレビュー画面へドラッグ＆ドロップ**してください。AviUtl2 公式のプラグイン導入方法です（本体添付の `aviutl2.txt` に記載があります）。
5. **AviUtl2 を再起動する。** 上部メニューから Nz-Videomni を開けるようになります。

#### サーバーの止め方

`run.bat` で開いた黒い画面を、**右上の × ボタンで閉じてください**。これでサーバーが止まります。

> **この案内に `Ctrl+C` を書き足さないでください（意図的な省略です）。** `Ctrl+C` で止めると、`cmd` が `Terminate batch job (Y/N)?` という英語のプロンプトを返すことがあり、本プロジェクトの想定利用者（PowerShell を自分で開けないリテラシー）はここで手が止まります。× で閉じる 1 通りだけを案内する、というのが決定事項です。

#### AviUtl2 から使うときの注意（重要）

**先に `run.bat` を起動して、その黒い画面を開いたままにしておいてください。** AviUtl2 のプラグインは、バックエンドのサーバーを自分で起動しません（プラグイン側にサーバーを立ち上げる仕組みは入っていません）。サーバーが動いていないと、プラグインの画面には**「サーバー未起動」**というバッジが出るだけで、生成はできません。黒い画面を閉じるとサーバーも止まるので、AviUtl2 から使っている間は閉じないでください。

#### 更新のしかた

1. VSCode の画面から `git pull`（同期）を実行して、最新のコードを取り込む。
2. **そのあと、`setup.bat` をもう一度実行する。**

**2 を省かないでください。** 依存パッケージの内容が変わっていた場合、`run.bat` は起動こそするものの中身が古いままで正しく動かない、という分かりにくい状態になります。`setup.bat` は 2 回目以降、すでに揃っているものを飛ばすので、変更が無ければ短時間で終わります。

自動更新の仕組みはあえて用意していません（更新確認の通信がウイルス対策ソフトに誤検知されるリスクを避けるためです）。

#### 設定ファイル `config.yaml` の扱い

`config.yaml` は **git の管理対象外**です。配布されるのはひな型の `config.yaml.example` で、`setup.bat`（および `run.bat`）が、`config.yaml` がまだ無いときにひな型から複製します。利用者が自分のマシンに合わせて書き換えた設定が、`git pull` のたびに衝突しないようにするための作りです。設定を変えたい方は `config.yaml` のほうを編集してください（ひな型を編集しても、動作中の設定には反映されません）。

すでに `config.yaml` を編集して使っている環境を移行するときの退避手順は、下の「models フォルダの構成」にあります。

### ハードウェア要件の詳細  

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA 製・**VRAM 16GB 以上**。対応世代は Turing（GeForce RTX 20系）／Ampere（同 30系）／Ada Lovelace（同 40系）／Hopper／Blackwell（同 50系） |
| GPU ドライバ | **R570 以上を推奨**（Blackwell では必須）。CUDA 12.x のマイナーバージョン互換だけを見れば Windows では 525 以上が下限ですが、本プロジェクトは cu128 ビルドの torch を使うため R570 以上を勧めます |
| メインメモリ | **32GB 以上、かつページファイルを有効にしておくこと**（下の「メインメモリとページファイル」が最重要）。**モデル骨格の常駐（`keep_resident`）を使う場合は 64GB 以上を推奨**します（LTX 2.3 では約 20GB、LTX 2.5 では約 7.7GiB を常時占有するため。既定は off なので、使わないかぎりこの要件は増えません。§5「モデル骨格の常駐（`keep_resident`）」）。**LTX 2.5 を使う場合も 64GB 以上を推奨**します——2.5 のワーカーは仕上げ工程のために重みをメインメモリへ持ち続ける設計（`cache_weights`、既定 on）で、**生成中のメインメモリの山が実測で約 26GiB** あるためです。この既定を off にすれば常駐は減りますが、そのぶん仕上げ工程の作り直しに時間がかかります。**LTX 2.5 でモデル骨格の常駐も同時に on にする場合は、安全側の見積りとして合計 約 34GiB を見ておいてください**（内訳は連結生成の OFF ピーク 26.33GiB ＋ 常駐増分 7.68GiB ≒ 34.0GiB。[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §69.19・§76.2(3)・**§76.8(3)**）。**さらに埋め込み処理器の常駐（`keep_resident_embeddings`。LTX 2.5 専用の項目です）も on にする場合は、合計 約 39GiB を見ておいてください**（同じ式に常駐増分 4.66GiB を足したもので、26.33 ＋ 7.68 ＋ 4.66 ≒ 38.7GiB。出所は同 **§92**。§5「[埋め込み処理器の常駐](#keep-resident-embeddings)」） |
| ストレージ | **このフォルダを置くドライブに約 50GB**（`setup.bat`のみ＝LTX 2.3を導入した場合の必要空き容量。内訳は下の「必要な空き容量の内訳」参照）。**これとは別に**、ページファイルを置いたドライブに 60GB 以上の空き |
| ストレージ（ベースモデルを追加する場合） | **LTX 2.5：追加で約 30GB**（`install-LTX25.bat`。下の「LTX 2.5 を追加する」参照） |
| attention（注意機構の計算方法） | 既定は全世代で **SDPA**（PyTorch 標準の実装）。**生成のたびに SageAttention へ切り替えられます**（§5「生成の高速化（Acceleration）」）。xformers・flash-attn は導入も使用もしません |

#### 対応する GPU 世代

エンジン venv の torch は **2.9.1+cu128** に固定してあり、この配布ビルドが同梱しているコンパイル済みカーネルの一覧（`torch.cuda.get_arch_list()` の実測値）は次のとおりです。

```text
['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

`sm_XX` は GPU の compute capability（世代を表す番号）です。Ampere（RTX 30系＝`sm_86`）・Hopper（`sm_90`）・Blackwell（RTX 50系＝`sm_120`、データセンター向け B200＝`sm_100`）は、この一覧にそのまま含まれています。**`sm_89`（Ada Lovelace＝RTX 40系）が無いのは、`sm_86` 向けにコンパイルされたカーネル（cubin）がそのまま Ada の GPU 上で動くためです**——NVIDIA の [Ada Compatibility Guide](https://docs.nvidia.com/cuda/archive/12.8.0/ada-compatibility-guide/index.html) が名指しの例として明記しています（逆方向はできません）。実機で確認できているのは Ada と Ampere の2世代で、残りは理論上の互換です（§7.3）。

attention は全世代で PyTorch の SDPA を既定にしており、xformers や flash-attn はインストールもしなければコードからも呼びません（16GB 環境で速度を決めているのは attention ではなく重みの転送であるため。仕様書 §5.4）。SageAttention に必要なパッケージは `setup.bat` が標準で入れるため、追加の作業は要りません（`sageattention` 2.2.0 と `triton-windows` 3.5.1.post24。後者は必要なコンパイラを同梱しているため、**Visual Studio の導入は不要**です）。

#### 必要な空き容量の内訳

上の「ストレージ」欄の **約 50GB**（LTX 2.3 のみ）が、本プロジェクトで統一している必要容量の数字です。合計の正本はこの欄だけで、他の場所には数値を書き写しません。`setup.bat`（`scripts/setup.ps1`）が起動時に出す空き容量の案内・失敗時の案内は、この数字に余裕を足ししきい値として使っています。

中身は `models/`（モデル一式。フォルダの意味は下の「models フォルダの構成」を参照）＋ Python 環境（`.venv/` ＋ `.venv-engine/` ＋ `.venv-engine-ltx25/` ＋ 共有キャッシュ `.uv_cache/` ＋ `.python/`。3つの venv は `.uv_cache/` へのハードリンクで実体を共有するため、単純な足し算にはならず、キャッシュの再構築回数によっても変わります）＋ `tools/`（`uv` ＋ `ffmpeg`、約 0.4 GiB）です。**3つ目の仮想環境（`.venv-engine-ltx25`、LTX 2.5 用）は `install-LTX25.bat` ではなく `setup.bat` の時点で、選んだベースモデルに関わらず作られます**——この約 50GB にすでに含まれています。

> **この約 50GB に LTX 2.5 の重みは含まれていません。** LTX 2.5 は `setup.bat` ではなく `install-LTX25.bat` で別に導入するもので、上表の「ストレージ（ベースモデルを追加する場合）」＝追加で約 30GB を見てください（詳しくは下の「LTX 2.5 を追加する」）。

> **ページファイル用の 60GB は、上の「ストレージ」欄の空きの代わりにはなりません。** ページファイルは別のドライブに置いていても構わない性質のもので（Windows の既定では C ドライブ）、用途もまったく別です。**両方**必要だと考えてください。たとえばこのフォルダを D ドライブへ置き、ページファイルが C ドライブにあるなら、D に約 50GB・C に 60GB の空きが要ります。

#### メインメモリとページファイル（最重要）

**実際に生死を分けるのはここです。** LTX 2.3 の推論では、Windows の「コミット」（プロセスが OS に確保を約束させた仮想メモリの総量。物理 RAM とページファイルの合計で裏打ちされる）が、物理メモリをはるかに超える量まで伸びます。開発機（メインメモリ 63.8GB ＋ ページファイル 48GB）で採取した実測値は次のとおりです。

| 時点 | コミット総量 | アイドル比 |
|------|--------------|-----------|
| アイドル（生成前） | 22.3GB | — |
| 生成ジョブ1のピーク | **70.1GB** | **+48GB** |
| 続けてジョブ2のピーク | 90.2GB | +68GB |
| 続けてジョブ3のピーク | 101.6GB | +79GB |

1回の生成でアイドル比 **+48GB**、さらに**連続して生成するとジョブごとに +12〜15GB ずつせり上がっていきます**（アプリを再起動すれば元に戻ります）。つまり **メインメモリ 32GB では物理メモリだけでは全く足りず、ページファイル（物理メモリが足りないときに中身をディスクへ退避させるための、ドライブ上の領域）が必須**です。設定は次のようにしてください。

- Windows のページファイル設定は **「システム管理サイズ」のままにする**（Windows の既定値。手動で固定サイズにしない）。
- ページファイルを置いているドライブに **60GB 以上の空き容量**を確保しておく。OS は足りなくなるとページファイルを
  自動的に広げますが、広げる先の空きが無いと失敗します。
- **ページファイルを無効化しない・小さな固定サイズにしない**。「SSD の寿命が心配」「メインメモリが潤沢だから要らない」
  といった理由で無効化・固定小サイズにしている環境では、**生成の途中でワーカープロセスがエラーメッセージを一切出さずに
  落ちます**。Python の例外もログも残らない（OS レベルのアクセス違反で即死するため）ので、この状態は原因究明が
  非常に困難です。「生成が途中で止まるのにログには何も出ていない」場合は、まずページファイルの設定を疑ってください。
  実測記録は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §8.8 にあります。

#### NVIDIA の「システム メモリ フォールバック」設定

**NVIDIA コントロールパネルの「CUDA - システム メモリ フォールバック ポリシー」は、既定（ドライバーの既定＝有効）のままにしておいてください。** これは VRAM（GPU が持つメモリ）が足りなくなったとき、あふれた分をメインメモリへ自動で逃がしてくれる仕組みです。これを「システム メモリ フォールバックを優先しない（Prefer No Sysmem Fallback）」へ変更しているとその逃げ道が無くなり、本来は**「遅くはなるけれど最後まで完走する」**種類の重い指定（長い尺に参照動画を組み合わせた生成など）が、`CUDA out of memory` というエラーでその場で失敗することがあります。アプリが生成前に出す「遅くなるおそれがあります」という警告も、この設定が既定のままであることを前提にした案内です。**生成が `CUDA out of memory` で止まるときは、まずこの設定が変更されていないか確認してください。**

### アプリ venv（`./.venv`, torch 無し）

> ここから先は**中で何が起きているかの説明**です。`setup.bat` が `scripts/install_ltx.ps1` を通じてほぼ同じことを自動で行うので、通常の利用では手で打つ必要はありません。

```powershell
# Python 本体もプロジェクト内に固定する（システムを汚さない）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"

uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev   # テストも動かす場合。実行だけなら素の uv sync でよい
```

これで `.venv/`（アプリ仮想環境）と `.python/`（uv管理のPython 3.12）がプロジェクト内に作成されます。`pyproject.toml` / [`requirements.txt`](requirements.txt) には FastAPI 側の依存のみ定義しています（`torch` は含みません）。これだけで **モック backend** で API/UI/テストが動きます（GPU 不要）。

### エンジン venv（torch+cu128）と実モデル

実生成には、アプリ venv とは別に**エンジン系統ごとの venv**と GGUF/component モデル群が必要です。**2つに分かれているのは、LTX 2.3 と LTX 2.5 が要求するパッケージのバージョンが同居できないためです**（`transformers` 4.57 と 5.x）。どちらも `setup.bat` が自動で作るので、通常は意識する必要はありません。

| venv | 対象 | 依存の宣言（再構築に使うファイル） |
|------|------|------------------------------------|
| `./.venv-engine` | LTX 2.3（`engine/worker.py`） | [`engine/engine-venv-pyproject.toml`](engine/engine-venv-pyproject.toml)（`[tool.uv.sources]` に torch cu128 index と `ltx-core`/`ltx-pipelines`/`diffusers` の git rev を記載）＋ [`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt)（`name==version` の完全スナップショット） |
| `./.venv-engine-ltx25` | LTX 2.5（`engine25/worker.py`） | [`engine25/engine25-venv-pyproject.toml`](engine25/engine25-venv-pyproject.toml) ＋ [`engine25/venv-engine-ltx25.freeze.txt`](engine25/venv-engine-ltx25.freeze.txt)（同じ作法。公式 LTX-2 v1.2.0 ＋ `transformers` 5.x） |

`.venv-engine` の provenance と再現手順の詳細は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md) を参照してください。

> **`setup.bat` を再実行したときのふるまい**: エンジン venv は「すでに存在するから飛ばす」のではなく、**ピン留めされた依存の内容が前回と変わっていないときだけ飛ばします**。freeze ファイルの中身と、`install_ltx.ps1` が持つ git リビジョンの指定をまとめてハッシュにし、venv 内の `.nz-engine-state` に記録した前回の値と突き合わせる方式です（**2つのエンジン venv はそれぞれ独立に判定されます**）。`git pull` で依存が変わっていれば自動で貼り直され、記録が無い場合（前回の導入が途中で中断した場合など）も貼り直しになります。「`git pull` のあとに `setup.bat` を再実行する」という更新手順は、この仕組みで成り立っています。

**必要なモデル**は `setup.bat` / `install_ltx.ps1` が自動でダウンロードします。**どのファイルがどこに要るかを宣言しているのは `config.yaml` ではなく「ベースモデル記述子」**＝`scripts/manifests/*.json` で、そこに書かれた相対パスは `config.yaml` の `model.models_dir`（既定 `./models`）を起点に解決されます。

| 要素 | 既定パス | 概算 | 役割 |
|------|----------|------|------|
| GGUF transformer (Q4_K_M) | `models/LTX23/Weights/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | ~17GB | 本番トランスフォーマー |
| GGUF Gemma (Q4_K_M) | `models/LTX23/TextEncoder/gemma-3-12b-it-Q4_K_M.gguf` | ~7.3GB | text encoder（GPU 推論・逐次オフロード） |
| component VAE / audio / text-projection | `models/LTX23/VAE/*.safetensors`（映像・音声 VAE と `prunavaed/` の枝刈りデコーダ）＋ `models/LTX23/TextEncoder/ltx-2.3_text_projection_bf16.safetensors` | ~3.9GB | 46GB モノリスを置換する小単体ファイル |
| spatial upsampler | `models/LTX23/Upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~0.95GB | 2段生成の x2 アップサンプラ |
| Gemma tokenizer dir (`gemma_root`) | `models/LTX23/TextEncoder/tokenizer/` | ~40MB | tokenizer/preprocessor のみ（`tokenizer.model` 等）。**重みは含まない** |
| IC-LoRA 2点 | `models/LTX23/IC-LoRA/{pixel-spatial-upscaler,union-control}/*.safetensors` | ~1.2GB | `config.yaml` の `ic_loras:` が登録するアダプタの実体（`pixel-spatial-upscaler-x2` と、同一の union-control ファイルを3つの名前で公開した `canny-control` / `pose-control` / `depth-control`） |
| IC-LoRA Deblur 1点 | `models/LTX23/IC-LoRA/deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | ~0.91GB | ピンぼけした動画をくっきりさせる `deblur` アダプタの実体（前処理を必要とせず、ぼけた参照動画をそのまま渡す） |
| IC-LoRA In-Outpainting 1点 | `models/LTX23/IC-LoRA/in-outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` | ~1.22GB | 動画のキャンバス拡張（Outpainting）で使う `in-outpainting` アダプタの実体 |
| DWPose 前処理器 2点 | `models/Preprocessors/DWPose/{yolox_l,dw-ll_ucoco_384_bs5}.torchscript.pt` | ~0.34GB | `pose-control` アダプタが参照動画から骨格を起こすときに使う姿勢推定モデル（`engine/preprocess/dwpose.py` が絶対パスで読む） |
| VDA 深度前処理器 2点 | `models/Preprocessors/VDA/video_depth_anything_vits.pth` ＋ `LICENSE` | ~0.12GB | `depth-control` アダプタが参照動画から深度マップ（手前と奥の距離を明暗で表した白黒映像）を起こすときに使う Video-Depth-Anything Small（`engine/preprocess/depth.py` が絶対パスで読む）。同梱の `LICENSE` は Apache-2.0 の全文で、この重みだけライセンスが異なるため必ず一緒に置かれる |

上記10要素はすべて、本プロジェクトが再ホストした3つの公開リポジトリ——[`Rootport/Nz-LTX23-weights`](https://huggingface.co/Rootport/Nz-LTX23-weights)（transformer・VAE・アップスケーラ・IC-LoRA・VDA）／[`Rootport/Nz-Gemma3-12B`](https://huggingface.co/Rootport/Nz-Gemma3-12B)（text encoder と tokenizer）／[`Rootport/Nz-DWPose`](https://huggingface.co/Rootport/Nz-DWPose)（姿勢推定）——から取得します。いずれも Public かつ非 Gated（ライセンス承諾の壁が無い）ため、**HuggingFace のアカウントもアクセストークンも一切必要ありません**。どのファイルをどこへ置くかは manifest（`scripts/manifests/*.json`）が決めており、リポジトリの中身はまず一時置き場（`models/.dl/`）へ落としてから上表の場所へ移します。一時置き場は成功するまで消さないので、途中で失敗しても再実行すればダウンロード済みの分は再取得されません。

インストールの最後に出る検証テーブルは、IC-LoRA・前処理器も含めて 1 ファイルずつ PASS/MISSING を表示します（**`setup.bat` が担当するのは LTX 2.3 と共用前処理器だけ**なので、この表に LTX 2.5 の行は出ません）。MISSING のまま気づかないと、UI にはアダプタ名が出るのに選んだ瞬間 404 になる——という分かりにくい壊れ方をするため、あえて検証の対象に含めてあります。生成の中核として実際にロードされるモデルは合計 **~28GB**（28.15GiB）で、`install_ltx.ps1` はこれに IC-LoRA・前処理器・枝刈りデコーダを加えた **約 33GB（32.51GiB）** をダウンロードします。

backend の選択は `config.model.backend`（`auto`/`mock`/`real`）で行います。既定 `auto` は「そのエンジンの python・worker スクリプト・記述子が宣言するロード対象ファイルが全て存在」すれば **real**、無ければ **mock** です（`_real_available`）。**判定材料は選んでいるベースモデルのエンジン系統ごとに違います**。

| | LTX 2.3（エンジン系統 `ltx`） | LTX 2.5（エンジン系統 `ltx25`） |
|---|---|---|
| python | `config.yaml` の `model.engine_python`（既定 `./.venv-engine/Scripts/python.exe`） | `config.yaml` の `model.engine_python_ltx25`（既定 `./.venv-engine-ltx25/Scripts/python.exe`） |
| worker | `model.engine_dir`（既定 `./engine`）の `worker.py` | `./engine25/worker.py`（**設定項目は無く固定**。`engine25/` は本リポジトリ同梱のため置き場所の選択肢がありません） |
| 重み | 記述子 `scripts/manifests/10-ltx23.json` の 4 カテゴリ＋固定ファイル 3 点 | 記述子 `scripts/manifests/20-ltx25.json` の 4 カテゴリ＋固定ファイル 1 点（空間アップスケーラのみ） |
| 実装 | [`services/engines/ltx/adapter.py`](services/engines/ltx/adapter.py)（旧パス `services/ltx_runner.py` は再エクスポート用の薄い層です） | [`services/engines/ltx25/adapter.py`](services/engines/ltx25/adapter.py) |

### 追加の transformer GGUF / LoRA を配置する

**transformer GGUF**: `models/LTX23/Weights/`（LTX 2.5 なら `models/LTX25/Weights/`）**直下**に `.gguf` を置くだけで、ファイル名から自動認識され UI/API のドロップダウンに列挙されます。サブフォルダに入れても再帰スキャンで拾われます。**どこを・どの拡張子で・再帰するかを決めているのはベースモデル記述子**（`scripts/manifests/*.json` の `categories.transformer` の `scan` / `extensions` / `recursive`）で、スキャンを実行するのが [`services/model_registry.py`](services/model_registry.py) です。登録名はファイル名（拡張子除く）で、既定の登録名と衝突する場合は親フォルダ名が `親フォルダ名__ファイル名` の形で前置されます。`config.yaml` の編集は不要です（`model.transformers` への明示登録は、スキャンでは拾えないファイルを公開するための上書き用の代替手段です）。

選択は UI の「Models」設定タブのドロップダウン、または API `GET /models`（登録名の一覧確認）→ `POST /pipeline/load`（body `{"models": {"transformer": "<登録名>"}}`）で行います。選択が現在ロード中のものと異なる場合のみワーカーが再構築されます。

GGUF の要件: (1) KVメタデータに `config`（モデル設定のJSON文字列）が埋め込まれていること、(2) テンソル名が LTX ネイティブの生キーであること、(3) `embeddings_connector` 層が非量子化（F32/BF16）であること。これらを満たさない外部配布 GGUF はロードに失敗します（条件を満たすのは QuantStack 製、および自家製変換ツール `Nz-GGUF-Converter-LTX23` の出力）。量子化タイプは既定の Q4_K_M に加え Q6_K / Q8_0 等にも対応します。

**LoRA**: ここで言う LoRA は、利用者が自分で用意する**画風・キャラクター系（スタイル LoRA）**のことです。`config.yaml` の `ic_loras:` に登録済みの IC-LoRA（`pixel-spatial-upscaler-x2` / `canny-control` / `pose-control` / `depth-control` / `deblur`）は `install_ltx.ps1` が自動取得するので、下記の手動配置の対象ではありません。**LoRA の置き場所はベースモデルで分かれていません**——LTX 2.5 を選んでいるときも、`models/LTX23/StyleLoRA/` と `config.yaml` の `ic_loras:` に登録した同じファイルがそのまま使われます（効き方の違いは §7.1）。

`models/LTX23/StyleLoRA/` に `.safetensors` を置くと自動認識されます（`GET /loras` で一覧確認、`POST /loras/reload` で明示再スキャン）。生成時は API の `loras: [{"name": ..., "strength": ...}]`、または Gradio UI のプロンプト内 `<lora:名前:強度>` 記法で適用します（強度は 0.05〜2、0は不可）。**指定した値がそのまま効きます（ComfyUI と同じ数え方です）。** 音声側の適用強度だけを映像側と別に指定したい場合は `<lora:名前:映像の強度:音声の強度>` の第3引数（0〜2。音声側だけ 0＝適用しない、を指定できる）を使います。省略時は音声側も映像側の値に追従します（詳細は [`Docs/LORA_AUDIO_STRENGTH_WORKORDER.md`](Docs/LORA_AUDIO_STRENGTH_WORKORDER.md)）。

**読み込める書式は2つです。** ComfyUI 形式（`diffusion_model.` プレフィックス＋ `lora_A`/`lora_B`）と、**kohya 形式**（LoRA 学習ツール kohya-ss 系が出力する書式。`lora_down`/`lora_up` ＋ `alpha`）で、kohya 形式のファイルは読み込むときに自動で変換します（`alpha ÷ rank` の倍率もそこで重みへ畳み込みます）。ただし kohya 形式でも、**鍵がドットではなくアンダースコアでつながれているもの**（`lora_unet_…`）は読めません。**読めない書式のファイルを指定した生成は、422 `LORA_FORMAT_UNSUPPORTED` ではっきり断ります**——上記のアンダースコア連結の鍵に加えて、DoRA・LoHa・LoKr がこれに当たります。どちらの書式も量子化 GGUF モデルにそのまま適用できます（実行時加算方式のため、モデル側の量子化と衝突しません）。

### models フォルダの構成

`models/` は「どのベースモデルのものか」を最上位で分ける構成になっています。置き場所がそのまま「このファイルは LTX 2.3 用です」という宣言になるため、ベースモデルが増えても、ファイルの中身を見分ける仕組みを足さずに並べていけます。**実際に `LTX23/` と `LTX25/` の2つが並んでいます。**

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

**LTX 2.5 の重みは、設計上 `setup.bat` の取得対象ではありません**（上の約33GBには含まれません）。`setup.bat` は Nz-Videomni 本体のインストーラで、「最初にお試しいただくAI」として LTX 2.3 だけを一緒に導入します。**別のベースモデルは、そのモデル専用のバッチファイルをダブルクリックして導入します**（下の「LTX 2.5 を追加する」）。

**古い構成のまま使っている場合**、`setup.bat` を再実行すると自動で新しい構成へ移ります。中身を**移動するだけ**（同じドライブ内なので数秒で終わり、再ダウンロードは発生しません）で、移動先に同名のファイルがある場合は上書きせずその場で止まります。自分で置いた LoRA・自家変換 GGUF・x4 アップスケーラーなども、対応表に無いものはそのまま持ち上がります。実行のたびに、何をどこへ移すかの一覧が画面に出て、1ファイル1行の記録が `logs\model_migration_日付_時刻.log` に残ります。`config.yaml` に対して行うのは、**旧構成のモデルパスの書き換え**と、**記述子へ移って読まれなくなった廃止済みキーの行の削除**の2種類だけで（どちらも実行前に `config.yaml.bak` を書き出します）、それ以外の設定値・コメント・空行には一切触れません。

> **元に戻したいとき**: 上の移行ログの `MOVE` 行は「移動元」「移動先」の順に並んだ表になっているので、ログを下から順に読み、各行の移動先を移動元へ戻す（PowerShell なら `Move-Item -LiteralPath models\<移動先> -Destination models\<移動元>`）だけで、移行前の配置に戻せます。`CONFIG` 行がある場合は `config.yaml.bak` を `config.yaml` へ戻してください。ただし、この作業は**古いコードへ戻す場合にだけ意味があります**（現在のコードは新しい構成のパスを見にいくため）。

### LTX 2.5 を追加する（`install-LTX25.bat`）

**先に `setup.bat` を済ませてください。** `install-LTX25.bat` は重みファイルを取ってくるだけのもので、Python 環境とダウンロード道具は `setup.bat` が用意します（済んでいないときは、その場でそう案内して止まります）。

準備ができたら、**リポジトリ直下の `install-LTX25.bat` をダブルクリックしてください。** 下表の 5 ファイルが `models/LTX25/<カテゴリ>/` へ自動で置かれ、そのあと `run.bat` で画面を開き直せば、ヘッダーのベースモデル一覧から「LTX 2.5」を選べるようになります（**画面を開いたままだと未導入のままに見えます**——一覧は画面を開いたときにしか読み直さないためです）。

- **ダウンロードの量はおよそ 25 GB**（必要な空き容量として上の「ハードウェア要件の詳細」に載せている約 30GB は、これに安全側の余裕を足した値）で、回線によっては1〜2時間かかります。途中でスリープしない設定にしておくと確実です（**バッチ自身も実行前に必要な空き容量を表示します**）。
- **揃っているファイルは取り直しません。** 途中で失敗しても、もう一度ダブルクリックすれば続きから再開します。`setup.bat` と続けて実行しても二重取得は起きません。
- 実行の記録は `logs\install_LTX25_<日時>.log` に残ります。うまくいかないときは、この記録を見てください。
- 検証に失敗したときは、**足りない 1 ファイルだけを消して**もう一度実行してください。`models\LTX25` をフォルダごと消さないでください（ご自身で置いた LoRA などが同居している場合、再取得できません）。
- **この導線は、まっさらな別のパソコンで、実際のダウンロードから生成まで通しで検証済みです。**

**LTX 2.5 が探すファイルは 5 本です**（記述子 `scripts/manifests/20-ltx25.json` が宣言している既定のファイル名。すべて `models/` からの相対パスです）。**下表はバッチが置く場所の内訳で、手で置きたい場合の対応表も兼ねています**——バッチを使わずに下記の公開リポジトリから手でダウンロードし、同じ場所へ同じファイル名で置いても、そのまま「導入済み」になります。ファイルごとの正確なサイズと SHA-256 は [`Docs/LTX25_RESEARCH_NOTES.md`](Docs/LTX25_RESEARCH_NOTES.md) 10節にあります。

| 役割 | 期待するパスとファイル名 | 取得元リポジトリ |
|------|--------------------------|------------------|
| transformer（本体） | `LTX25/Weights/LTX-2.5-22B-distilled-transformer.gguf` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| テキストエンコーダ（Gemma 4） | `LTX25/TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf` | [`Rootport/Nz-Gemma4-12B-LTX25`](https://huggingface.co/Rootport/Nz-Gemma4-12B-LTX25) |
| 映像 VAE（畳み込みデコーダ版） | `LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| 音声 VAE | `LTX25/VAE/ltx-2.5-audio-vae-bf16.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |
| 空間アップスケーラ | `LTX25/Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | [`Rootport/Nz-LTX25-weights`](https://huggingface.co/Rootport/Nz-LTX25-weights) |

**どちらのリポジトリも Public かつ非 Gated** なので、HuggingFace のアカウントもアクセストークンも要りません。リポジトリ内のフォルダ名（`Weights/` ・ `TextEncoder/` ・ `VAE/` ・ `Upscaler/`）は上表のフォルダ名とそのまま対応しているので、ダウンロードしたファイルを同じ名前のフォルダへ入れるだけで済みます。テキストエンコーダだけリポジトリが分かれているのは、基になったモデル（Gemma 4）のライセンスが異なるためです。各フォルダには `put_〇〇_here.txt` という案内ファイルが1つ入っていて、ファイル名がそのまま「ここに置ける形式」の掲示になっています（この案内ファイル自体はモデルの読み込み対象になりません）。

> **`install-LTX25.bat` を実行するまで、LTX 2.5 は「選べるが未導入」として画面に出ます。** LTX 2.3 だけを使うぶんには何の影響もありません。テキストエンコーダのトークナイザは GGUF の中に入っているので別途置く必要はありません。**隣にできる `*.assets.safetensors` は、バックエンドが初回のモデル読み込み時に GGUF の中身から自動で作るもの**なので、ダウンロードする必要はありませんし、消しても次回に作り直されます。

---

## 2. 起動

**通常は `run.bat` をダブルクリックしてください。** 起動が終わると、黒い画面に囲み枠つきで `http://127.0.0.1:18620/ui` のようなアドレスが表示されるので、それをブラウザで開きます（このアドレスを表示するのは `main.py` の起動バナーで、そこが表示の正本です）。止めるときは、その画面を × ボタンで閉じてください。

`run.bat` は、リポジトリ直下の `run.ps1` を実行ポリシーの制約を受けない形で呼び出すだけの薄いラッパーです。`run.ps1` は残っており、PowerShell から直接実行することもできます（引数はどちらからでも同じように渡せます）。

```powershell
# 起動スクリプト（環境変数設定 → tools/ を PATH の先頭へ → アプリ venv → main.py 起動 を一括）
./run.ps1

# もしくは直接（環境変数と PATH は自分で用意することになります）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
.\.venv\Scripts\python.exe main.py
```

**GPU のメモリまわりの環境変数は、設定する必要はありません**（`PYTORCH_CUDA_ALLOC_CONF` の `expandable_segments:True` は Windows では効かないため、プロジェクトのどこからも設定していません）。メモリの断片化への対策は、実際に断片化が起きる場所——生成の直前と、モデルの部品を GPU へ出し入れする仕組み——で行っています。

`run.ps1` は **アプリ**（`./.venv` の `main.py`）を起動します。real backend が選ばれると、アプリが、選ばれているモデルに対応する worker を subprocess として自動 spawn します（手動起動は不要）。起動を軽く保つため、**依存の再同期（`uv sync` など）は行いません**。`git pull` のあとは `setup.bat` を実行してください（§1「更新のしかた」）。なお `.venv` がまだ無い場合は、その旨を表示して終了します（自動では作りません。`.venv` だけ作ってもモデルもエンジン環境も無く、実生成はできないためです）。

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

**2プロセス分離**（凍結 API 層 + 実エンジン worker）です。**worker は選んでいるベースモデルのエンジン系統によって中身が入れ替わります**（同時に生きているのは常に1つ）。

```text
[Gradio /ui]──HTTP──┐
                    ▼
┌───────────────────────────────────────────────┐
│  FastAPI アプリ (./.venv, torch 無し)          │
│   api/ ・ services/ ・ gradio_ui/ ・ mcp_server/│
│   services/engines/<系統>/adapter.py            │
│        ├─ _MockBackend (合成クリップ・GPU不要)  │
│        └─ _RealBackend ── subprocess.Popen ──┐ │
└──────────────────────────────────────────────┼─┘
                                                │ JSON-lines (@@LTX@@ frames)
                                                ▼ stdin/stdout
   ┌─────────────────────────────┬─────────────────────────────┐
   │ LTX 2.3 の worker（系統 ltx）│ LTX 2.5 の worker（系統 ltx25）│
   │  ./.venv-engine             │  ./.venv-engine-ltx25        │
   │  python -m engine.worker    │  python -m engine25.worker   │
   │  ログ: logs/ltx_worker.log   │  ログ: logs/ltx25_worker.log  │
   └─────────────────────────────┴─────────────────────────────┘
```

- **`engine/`（LTX 2.3）と `engine25/`（LTX 2.5）はどちらも first-party**（project root 直下・git 追跡）で、由来と provenance は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md) にあります。上流参照用の `vendor/LTX-2` は温存しています。
- **プロトコル**: アプリは real backend でも torch/LTX を import しません。`_RealBackend` がそのエンジンの python で worker を常駐起動し、`{"op":"load",...}` → `@@LTX@@{"event":"ready"}` → `{"op":"generate",...}` → `@@LTX@@{"event":"done",...}` と往復します。worker がモデルを **1度だけ**構築してジョブを使い回し、mp4 は worker が直接ディスクへ書きます（制御 JSON のみパイプを渡る）。**この受け答えの作法は2系統で共通**で、違うのは起動する python とモジュール、ロードペイロードの項目名、そしてワーカーログの名前だけです。
- **16GB 技術**: GGUF Q4_K_M transformer + block-swap（GPU 常駐 8 ブロック）+ GGUF Gemma の逐次 per-layer CPU オフロード（`--te-offload`）+ DiT の CPU 構築（`--dit-cpu-load`）+ VAE タイリング + component-file 経路。512×320 で peak_vram ~9.2GB、720p（1280×768→crop）実証済みです（内部の詳細は仕様書 §9「低VRAM戦略」）。
- **モック backend** は `./.venv` のみで動く合成クリップ生成で、テストと GPU 無し開発に使います（`GenerationOutcome.backend` だけが real と異なり、API/スキーマ/出力構造は同一）。

---

## 4. API 概要（`/api/v1`）

> **この表は主要なものだけを載せた抜粋です。** 全ルートの一覧と各フィールドの詳細は、同じリポジトリ内のフロントエンドにある `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md` が正本です。稼働中のサーバーであれば `http://127.0.0.1:18620/docs`（Swagger UI）でも全ルートを確認できます。

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
- 幅・高さは **64の倍数**（two-stage distilled が stage1 を半解像度で生成し x2 アップサンプルするための帰結。由来の解説は [`Docs/LTX23_REFERENCE.md`](Docs/LTX23_REFERENCE.md) §3、検証は [`api/models.py`](api/models.py) の `GenerateRequest` validator）。
- フレーム数は **8n+1**（9, 17, 25, … 121）。尺 cap は **20s（481f=8×60+1）@24fps** まで許容。溢れ/低速/非実用は**クライアント UI 警告に委ねる**方針で、解像度別 spill-free フレーム数を `GET /api/v1/config` の `limits.spill_free_frames` に露出します（**値の正本は `config.yaml` の `limits.spill_free_frames` と [`Docs/COMFORT_LIMIT_TABLE.md`](Docs/COMFORT_LIMIT_TABLE.md)**。閾値の考え方は同書 §付記、実測の経緯は [`Docs/RESOLUTION_DURATION_CAPABILITY.md`](Docs/RESOLUTION_DURATION_CAPABILITY.md) §8.4/§8.6）。これを超えると shared へ溢れ ~2-4x 低速化します（OOM はしません）。1080p の長尺は非実用（~40分・commit リスク）のため **720p 生成＋外部 upscale** 推奨です。
- Distilled は **8 steps / CFG=1.0** 固定。
- I2V のキーフレーム画像は **最大5枚**（画像なし=T2V、1枚以上=I2V）。`frame_idx` は `0`（開始フレーム）か 8n+1 で、それ以外の値を送っても 422 にはならず、8n+1 グリッドへ丸めて `[1, num_frames-8]` の範囲へ収められます。
- `crop_output` を指定すると、任意の非64サイズ（例 960×540, 1280×720）を中央クロップで得ます。

---

## 5. 生成のしかたと高速化（16GB 向け生成テスト）

> **本節は LTX 2.3 を選んでいるときの説明です。LTX 2.5 を選んでいるときの違いは §7.1 にまとめてあります。**

出力は `outputs/{job_id}/output.mp4` と `outputs/{job_id}/metadata.json` に保存されます。`peak_vram_mb` は `metadata.json` またはワーカーログの `GENERATED_OK peak_vram_mb=` から取得できます（jobs API 応答には含まれません）。**ワーカーログはエンジン系統ごとに別ファイル**です（§7.4）。

動作確認用の `smoke_test`（384×256/17f）・`minimal`（512×320/49f）・720p・最小I2V を API から叩くときの curl / PowerShell の例は、[`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) §16.1 にあります。

**使うときに困りやすい注意が3つあり、それぞれ正本は次の場所です。** ①[`sage attention` は選ぶと絵の細部が変わる](#sage-seed-note)／②[PrunaVAED も同じく絵が変わり、LTX 2.5 では使えない](#prunavaed-quality-note)／③[素材（末尾）はクリップ1本での使用を推奨](#end-source-note)。

### worker 単体スモーク（engine を直接叩く場合）

```powershell
$env:PYTHONPATH = (Get-Location).Path
& ".\.venv-engine\Scripts\python.exe" -m engine.worker   # LTX 2.3。{"op":"load",...} を stdin へ → @@LTX@@{"event":"ready"}

# LTX 2.5 は別の仮想環境・別のモジュール（受け答えの作法は同じ）
& ".\.venv-engine-ltx25\Scripts\python.exe" -m engine25.worker
```

### Gradio UI

`/ui` を開き、画像なしで「生成」→ T2V、画像1枚を指定して「生成」→ 最小I2V です。

**A2V（音声から動画生成）**: Generate タブの「A2V（音声から動画生成）」アコーディオンに音声ファイルを添付します。`.wav` 推奨で、添付すると音声長に収まる最大フレーム数が自動で Frames に入ります（他形式は自動調整の対象外でサーバー側チェックに委ねます）。画像でキャラクター等を固定したい場合は「キーフレーム画像」アコーディオン（5スロットとも A2V と併用可）を使います。**スタイル LoRA とも、参照動画を必要とする control 系 IC-LoRA とも併用できます。** Clip Chain タブでの連結生成では、**長い参照動画を1本だけ添付すると各クリップが担当する区間をサーバーが自動で切り出して stage-1 にのみ注入する「長尺 IC-LoRA」**として働きます（参照が生成の尺より短ければ、足りない分は参照なしで生成されます）。ただし `depth-control` だけは前処理（Video-Depth-Anything）が全編一括設計でメモリに載らないため、2クリップ以上のチェーンでは `422 LORA_DEPTH_CHAIN_UNSUPPORTED` で断ります（クリップ1本のチェーンと単発生成は使えます）。

**Batch A2V（就寝中の一括生成）**: A2V の下にある「Batch A2V」アコーディオンの Enable をオンにすると、ゆっくり実況・VOICEROID実況（音声合成ソフトによるキャラクター実況動画）向けに、音声フォルダの中身をまとめて一括生成できます。音声フォルダ・画像フォルダ・出力先を入力して「Set audios」を押すと音声ファイルごとの行を持つ表ができ、行ごとにプロンプトや使用画像を編集して「Start a2v batch」で開始します。Generate タブの現在の設定がその時点のスナップショットとして全行に適用され、1件ずつ順番に生成されます。実処理は**サーバー側のバックグラウンドスレッド**で回るので、ブラウザを閉じても止まりません。進行は音声フォルダ直下の CSV マニフェスト（`batch_a2v_manifest.csv`）へ逐次記録され、電源断やアプリ再起動のあとも「Set audios」→「Start a2v batch」で続きから再開できます（Done/Skip 済みの行は再実行されません）。開始前には入力忘れの検査が走り、問題があれば理由を表示してバッチ自体を開始しません。詳しい仕様・設計上の決定事項は [`Docs/BATCH_A2V_WORKORDER.md`](Docs/BATCH_A2V_WORKORDER.md)、CSV の列定義・文字コードなど相互運用の共通仕様は [`Docs/BATCH_A2V_CSV_SPEC.md`](Docs/BATCH_A2V_CSV_SPEC.md) が正本です。

### 非CFGネガティブプロンプト（NAG）

蒸留版の LTX は CFG（Classifier-Free Guidance。正負2パスの denoise でネガティブプロンプトを効かせる従来手法）が `guidance_scale=1.0` に凍結されているため、従来型のネガティブプロンプトは何も効きません。**NAG（Normalized Attention Guidance）** は、cross-attention（テキストと映像/音声の対応を取る注意機構）の出力レベルで正負のプロンプト出力を外挿・正規化・ブレンドすることで、CFG の2パス化なしに1パスのままネガティブプロンプトを効かせる非CFG手法です。単発 Generate（`/generate`）・Clip Chain（`/generate/chain`）・バッチA2V の**全経路で使えます**。NAG と VSF の使い分けは [`Docs/VSF_README_NOTES.md`](Docs/VSF_README_NOTES.md) が正本です。

**API**（`GenerateRequest` / `GenerateChainRequest` 共通）:

| フィールド | 型 | デフォルト | 範囲 | 説明 |
|-----------|----|-----------|------|------|
| `nag_enabled` | bool | `false` | — | ONにするとNAGが有効になる。ONで`negative_prompt`が空だと422 |
| `nag_scale` | float | `11.0` | `1.0`〜`20.0` | 負プロンプトをどれだけ強く外挿するか |
| `nag_tau` | float | `2.5` | `1.0`〜`10.0` | ノルムの頭打ち上限（暴れ防止） |
| `nag_alpha` | float | `0.25` | `0.0`〜`1.0` | 正出力とのブレンド比率 |

`negative_prompt` にも `max_length=2000`（`prompt` と同じ上限）が付いています。既定値 11.0/2.5/0.25 は、先行実装 [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) の `LTX2_NAG` に準拠したものです。**`nag_alpha` を 0 にしても「NAG なし」とまったく同じ絵にはなりません**——切りたいときはチェックボックス（`nag_enabled`）を外してください。

**GUI**: 共有プロンプト欄の直下（Generate/Clip Chain 両タブの外）にある「Negative Prompt」アコーディオンから使います。テキスト欄は既定では入力不可（グレーアウト）で、「non-CFG Negative」チェックボックスをONにすると編集可能になり、下の3スライダー（scale/tau/alpha）も効くようになります。**コスト**: cross-attention の計算が正負2回に増えます（自己注意は増えないため全体では数%〜15%程度の増加）。VRAM は negative context とゲート用の中間テンソル分だけ増えます（768p 帯で数百MB程度）。設計判断・非対称設計（AdaLN変調の扱い）の根拠は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §38 にあります。

### 生成の高速化（Acceleration）

生成そのものを速くするための切替を、設定画面の「Acceleration（生成の高速化）」という区画にまとめてあります（AviUtl2 の操作パネルなら Settings、Gradio UI なら Settings タブ）。**項目は6つです。** ただし **Gradio UI の Settings タブは5項目**で、6つ目（埋め込み処理器の常駐）は AviUtl2 の操作パネルにだけあります（意図的な差で、実装漏れではありません）。**「どちらのモデルでも使えない項目」はありませんが、「そのモデルでは使えない項目」がお互いに1つずつあります**——LTX 2.5 では PrunaVAED が、LTX 2.3 では埋め込み処理器の常駐が使えません。

| 項目 | 選択肢 | LTX 2.3 での状態 | LTX 2.5 での扱い |
|------|--------|------------------|------------------|
| Fused GGUF Dequantization Kernel（GGUF逆量子化の1カーネル化） | On / Off | **実装済み**。既定は on | **実装済み**。既定は on |
| Attention（注意機構の実装） | `sdpa` / `sage attention` | **実装済み**。既定は `sdpa` | **実装済み**。既定は `sdpa` |
| Block-swap prefetch（先読みblock swap） | On / Off | **実装済み**。既定は on | **実装済み**。既定は on |
| モデル骨格の常駐（keep_resident） | On / Off | **実装済み**。既定は off | **実装済み**。既定は off |
| 埋め込み処理器の常駐（keep_resident_embeddings） | On / Off | **この項目はありません**。**AviUtl2 の操作パネルでは、この行そのものが見えません**。API へ指定すると **422** | **実装済み**。既定は off |
| VAE（映像の復元処理） | Default / PrunaVAED | **実装済み**。既定は Default（＝off） | Default のみ。PrunaVAED は **422**。**AviUtl2 の操作パネルでは、この行そのものが見えません**。Gradio UI では従来どおり見えているので、選ぶと 422 になります |

**GGUF逆量子化の1カーネル化と先読み block swap を LTX 2.5 で併用したときの生成時間は、実測で 102.86 秒 → 39.28 秒＝61.8%短縮です。** 出力そのものは変わらず、**固定の検証用ジョブ17本すべてで動画のファイルが1バイトも変わっていない**ことを確認しています。残る PrunaVAED だけは LTX 2.5 で **422** で断られます——利用者が意図して on にしたときだけ付くものなので、効かないまま黙って通すより断ったほうが親切だという判断です（詳しくは §7.1 と [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §75・§76・§77）。

**選び方**: `sdpa` は PyTorch 標準の実装で、これまでどおりの結果が出ます。`sage` は [SageAttention 2.2.0](https://github.com/thu-ml/SageAttention)（量子化を使って注意機構の計算そのものを速くする外部カーネル）を使い、**生成が速くなります**。切替はラジオボタンを押すだけで、サーバーの再起動もモデルの読み込み直しも要りません。選んだ内容はブラウザ側に保存され、次に開いたときも保たれます。

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

> **⚠ 効き目は動画の大きさに強く左右されます。** `sage` が速くするのは注意機構の計算だけなので、**その計算量が大きいほど効きます**——大きい画面・長い尺ほど効き、**小さい画面（512×320 級）ではほとんど効かないか、かえって遅くなることがあります**（上の表の 0.95 倍の行がそれです）。また、`sage` が触れない区間（文章の読み取りと動画への書き出し）が全体の4分の1ほどあるため、二段目だけを見た倍率がそのまま全体の倍率になることはありません。**「on にすれば必ず速くなる」ものではない**とお考えください。

> <a id="sage-seed-note"></a>**⚠ `sage` を選ぶと、同じシードを指定しても生成結果の細部が変わります。** 計算に使う数値の精度が違うためで、
> 不具合ではありません。構図や被写体といった大枠は同じままで、質感やノイズの出方といった細かいところが変わります
> （実測での差は PSNR〔ピーク信号対雑音比。元の絵との違いを 1 つの数値にしたもので、単位は dB。数値が大きいほど元の絵に近い〕で 27〜28dB 程度）。**以前つくった動画とまったく同じものを作り直したい場合は、`sdpa` を
> 選んでください。** 既定を `sdpa` のままにしてあるのは、アップデートによって利用者の生成結果が黙って変わることを
> 避けるためです。**この注意書きが `sage` についての正本**で、他の箇所（§7 の制限事項や §8 の MCP 注意事項）はここを参照します。
> **この注意は LTX 2.3 でも LTX 2.5 でもそのまま当てはまり、既定はどちらも `sdpa` です。**

**入っていない環境ではどうなるか**: `sageattention` は `setup.bat` が標準で入れますが、何らかの事情で入っていない環境（古い手順で作った仮想環境など）では、`sage` を選んでも**エラーにはならず、自動的に `sdpa` に切り替わって最後まで生成されます**（**この降格のふるまいは2系統でまったく同じです**）。利用可否は `GET /status` の `acceleration.sage_available` で確認でき、操作パネル側は使えないときはボタン自体が選べなくなります。実際にどちらで生成されたかは `outputs/{ジョブID}/metadata.json` の `attention_used`（`"sdpa"` / `"sage"` / `"sage->sdpa"`）で確認できます。

> **最初の1本だけ遅く感じることがあります。** `sage` は内部で triton という仕組みを使い、その初回にだけ「JIT コンパイル」（実行時にカーネルを組み立てる処理）が走るためです。2本目以降は本来の速さになります。速度を測って比べたい場合は、この点と、**worker プロセスの最初のジョブだけ約 8% 速い**という癖の両方を踏まえて、`sdpa` と `sage` を交互に流して隣り合う組で比べてください（詳細は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §43.6）。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `attention_backend`（`"sdpa"` または `"sage"`、既定 `"sdpa"`）を指定できます。詳しい設計判断・検証結果は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §43 を参照してください。

### 先読みblock swap（`block_swap_prefetch`）

block swap（VRAM を節約するために transformer のブロックを CPU と GPU のあいだで出し入れする既定の仕組み）の転送を、計算とは別の CUDA stream で先回りさせて待ち時間を隠す機能です。Settings の Acceleration 区画にある「Block-swap prefetch」トグルで、Attention と同じくジョブ単位で切り替えられます。**既定は on** です。

`sage` と違い、**転送方式だけを変えるので出力は変わりません**——同じシードなら off/on でビット単位で完全に同じ動画になります。LTX 2.3 の実測（768p/257フレーム、交互対比較3組）で平均約14.7%短縮、VRAM の増加はほぼ0です。off にすると従来の同期スワップへ戻ります（block swap 自体が無効な設定では、on/off にかかわらず何も起きません）。LTX 2.5 で単独で使ったときは**平均 25.3%短縮**で、こちらも**固定の検証用ジョブ17本すべてで動画のファイルが1バイトも変わりません**（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §44・§75）。

> **メインメモリを約416MB（207.9MB × 2枠）、サーバーを起動しているあいだずっと使います。** 重みを CPU から GPU へ
> 速く渡すために、OS が動かせない形で確保した受け渡し用の枠（ページ固定メモリ）を2つ持ち続けるためです。
> **この枠はジョブが終わっても返しません。off にしても返りません。解放されるのはサーバーを再起動したときだけです**
> ——枠を手放す処理は「block swap の仕組みそのものを取り外す」経路にしかなく、本番の運転ではその経路を通らないためです
> （枠を作り直すには 100ms 前後かかるうえ、長く動かしたプロセスほど確保に失敗しやすくなるので、
> 作っては返すのではなく持ち続ける設計にしてあります）。枠の大きさはモデルの最も大きなブロックに合わせて決まるので、
> LTX 2.3 では 253.8MB × 2枠（約508MB）になります。**LoRA を使うと最も大きなブロックがそのぶん育つので、枠も一緒に
> 大きくなります**（LTX 2.3 の実測で 260.8MB／266.8MB／277.9MB × 2枠。使う LoRA の大きさによって変わります）。

### GGUF逆量子化の1カーネル化（`fused_gguf_dequant_kernel`）

GGUF ファイルの中で圧縮された形で持っている重みを計算に使える形へ展開する処理（逆量子化）を、**これまでの18〜33個の細かい GPU 処理から、量子化形式ごとに1個の GPU 処理へまとめた**機能です。Settings の Acceleration 区画のトグルで、他の項目と同じくジョブ単位で切り替えられます。**既定は on** です。

先読みblock swap と同じく、**展開の手順しか変えないので出力は変わりません**。LTX 2.3 の実測（768p/257フレーム、交互対比較3組）で**約17.5%短縮**（144.5秒 → 119.2秒）、LTX 2.5 では生成時間の中央値が **99.95秒 → 65.72秒**で、**映像を作る本体だけでなく、文章を読み取る部分（テキストエンコーダ）にも効いて**います。この環境で動かせない場合（Triton が入っていない等）は自動的に従来の方法へ戻り、生成そのものは止まりません。実際に効いたかどうかはメタデータの `fused_gguf_dequant_kernel_used`（`"off"` / `"on"` / `"on->off"`）で確認できます（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §51・§75）。

### モデル骨格の常駐（`keep_resident`）

生成のたびに作り直している**モデルの「骨格」**（GGUF ファイルから組み上げた重みの一覧とモジュールの構造。DiT 約 16.5GB ＋ Gemma 約 8〜9GB）を、次のジョブでもそのまま使い回す機能です。Settings の Acceleration 区画にある「モデル骨格の常駐（ジョブ間キャッシュ）」トグルで切り替えます。**既定は off** です。

効くのは毎回の生成の先頭に乗っている待ち時間（前処理）だけで、開発機での実測は **68.6〜75.0 秒 → 5.32 秒**でした。2本目以降の生成から効くので、シードを変えて何本も試すときやバッチを流すときに差が出ます。**生成結果は変わりません**——同じシードなら on/off でビット単位まで同じ動画になります。GPU 側には何も置かないため、VRAM の使用量も変わりません。

> **⚠ 使うときはメインメモリ 64GB 以上を推奨します。** 骨格を持ち続けるぶん、**約 20GB のメインメモリを常時占有**します（実測 19.4GB）。既定を off にしてあるのはこのためで、**off に戻せば解放されます**（ただし OS がメモリを完全に返すまでには少し間があります）。32GB 級のマシンで on にすると、ページファイルへの追い出しが起きて逆に遅くなることがあります。

一部の設定とは併用できません。`dit_cpu_load` か先読み block swap を off にしている場合は、警告を出したうえで**そのジョブだけ自動的に off** になって最後まで生成されます（メモリが二重に必要になるため）。GGUF の逐次量子化を off にしている場合だけは、生成結果が壊れる（LoRA を融合した重みがキャッシュに焼き付く）ので**エラーで止まります**。実際にどう扱われたかは `metadata.json` の `keep_resident_used`（`"off"` / `"on"` / `"on->off"`）で確認できます。**API から使う場合**は `POST /generate` と `POST /generate/chain` のどちらにも `keep_resident`（真偽値・既定 `false`）を指定できます（[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §48）。

#### LTX 2.5 での常駐

**LTX 2.5 でもこのトグルが効きます。既定は off です。** ただし**同じ名前でも中身は別物**です——LTX 2.3 が抱え込むのは全部品の骨格（約 20GB）ですが、**LTX 2.5 が抱えるのは文章を読み取る部分（Gemma 4 テキストエンコーダ）の重み1つだけ**で、**常駐に使うメインメモリは実測 7.7GiB** です。

- **効くのは2本目以降です。** 2本目からは重みの組み立て（6〜8秒）が **0.4秒**になり、**生成そのものが実測で 27.6秒 → 20.5秒＝約25%短縮**されます（連結生成では約5秒の短縮）。**出来上がる動画は1バイトも変わりません。**
- **VRAM は増えません。** off に戻すとそのジョブの先頭で解放され（ログに「released 7.68 GiB」と出ます）、もう一度 on にすると組み立てを1回だけ払い直します（実測 4.4秒）。
- **併用の制限も自動 off もありません。** `keep_resident_used` に出る値は **`"on"` か `"off"` の2つだけ**で、LTX 2.3 で出ることのある `"on->off"` は出ません。
- **メインメモリの目安**: LTX 2.5 は常駐 off のときでも生成中に約 26GiB まで使います。常駐を on にする場合は、安全側に見積もって **合計 約 34GiB** を見ておいてください（§1「ハードウェア要件」。実測は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §76）。

<a id="keep-resident-embeddings"></a>
### 埋め込み処理器の常駐（`keep_resident_embeddings`）

**この項目は LTX 2.5 専用です。** 上の「モデル骨格の常駐」と違い、**LTX 2.3 にはこの部品自体がありません**——そのため**LTX 2.3 を選んでいるあいだは、操作パネルの Settings からこの行そのものが見えません**（PrunaVAED を LTX 2.5 で隠すのと同じ扱いで、向きだけが逆です）。API へ直接指定した場合は LTX 2.3 で **422** になります。

**埋め込み処理器**（embeddings processor）は、プロンプトを読み取ったあとの内部表現を整える部品です。生成のたびに自分の GGUF ファイルから組み直していたこの部品を、**次のジョブでもそのまま使い回す**のがこの機能です。Settings の Acceleration 区画にある「Embeddings processorの常駐（LTX 2.5）」トグルで切り替えます。**既定は off** です（画面のラベルだけ原語のままです。この文書では日本語で「埋め込み処理器」と呼びます）。

- **効くのは2本目以降です。** 1本目は組み立てるので、これまでどおりの時間がかかります。**2本目からはその組み立てが 0.58秒になります**（1本目は 5.47秒でした）。組み立てに払っていた**3〜5秒がまるごと省かれる**ということです。
- **出来上がる動画は1バイトも変わりません。** 同じ部品を作り直さずに使い回すだけなので、速くなるだけです。
- **GPU（VRAM）は増えません。代償はメインメモリで、常駐は実測 4.66GiB（およそ 5GB）です。**
- **上の「モデル骨格の常駐」とは別のスイッチで、常駐する相手も別物です**（あちらは文章を読み取る部分＝テキストエンコーダの重み）。**両方 on にすると、メインメモリの増分は足し算になります**——片方が他方を含んでいるわけではありません。**メモリに余裕がない場合は、片方だけを on にする使い方もできます。** 両方 on にするときの合計の目安は §1「ハードウェア要件」にあります。
- **off に戻すと、そのジョブの先頭で解放されます**（ログに「released 4.64 GiB」と出ます）。
- **併用の制限も自動 off もありません。** `keep_resident_embeddings_used` に出る値は **`"on"` か `"off"` の2つだけ**です。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `keep_resident_embeddings`（真偽値・既定 `false`）を指定できます。**連結生成でも1つの設定がチェーン全体に効き、組み立ては1ジョブにつき1回**なので、節約されるのは**次のジョブの組み立て**であって、チェーンの内側ではありません（`keep_resident` と同じ読み方です）。実測とゲートの記録の正本は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §92 です。

### 枝刈り版の映像VAEデコーダ（PrunaVAED・`vae_mode`）

映像の復元処理（生成の最後に、内部表現から実際の映像フレームを作る処理）を、**枝刈り**（pruning＝寄与の小さい部分を削ること）と**蒸留**（distillation＝軽くしたモデルに元の出力を教え込むこと）を施した軽量なデコーダへ差し替える機能です。上流は [Pruna AI](https://huggingface.co/PrunaAI/PrunaVAED) が公開している LTX-2.3 専用のデコーダで、これを当方で LTX-2.3 のエンジンが直接読める形（約690MB の単体ファイル）へ変換し、`setup.bat` が他のモデルと一緒に取得します。Settings の Acceleration 区画にある「VAE」の Default / PrunaVAED で、他の項目と同じくジョブ単位で切り替えられます。**既定は Default です。**

> <a id="prunavaed-quality-note"></a>**⚠ PrunaVAED を選ぶと、出力品質がわずかに低下する可能性があります。** 別のデコーダで映像を作るので、`sage` と
> 同じく「絵が変わる」種類の切替です。同一シードでの見比べでは「劣化は肉眼ではほとんど分からない」水準
> でしたが（客観指標では PSNR 36.06dB・SSIM〔構造類似度。2 つの映像がどれだけ同じ構造をしているかを 0〜1 で表す指標で、1 に近いほど同じ〕0.9854〔輝度〕）、**以前つくった動画とまったく同じものを作り直したい
> 場合は Default を選んでください**。既定を Default のままにしてあるのは、アップデートによって利用者の生成結果が
> 黙って変わることを避けるためです。**この注意書きが PrunaVAED についての正本**で、他の箇所（§7・§8）はここを参照します。
>
> **LTX 2.5 を選んでいるときは PrunaVAED そのものが使えません**——重みファイルの有無にかかわらず 422 で断られます
> （枝刈りデコーダは LTX 2.3 用のもので、LTX 2.5 には対応物がありません。§7「制限事項」）。

**PrunaVAED に対応していないベースモデルを選んでいる間は、AviUtl2 の操作パネルの Settings から「VAE」の行と注意書きがまるごと見えなくなります**（灰色にするのではなく非表示です。422 で断られる項目であって「受け付けるけれど効かない」たぐいではないため、こちらにしてあります）。同時に、ブラウザに残っていた選択も Default へ書き戻されます。**そのかわり、LTX 2.3 → LTX 2.5 → LTX 2.3 と往復すると、PrunaVAED の選択は Default に戻っています**（これは仕様です）。使いたいときは手で選び直してください。選び直すまでは快適上限のマーカーが Default 構成の低いほうの線を指しますが、**これは実際の設定を正しく映した結果です**。**Gradio UI の Settings タブには、この非表示の仕組みが入っていません**——LTX 2.5 を選んでいても「VAE」が見えているので、そこで PrunaVAED を選んで生成すると **422 で断られます**。

**どれくらい速いか**: 720p（1280×768）・257 フレームの実測（交互対比較4組）で、**1本あたり平均 12.5 秒短縮**（119.2 秒 → 106.7 秒＝約10.5%）。映像の復元処理そのものは 32.5 秒 → 20.2 秒（**1.61 倍**）で、短縮のほぼ全部がこの区間で説明できます。**VRAM の予約量は約 2.6GB 減ります**（13,911MB → 11,276MB）。動画が長いほど・解像度が高いほど効きやすくなります。**枝刈りデコーダのファイルが見つからない場合**は、エラーにはならず通常のデコーダで最後まで生成されます（ファイルを戻せば次のジョブからまた使われます。サーバーの再起動は要りません）。

**API から使う場合**: `POST /generate` と `POST /generate/chain` のどちらにも `vae_mode`（`"default"` または `"prune_vaed"`、既定 `"default"`）を指定できます。**フィールドの値が `"prune_vaed"` と綴られているのは、この機能を用意した当時の名前の誤記がそのまま外部仕様になったためです**（正しい名前は PrunaVAED）。値を変えると既存の利用者を壊すので、表示名だけを直してあります。なお、VRAM 節約のための `vram_optimization.vae_tiling`（タイル分割）とは**まったく別の設定**です。実際にどちらで生成されたかは `metadata.json` の `vae_mode_used` で確認できます（**LTX 2.5 では意味が変わり、載っているデコーダの実名〔現行の構成では `"conv"`〕がそのまま入ります**。[`Videomni_Backend_Specification.md`](Videomni_Backend_Specification.md) §6.6）。設計判断・重みの変換手順・実機ゲートの実測値は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §52 にあります。

---

## 6. テスト

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

pytest は **アプリ venv（`./.venv`, torch 無し）** で動きます。`tests/conftest.py` が `model.backend="mock"` を強制するため、GPU/モデル無しで T2V/I2V のバリデーション（64倍数・8n+1・キーフレーム画像の上限5枚・`frame_idx` の丸め）とモックランナーによる生成疎通、`GET /status` の `vram_optimization` 契約を検証します。

`setup.bat` で作った環境には、インストーラが常に `uv sync --extra dev` で同期するため pytest などの開発用依存が最初から入っています。素の `uv sync` を単体で実行するとそれらは取り除かれるので、その場合は `uv sync --extra dev` で入れ直してください。

---

## 7. 制限事項

**この節は「できないこと・気をつけること」だけを集めた場所です。** 長いので、先に中身を並べておきます。

| 節 | 内容 |
|----|------|
| [7.1](#limit-ltx25) | **LTX 2.5 を選んでいるときの制限**（使えない機能・無視される設定） |
| [7.2](#limit-onejob) | **同時に走る生成は1本だけ**（409 の理由・キャンセルの効き方） |
| [7.3](#limit-hardware) | 動作確認済みのハードウェア |
| [7.4](#limit-server) | サーバーとファイルの扱い（設定・ジョブ履歴・保存領域・ログ） |
| [7.5](#limit-generation) | 生成そのものの限界 |

<a id="limit-ltx25"></a>
### 7.1 LTX 2.5 を選んでいるときの制限

**LTX 2.5 で使えないのは、下の2つだけです。** それ以外——基本生成（テキストから動画・画像から動画）、クリップ連結（Chained）、V2V（素材（冒頭）に動画を使って続きを作る）、A2V（音声から動画。長尺・バッチを含む）、撮り直し（Retake）、素材（末尾）（End source）、スタイル LoRA・IC-LoRA（参照動画による制御。長尺を含む）、キャンバス拡張（画角拡張・Outpainting）、ネガティブプロンプト（NAG／VSF）、生成の高速化6項目——は**すべて LTX 2.5 でそのまま使えます**（**このうち埋め込み処理器の常駐は LTX 2.5 専用で、逆に LTX 2.3 では使えません**。§5「[埋め込み処理器の常駐](#keep-resident-embeddings)」）。**画面で灰色になるタブ・パネル・サブタブは1つもありません。** ここに挙げた範囲は実機で検証済みです（結果は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §69.21・§72.10・§73.10・§74.12・§77.10・§78.14・§79.11・§80.10・§92）。

下記は要求すると **422 で断られます**。

| 断られる機能 | 対応するリクエスト項目 |
|--------------|------------------------|
| 枝刈り版の映像VAEデコーダ（PrunaVAED） | `vae_mode` |
| 高品質パイプライン（`two_stage_hq`） | `pipeline` |

**この2つは、画面のタブやパネルとして出ている「機能」ではありません。** PrunaVAED は **LTX 2.5 版の枝刈り済み重みが世の中に無い**ため、非蒸留モデルは **それを動かせる計算機が手元に無い**ため、どちらも先送りです。**断り方は「入口ごとまとめて」ではなく項目ごとです**——連結生成（`POST /generate/chain`）そのものは通り、上の2項目を指定したときだけ 422 になります。

**使うときに知っておいてほしいことが3つあります。**

1. **「同じ設定なら同じ動画」にならない設定が2つあります。** `sage attention` は、同じシードでも出来上がる絵の細部が変わります（§5 の[注意書き](#sage-seed-note)。効き目は動画の大きさに強く左右され、512×320 級ではほとんど効かないか、かえって遅くなることがあります）。もう1つは**ネガティブプロンプトの `nag_alpha`** で、0 にしても「NAG なし」とまったく同じ絵にはなりません——切りたいときは**チェックボックス（`nag_enabled`）を外してください**（§5「非CFGネガティブプロンプト（NAG）」）。
2. **撮り直しと素材（末尾）には、使い方の推奨があります。** **撮り直しは、窓（作り直す範囲）を 121 フレーム以上にしてください**——73 フレームだと前後ののりしろ（25＋24フレーム）を除いて自由に作り直せるのが 24 フレームしか残らず、出来上がりが元とほとんど変わりません（**LTX 2.3 のときからそうで、LTX 2.5 で悪くなったわけではありません**）。**素材（末尾）はクリップ1本での使用を推奨します**（§7.5 の[注意書き](#end-source-note)）。
3. **画面の見え方で戸惑いやすいところが2つあります。** **キャンバス拡張では、生成の進み具合の表示が 50% あたりで15秒ほど止まって見えます**が、不具合ではありません（内部で映像を組み直している区間で、進み具合の計算がこの区間を数えないためです。**LTX 2.3 でも同じです**）。**LoRA は LTX 2.5 でやや弱く効く傾向があります**——既定の強度は LTX 2.3 と同じ 1.0 のままなので、効きが弱いと感じたらプロンプト内の LoRA タグで `1.3` のように指定してください。**スタイル LoRA はトリガー語（学習時に使われた合言葉）をプロンプトに入れないと効きが穏やか**で、**参照動画で輪郭線制御（canny）を使うときは、ぼけていない鮮明な素材を渡してください**（輪郭がほとんど検出されません）。

**指定しても断らず、黙って無視して生成を続ける項目もあります**——`guidance_scale` と `num_inference_steps` の2つです。蒸留版の LTX 2.5 は工程数が固定で、CFG（プロンプトへの従い具合の制御）そのものも無いためです。**無視したことは `logs/server.log` に1行残ります**（ただし**この2つは、LTX 2.5 が受け付けるリクエストでは既定値から動かせないので、実際にこの行が出ることはありません**）。**LTX 2.3 を選んでいるあいだは、これらはすべて従来どおり使えます。** LTX 2.5 の導入手順は §1「LTX 2.5 を追加する（`install-LTX25.bat`）」にあります。

<a id="limit-onejob"></a>
### 7.2 同時に走る生成は1本だけ

- **同時実行は 1 ジョブのみ**。実行中に新しい `POST /generate`（および `POST /generate/chain`）を投げると **409 Conflict**（`JOB_BUSY`）が返ります。**ジョブキューはありません（同時に走る生成は1本だけです）**——単一ユーザー向けのローカルツールという前提で、「1ジョブ＋busy 409」を正式な仕様にしてあります（仕様書 §13.5）。
- **実行中ジョブのキャンセルは best-effort**。PyTorch 推論を安全に中断できないため、`running` のジョブは推論完了後に `cancelled` へ遷移します。まだ実行に移っていない `queued` のジョブは、`DELETE /jobs/{id}` で**即座に** `cancelled` になり単一ジョブガードが解放されます。
- **MCP サーバー（§8）経由でも同じ制約です。** 複数エージェント・複数セッションからの並行操作は非対応です。

<a id="limit-hardware"></a>
### 7.3 動作確認済みのハードウェア

**動作確認済みのハードウェアは 2 構成です。**

- **開発機**: RTX 4070 Ti SUPER 16GB（Ada Lovelace）／メインメモリ 64GB／ページファイル 48GB。日常的な開発と検証はすべてこの 1 台で行っています。
- **サブマシン**: RTX 3080 mobile 16GB（Ampere）／メインメモリ 32GB。**AviUtl2 を導入していない新規環境**で、導入から生成まで一通りを実測し、全項目に成功しました（`setup.bat` での導入 → `run.bat` での起動 → ブラウザで WebUI を開く → `smoke_test` サイズの生成 → **IC-LoRA の DWPose（pose-control）と canny をそれぞれ 768p・257 フレームで制御生成** → `NzVideomni.aux2` を AviUtl2 のプレビュー画面へドラッグ＆ドロップして導入 → 再起動後に操作パネルを表示 → タイムラインからの生成と、生成済み動画の右クリックからのタイムライン配置）。
- 残る GPU 世代（Turing・Hopper・Blackwell）は、torch 2.9.1+cu128 が同梱するカーネルの一覧と CUDA のバイナリ互換性から**理論上は動作するはずですが、実機では未検証**です。

**メインメモリ 32GB では、上記サブマシン 1 台での実測合格があります**（最小構成の生成だけでなく 768p・257 フレームの IC-LoRA 制御生成まで通りました）。ただしこれは**この 1 台での実測結果**であり、あらゆる 32GB 環境での動作を保証するものではありません。§1 の「メインメモリとページファイル」の案内は引き続き必ず守ってください。**LTX 2.5 を使う場合は 64GB 以上を推奨**します（§1 のハードウェア要件）。

<a id="limit-server"></a>
### 7.4 サーバーとファイルの扱い

- **`low_vram_mode=true` がデフォルト**。16GB 環境前提。`low_vram_mode=false` は高VRAM/クラウド用の任意検証で、16GB成功は保証しません。
- ジョブ履歴は in-memory（再起動で消える）。`outputs/{job_id}/metadata.json` はディスクに残ります。
- **`uploads/`（アップロードした画像・動画・音声）に自動削除はありません**。ディスクに残るファイルの区別と整理のしかたは [`Docs/STORAGE_POLICY.md`](Docs/STORAGE_POLICY.md) にまとめてあります（`outputs/` は成果物、`uploads/` は入力素材のキャッシュ）。
- **ワーカーのログはエンジン系統ごとに別ファイル**です——LTX 2.3 は `logs/ltx_worker.log`、LTX 2.5 は `logs/ltx25_worker.log`。アプリ側のログは `logs/server.log` で、LTX 2.5 が「この設定は無視した」と書くのもこちらです。切り替えても両方の記録が残るように分けてあります。

<a id="limit-generation"></a>
### 7.5 生成そのものの限界

- transformer は Q4_K_M 量子化のため、フル bf16 公式とビット一致ではありません（聴感・視感は良好）。
- **キーフレーム画像で「最終フレームちょうど」を条件にすることはできません**（`frame_idx` は `num_frames-8` にクランプされます）。末尾を指定したいときは end source（素材（末尾））を使ってください。
- **「生成の高速化（Acceleration）」のうち2項目は、選ぶと絵が変わります**——`sage attention` と PrunaVAED です。詳しくは §5 の[`sage` の注意書き](#sage-seed-note)と[PrunaVAED の注意書き](#prunavaed-quality-note)を参照してください（`sage attention` は LTX 2.3・LTX 2.5 のどちらでも使えます。PrunaVAED は LTX 2.5 では 422 になります）。

> <a id="end-source-note"></a>**素材（末尾）（end source）は、クリップ1本での使用を推奨します。** 2本以上つなぐ使い方（逆順）も受理しますが**推奨外**で、
> つなぎ目の音が段差状に聞こえます（仕様として許容しているものです）。また、素材が本体のシーンと意味論的に遠いと、
> クロスフェードやカットで繋がります（モデルの限界で、素材の選び方で回避します）。**長くつなぎたいときの運用手順が2つあります**
> （どちらもコード変更不要・出荷済み機能の組み合わせです）。①**手動リレー**——end source をクリップ1本ずつ使い、
> 生成物の冒頭を次の素材にして過去へ遡る。②**正順Chained＋補間仕上げ**（**未実機検証**）——本体は通常の正順 Chained で
> 生成し、最終クリップだけを end source で仕上げる。**手順の正本は
> [`Docs/CHAIN_STAGE2_RESEARCH_NOTES.md`](Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11** で、音声錨の幅と `context_frames` の
> 関係もそちらにあります。

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
| `upload_video` | ローカルの動画ファイルをアップロードする（V2V・参照動画用）。`max_frames` を渡すと**尺（フレーム数）とフレームレートを実測して返す**ので、撮り直しの窓の開始秒を決める下調べに使える（先頭Nフレームだけ残す切り詰めも兼ねる引数なので、測るだけのときは元の尺より確実に大きい値を渡してください） |
| `upload_audio` | ローカルの音声ファイルをアップロードする（A2V用） |
| `submit_generate` | 単発の動画生成ジョブを登録する（T2V/I2V、`POST /generate`）。`attention_backend` ほか生成の高速化6項目に加え、**画角拡張（Outpainting）の6引数**も指定できる |
| `submit_chain` | クリップチェーン生成ジョブを登録する（V2V/A2V/連結、`POST /generate/chain`）。同じく生成の高速化6項目に加え、**撮り直し（Retake）の5引数**も指定できる |
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

各ツールの引数の意味はツールの説明文（docstring）が正本で、設計の理由は [`Docs/MCP_SERVER_DESIGN.md`](Docs/MCP_SERVER_DESIGN.md) にあります。

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
4. 以降は T2V と同じ。**切り替え直後の1本目だけは通常の約2倍**かかるので、`wait_for_job` を多めに呼び直してください。
5. LTX 2.3 へ戻すときは `load_pipeline(base_model="LTX23")`。**`submit_chain` が投げられるモードは LTX 2.5 でも全部通ります**（§7.1）。

**A2Vバッチ（音声フォルダの一括生成）**:
1. `plan_a2v_batch` で音声フォルダを走査し、行ごとの計画（音声パス・提案フレーム数・同stem画像等）を得る。
2. 各行について **順番に**（同時1ジョブ制約のため直列で）: `upload_audio` → `submit_chain` → `wait_for_job` を繰り返し呼ぶ → `save_job_video` で任意の出力フォルダへ保存する。

### 注意事項

- **同時実行は1ジョブまで**: ジョブ実行中に新しい `submit_generate` / `submit_chain` を呼ぶと **409 JOB_BUSY** になります。**ベースモデルの切り替え**（`load_pipeline` の `base_model` 引数）も**ワーカーの載せ替え**を伴うため、ジョブ実行中は同じく 409 で断られます。
- **`wait_for_job` は最大45秒でタイムアウト**します。エラーにはならず `timed_out: true` とその時点の進捗を返すので、終端状態になるまで繰り返し呼んでください。
- **`config.yaml` を変更した場合は MCPサーバーの再起動が必要**です（設定は起動時に1回だけ読み込みます）。MCPサーバーは Claude Code のプロセス内で管理されるサブプロセスなので、**Claude Code 自体を再起動**すれば再読み込みされます。
- 生成された動画は base64 等で埋め込まれず、**常にローカルの絶対パス**で返されます（`save_job_video` で任意のフォルダへコピーも可能）。パスは MCP サーバーを動かしているマシン上のものです。
- **「生成の高速化（Acceleration）」の6項目は、すべて MCP のツールに公開しています**（`attention_backend` / `block_swap_prefetch` / `keep_resident` / `fused_gguf_dequant_kernel` / `vae_mode` / `keep_resident_embeddings`）。**LTX 2.5 で 422 になるのは `vae_mode`（PrunaVAED）の1つだけ**、**LTX 2.3 で 422 になるのは `keep_resident_embeddings`（LTX 2.5 専用）の1つだけ**です（§7.1。ツール側からは `list_models` の `base_models[].unsupported_features` でも確認できます）。`attention_backend="sage"` は選ぶと絵の細部が変わり（[注意書き](#sage-seed-note)）、`vae_mode="prune_vaed"` も同様です（[注意書き](#prunavaed-quality-note)）。実際に使われた方式はメタデータの `attention_used` / `vae_mode_used` に記録されます。

---

## ライセンス

本リポジトリは **Apache-2.0** です（[LICENSE](LICENSE) を参照してください）。`engine/` は LTX-2 / LTX-Desktop 由来の派生コードを含むため、再配布時は上流の帰属表示を保持してください（[`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)）。モデルウェイトは LTX-2 Community License に従ってください。
