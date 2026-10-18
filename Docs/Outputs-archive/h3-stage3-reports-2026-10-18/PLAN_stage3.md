# MiniMax H3 対応 α版・第 3 弾の計画（`install-MiniMaxH3.bat`・README）

作成: 2026-10-18（敵対的レビュー反映済み。採否は §9）。台帳 §1-93。設計正本 `Docs/MINIMAX_H3_DESIGN.md` §10・§11、土台 `Docs/MULTI_ENGINE_DESIGN.md` §6.2（導入バッチ）。第 2 弾の記録は `Docs/VERIFICATION_LOG.md` §166。事前調査 4 本（I1 インストーラの機構／I2 配布元の実物／I3 GPU 世代／I4 README）とレビューは一時領域 `scratchpad\h3\stage3\` と plans フォルダにあり、了承後に `Docs/Outputs-archive/h3-stage3-reports-2026-10-18/` へ写す。

## 1. 結論（何を作り、何で合格とするか）

**到達点（α版の完結条件の残り。裁定 14）**: `install-MiniMaxH3.bat` を 1 回実行すると、専用の仮想環境 `.venv-engine-h3` の構築と、重み 4 本＋tokenizer 6 ファイル（約 51GB）の取得が終わり、操作パネルで MiniMax H3 が「導入済み」になる。README に H3 の導入手順・要件・制限・ライセンスが書かれ、第 2 弾で誤りになった 5 か所が直っている。

**合格の条件**: 開発機の門（3-G1 構築・3-G2 指紋）に合格し、**サブマシン（裁定 39）で実際の取得から T2V 1 本までが通る**こと。合格後に dev → main を `--no-ff` で merge し、台帳 §1-93 を CLOSED へ移す（α版の完結）。

**変えないもの**: `setup.bat` の挙動（LTX 2.3＋前処理の集合。uv の環境変数も足さない）、`install-LTX25.bat`・`install-UETrack.bat`、`Ensure-EngineVenv` の関数本体、README の凍結ゾーン（1〜133 行目。境界は 134 行目の見出し `# Nz-Videomniの詳細`。裁定 40）、§0〜§8 とライセンスの番号と見出し名、アンカー 3 つ、`setup.ps1` が名指しする見出し「必要な空き容量の内訳」と「約 50GB」の文字列。

## 2. 裁定（研究ノート §10 に 36〜42 として記録する）

- 36【オーナー裁定 2026-10-18】ライセンスは README に明記（名称・適用除外地域・取得者の自己責任）。インストーラは取得前に注意文とライセンスの URL を表示するだけで、同意の入力は作らない。
- 37【オーナー裁定】README で H3 を「α版」と明示し、できること・できないこと・快適上限が未較正で大きな画角では遅くなること・推奨（VRAM 16GB・主メモリ 64GB）を書く。
- 38【オーナー裁定】GPU 世代の文言は正直に: 「ドライバ R580 以上・RTX 20 以降（GTX 10 以前は動かない）。確かめたのは RTX 40 だけ。RTX 30・50 は未確認。RTX 20 は int8 が遅い経路になり SageAttention も使えず、動作も未確認」（I3）。
- 39【オーナー裁定】サブマシンでの検証を門にする。構築と取得と UI の導入済み表示までが必須、常駐なしの T2V 640×384 が 1 本通れば合格（VRAM が足りなければ前者まで）。
- 40【オーナー裁定】README の凍結ゾーンは触らない。H3 の話題は詳細側に書く。
- **41（了承時に確認。監督の提案）**: インストーラが `MiniMaxAI/MiniMax-H3` の `LICENSE` も取得して `models/MiniMaxH3/LICENSE` に置く。本製品はモデルを再配布しないが、ライセンスは第三者へ渡すときに写しと NOTICE を求めている（I2）。写しを手元に置くのは安全側で費用はほぼ無い。NOTICE は作らない。
- **42（了承時に確認。裁定 39 の補足）**: サブマシンの T2V が **主メモリ 32GB に起因して**失敗した場合（ページファイルの不足など）も「構築と取得と UI まで」で合格とし、失敗の事実を記録簿に書く。記録上サブマシンの VRAM は 16GB（§82.4）なので、落ちるとすれば主メモリのほう。

## 3. 設計の決めごと（レビュー済み）

### 3.1 インストーラ

- **bat**: `install-UETrack.bat` を写し、id を `MiniMaxH3`・表題を H3 に。ASCII・CRLF。
- **`scripts/install_model.ps1`**: 記述子駆動なので受け付けは変更不要。日本語の案内文の 3 か所（「重みだけ追加」「1〜2 時間」「仮想環境を作る例外は UETrack だけ」）を現行化。H3 では**取得前にライセンスの注意文と URL を表示**（裁定 36。判定は記述子の `id` が `MiniMaxH3` のときに限る分岐で、記述子に印は足さない。文面は README のライセンス節と同じ要点）。失敗時の共通の案内に**「仮想環境の構築で失敗したときは何も消さずに再実行する」の 1 行を足す**（MISSING の行が出ない失敗のため）。`Test-SetupDone` は現状のまま。空き容量の見積りは現状の `files[].min` 合算のまま（仮想環境ぶんは README の必要容量に書く）。
- **`scripts/install_ltx.ps1`**: H3 の仮想環境を作る 3 回目の呼び出しを 1 ブロック足す（UETrack と同じ `if ($foundIds -contains 'MiniMaxH3') { Ensure-EngineVenv ... -IgnoreSkipVenv }`）。渡すもの: freeze `engine_h3/venv-engine-h3.freeze.txt`（`Ensure-EngineVenv` は freeze と直接ピンを受ける。`engine_h3-venv-pyproject.toml` は人間向けの仕様で渡さない）・`-IndexUrl https://download.pytorch.org/whl/cu130`・直接ピン 3 本（`torch==2.13.0+cu130`・`torchvision==0.28.0+cu130`・SageAttention の cu130 wheel の URL〔正本は pyproject の注記〕）・`-DirectPinArgs` は LTX 2.5 と同じ `--index`＋`--index-strategy unsafe-best-match`（`--no-sources` は付けない）。検証の表に `h3_python` と `engine_h3/worker.py` の 2 行を足す（UETrack 型）。英語コメントは ASCII のみ・BOM なし。
- **torch のタイムアウト対策**: 順番実行（仮想環境 → 重み）で第 1 弾の原因は起きない。**uv の環境変数は足さない**（`setup.bat` にも効いて「変えないもの」と食い違う）。uv の呼び出しを包む再試行も作らない。失敗したら案内文（上の 1 行）に従って再実行。
- **ドライバと GPU 世代の確認**: インストーラは GPU を見ない方針のまま。README の要件に書く。
- **記述子 `scripts/manifests/40-minimaxh3.json`**: パス・サイズ・`map` は I2 で検証済みなので変えない。`_note` を「検証済み（取得元のリビジョン: Comfy-Org `e5eb578`・MiniMaxAI `42ed227`）」に。裁定 41 が通れば `LICENSE` の 1 行を `downloads` に足す（`MiniMaxAI/MiniMax-H3` の `LICENSE` → `MiniMaxH3/LICENSE`。`files[].key` は `h3_license`。`INSTALLED_PATHS.txt` に 1 行増える）。リビジョンの固定は作らない。tokenizer の `include` は 1 ファイルずつのまま。
- **開発機の既存の仮想環境**: 第 1 弾で手で作った `.venv-engine-h3`（完了の印なし・torch はローカル wheel 由来・uv 0.6.4）は、門の前に**リポジトリの外** `S:\OriginalApps\12_Nz-LTX23-AviUtl2\_venv-engine-h3.stage1\` へ移して退避し、インストーラに新規構築させる（`.gitignore` の対象外に残さないため）。`config.yaml` の `engine_python_h3` は絶対パスで `.venv-engine-h3` を指すので影響なし。退避の削除はオーナー判断。
- **freeze の比較と取り直し**: インストーラが作った環境の `uv pip freeze` を第 1 弾の freeze と比べる。比べ方: `name==version` の集合で比べ、SageAttention の行は `name @ URL` の形で出るので名前と版だけを見る。段 (b) は freeze の版を `==` でそのまま入れるので版の差は起きない見込みで、差があるとすれば段 (a) が残す freeze に無いパッケージ。差があれば freeze を新しい結果に更新し、3-G2 の指紋で出力が変わらないことを確かめ、**更新後に `install-MiniMaxH3.bat` をもう一度実行して全部 SKIP になること（3-G1 の冪等性）を確かめ直す**（freeze のピンの行が変わると完了の印のハッシュが変わるため）。

### 3.2 README（I4 の A1・A2・B1〜B21・R-1〜R-5）

- **新設 2 か所**: §1 に「MiniMax H3 を追加する（`install-MiniMaxH3.bat`）」（「LTX 2.5 を追加する」の直後・同じ形。違い＝仮想環境もこのバッチが作る・取得元は Comfy-Org と MiniMaxAI の公開リポジトリ・ライセンスは 1 文で済ませライセンス節を指す・**更新（`git pull`＋`setup.bat`）では H3 の仮想環境は当て直されないので、freeze が変わったときは `install-MiniMaxH3.bat` をもう一度実行する**〔UETrack と同じ。更新手順の節にも 1 文〕）。§7 に「7.6 MiniMax H3（α版）を選んでいるときの制限」（アンカー `limit-h3`。目次の表に 1 行。7.1〜7.5 の番号は不変）: できること（T2V・先頭フレームの I2V・24fps・17n+5 で 22 以上・32 の倍数・ステップ数と shift の 3 本のスライダー）／できないこと（連結・Edit・LoRA・NAG・参照動画・末尾と中間のキーフレーム・A2V。操作パネルでは灰色か非表示）／快適上限は未較正でマーカーが出ない・大きな画角では VRAM が溢れて遅くなる（目安は書かない）／常駐は DiT と復号器と音声 VAE を主メモリに持つ（64GB 推奨）／生成時間の目安は書かない。
- **既存への追記（B1〜B21）**: ハードウェア要件の表（ストレージに H3 の行＝重み約 51GB＋仮想環境約 3.4GB／ドライバ R580 以上／GPU 世代は裁定 38 の文言／主メモリ「H3 は 64GB 推奨・32GB 機は対象外（未検証）」）・GPU 世代の段落・「必要な空き容量の内訳」の直後に「H3 は含まれない」の引用（見出しと「約 50GB」は不変）・仮想環境の列挙と表に `.venv-engine-h3`（数を書かない言い方へ）・系統の表に H3 の列・`models/` のツリーに `MiniMaxH3/`・物体追尾の節とアーキテクチャの図の「数」の言い方・操作パネルの Base model に H3・§5 冒頭に「H3 は §7.6」・worker 単体スモークに H3 の 1 行・高速化の表に「MiniMax H3 での扱い」の列・7.3 に推奨構成・7.4 に `logs/h3_worker.log`・7.5 は LTX の話と明示・§8 の 422 の項目に H3・ライセンス節に 1 段落（名称「MiniMax H3 COMMUNITY LICENSE AGREEMENT」・利用できる地域は EU・英国・韓国・米国を除く全世界・取得者の自己責任・本文の URL）。
- **R-1〜R-5**: 付録 A の凍結 API 契約は見出しを保ち、各行に【LTX の規則】の札と検証先 `services/engines/ltx/request_rules.py`、H3 の値は §7.6 を指す。R-4 は無視の項目に `shift_video`・`shift_audio` を足し、件数を書かず仕様書 §6.10(d) を指す。
- **既存の不変条件違反 2 件も直す**: 468 行目の日付・362 行目の人物主語。
- **3-G3 の結果による更新**: サブマシンの T2V が通ったら「RTX 30 は未確認」「32GB 機は未検証」の 2 文言が誤りになる。門の後に文言の案（例: 「RTX 30 系では 1 本の生成を確認」「主メモリ 32GB ではページファイルを前提に 640×384 の生成を確認。推奨は 64GB」）を示し、オーナーが決めてから直す（裁定 38 に関わる）。
- README は日付ゼロ・人物主語ゼロ・内部台帳へのリンクゼロ・数値は正本参照（容量の表だけ概数）。

### 3.3 文書（README 以外）

- 操作パネル側 `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/README.md` 4〜6 行（2 つに言い切っている箇所）と `webui/README.md` 18 行。
- 仕様書 `Videomni_Backend_Specification.md` §2.5（「現行は `install-LTX25.bat` の 1 本」）。`MULTI_ENGINE_DESIGN.md` §6.2 に H3 の導入バッチの 1 段落。設計正本 `MINIMAX_H3_DESIGN.md` §10（利用者側の条件は README に書いた、と現在形）・§11（第 3 弾の行を完了に）・§13（サブマシンのドライバの項を消す）。
- 記録 `VERIFICATION_LOG.md` §167（門の結果・freeze の比較・サブマシンの実測）。台帳 §1-93 を CLOSED 3-239 へ（α版の完結。§1-95・§1-97・§1-100〜§1-102 は残る）。研究ノート §10 に裁定 36〜42。事前調査 4 本と本計画とレビューを `Docs/Outputs-archive/h3-stage3-reports-2026-10-18/` に保管。

## 4. 作業項目と担当（1 ファイル 1 所有者）

- **C1（Opus）インストーラ**: `install-MiniMaxH3.bat`（新設）・`scripts/install_ltx.ps1`（H3 のブロック・検証の表）・`scripts/install_model.ps1`（文面・ライセンスの注意文・失敗時の案内）・`scripts/manifests/40-minimaxh3.json`（`_note`・LICENSE）。開発機の門 3-G1・3-G2 の実行と記録（`outputs/h3-gate3/`・複写 `Docs/Outputs-archive/h3-gate3/`）。
- **C2（Opus）README と文書**: `README.md`・操作パネル側 README 2 本・仕様書 §2.5・MULTI_ENGINE §6.2・設計正本 §10／§11／§13。
- **監督**: 旧環境の退避・aux2 の再ビルドと配置（第 2 弾の後に画面の文字列が変わっているため。配布用コピーのコミットはオーナー）・コミット（dev）・push・サブマシンの手順書・main merge。
- **文書係（Sonnet）**: 研究ノート §10・台帳・VERIFICATION_LOG §167・保管（門の結果が揃ってから）。

## 5. 門

| 門 | 内容 | どこで |
|---|---|---|
| **3-G1** | 開発機で `install-MiniMaxH3.bat`: 退避後に `.venv-engine-h3` が `tools\uv` で新規構築される（cu130 の index・直接ピン 3 本・SageAttention の wheel）。重みは既にあるので取得段は SKIP（裁定 41 の LICENSE だけは取りに行く）。検証の表の H3 の行（重み 5 行＋python＋worker。LICENSE 込みなら 6 行）が PASS。`INSTALLED_PATHS.txt` に H3 の行。完了の印。`uv pip freeze` と第 1 弾の freeze の差を §3.1 の比べ方で記録。**freeze を更新した場合はその後に**、2 回目の実行で全部 SKIP（冪等） | 開発機（GPU 不要） |
| **3-G2** | 新しい環境で H3 のテスト 86 件と、`--selftest-generate` の T2V 1 本（第 1 弾の G5 と同じ条件＝セルフテストの既定: 640×384・124f・20 ステップ・seed 42・sdpa・`deterministic` は既定）で指紋 `0da6111cdca853ce…` と一致。`list_backends()["cuda"]` に `int8_linear`・`probe_sage` が真 | 開発機（**GPU 1 本。本計画の了承で了承とみなす**） |
| **3-G3** | **サブマシン（オーナー）**: 事前に `nvidia-smi` でドライバが R580 以上と VRAM（記録上 16GB）、空き容量 60GB 以上、ページファイルが「システム管理」であることを確認。dev を pull → **`setup.bat` を再実行**（アプリ側の環境を揃える）→ **aux2 を AviUtl2 の Plugin フォルダへ置く**（リポジトリの配布用コピー `AviUtl2-Plugin/NzVideomni.aux2`。再ビルド済みのもの）→ `install-MiniMaxH3.bat`（実際の取得 約 51GB・仮想環境の構築）→ 検証の表が PASS → AviUtl2 を起動し H3 が「導入済み」で選べる → 常駐 off で T2V 640×384・124f を 1 本。記録: 所要時間・取得の再試行の有無・生成の秒数・主メモリ（タスクマネージャのコミット量と作業セット）・VRAM。失敗の扱いは裁定 39・42 | サブマシン（RTX 3080 Laptop 16GB・主メモリ 32GB） |
| **3-G4** | README の読み合わせ: 「初めて H3 を足す利用者」の視点のレビュー（Opus）で、手順どおりに進めて迷う箇所・矛盾・凍結ゾーンへの侵入・不変条件違反が無いこと。アプリ側のテスト全件（記述子の変更を見るテストを含む）と `test_h3_format` が通ること | 開発機 |

## 6. 進め方

```text
了承 → 監督: 旧環境を退避（リポジトリ外）・aux2 を再ビルドして配置
     → C1（インストーラ）と C2（README と文書）を並行
     → C1: 実装 → 3-G1 → 3-G2（GPU 1 本）→ freeze の比較と更新 →（更新したら）3-G1 の冪等の確かめ直し → コードレビュー（Opus）→ 修正
     → C2: 実装 → 3-G4 のレビュー（Opus）→ 修正
     → 監督: dev にコミット（(1) インストーラと記述子 (2) README と文書）→ push → オーナーへサブマシンの手順書（配布用 aux2 のコミットはオーナー）
     → 3-G3（オーナー・サブマシン）→ README の 2 文言の更新案 → オーナー判断 → 文書係（§167・台帳 CLOSED・裁定・保管）→ コミット (3) → main merge（オーナーの合図）
```

- GPU の了承: 本計画の了承が 3-G2 の 1 本を兼ねる。3-G3 はオーナー自身の実行。

## 7. 間違えやすい点（担当への申し送り）

- `install_ltx.ps1` は ASCII のみ・BOM なし（非 ASCII は PS 5.1 で化ける）。`install_model.ps1`・`setup.ps1` は UTF-8 BOM 付き。CRLF の ps1 を GNU `sed -i` で触らない（Edit か PowerShell）。
- `$ErrorActionPreference='Stop'` 下で `& hf.exe ... 2>&1` は NativeCommandError。`-File` に `-BaseModel a,b` は 1 文字列。
- `setup.ps1` は `Ensure-EngineVenv` を直接呼ばない（`install_ltx.ps1` が 2 回呼ぶ）。H3 の 3 回目は `install_ltx.ps1` に足す。`Ensure-EngineVenv` は freeze と直接ピンを受け、pyproject は受けない。
- SageAttention の wheel は依存を持たない。直接ピンの段で torch を引き込まないことを確かめる。
- `Get-EngineStateHash` は行頭コメントと CRLF を除いて計算する。freeze のピンの行を変えるとハッシュが変わり、再実行で当て直しになる（それが正しい）。
- tokenizer の `include` は 1 ファイルずつ（`FL2VA/text_encoder/*` は 65GB の bf16 テキストエンコーダを引き込む）。
- 開発機の重みは揃っているので取得段は SKIP になる（LICENSE を除く）。取得の経路を通す門はサブマシンだけ。
- README: 凍結ゾーン（1〜133 行）は 1 文字も触らない。§7 の番号を動かさない。見出し「必要な空き容量の内訳」と「約 50GB」の文字列は不変。日付・人物主語・台帳リンクを入れない。生成時間や VRAM の数値を写さない。
- `config.yaml` の `engine_python_h3` は絶対パス。退避は改名でなく移動（リポジトリ外へ）。

## 8. 危険と、分からない点

1. **uv 0.11.32 と第 1 弾の uv 0.6.4 で依存の解決結果が違う可能性**（freeze の差）。3-G2 の指紋で担保する。差が出て指紋も違えば止まって切り分ける。
2. **32GB 機の生成**: 作業セット約 22GiB・コミット量約 34.5GiB（第 2 弾の実測）。ページファイルが「システム管理」なら動く見込み。失敗の扱いは裁定 42。
3. **取得時間**: 約 51GB。回線次第で 1 時間前後〜数時間。`hf download` の再試行 5 回は既存。
4. **ライセンスの解釈**: 直接取得が「配布」に当たるかは未確認（I2）。裁定 41 の写しの取得は安全側の手当てで、法的判断ではない。
5. **README の 1 回の大きな改修**: 21 か所の追記で既存の文の整合が崩れやすい。3-G4 のレビューで「初めて足す利用者」の視点を取る。
6. **Ampere での初めての実測**: サブマシンで動かなかった場合、README の「RTX 30 は未確認」はそのままで、記録簿に事実を書く。

## 9. 敵対的レビューの結果と採否

レビュー（Opus・`plans/shimmering-twirling-octopus-agent-a58d2ac51b48d41ef.md`）: Critical 0・Major 8・Minor 11。

| 指摘 | 採否 | 反映 |
|---|---|---|
| M-1 冪等の確認は freeze 更新の後に | 採用 | §3.1・3-G1・§6 |
| M-2 freeze の比べ方（sage は `name @ URL`） | 採用 | §3.1 |
| M-3 退避先が `.gitignore` の対象外 | 採用（リポジトリ外へ移動） | §3.1・§7 |
| M-4 サブマシンの手順の抜け 3 つ | 採用 | 3-G3・§4・§6 |
| M-5 VRAM は 16GB・主メモリ起因の失敗の扱い | 採用（裁定 42 として提案） | §2・3-G3 |
| M-6 門の後の README の 2 文言 | 採用 | §3.2・§6 |
| M-7 更新手順の例外 | 採用 | §3.2 |
| M-8 失敗時の案内 | 採用 | §3.1 |
| m uv の環境変数は入れない／LICENSE は SKIP の例外／3-G2 の条件はセルフテストの既定／pyproject は渡さない ほか | 採用 | §1・§3.1・3-G1・3-G2・§7 |
