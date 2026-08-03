# IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）追加 ワークオーダー

> **【ステータス 2026-08-03】実装完了・機械検証（pytest・自己診断・型検査・vitest）全PASS・G1ゲート（深度前処理の性質チェック）全項目PASS・G2ゲート（モック通し）全項目PASS。コミット・プッシュ済み（backend `fd6d43f` / frontend `d375028`）。ただし G3（オーナー実機real）・G4（既存canny/pose/upscalerの回帰）は**未実施**であり、本テーマは**実機ゲート待ち**の状態である。デプロイも未実施。**「完了」ではない。**
>
> 併読: [`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)（制御系IC-LoRAと前処理段そのものの完成物。本テーマはその上に2アダプタを足したもの）／[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（LoRA適用機構）／フロントエンド台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-8（起票元）／機械検証と実測値の記録＝本リポジトリ[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49。

- 作成: 2026-08-03（実装と同時に起票）
- 正本ポインタ: 本テーマの仕様・設計判断＝**本書**／実測値と検証記録＝[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49／未対応アダプタの台帳＝フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-8

---

## 1. 背景

フロントエンド台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-8には、LTX 2.3の公式IC-LoRA（参照動画を手がかりに新しい動画を作るための追加学習アダプタ）のうち**未対応の4種**——depth（深度）・Motion-Track（動きの軌跡追従）・In-Outpainting（画面外の描き足し）・Deblur（ぼけ除去）——が残っていた。このうち本テーマで**DepthとDeblurの2種を製品化**する。

- **Depth** が塞がれていた理由は「**前処理器（深度推定モデル）の選定が収束していなかった**」ことである。公式スタッフのDepthCrafter言及と、調査側のVideo-Depth-Anything推奨とで意見が割れていた。今セッションで**公式ComfyUIワークフローのJSONを直接解読**し、公式実装もVideo-Depth-Anythingの**Small版**を使っていることを実値で確認した（§3）。これで選定が確定した。
- **Deblur** は前処理を一切必要としない。Lightricks公式のモデルカードに「ぼけた動画をそのまま参照として渡す」と明記されており、実際に前処理器の追加なしで通る（§4）。

なお`conditioning_attention_mask`（参照条件を画面の場所ごとに効かせ分けるマスク）のAPI露出は**本テーマのスコープ外**である。フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-25（AviUtl2側で作成したマスク動画のバックエンド接続）と相互参照する形で§4-8に残している。

計画は着手前にOpusによる敵対的レビューを2ラウンド通しており、「過剰設計の回避」「裏付けの有無」の2観点で指摘を取捨選択したうえで実装に入っている。

---

## 2. 確定方針（オーナー承認済み）

| # | 決定 | 理由 |
|---|---|---|
| 1 | 深度前処理は **Video-Depth-Anything Small（vits）**。ライセンスはApache-2.0 | 公式ComfyUIワークフローと同一の選択（§3）。Apache-2.0なので再ホスト・同梱に制約が無い |
| 2 | 公式の**窓方式推論関数（`infer_video_depth`）をそのまま呼ぶ** | 32フレームの移動窓・10フレームの重なり・窓どうしの整合処理が時間方向の安定性を生んでいる。簡易版を自作するとかえってコード量が増え、しかも安定性を失う |
| 3 | 深度の推奨値0.6は**ヒント文の表示のみ**（値の自動セットはしない） | オーナー決定。利用者が意図して選ぶ余地を残す |
| 4 | 推奨値0.6の対象は②`conditioning_attention_strength`（制御追従度）であり、③LoRAアダプタ強度は**1.0のまま**である旨をヒント文に明記 | この2つは別のつまみである。③を下げると参照が滲み込む（bleed-through）と公式が警告している。取り違えを防ぐため区別を文言で明示した。3つのノブの切り分けは[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §28.1で照合済み |
| 5 | API側の解像度バリデーションは**128の倍数のまま維持** | 既存の制御系アダプタと同じ制約であり、緩める理由が無い |
| 6 | 新規の重みは **`Rootport/Nz-LTX23-weights` へ再ホスト**する | Deblurの上流はgated（ライセンス同意の壁がある）ため、利用者にHuggingFaceアカウントを要求しない既存方針を守るには再ホストが必須 |

---

## 3. 公式ComfyUIワークフローの解読結果（一次資料）

Lightricks公式配布のワークフローファイル **`LTX-2.3_ICLoRA_Union_Control_Distilled.json`** をJSONとして直接解読し、深度前処理ノードのパラメータ実値を読み出した。以下は転記であり、推測は含まない。

| 項目 | 公式ワークフローの実値 | 本実装での採用 |
|---|---|---|
| モデル種別 | `vits`（Small） | 同一（`engine/preprocess/depth.py` の `_ENCODER = "vits"`） |
| `input_size` | `518` | 同一（`_INPUT_SIZE = 518`） |
| `max_res` | `960` | 同一（`_MAX_RES = 960`）。推論入力は長辺が960以下になるよう縮小し、出力は元解像度へ戻す |
| 精度 | `fp32`（autocast無効） | 同一（`_FP32 = True`） |
| 出力形式 | グレースケール（近＝白） | 同一。クリップ全体でのmin-max正規化を1回だけ行う |

**重要な留保**: 同ワークフローで実際に配線が完成しているのは**canny（輪郭抽出）の経路だけ**である。depthとposeのノードは置かれてはいるが「例示」の位置づけで、canny経路のように最後まで繋がってはいない。したがって上表は「**公式がdepthに使うと示している前処理器とそのパラメータ**」の一次資料としては有効だが、「公式が完成品として動作保証している深度経路」ではない。この区別は、公式実装との数値一致をG1ゲートの合格条件に採用しなかった理由でもある（§7.1）。

---

## 4. Deblurアダプタのメタデータ実測

公式 **`LTX-2.3-22b-IC-LoRA-Deblur`**（ファイル名 `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors`・906,071,437バイト）の safetensors ヘッダを実測した結果は次のとおり。

| 確認項目 | 実測結果 | 結論 |
|---|---|---|
| `reference_downscale_factor` キー | **実在する。値は `"1"`** | 既存のメタデータキーによる判定（`services/lora_registry.py`）がそのまま `kind=control`（制御系）へ自動分類する。**configスキーマの拡張は不要**と確定した |
| `reference_temporal_scale_factor` キー | **無し** | 時間方向の係数を適用できないwheelの制約に抵触しない。実装方針の再検討は不要 |

`reference_downscale_factor="1"` は「**参照を縮小せずに（縮小係数1で）条件付けに使う**」ことを意味する。既存のcanny/pose（union-control）とpixel-spatial-upscalerはいずれも2であり、参照を半分に縮めてから使う。Deblurが1であることは次の2つの帰結を持つ。

1. **エンジン側の縮小係数ガードを緩める必要があった**（§5.1）。従来は「1以下は拒否」という実装で、これは事実上「Deblurを弾く」ことを意味していた。
2. **VRAM消費が増える**。参照が縮まないぶん、stage-1（1段目の生成）の総トークン数が、**参照なしを1として、縮小係数2で1.25倍・係数1で2.0倍（既存の制御系比で約1.6倍）**になる。ヒント文にこの旨を明記した（実測はG3で行う。§7.1）。

**プロンプトの書式**は公式モデルカードに従い「2段構成」である。前半で参照動画の状態（ピンぼけしている）を述べ、`DEBLUR` という語を挟んで後半で望む結果（同じ場面がくっきりしている）を述べる。**対象はデフォーカスぼけ（ピンぼけ）専用**で、モーションブラー（被写体ぶれ・手ぶれ）には効かない。

---

## 5. 実装内容

### 5.1 エンジン — 縮小係数の全LoRA走査（`engine/pipeline/fast_video_pipeline.py`）

従来は「登録されたLoRAの**先頭1本だけ**からメタデータを読む」実装だった。これを**全LoRAを走査する**方式へ置き換えた。

wheel（固定しているLTXのパッケージ）が提供する読み取り関数は、「1と宣言されている」場合と「キー自体が存在しない」場合の**両方で1を返す**。したがってキーの有無を安全に区別するには、`safetensors` のヘッダを別途開いて `reference_downscale_factor` キーの存在そのものを確かめる必要がある。実装した規則は次の3つ。

- **キーを持つLoRAだけが投票する**。キーを持たないもの（画風LoRAなど）は投票せず、無視する。ヘッダを開けなかったLoRAも投票しない（wheelの読み取り関数も同じように劣化するうえ、レジストリがconfigの全エントリを登録時にヘッダ検証済みである）。
- **宣言値が2種類以上あったら矛盾としてエラー**にする。参照動画は1つの解像度で1回だけ読み込まれるので、異なる倍率を期待するアダプタを混ぜることはできない。**1と2の混在も矛盾**として扱う（片方のアダプタに、訓練時と違うスケールの参照を黙って食わせることになるため）。
- **どのLoRAもキーを持たなかったらエラー**にする。宣言のない状態で参照条件付けを走らせない。

あわせて `factor <= 1` を拒否していたガードを **`factor >= 1` 許容**へ緩め、下流にあった `assert` を**明示的な検証と例外送出**へ置き換えた（`python -O` で最適化実行するとassertは消えてしまうため）。割り切れ判定のロジックには手を触れていない（係数1では元々発火せず、動作上の変化が無いため）。チェーン側（`engine/pipeline/chain_pipeline.py`）は同じメソッドを共用しているので、独自の修正は不要である。

### 5.2 API — 新エラー `REFERENCE_REQUIRES_CONTROL_LORA`（`api/errors.py`・`api/generate.py`・`api/generate_chain.py`）

§5.1のガード緩和には副作用があった。従来は「参照動画＋画風LoRAだけ」という**誤用**を、この縮小係数ガードが偶然弾いていた（生成の途中で例外になっていた）。緩和するとこの網が消える。

そこで**API層に逆方向のチェックを新設**した。`reference_video_id` が指定されているのに制御系のLoRAが1本も無いリクエストを、**422**（リクエストの形式は正しいが、必要な相棒の入力が欠けている）で拒否する。**単発生成（`/generate`）とチェーン生成（`/generate/chain`）の両方**に入れてある。

これは既存の逆向きのチェック（制御系LoRAがあるのに参照動画が無い＝`LORA_REQUIRES_REFERENCE`）と対になるもので、フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-11の①として起票されていた「アップロードもジョブ開始も済んだあとで落ちるので、リクエスト時点で422にするほうが親切」という指摘を、本テーマの副産物として解消したことになる。

### 5.3 前処理 — Video-Depth-Anythingの取り込みと動画単位経路

**(a) `engine/preprocess/vda/`（ベンダリング）**

Video-Depth-Anythingの**推論コアだけ**をこのディレクトリへ取り込んだ。`vendor/` 配下は `.gitignore` の除外対象なので使えず、git追跡対象となる `engine/preprocess/vda/` を新設している。Apache-2.0のライセンス全文（`LICENSE`）と、**改変点の完全な一覧を含むREADME**（`README.md`）を同梱した。上流のディレクトリ構造（`video_depth_anything/` ＋ `utils/`）は将来の差分確認が読みやすいようそのまま保っている。チェックポイント（`.pth`）はgitに入れず、インストーラ経由で取得する。

クラス名・モジュール構成・属性名・コンストラクタ引数は**一切変えていない**（チェックポイントを `strict=True` で読み込むため、名前を変えると壊れる）。加えた改変は次の4点のみである。

1. **相対importへの変更**。`from utils.util import ...` というトップレベルimportは、`sys.path` 上にたまたま存在する別の `utils` パッケージへ黙って結びつく危険があったため `from ..utils.util import ...` にした。
2. **`easydict` の除去**。キーワード引数の入れ物としてしか使われていなかったので素の `dict` に置き換えた。エンジン用仮想環境に `easydict` は入っていない。
3. **SDPA（PyTorch標準の高速attention）への差し替え（2箇所）**。詳細は§6.1のG0スモークを参照。
4. **xformersのimportガードの削除（3ファイル）**。ガードが握っていた `except ImportError` 分岐は素の `print(...)` を実行するもので、ワーカーの標準出力はフレーム化されたプロトコル通信路であるため、そこへ文字列が漏れると通信が壊れる。削除後に残る経路は、xformersが無いときに上流が元々通っていた経路そのもの（＝G0スモークで実測した構成）である。

**(b) `engine/preprocess/depth.py`（`DepthProcessor`）**

公式の `infer_video_depth` を呼ぶ薄いラッパー。重いimport（ベンダリングしたVDA。torchvisionとDINOv2のスタックを引き込む）は `_ensure_loaded()` の中に**遅延**させてある。これはDWPose（`dwpose.py`）と同じ作法で、トップレベルでのimport失敗がcanny/poseまで道連れにするのを防ぐためである。ジョブごとにロードし、制御動画を書き終えたら `release()` でGPUから追い出す（16GBぎりぎりの生成デノイズ中に深度の重みを残さない）。

**(c) `engine/preprocess/base.py`・`driver.py`（動画単位経路と `frame_cap`）**

深度はフレームを1枚ずつ処理できない。時間方向の移動窓を使い、正規化もクリップ全体で1回だけ行うためである。そこで従来の `FrameProcessor`（フレーム単位）に加えて **`VideoProcessor`（クリップ単位）** のプロトコルを新設し、ドライバがどちらを持つかで分岐するようにした。デコード・エンコード・FPS保持・`release()` の呼び出しはどちらの分岐でも共通である。

さらに **`frame_cap`（デコードを打ち切るフレーム数）** をドライバに追加した。深度は与えられた分をすべて正規化に使うので、生成で使わないフレームまで処理すると時間を無駄にするうえ**グレーの割り当てレンジがずれる**。値は `engine/worker.py::_preprocess_frame_cap` が決め、**単発生成では `num_frames`、チェーン生成では `clips[0]["num_frames"]`** を使う（参照条件は先頭クリップのstage-1にしか付かないため）。

**canny/dwposeの経路はバイト単位で不変**である。`frame_cap` は深度のときだけ渡し、ログ行も `cap=` の部分は該当するときにしか追加しない（Phase Cで記録した実行ログと文字単位で一致し続ける）。

**(d) 登録**

- `config.yaml.example`: `depth-control`（**union-controlのファイルを共用**・`preprocess: depth`）と `deblur`（前処理不要なので**文字列形式**でパスのみ）の2エントリを追加。
- `config.py`: `IcLoraEntry.preprocess` のリテラル型に `"depth"` を追加。
- `engine/preprocess/driver.py`: `_FACTORIES` に `"depth"` を遅延importのラッパー経由で登録。
- `engine/preprocess/__init__.py`: `VideoProcessor` と `DepthProcessor` を再輸出。
- Gradioの静的フォールバック一覧（`/config` が届かなかったときだけ使われる死に枝）には手を触れていない。

### 5.4 UI

**Gradio（バックエンド同梱の検証用Web UI）**: `gradio_ui/adapters.py` の `ADAPTER_FRIENDLY` に `depth-control` と `deblur` の表示名を追加。`gradio_ui/i18n.py` に英日それぞれ3種のヒント文（`note_iclora_aspect` / `note_iclora_depth` / `note_iclora_deblur`）を追加し、`gradio_ui/ui.py` の参照動画欄に常時表示の注記として並べた。アダプタの選択に連動して出し分ける仕組みは既存に無く、そのためだけに新しいUI機構を足すことは避けた（既存の `note_ref128` と同じ扱いにしている）。

**AviUtl2フロントエンド（webui）**: 同じ内容を `webui/src/i18n/strings.ts` の英日へ追加し、`webui/src/modes/create/GenerationForm.tsx` の参照動画セクションに3行のヒントとして表示。`webui/src/api/types.ts` の `IcLoraEntry.preprocess` に `"depth"` を追加し、`webui/src/bridge/mockBridge.ts` のフィクスチャに `depth-control`（マップ形式）と `deblur`（文字列形式）の2件を足した。アダプタの選択肢そのものは `/config` 経由で自動的に増えるので、これ以外の実装は不要である。

**ヒント文の3種**は次の内容である。

1. **アスペクト比**: 参照動画は出力解像度へ単純にリサイズされるため、比が違うと映像が歪む（Depth controlで特に目立つ）。
2. **Depthの推奨値**: 公式推奨は制御追従度（`conditioning_attention_strength`）＝0.6。**アダプタ強度（LoRA強度）は1.0のまま**にすること。下げると参照が滲み込む。この2つは別のつまみである。
3. **Deblurの書式とVRAM**: プロンプトの2段構成の実例、デフォーカスぼけ専用でモーションブラーには効かないこと、参照を縮小せずに条件付けに使うためVRAM消費が他のアダプタより大きいこと。

### 5.5 配布 — 再ホストとインストーラ

**再ホスト先**: `Rootport/Nz-LTX23-weights`（既存の公開・非gatedリポジトリ）。追加したのは次の2ディレクトリで、SHA-256が上流と一致することと、匿名（ログイン無し）でダウンロードできることを検証済み。リポジトリのREADMEとNOTICEも更新済みである。

| ディレクトリ | 内容 | サイズ |
|---|---|---|
| `ltx-2.3-ic-lora-deblur/` | `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | 906,071,437バイト |
| `preprocessors-vda/` | `video_depth_anything_vits.pth` ＋ `LICENSE`（Apache-2.0の全文） | 116,452,112バイト（2ファイル計） |

VDAのチェックポイントに `LICENSE` を同梱しているのは、これが重みリポジトリ内の他ファイル（LTX-2 Community Licence）と**異なるライセンス**だからで、`.pth` と必ず一緒に運ばれる必要がある。

**`scripts/install_ltx.ps1`**: 独立したダウンロード呼び出しを2本追加し、検証表（`$required`）にも2行追加した（表は14項目から16項目になった）。

> **なぜ既存ディレクトリの子ではなく「兄弟ディレクトリ」に置くのか（重要）**
>
> インストーラのスキップ判定は、Checkに指定したディレクトリの**再帰的なサイズ合計**で行われる。もしDeblurの906MBを既存の `models/ltx-2.3-ic-lora/` の**中**に置くと、そのディレクトリの合計は1,308,930,638から2,215,002,075へ増える。すると**654MBのunion-controlファイルを失っているマシンでも合計1,560,536,723となり、既存のMin値1,000,000,000を上回ってダウンロードをスキップしてしまう**。以後は再実行するたびに「union-control MISSING」が出続け、しかも自力では直せないという恒久的な行き詰まりになる。これはGemmaのtokenizerで実際に起きた事故の**逆方向の再発**である（あちらは大きなファイルが不在の兄弟ディレクトリを覆い隠した。こちらは新しいファイルが不在の兄弟ファイルを覆い隠す）。**新しいCheckディレクトリは自分の中身だけで測られる**ので、この事故が構造的に起こりえない。
>
> 命名も僅差で助かっている。`models/preprocessors-vda` はDWPoseの `models/preprocessors` の**兄弟**であって子ではないため、互いのサイズ合計が混ざらない。もし `models/preprocessors/vda/` という名前にしていたら、その116MBがDWPoseの判定を水増しして、欠けた135MBの `dw-ll_ucoco` を覆い隠していた。ダウンロード対象を絞るglobも `preprocessors/*` は `preprocessors-vda/...` に一致しないため、こちらも混線しない。

Min値は既存の規則（「ディレクトリ合計 −（検証表が見る最小のファイル）」より大きく、ディレクトリ合計以下）に従って `ltx-2.3-ic-lora-deblur` を900,000,000、`preprocessors-vda` を110,000,000とした。後者は `LICENSE`（11,356バイト）が検証表の対象外＝欠けてもMISSINGにならないため、**LICENSEの有無で判定が変わらない**ようにこの値を選んである。疑似環境で3ケースの実測トレース（両方あり／LICENSEのみ欠損／`.pth` 欠損）と反例テストを行って確認済みである。

---

## 6. 検証状況

**済んでいるものと済んでいないものを取り違えないこと。** 実測値の詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49に記録した。

### 6.1 ✅ G0 前処理単体スモーク（実施済み）

エンジン用仮想環境（torch 2.9.1／numpy 2.4、xformersとdecordは無し）で、**追加のインストールを一切せずに動く**ことを確認した。ここで見つかった問題と手当てが次の2点である。

- **xformers不在時のフォールバックが致命的だった**。上流の実装は、xformersが無いときにN×Nのスコア行列を丸ごとメモリ上に作る。1920×1088で実測したところ **reserved 14.7GB** に達した。**SDPAへ差し替えると約4.4GBまで下がり、しかも速度も向上**した。数値の差はfp16の許容誤差の1/20以下であった。この結果を受けてSDPA化を**必須**として実装に組み込んだ（§5.3(a)の改変3）。
- **`easydict` はキーワード引数の入れ物としてしか使われていなかった**ので素の `dict` に置き換えた（改変2）。

実測性能: **VRAM 約4GB固定・1280×768で約15fps**。

### 6.2 ✅ G1 深度制御信号の性質ゲート（実施済み・全項目PASS）

`engine/preprocess/depth_g1_gate.py` を新設して実施した。**本番と同じ経路**（`get_processor("depth")` ＋ `driver.preprocess_video`）を1回走らせ、mp4へ書き出す直前のフレームを取り出して測る作りなので、測っているものは実ジョブが作るものと同一である。条件は実クリップ1280×720・41フレーム。

| 項目 | 内容 | 実測 | 閾値 | 判定 |
|---|---|---|---|---|
| (a) | 近い被写体が白になっているか（近／遠の2つの矩形を指定して平均輝度を比較） | 近 **184.07** ／ 遠 **15.49** | 近 > 遠＋余裕 | ✅ PASS |
| (b) | 静止画素のフリッカ（元動画がほとんど動いていない画素だけで測った、フレーム間の深度の揺れ） | 平均 **0.561** ／ 95パーセンタイル **0.693** | 2.0 | ✅ PASS |
| (c) | mp4書き出し往復の階調劣化 | PSNR **35.67dB** | 6.0 | ✅ PASS |

速度は41フレームを **4.35秒** で処理。

(c)の閾値6.0は、**既存のコーデック（`mp4v`。canny/poseが既に通っているもの）の実測から較正した値**である。この項目が捕まえるのはコーデック由来の一般的な劣化ではなく、**階調が崩壊する事態**である。

**G1が公式実装との数値比較を採らなかった理由**: 比較には別環境の構築が必要で、しかも§3のとおり公式ワークフローのdepth経路は配線が完成していない。最終的な「絵として正しいか」の判断は、既存の `outputs/visual_review/` 運用に従い、G3のオーナー目視に委ねる設計とした。

### 6.3 ✅ 自動テスト（実施済み）

| 対象 | 結果 | 着手前 |
|---|---|---|
| アプリ用仮想環境のpytest | **910 passed / 9 skipped** | 900 |
| エンジン用仮想環境（`test_ic_lora_engine_conditioning.py` ＋ forward系） | **28 passed** | 15 |
| `engine/preprocess/preprocess_selfcheck.py`（前処理の自己診断） | **11 / 11 PASS** | — |
| Gradio系のテスト | **277 緑** | — |
| フロントエンド vitest | **1673 緑** | — |
| フロントエンド `npm run typecheck`（`tsc -b`） | **0エラー** | — |

`preprocess_selfcheck.py` はpytestではなくスクリプトとして置いてある。エンジン用仮想環境には `fastapi` が無いので `tests/conftest.py` を読み込めず、アプリ用仮想環境には `cv2` も `torch` も無いためである（`block_swap_prefetch_selfcheck.py` と同じ切り分け）。GPUもモデルの重みも要らず、ドライバは偽のプロセッサで、`DepthProcessor` は純粋な配列変換ヘルパだけを叩く。

### 6.4 ✅ 再ホストの検証（実施済み）

上流とのSHA-256一致、匿名アクセスでのダウンロード成功、README・NOTICEの更新を確認済み（§5.5）。

### 6.5 ✅ G2 モック通し（実施済み・全項目PASS）

`backend: "mock"` の**隔離config**（scratchpadへコピー・**ポート18902**・出力先とアップロード先もscratchpadへ隔離）で**実uvicornを起動**し、HTTP経由で6項目を確認した。**リポジトリ側のファイルおよび `outputs/`・`uploads/` への書き込みはゼロ**である。実施方法は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §23.3／§34.5の前例に倣った。

| # | 内容 | 実測 | 判定 |
|---|---|---|---|
| 1 | `depth-control` ＋ 参照動画 ＋ 制御追従度 `0.6` の単発生成 | 202 → `completed`。`metadata.json` に `preprocess: depth`／参照ID／`0.6` が正記録 | ✅ PASS |
| 2 | `deblur` ＋ 参照動画の単発生成 | 202 → `completed` | ✅ PASS |
| 3 | 1クリップchain（Create画面のA2V相当） | 202 → `completed`（`num_frames=25`） | ✅ PASS |
| 4 | 2クリップchain ＋ 制御系 | **422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`**（既存仕様の維持） | ✅ PASS |
| 5 | 参照動画 ＋ 画風系LoRAのみ | **単発・chainの両方で 422 `REFERENCE_REQUIRES_CONTROL_LORA`** | ✅ PASS |
| 6 | `/config`・`/loras` への新2件の出現 | 両方に出現。**`deblur` は実メタデータ由来で `kind: control`** | ✅ PASS |

- **項目3の `num_frames=25` について**: 17で投げると422になるが、これは `overlap_frames=3` との**既存の境界バリデーション**が正しく働いた結果であり、**本改修のバグではない**。
- **項目4は既存仕様の維持確認である。** 複数クリップのチェーンで制御系を使えるようにすることは本テーマのスコープ外であり、**台帳には未起票**である（フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-7は合成制御信号の別テーマで、この件ではない）。
- **項目5は§5.2の新設チェックが実サーバー上で効いていることの確認**であり、同時に[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §34.6の既知事項#2（参照＋画風系のみがワーカー内 `RuntimeError` になる穴。早期422化が将来課題として残されていた）を**解消した**ことの実機確認でもある。
- **`control_depth.mp4` の実生成確認はG2の対象外**である。**モックのエンジンは設計上、前処理を実行しない**（`_MockBackend` は `lora_paths`／`reference_video_path` を受理はするが無視する）。深度前処理が実際に走って制御動画が書き出されることの確認は**G3の対象**として残る（前処理そのものの性質は§6.2のG1で実測済み）。

詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49.7。

### 6.6 ⬜ 未実施のもの

**G3（オーナー実機real）・G4（既存canny/pose/upscalerの回帰）は未実施である。** デプロイも行っていない。定義は§7に置く。

---

## 7. 残ゲートの定義

**G2は2026-08-03に実施し全項目PASSした**（実測は§6.5）。**残るのはG3とG4の2つである。**

### 7.1 ⬜ G3 オーナー実機（real環境・実GPU）

**本テーマの本命ゲートである。** 以下の4点を確認する。**深度前処理が実際に走って `control_depth.mp4` が書き出されることの確認も、ここが初出となる**（G2はモックのため前処理が実行されない。§6.5末尾）。

1. **`depth-control` の通常経路**を1本以上。参照動画をアップロードして生成し、深度が制御として効いていること（元の動きや構図を保ったまま内容が置き換わること）を目視で確認する。制御追従度（`conditioning_attention_strength`）は公式推奨の0.6を出発点にする。**出力ディレクトリに `control_depth.mp4` が生成されていること**もあわせて確認する。
2. **`depth-control` のA2V経由**を1本以上。音声から動画を作る経路でも同じ前処理が通ること。
3. **`deblur`** を1本以上。ぼけた動画（**デフォーカスぼけ**であること。モーションブラーでは効かない）を参照に渡し、§4のプロンプト2段構成でくっきりした出力が得られること。
4. **DeblurのVRAM実測**。`reference_downscale_factor=1` のため、stage-1の総トークン数は**参照なしを1として、縮小係数2で1.25倍・係数1で2.0倍（既存の制御系比で約1.6倍）**になる。実測値を`config.yaml`の`limits.spill_free_frames`（根拠[`RESOLUTION_DURATION_CAPABILITY.md`](RESOLUTION_DURATION_CAPABILITY.md) §8.4）と突き合わせ、**ヒント文のVRAMに関する記述を確定させる**（現在は「他のアダプタよりVRAM消費が大きくなります」という具体値のない固定文言にしてある）。表をdeblur向けに分けるかは実測後判断。

- **残タスク**: UIヒント文の「出力と同じ解像度で処理」はG3のVRAM実測後の文言確定時に「縮小せずに条件付けに使う」へ修正する（`gradio_ui/i18n.py`・webui `strings.ts` の両方。webuiは再ビルド・再デプロイ要）。

### 7.2 ⬜ G4 既存アダプタの回帰

`canny-control`・`pose-control`・`pixel-spatial-upscaler-x2` が本改修前と同じに動くこと。**縮小係数2の経路がバイト単位で不変**であることを確認する。§5.1でメタデータの読み取り方を全走査に変え、§5.3で `frame_cap` をドライバに足しているため、これらが既存経路に影響していないことを実測で押さえる必要がある。

- **実施者**: G3と同じ実機セッションで行う（別立てのセッションにはしない）。
- **比較基準**: 改修前コミット `24c67fd` の同一seed・同一パラメータ出力とのbyte一致。
- **前例**: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §22のG1回帰（Phase Cで同じ形の回帰を通している）。

---

## 8. 実装ファイル一覧

バックエンドリポジトリ（`git status` で確認できる）。

**変更**: `api/errors.py`（新エラー）／`api/generate.py`・`api/generate_chain.py`（逆方向チェック）／`config.py`（`"depth"` リテラル）／`config.yaml.example`（2エントリ登録）／`engine/pipeline/fast_video_pipeline.py`（全LoRA走査・ガード緩和・assert撤去）／`engine/preprocess/__init__.py`・`base.py`・`driver.py`（`VideoProcessor` と `frame_cap`）／`engine/worker.py`（`_preprocess_frame_cap` と配線）／`gradio_ui/adapters.py`・`i18n.py`・`ui.py`（表示名とヒント文）／`scripts/install_ltx.ps1`（ダウンロード2本と検証表2行）／`services/lora_registry.py`（型注釈）／`tests/test_chain_reference.py`・`tests/test_ic_lora_api.py`・`tests/test_ic_lora_engine_conditioning.py`

**新規**: `engine/preprocess/depth.py`／`engine/preprocess/depth_g1_gate.py`／`engine/preprocess/preprocess_selfcheck.py`／`engine/preprocess/vda/`（ベンダリング一式＋`LICENSE`＋改変一覧つき`README.md`）／`tests/test_preprocess_depth.py`

フロントエンドリポジトリ（`Nz-LTX23-frontend-AviUtl2`）: `webui/src/api/types.ts`／`webui/src/bridge/mockBridge.ts`／`webui/src/i18n/strings.ts`／`webui/src/modes/create/GenerationForm.tsx`。詳細は[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §59。

---

## 9. 参照

- [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) **§49**（本テーマの検証記録・実測値の正本）／**§28.1**（制御追従度の3つのノブの切り分け。深度0.6推奨の出典もここで照合済み）／**§22**（Phase Cの全ゲート詳細）
- [`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)（制御系IC-LoRAと前処理段の完成物。本テーマの土台）
- [`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md)（利用者向けのIC-LoRA系統解説）
- [`../LTX23_Backend_Specification.md`](../LTX23_Backend_Specification.md) §5.1b・§11.2（重みの配置と `ic_loras` の登録表）
- フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§4-8**（未対応IC-LoRAの台帳。本テーマの起票元）／**§4-11**（参照動画まわりの早期バリデーション。①を本テーマで解消）／**§4-25**（マスク動画の受け口。`conditioning_attention_mask` の相互参照先）
- 上流: [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)（Apache-2.0・Bytedance Ltd.）／Lightricks `LTX-2.3-22b-IC-LoRA-Deblur`（モデルカード）／公式ワークフロー `LTX-2.3_ICLoRA_Union_Control_Distilled.json`
