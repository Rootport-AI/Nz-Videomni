# MiniMax H3 事前調査ノート（参考資料）

## 本書の位置づけ

**2026-09-06に実施した、MiniMax H3を第3のベースモデルとして追加するための事前調査の記録である。設計資料ではなく参考資料として収蔵する。**

- マルチエンジン機構（ドロップダウンによるベースモデル切り替え）の設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)である。MiniMax H3固有の設計正本は本書執筆時点では未執筆であり、執筆後に本書と記述が食い違った場合は設計正本を採ること。
- 台帳（[`PENDING_TASKS.md`](PENDING_TASKS.md)）への起票は未実施（設計段で行う）。
- 調査の内訳: WEB調査4本（公式情報・ComfyUI事例・配布地域制限・実装ベース比較）＋配布ファイルのヘッダ実測（GGUF 13本・safetensors 7本をHTTP Rangeリクエストで部分取得して解析）＋当リポジトリの受け皿の棚卸し。数値・状況はすべて2026-09-06時点のものである。
- 本書内では「一次資料で確認済み」「実測」「二次情報」「推測」を区別して記す。

---

## 結論

MiniMax H3の追加は実現可能である。VRAM 16GBでの動作は、diffusers公式ドキュメントにレシピが実在し、16GBカード（RTX 5070 Ti）での一次実測（VRAMピーク14.5〜15.8GiB）も複数存在する。

有望な経路は「**コミュニティ配布のpruned GGUF Q4（DiT本体約11.6GB・VRAM常駐可）を、manifestから直接ダウンロードし、DiffSynth-Studioの実装を土台に専用ワーカーへ移植する**」形である。事前に懸念されていた「K量子化の罠」（Q4_K_M表記の実体がQ4_0である疑い）は、狙っているpruned版についてはヘッダ実測で**否定**された。

一方で、前提となる工事が1つ判明した。コミュニティGGUFはKVメタデータ（`general.architecture`／`model_version`）が全滅しており（偽装・欠落）、既存の2キー契約では系統判別ができない。**テンソル名の指紋による判別への契約拡張が必須**になる。

最大の未解決課題はテキストエンコーダ（Qwen3-VL-32B）の載せ方で、3案（§8）の実測比較が必要である。

---

## §1 モデルの正体（一次資料で確認済み）

| 項目 | 内容 |
|---|---|
| 正式名称 | MiniMax H3（構成名 H3-Omni-Transformer / H3-Base。「Hailuo 3.0」は二次情報側の通称で公式表記ではない） |
| 発表・公開 | 2026-07-31発表（公式ブログ）、2026-08-03重み公開 |
| 配布 | オープンウェイト（`MiniMaxAI/MiniMax-H3`）とAPI提供の併存 |
| 方式 | 拡散（rectified flow）。自己回帰ではない |
| 最大の特徴 | **映像と音声（32kHzステレオ。台詞・劇伴・効果音・環境音）を、1本のパックされたシーケンスとして同一トランスフォーマで同時にデノイズする** |
| 蒸留 | **ガイダンス蒸留済み**。negative promptもguidance_scaleも存在せず、1ステップ=1フォワードパス |
| スケジューラ | rectified-flow Euler。映像用shift=12.0と音声用shift=3.0の**2本立て**を1回のトランスフォーマ呼び出しの中で進める |

公開されていないもの（重要）: 公式が「強く推奨」する前処理 **H3-Context-IR**（長大な文脈を4Kトークンに蒸留するキャプショニング系）と、2K化モジュール **H3-Regenerate-2K** は未公開。**ローカル実行は短辺768px止まりで、ホストAPI版とは品質差がある。**

## §2 生成仕様（一次資料で確認済み）

- 解像度: 短辺768pxが既定（学習canvasは1344×768）。縦横は**32の倍数**。アスペクトは21:9/16:9/4:3/1:1/3:4/9:16。
- 尺: 4〜15秒（diffusers実装上は5〜15秒）。**24fps固定**（他の値は上書きされる）。
- **フレーム規約: `17n+5`**（フレーム数は17n+5グリッドへ切り上げスナップ、最小5）。LTXの`8n+1`とは異なる。diffusers・DiffSynth（`time_division_factor=17, remainder=5`）・stable-diffusion.cppの3実装で一致確認済み。
- タスク:
  - **T2VA**: テキストのみ→映像+音声（txt2vid。文章から動画生成）
  - **FL2VA**: 先頭/末尾キーフレーム0〜2枚→映像+音声（img2vid。画像から動画生成を包含。**T2VAと同一重み**）
  - **Ref2VA**: 順序付き参照（画像≤9枚・動画≤3本・音声≤3本、計12点）→生成。**別重み**（`transformer_ref/`）
- 既定ステップ数20。Turbo蒸留LoRA使用時はT2V/I2Vで8ステップ、R2Vで4ステップ。

## §3 構成コンポーネントと配布物

公式配布（bf16、`MiniMaxAI/MiniMax-H3`、リポジトリ全体498GB）:

| コンポーネント | 中身 | サイズ |
|---|---|---|
| `transformer/` | T2VA・FL2VA用DiT本体。33Bパラメータ（うち**約13BがAdaLN変調枝**） | 66.3GB |
| `transformer_ref/` | Ref2VA用DiT本体 | 同規模 |
| `text_encoder/` | **Qwen3-VL-32B**。実際に使うのは**第50デコーダ層の非正規化隠れ状態**のみ（LM head未使用） | 66.7GB |
| `vae/` | H3-VisualVAE（空間16×・時間4×圧縮・潜在24ch） | 約10.4GB |
| `audio_vae/` | H3-AudioVAE（ステレオ） | 605MB |

ComfyUI公式再パッケージ（`Comfy-Org/MiniMax-H3`）は、**AdaLN枝の刈り取り（pruning）＋int8量子化でフル123.6GB→42.5GBに約66%削減**している。

**pruningの正体（実測で構造分解済み）**: timestepはスカラーなので「time_embedder→AdaLN射影」の写像は本質的に1次元曲線である。これを1,025点グリッドで先計算し、ランク8で因子分解したものが `adaln_t_table [1025, 8]`（F32）×`adaln_proj [96768, 8]` である。削減量は13.0Bパラメータ（全体の39.3%）。品質を落とす一般的な枝刈りとは意味が違い、「機能等価」の主張は構造的に妥当（原典の自己申告で相対RMS誤差1.45e-5。Comfy-Org開発者評は「同一シードで絵は変わるが優劣はない」）。ただし**ビット単位の同一ではない**。独立した第三者の定量比較は見つかっていない。

16GB向け最小構成の目安（ファイルサイズはHugging Face実測）:

```text
DiT: Abiray/MiniMax-H3-Pruned-GGUF の FL2VA Q4_K_M   11.56GB（VRAM常駐可）
テキストエンコーダ: §8の3案から選定                     7.9〜19.8GB
映像VAE + 音声VAE                                     数GB（配布バリアントにより変動）
合計                                                  おおむね30GB台
```

## §4 ライセンスと配布方針

**MiniMax H3 Community License Agreement**（Apache 2.0ではない。一次資料で確認済み）:

- **適用地域から米国・EU・英国・韓国を除外**。日本は除外対象に含まれず、日本国内での使用・改変は適用地域内。
- 派生物（量子化版を含む）の作成・配布は明示的に許諾。配布時の義務は①本契約の写しの提供②改変表示③NOTICE文の添付。
- 商用製品はUI上に「MiniMax H3」の目立つ表示。年間収益2,000万米ドル超は別途書面許諾。

**配布プラットフォーム側で地域条項を技術的に強制する手段は実質存在しない**（第2次調査で確認）。Hugging Faceの無償機能はEU限定IPブロックのみ（米英韓は対象外）で、任意国ブロックは有料のEnterprise Plus契約かつ組織全体一括適用。CivitAIには地域制限機能自体が見当たらない。**MiniMax本家自身がHF上で技術ブロックを一切使っておらず**（gated repoすら不使用、ライセンス文中の許諾申請リンクのみ）、コミュニティの量子化再配布勢も同様。業界の実務は「ライセンス文の明記＋ダウンロードする側の自己責任」で統一されている。

この実態を踏まえ、当プロジェクトは**自前の再ホストを行わず、manifestからコミュニティ/Comfy-Org公式のHFリポジトリを直接ダウンロードする**方式を採る（オーナー裁定、§10）。必要な配布元はすべてHF上にあり、既存の`hf download`駆動インストーラ（[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §4の記述子機構）がそのまま使える。

なお、LTX 2.3で自前GGUF変換を始めた経緯は「コミュニティGGUFが動かなかったから」ではない（文書裏取り済み）。公式ベースモデルのGGUFは現在もコミュニティ配布物（QuantStack/LTX-2.3-GGUF）を直用しており、自前変換の真の理由は「使いたいファインチューン版にGGUF版が存在しなかった」＋「Pythonエコシステムに当時K量子化の書き込み実装が無かった」である。したがってコミュニティGGUF直参照の路線に過去の経緯からの障害はない。

## §5 VRAM 16GBでの動作実態

### 公式レシピ

- diffusers公式ドキュメントに12〜16GB向けレシピが実在（int8量子化＋グループオフロード＋映像VAEもオフロード＋960×544等の小キャンバス）。ただし但し書きとして「**重みはホストRAMに住む。int8で約75GB**」。この数値はdiffusers方式（フルのテキストエンコーダ62GBをRAMに持つ）に固有であり、pruned＋逐次ロードの自前ワーカーでは大きく下回る見込み（推測）。
- ComfyUI公式ブログは「RTX 3060級のコンシューマ機でも動作可能」と明記。

### 16GBカードの一次実測（いずれもBlackwell世代。Linux）

- RTX 5070 Ti 16GB: pruned int8構成でVRAMピーク**14,471〜15,758MiB**。640×384・226フレーム・20ステップで**170秒**。参照映像をフル解像度のまま渡すと1,027秒に悪化（6倍）し、8GB制限下ではOOM——**参照素材の解像度が最大の地雷**。
- RTX 3090 24GB: 832×480・362フレーム（15秒）で23分17秒・VRAMピーク約19.8GB。サンプリングが実時間の約95%を占め、フレーム数に対し超線形（226→362フレームでステップ単価5.6倍）。

**注意: Ada世代（RTX 4070 Ti SUPER級）とAmpere 16GB級の一次実測は、調査時点で世の中に存在しない。** 開発機・サブマシンでの初回実測は当プロジェクトが自前で行うことになる。

### 世代別の量子化カーネル対応

| 形式 | Blackwell | Ada (sm_89) | Ampere (sm_86) |
|---|---|---|---|
| NVFP4 | ネイティブ | エミュレーション（**速度利得ゼロ**） | エミュレーション |
| INT8 ConvRot | ネイティブ | ネイティブ（CUDA 13系＋comfy-kitchen前提） | 動作可 |
| FP8 | ネイティブ | ネイティブ | **非対応** |
| GGUF（逆量子化方式） | 可 | 可 | 可 |

開発機（Ada）とサブマシン（Ampere）の両方で成立するのは**GGUF系（自前の逆量子化カーネルで処理）またはINT8/W4A4系**。FP8とNVFP4は選ばない。

### 運用上の落とし穴（一次実測・公式issue由来）

- **ページロックメモリの無効化が必須**: ComfyUIは待機中の重みをページロックされたシステムRAM（スワップ不可領域）に置くため、RAM 31GB機でOOM killの実例。`--disable-pinned-memory`相当でホストRAM消費が29.9GB→7.5GBに落ちた実測あり。自前ワーカーの設計でも同じ罠を踏まないこと。
- **Windows固有**: ComfyUI issue #15488（未修正）——RAM 64GBが見える状態＋comfy-kitchen量子化ストリーミングの組み合わせでGPUがPCIeバスから消える（TDR）。回避策はOSから見えるRAMを32GBに制限するか、ストリーミングを使わないこと。**当プロジェクトはcomfy-kitchenを使わずDiT常駐方式を狙うため、この問題の発生条件自体を避ける方向**（推測を含む）。
- int8ストリーミング構成では1生成あたり**ディスク読み270GiB**の実測あり（NVMe必須）。DiTがVRAM常駐できるpruned Q4構成なら毎ステップのストリーミングが不要になり、この負荷はロード時に限られる見込み（推測）。
- H3の**バンディング（縞状の色むら）報告はVAEのタイル処理のアルゴリズム問題**（ComfyUI issue #15416。「dtype非依存＝精度でなくアルゴリズム」と分析されている）であり、量子化由来ではない。VAEタイリングを自前実装する際の要注意点。

## §6 量子化配布物の実物検分（ヘッダ実測）

GGUF 13本・safetensors 7本のヘッダをHTTP Rangeリクエストで取得・解析した（ファイル全体はダウンロードしていない）。

### 「K量子化の罠」の真偽——prunedか否かで反転する

事前のWEB調査に「H3のデノイザは幅2,688が256で割り切れないため真のK量子化が使えず、Q4_K_M表記の実体はQ4_0/Q4_1」という二次情報があった。実測の結果:

- **隠れ幅は5,376**（=21×256、割り切れる）。2,688の正体は**time_embed_dim**（AdaLN射影の入力次元）で、256で割り切れないのはAdaLN射影テンソル1種類だけ。
- **pruned版（狙っているファイル）では仮説は誤り**: Abiray pruned Q4_K_Mは重量テンソル208本すべてが**本物のQ4_K**（ggml型12）。pruningがAdaLN射影を[96768, 8]に畳むため障害自体が消滅し、8幅の小テンソルと`adaln_t_table`はF32温存。ファイルサイズの独立計算とも一致。
- **非pruned版では仮説どおりの実例あり**: leejet非pruned Q4_K_Mは2,688幅の50本だけQ4_0に落ちている。Abiray非prunedは当該テンソルを[256, 1016064]にリシェイプして強引にQ4_K化（スーパーブロックが行境界をまたぐ理論上の不利あり）。
- `_S`/`_M`サフィックスはH3のDiTでは実質無意味（llama-quantizeの混合ルールがLLM用テンソル名にマッチせず不発、という推測。Abiray prunedのS/Mはバイト単位で同一）。

### KVメタデータ契約はコミュニティGGUFで成立しない（最重要）

当プロジェクトの2キー契約（`general.architecture`＋`model_version`、[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §2）に対し、実測13本の状況:

- `model_version`: **13本すべてに存在しない**（自前変換GGUFだけの独自規約であることが確定）。
- `general.architecture`: **`'wan'`偽装**（Abiray・vantagewithai。city96ローダーのアーキテクチャ・ホワイトリストを通過するための偽装であることをPR #476が明言）／`'minimax_h3'`（joeygambino）／**キー自体が無い**（unsloth・leejet、KV数0）——と4通りに割れている。2キー契約をそのまま使うと**H3をWanと誤認する**。

**代わりに使うべきはテンソル名の指紋**である。ComfyUI本体（`model_detection.py`）・stable-diffusion.cpp・leejetフォークの3実装がすべてこの方式で一致している:

- H3系統の判定: `video_patch_proj.weight` **かつ** `audio_patch_proj.weight` の同時存在（映像+音声同時生成モデル固有の組で、既存系統と衝突しない）
- pruned/非prunedの判定: `adaln_t_table` の有無
- 形状情報: KVではなくテンソル形状から導出（`video_patch_proj.weight`のshape[0]=hidden_size等）

これは[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §2.4が予告していた「ComfyUI方式のヘッダキー指紋」のGGUF版に相当し、**H3対応の可否と独立に、コミュニティGGUFを受け入れるなら必ず発生する契約拡張**である。

### その他の実測事実

- **FL2VAとRef2VAはヘッダで区別できない**（532本のテンソル名・形状・サイズが完全一致）。manifest側でファイル名を固定して縛るしかない。配布元ごとにファイル命名もバラバラ。
- テキストエンコーダのGGUF（joeygambino Q4_K_M=19.8GB）は素のllama.cpp製（`'qwen3vl'`、本物のQ4_K+Q6_K混合、トークナイザ完備）。ただし**64層フルのQwen3-VL-32Bで、H3が使う50層に対し14層分の死荷重つき**。
- Comfy-Org公式repackのsafetensorsは、`__metadata__`（configつき）が非pruned bf16版にしか無く、pruned系は空。safetensors直読みでもテンソル名指紋が必要。int8_convrot版は`comfy_quant`独自記述子＋回転量子化（ConvRot）で、直読みには逆回転の自前実装が要る。
- GGUFローダー事情: city96/ComfyUI-GGUF本家は2026-01以降実質停止でH3**非対応**（issue #471オープンのまま）。読めるのはleejetフォーク等。ComfyUI本体はGGUF非対応（ネイティブ対応はsafetensors経路のみ）。
- 品質情報: Q4量子化のH3固有の統制された品質比較は**存在しない**（唯一の定量報告は本人が撤回）。先行実装からの落とし穴情報として「**`adaln_t_table`はF32のまま保つこと。F16化すると1ステップ目でクラッシュする**」という報告あり。

## §7 実装ベース3候補の比較

| 観点 | A: diffusers（公式） | B: ComfyUI本体 | C: DiffSynth-Studio |
|---|---|---|---|
| ライセンス | Apache-2.0 | **GPL-3.0** | Apache-2.0 |
| H3固有コード量 | 約10,000行（Modular Diffusers枠組みに全面依存） | 約2,600行（ただしコア4本・計約390KBに絡みつき） | **約3,000行・自己完結** |
| pruned形式の読み込み | **非対応** | 対応 | **対応**（78行のアダプタ。Comfy-Org配布のpruned safetensorsを直接ロードする実例が公式リポジトリにある） |
| GGUF | 構造的に不可 | 本家非対応（フォーク頼み） | 非対応（NF4/FP8/INT8のみ） |
| テキストエンコーダ | フル62.1GBをロード（50層目だけ読むのに全層を持つ） | 自前実装・50層切り詰め済み | **transformersを`num_hidden_layers=50`で構築**（必要な50層だけ確保） |
| 省VRAM機構 | group offload（ホストRAM約75GB前提） | 動的オフロード（コアと不可分） | `AutoTorchModule`（4段のdtype/device指定＋ディスクオフロード。約25KBで切り出しやすい） |
| torch要件 | ≥2.6 | 2.7以上（最適化パスはcu130必須） | **≥2.0** |

**推奨はC（DiffSynth-Studio）**。pruned形式を読める唯一の非GPL実装であり、テキストエンコーダを50層だけ構築し、コード量最小・オフロード自己完結・torch要件が緩い。弱点のGGUF非対応は、当プロジェクトが既に持つ融合GGUF逆量子化カーネルの文化と接続して埋める（`adaln_t_table`のF32厳守に注意）。

- **A（diffusers）はMiniMax公式が指名するリファレンス実装**なので、検証ゲートの数値オラクル（正解データ源）として使う。
- **B（ComfyUI）とKJNodesはGPL-3.0のため、コードの移植は行わない**。仕様理解のための読解に限る（特にテキストエンコーダのトークナイズ規約とテンソル名指紋の実装は仕様書として価値が高い）。
- torch整合: 既存2エンジンはtorch 2.9.1+cu128（freeze実測）。DiffSynth要件（≥2.0）を満たし、**「全エンジンでtorchを揃える」方針は維持できる**。Qwen3-VLはtransformers 4.57.0以降（既存venvは4.57.6/5.14.1でいずれも充足）。
- ComfyUIの16GB実測が使うSageAttention 2.2.0は既存2エンジンと同一wheelで、当プロジェクトに導入済み。

## §8 テキストエンコーダの載せ方（未裁定・3案）

16GB成否を分ける最大の設計課題。DiT（約11.6GB）とテキストエンコーダは16GBに同居できないため、**逐次ロード（エンコード→解放→DiT）が必須**（同時1ジョブ思想と整合）。エンコード段は16GBの境界ぎりぎりという理論試算あり（Windowsでは予約分により0.09GiB超過という未検証の試算）。

| 案 | サイズ | 利点 | 懸念 |
|---|---|---|---|
| ① GGUF Q2_K | 7.91GiB | 16GBに余裕で収まる。公式エコシステム索引の12〜16GB推奨構成 | 2bit級の品質が未知（統制比較なし） |
| ② GGUF Q4_K_M | 19.8GB | 品質面で無難 | VRAMに収まらず**層オフロードの新設が必要**（LTX 2.3の`--te-offload`=GGUF Gemmaの逐次per-layer CPUオフロードという同型の前例が当プロジェクトにある）。64層フルで14層死荷重 |
| ③ NF4（DiffSynth配布） | 15.32GB | 50層切り詰め済み・DiffSynthでそのまま動く実績 | 16GBぎりぎり。**bitsandbytesのWindows動作が未検証** |

次段で実測比較して決める（§11）。

## §9 当プロジェクトの受け皿との噛み合わせ（棚卸し結果の要約）

詳細は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)の該当節を参照。ここでは結論のみ記す。

- H3は同設計書§3.3の「依存関係やコードが同居できない系統」パターン（LTX 2.5と同型）に該当し、**専用ワーカー・専用仮想環境・専用アダプタを新設**する。レジストリ（`services/engines/__init__.py`の3辞書）・`config.py`のエンジン別インタプリタ設定・`install_ltx.ps1`の仮想環境構築呼び出し・記述子（manifest）＋`install-<ID>.bat`が追加箇所。**`install_model.ps1`とフロントエンドのドロップダウン／`baseModelInstaller(id)`はmanifest追加だけで自動追随する**（既存設計の狙いどおり）。
- **フレーム規約`17n+5`への対応が本丸**。LTXの`8n+1`は`api/models.py`に独立リテラル5箇所＋フロントエンド`mockBridge.ts`にも散在しており（同設計書§7の予告どおり）、エンジン別パラメータ制約の整理が必要になる。24fps固定・尺4〜15秒・32の倍数解像度も同様にエンジン別制約。
- H3は**3例目**であり、同設計書§5.2が予告する「汎用EngineRunnerとアダプタ分割の要否を再判断するタイミング」に当たる。
- `comfort_budgets`（快適上限の配信テーブル）は、新系統は較正が済むまで行を持たせず、`outpaint_budget`は`None`のまま（裁定J1「前例をコピーしない」、`config.py`のコメントに明記済み）。
- ガイダンス蒸留済み（negative promptなし）・音声同時生成という性質は、UI表示テーブル（`featureScope.ts`）とエンジン別制約の設計に反映が必要。
- mockバックエンドはLTXの出力形式・窓計算を模しているため`ltx`アダプタ内にあり、真に別規約のH3では改めて判断が必要（同設計書§9.1の注記どおり）。

## §10 オーナー裁定の記録（2026-09-06）

H3固有の設計正本が未執筆のため、暫定的に本書へ記録する。**設計正本の執筆後はそちらへ移管し、本書の本節は参照に置き換えること。**

1. 開発機のシステムRAMは64GB、サブマシンは32GB。どちらもページファイル利用可。
2. 配布方式は「**manifestからコミュニティ/Comfy-Org公式のHFリポジトリを直接ダウンロード**」（自前再ホストはしない。ライセンスの地域条項があるため）。
3. Single v1のタスク範囲は「**FL2VA重み1本でtxt2vid（文章から動画生成）とimg2vid（画像から動画生成）の両対応。Ref2VA（参照生成）は後回し**」。
4. DiT重みの第一候補は**pruned GGUF Q4**。考慮すべきはモデルの性能・精度。
5. 推論エンジンはまずSingle用のみを作る。インストーラーを先に作る（インストール作業の検証が事実上不要になるため）。

## §11 未検証事項と次段の調査候補

- テキストエンコーダ3案（§8）の実測比較（品質・実VRAM/RAM・速度）。**GPUを使う実測は着手前に毎回オーナーの了承を得る**。
- 量子化の実際の出力品質（本調査はヘッダの型分布とサイズ計算まで。生成品質は一切測っていない）。
- bitsandbytesのWindows動作（③NF4案を採る場合のみ）。
- VAEタイル処理の品質（§5のバンディング問題の回避方法）。
- Ada世代・Ampere世代での初回実測（世の中に一次データが無いため自前で測ることになる）。
- FL2VAとRef2VAの重み内容が実際に異なることの確認（ヘッダが同一なだけでデータ部は未取得）。
- 実装ベースをDiffSynth土台で確定させる場合の、attention実装の差し替え方針（DiffSynthのattentionはセグメントごとのPythonループによる素朴な実装で、そのままでは速度が出ない可能性がある——推測）。

## §12 主要出典

一次資料:

- [MiniMax公式ブログ: MiniMax H3](https://www.minimax.io/blog/minimax-h3)（2026-07-31）
- [MiniMaxAI/MiniMax-H3（重み・LICENSE・QA-about-License.md）](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax-AI/MiniMax-H3（GitHub）](https://github.com/MiniMax-AI/MiniMax-H3) / [awesome-minimax-h3-integration（公式エコシステム索引）](https://github.com/MiniMax-AI/awesome-minimax-h3-integration)
- [diffusers公式ドキュメント: MiniMax-H3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)（VRAMレシピの正本） / [diffusers PR #14355](https://github.com/huggingface/diffusers/pull/14355)
- [ComfyUI公式ブログ: Day-0 Support](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)（2026-08-03） / [ComfyUI PR #15224](https://github.com/Comfy-Org/ComfyUI/pull/15224) / [comfy/ldm/minimax/model.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)
- [Comfy-Org/MiniMax-H3（公式repack）](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [DiffSynth-Studio（modelscope）](https://github.com/modelscope/DiffSynth-Studio) / [DiffSynth-Studio/MiniMax-H3-NF4](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4)
- [Tomiigo/minimax-h3-16gb（RTX 5070 Ti 16GB実測）](https://github.com/Tomiigo/minimax-h3-16gb) / [tonyd2wild/minimax-h3-local（RTX 3090実測）](https://github.com/tonyd2wild/minimax-h3-local)
- [ComfyUI issue #15488（Windows GPU lost）](https://github.com/Comfy-Org/ComfyUI/issues/15488) / [issue #15529（int8アライメント）](https://github.com/Comfy-Org/ComfyUI/issues/15529) / [issue #15416（VAEタイルのバンディング）](https://github.com/Comfy-Org/ComfyUI/issues/15416)
- [city96/ComfyUI-GGUF issue #471](https://github.com/city96/ComfyUI-GGUF/issues/471) / [leejetフォーク](https://github.com/leejet/ComfyUI-GGUF)
- [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) / [Abiray/MiniMax-H3-GGUF](https://huggingface.co/Abiray/MiniMax-H3-GGUF) / [joeygambino/MiniMax-H3-encoder-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-encoder-GGUF)
- [Hugging Face: Gated models（公式ドキュメント）](https://huggingface.co/docs/hub/en/models-gated) / [Gating Group Collections](https://huggingface.co/docs/hub/en/enterprise-gating-group-collections)
- [multimodalart/MiniMax-H3-Pruned（枝刈りの原典・誤差の自己申告）](https://huggingface.co/multimodalart/MiniMax-H3-Pruned)
- GGUF/safetensorsヘッダの実測データ（2026-09-06、HTTP Range取得。本書§6が集計の正本）

二次情報（補助的に使用）:

- [ComfyUI Wiki（community quants等の各記事）](https://comfyui-wiki.com/) / [SD.Next Wiki: MiniMax](https://github.com/vladmandic/sdnext/wiki/MiniMax) / [Hugging Faceフォーラム: Quantisation RAM/VRAM](https://discuss.huggingface.co/t/minimax-h3-quantisation-ram-vram/178461) / note.com実践記事（RTX 5070 Ti、2026-08-07）
