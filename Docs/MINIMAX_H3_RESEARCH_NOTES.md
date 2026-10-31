# MiniMax H3 事前調査ノート（参考資料）

## 本書の位置づけ

**2026-09-06に実施した、MiniMax H3を第3のベースモデルとして追加するための事前調査の記録である。設計資料ではなく参考資料として収蔵する。**

- マルチエンジン機構（ドロップダウンによるベースモデル切り替え）の設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)である。MiniMax H3固有の設計正本は本書執筆時点では未執筆であり、執筆後に本書と記述が食い違った場合は設計正本を採ること。
- 台帳（[`PENDING_TASKS.md`](PENDING_TASKS.md)）への起票は未実施（設計段で行う）。

  > **更新（2026-10-16）**: 起票済み（§1-93）。

- 2026-10-16 に鮮度の確認と裁定の更新を行った。更新内容は §13、裁定の更新は §10 の追記にある。§1〜§9 の本文は 2026-09-06 時点のままで、古くなった箇所には訂正の引用を添えてある。
- 2026-10-17 に計画のセッションを開始し、オーナーの裁定 13〜21 を §10 に追記した。§13.7 の 1 と §13.8 に訂正と更新の引用を添えてある。
- 設計正本は [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md)（2026-10-17 新設）。
- 2026-10-28 に、Ref2VA（参照素材からの生成）の一次資料の調査を §15 に追記した。
- 調査の内訳: WEB調査4本（公式情報・ComfyUI事例・配布地域制限・実装ベース比較）＋配布ファイルのヘッダ実測（GGUF 13本・safetensors 7本をHTTP Rangeリクエストで部分取得して解析）＋当リポジトリの受け皿の棚卸し。数値・状況はすべて2026-09-06時点のものである。
- 本書内では「一次資料で確認済み」「実測」「二次情報」「推測」を区別して記す。

---

## 結論

MiniMax H3の追加は実現可能である。VRAM 16GBでの動作は、diffusers公式ドキュメントにレシピが実在し、16GBカード（RTX 5070 Ti）での一次実測（VRAMピーク14.5〜15.8GiB）も複数存在する。

有望な経路は「**コミュニティ配布のpruned GGUF Q4（DiT本体約11.6GB・VRAM常駐可）を、manifestから直接ダウンロードし、DiffSynth-Studioの実装を土台に専用ワーカーへ移植する**」形である。事前に懸念されていた「K量子化の罠」（Q4_K_M表記の実体がQ4_0である疑い）は、狙っているpruned版についてはヘッダ実測で**否定**された。

> **訂正（2026-10-16）**: 2026-10-16 のオーナー裁定で、DiT 重みの第一候補は枝刈り GGUF Q4 から**公式系 FL2VA の枝刈り int8 ConvRot**（Comfy-Org 配布）に変わった。コミュニティ配布の主流も枝刈り int8 ConvRot の safetensors（約 20.97GB）に移っている。DiffSynth-Studio はこの形式を comfy-kitchen 経由で直読みできる。詳細は §10 の追記と §13。

一方で、前提となる工事が1つ判明した。コミュニティGGUFはKVメタデータ（`general.architecture`／`model_version`）が全滅しており（偽装・欠落）、既存の2キー契約では系統判別ができない。**テンソル名の指紋による判別への契約拡張が必須**になる。

最大の未解決課題はテキストエンコーダ（Qwen3-VL-32B）の載せ方で、3案（§8）の実測比較が必要である。

> **訂正（2026-10-16）**: 裁定 8 により、まず Comfy-Org の int8 ConvRot 27.14GB で動かす。3 案は将来の選択肢（§8 の訂正を参照）。

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

> **訂正（2026-10-16）**: 「ガイダンス蒸留済み」は要確認になった。ComfyUI の issue #15563 の再現グラフは cfg 4、PinkCherry の メタデータ は `guidance_scale 4.0` だった（二次情報。真偽は未確認）。詳細は §13.9。

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

> **訂正（2026-10-16）**: 上の構成は GGUF Q4_K_M が前提だった。コミュニティ配布の主流は枝刈り int8 ConvRot の safetensors 20.97GB と、公式の w6a8 15.98GB／w4a8 約 12.5GB に移った。公式リポジトリ（Comfy-Org）には枝刈り fp8_scaled と w6a8 も加わっている。2026-10-16 の裁定で、DiT は公式系 FL2VA の枝刈り int8 ConvRot、テキストエンコーダは int8 ConvRot 27.14GB に変更した（§10 の追記）。詳細は §13.2・§13.3。

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

> **訂正（2026-10-16）**: 16GB の Ada 世代（RTX 4060 Ti 16GB・4080・4070 Ti SUPER）の第三者による実測が出てきた。Ampere 16GB はまだ見つからない。詳細は §13.5。

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

  > **訂正（2026-10-16）**: ComfyUI issue #15488 は 2026-10-16 の時点で 404 になっている。また 2026-10-16 の裁定で int8 ConvRot を採ったため、comfy-kitchen を使う可能性が出てきた（使うかどうかは計画のセッションで決める）。詳細は §13.4。
- int8ストリーミング構成では1生成あたり**ディスク読み270GiB**の実測あり（NVMe必須）。DiTがVRAM常駐できるpruned Q4構成なら毎ステップのストリーミングが不要になり、この負荷はロード時に限られる見込み（推測）。

  > **訂正（2026-10-16）**: 裁定 6 の int8 ConvRot 版（20.97GB）は 16GB の VRAM に常駐できないので、この前提は成り立たない。ブロック転送が前提になる（§13.7）。
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

  > **訂正（2026-10-16）**: テンソル名と形状が同じことは今も正しい。ただし重みのデータは違うことが確定した（ファイルの sha256 が異なる）。公式品は `adaln_t_table` の値（32KB）で区別できる。さらに FL2VA と Ref2VA を混ぜた Hybrid が普及したため、「FL2VA か Ref2VA かの 2 値で区別する」前提そのものが崩れている。詳細は §13.2・§13.3。
- テキストエンコーダのGGUF（joeygambino Q4_K_M=19.8GB）は素のllama.cpp製（`'qwen3vl'`、本物のQ4_K+Q6_K混合、トークナイザ完備）。ただし**64層フルのQwen3-VL-32Bで、H3が使う50層に対し14層分の死荷重つき**。
- Comfy-Org公式repackのsafetensorsは、`__metadata__`（configつき）が非pruned bf16版にしか無く、pruned系は空。safetensors直読みでもテンソル名指紋が必要。int8_convrot版は`comfy_quant`独自記述子＋回転量子化（ConvRot）で、直読みには逆回転の自前実装が要る。

  > **訂正（2026-10-16）**: 3 点を訂正する。①`__metadata__` は非枝刈り int8 ConvRot 版にも config つきで入っている。テキストエンコーダ int8 版と映像 VAE int8 版にもそれぞれ専用のメタデータがある。空なのは枝刈り版だけ（実測）。②`comfy_quant` は独自記述子ではなく、ComfyUI の `QUANTIZATION.md` で公開された仕様である。groupsize 256 と 64 が 1 ファイル内で混在しうる。③「逆回転の自前実装が要る」は半分だけ正しい。行列積の経路では活性化のほうをアダマール回転すれば済み、逆回転が要るのは bf16 に逆量子化するときだけ（推測。comfy-kitchen の実装と同じ考え方）。詳細は §13.2・§13.4。
- GGUFローダー事情: city96/ComfyUI-GGUF本家は2026-01以降実質停止でH3**非対応**（issue #471オープンのまま）。読めるのはleejetフォーク等。ComfyUI本体はGGUF非対応（ネイティブ対応はsafetensors経路のみ）。
- 品質情報: Q4量子化のH3固有の統制された品質比較は**存在しない**（唯一の定量報告は本人が撤回）。先行実装からの落とし穴情報として「**`adaln_t_table`はF32のまま保つこと。F16化すると1ステップ目でクラッシュする**」という報告あり。

  > **訂正（2026-10-16）**: `adaln_t_table` を BF16 で保存した配布物（Eros beta4）が流通している。ComfyUI は F32 のバッファに読み込んでいる。「F32 厳守」は保存時の dtype ではなく、計算時の dtype についての注意と読むべきである（推測）。詳細は §13.3。

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

> **訂正（2026-10-16）**: 表の「C: DiffSynth-Studio の GGUF 欄（非対応。NF4/FP8/INT8 のみ）」について。DiffSynth-Studio は 2026-08-20（#1620）の時点で、Comfy-Org の int8 ConvRot（枝刈り版を含む）と Comfy 版テキストエンコーダを comfy-kitchen 経由で直読みできていた（9 月の調査の見落とし）。また diffusers に単一ファイル読み込みが入った（2026-10-01）が、枝刈り版は明示的に拒否する。詳細は §13.4。

**推奨はC（DiffSynth-Studio）**。pruned形式を読める唯一の非GPL実装であり、テキストエンコーダを50層だけ構築し、コード量最小・オフロード自己完結・torch要件が緩い。弱点のGGUF非対応は、当プロジェクトが既に持つ融合GGUF逆量子化カーネルの文化と接続して埋める（`adaln_t_table`のF32厳守に注意）。

- **A（diffusers）はMiniMax公式が指名するリファレンス実装**なので、検証ゲートの数値オラクル（正解データ源）として使う。
- **B（ComfyUI）とKJNodesはGPL-3.0のため、コードの移植は行わない**。仕様理解のための読解に限る（特にテキストエンコーダのトークナイズ規約とテンソル名指紋の実装は仕様書として価値が高い）。
- torch整合: 既存2エンジンはtorch 2.9.1+cu128（freeze実測）。DiffSynth要件（≥2.0）を満たし、**「全エンジンでtorchを揃える」方針は維持できる**。Qwen3-VLはtransformers 4.57.0以降（既存venvは4.57.6/5.14.1でいずれも充足）。

  > **訂正（2026-10-16）**: 2026-10-16 の裁定 11 で、「全エンジンで torch を揃える」方針は H3 には適用しないことになった。H3 の仮想環境は、comfy-kitchen の CUDA 拡張が効く torch の版を選ぶ（CUDA ランタイム 13.0 以上・ドライバ R580 以上が要る）。torch 2.9.1+cu128 のままでは CUDA 拡張が使えない。そのため次の行の SageAttention も「既存と同一 wheel」とは限らなくなる（§13.9）。詳細は §13.4。
- ComfyUIの16GB実測が使うSageAttention 2.2.0は既存2エンジンと同一wheelで、当プロジェクトに導入済み。

## §8 テキストエンコーダの載せ方（未裁定・3案）

16GB成否を分ける最大の設計課題。DiT（約11.6GB）とテキストエンコーダは16GBに同居できないため、**逐次ロード（エンコード→解放→DiT）が必須**（同時1ジョブ思想と整合）。エンコード段は16GBの境界ぎりぎりという理論試算あり（Windowsでは予約分により0.09GiB超過という未検証の試算）。

| 案 | サイズ | 利点 | 懸念 |
|---|---|---|---|
| ① GGUF Q2_K | 7.91GiB | 16GBに余裕で収まる。公式エコシステム索引の12〜16GB推奨構成 | 2bit級の品質が未知（統制比較なし） |
| ② GGUF Q4_K_M | 19.8GB | 品質面で無難 | VRAMに収まらず**層オフロードの新設が必要**（LTX 2.3の`--te-offload`=GGUF Gemmaの逐次per-layer CPUオフロードという同型の前例が当プロジェクトにある）。64層フルで14層死荷重 |
| ③ NF4（DiffSynth配布） | 15.32GB | 50層切り詰め済み・DiffSynthでそのまま動く実績 | 16GBぎりぎり。**bitsandbytesのWindows動作が未検証** |

次段で実測比較して決める（§11）。

> **訂正（2026-10-16）**: 裁定 8 により、まず Comfy-Org の int8 ConvRot 版 27.14GB（50 層に切り詰め済み。視覚塔と埋め込みは bf16）で動かすことを目指す。上の 3 案は将来の選択肢として残す。中身の実測は §13.2。

## §9 当プロジェクトの受け皿との噛み合わせ（棚卸し結果の要約）

詳細は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)の該当節を参照。ここでは結論のみ記す。

- H3は同設計書§3.3の「依存関係やコードが同居できない系統」パターン（LTX 2.5と同型）に該当し、**専用ワーカー・専用仮想環境・専用アダプタを新設**する。レジストリ（`services/engines/__init__.py`の3辞書）・`config.py`のエンジン別インタプリタ設定・`install_ltx.ps1`の仮想環境構築呼び出し・記述子（manifest）＋`install-<ID>.bat`が追加箇所。**`install_model.ps1`とフロントエンドのドロップダウン／`baseModelInstaller(id)`はmanifest追加だけで自動追随する**（既存設計の狙いどおり）。
- **フレーム規約`17n+5`への対応が本丸**。LTXの`8n+1`は`api/models.py`に独立リテラル5箇所＋フロントエンド`mockBridge.ts`にも散在しており（同設計書§7の予告どおり）、エンジン別パラメータ制約の整理が必要になる。24fps固定・尺4〜15秒・32の倍数解像度も同様にエンジン別制約。
- H3は**3例目**であり、同設計書§5.2が予告する「汎用EngineRunnerとアダプタ分割の要否を再判断するタイミング」に当たる。
- `comfort_budgets`（快適上限の配信テーブル）は、新系統は較正が済むまで行を持たせず、`outpaint_budget`は`None`のまま（裁定J1「前例をコピーしない」、`config.py`のコメントに明記済み）。
- ガイダンス蒸留済み（negative promptなし）・音声同時生成という性質は、UI表示テーブル（`featureScope.ts`）とエンジン別制約の設計に反映が必要。

  > **訂正（2026-10-16）**: 「ガイダンス蒸留済み」は要確認（§13.9）。それ以外の本節の記述（特にフレーム規約 `17n+5` が本丸であること）は今も有効である（§13.7）。
- mockバックエンドはLTXの出力形式・窓計算を模しているため`ltx`アダプタ内にあり、真に別規約のH3では改めて判断が必要（同設計書§9.1の注記どおり）。

## §10 オーナー裁定の記録（2026-09-06）

設計に関わる裁定は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 に移管済み（各行の → 印）。本節は進め方の裁定と、裁定の原文の記録として残す。

1. 開発機のシステムRAMは64GB、サブマシンは32GB。どちらもページファイル利用可。
2. 配布方式は「**manifestからコミュニティ/Comfy-Org公式のHFリポジトリを直接ダウンロード**」（自前再ホストはしない。ライセンスの地域条項があるため）。
3. Single v1のタスク範囲は「**FL2VA重み1本でtxt2vid（文章から動画生成）とimg2vid（画像から動画生成）の両対応。Ref2VA（参照生成）は後回し**」。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
4. DiT重みの第一候補は**pruned GGUF Q4**。考慮すべきはモデルの性能・精度。
5. 推論エンジンはまずSingle用のみを作る。インストーラーを先に作る（インストール作業の検証が事実上不要になるため）。
   > **注記（2026-10-18）**: 後半（インストーラを先に作る）は裁定 14（インストーラは第 3 弾）で置き換わった。

### 2026-10-16 の裁定（追記）

オーナーが外部の α版ガイド（別のエージェントとの議論の結果。§13.8）を読んだうえで下した裁定である。**9 月の裁定 4（DiT 重みの第一候補は枝刈り GGUF Q4）は取り下げ、裁定 6 で置き換える。9 月の裁定 2（再ホストしない）は維持する**（裁定 7）。9 月の裁定 1・3・5 は変更しない（今回の裁定と矛盾しないため維持とみなす。監督の判断）。

6. 重みの形式は **int8 ConvRot**。利用者が CivitAI などから落としてフォルダに置くだけで動かせることが理想。理由: ファインチューンが大量に出回っており、コミュニティの主流が int8 ConvRot だから（§13.3）。置くだけで動かしたい（オーナーの言葉の趣旨）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
7. 配布方針は**公開リポジトリから直接ダウンロード**。再ホストはしない（9 月の裁定 2 を維持）。外部ガイド §18 の「Nz-Videomni 用の Hugging Face リポジトリから取得する」は採らない。理由: 9 月の裁定 2 と同じ（ライセンスの地域条項）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
8. テキストエンコーダは、まず **int8 ConvRot 27.14GB** で動かすことを目指す。nvfp4 や bf16 は将来の拡張とする。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
9. 動画の尺は「**快適上限を測る**」方針に揃える。外部ガイドの 1344×768・345 フレームは正式目標にしない。理由: 既存のエンジンと同じく、実測で上限を決める方針に揃えるため。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
10. **LoRA は後回し**。理由: Turbo LoRA（少ないステップ数で生成するための蒸留 LoRA）への対応は Style LoRA（画風などを足す LoRA）への対応とほぼ同じ仕組みになるはずなので、α版に載せるより Style LoRA に対応するときに一緒に進める。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
11. H3 の仮想環境は、**torch の版を LTX に揃えなくてよい**。comfy-kitchen の CUDA 拡張が効く版を選ぶ（§13.4）。利用者側の条件（ドライバ R580 以上と、ディスクの増加）は README に書く。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
12. 計画と実装は別のセッションで行う。2026-10-16 のセッションは、台帳の起票・鮮度の確認・文書の更新までとする。

### 2026-10-17 の裁定（追記）

計画のセッションの冒頭で、監督の確認 8 点にオーナーが答えた裁定である。**§13.7 の 1 の監督査定（α版は既存の `engine/sft_quant` の方式で通し、W8A8 は計測の後）は誤りで、裁定 16 で置き換える。** 裁定 1〜12 は変更しない。

13. このセッションの到達点は、計画の確定と、了承後の第 1 弾の実装まで。
14. 第 1 弾の範囲は、ワーカーのセルフテストで 1 本生成すること。α版の完結条件は、AviUtl2 の操作パネルから T2V と I2V が生成できることと、`install-MiniMaxH3.bat` と README まで進むこと。快適上限の較正と配信値は α版に含めず、インストーラと README より優先順位が低い（台帳 §1-95）。α版では快適上限マーカーを非表示にする。
15. インストーラの名前は `install-MiniMaxH3.bat`（`install-H3.bat` ではない）。
16. int8 の実行は comfy-kitchen の CUDA 拡張に任せる。H3 の仮想環境の torch は、MiniMax H3 を動かすことに最も適した版を選ぶ（裁定 11 の確認。本製品はモデルごとに仮想環境を持てる設計なので、既存の torch の版にこだわる必要はない）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
17. 実装の土台は 9 月の推奨（DiffSynth-Studio を土台・diffusers は正解データ源・ComfyUI は仕様書として読むだけ）のままとし、次の修正点 4 つを計画に織り込む。(1) int8 の実行は comfy-kitchen（Apache-2.0）に任せる候補を正面に置く。既存の読み込み（呼び出しごとに bf16 へ戻す・groupsize 256 のみ）はヘッダの判定には使えても H3 の実行経路には向かない。(2) attention は DiffSynth の区間ごとの Python ループを使わず、diffusers の「1 リクエストを 1 つの文書に詰めてマスクなしで渡す」形を手本にして書き、本製品に導入済みの SageAttention を効かせる。(3) CPU オフロードは DiffSynth の仕組みではなく、本製品の block swap と先読みを接続する。(4) ローダを最初から「揺れを吸収する」作りで書く案を、計画で検討する（DiffSynth の変換表を出発点に、ComfyUI の判定の振る舞いを仕様として自前で書く）。あわせて、次の論点 4 つを計画で決める。(a) DiffSynth を移植するか、版を固定した依存として入れて本体のモデル部品だけ使うか。(b) 映像 VAE は Comfy-Org の fp16 版か、DiffSynth の例が使う MiniMax 公式の VAE か（中身が同じかは未確認）。(c) 「ガイダンス蒸留済みで negative prompt なし」が本当か（cfg 4 を使う利用例が複数あり、本当でなければ CFG の経路が要る）。(d) comfy-kitchen の triton 経路の未解決の問題（#136）。CUDA 拡張の経路なら影響しない見込みだが、計測で確かめる。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
18. α版では、ステップ数と shift を定数で固定し、画面に出さない。画面に出す改修は台帳 §1-96（第 2 弾へ吸収済み。裁定 22）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
19. α版の I2V は先頭フレームのみ。末尾キーフレームと中間キーフレームは台帳 §1-97。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
    > **更新（2026-10-19）**: 末尾と中間のキーフレームは実装済み（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-241、設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3）。
20. ファイル群の取得と仮想環境の構築は 1 回の了承で進める。GPU を使う実測は、弾ごとにまとめて了承を得る。
21. Turbo を後回しにするため生成が遅くなることは了承済み。快適上限の較正に支障が出るほど遅ければ、LoRA 対応の優先順位を上げる。
22. ステップ数と shift（映像・音声）の 3 本のスライダーを、第 2 弾で操作パネルと Gradio に出す。既存の CFG スライダーのモックを差し替え、項目名・スライダー本体・数値を 1 行に並べる。既定値は第 2 弾の着手時に決める（→ 既定値は裁定 24）。背景は、Turbo を融合した低ステップの配布物が多く、LoRA 未対応でも速く生成できる見込みがあること。**裁定 18 のうち「画面に出さない」は取り下げる**（第 1 弾でステップ数と shift を定数の既定値とし、ワーカーが引数で受け取る部分は残る）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管

### 2026-10-17 の裁定（第 2 弾の着手時・追記）

第 2 弾（I2V のセルフテスト・アダプタ・操作パネル）の計画のセッションの冒頭で、監督の確認 5 点と提案 4 点にオーナーが答えた裁定である。裁定 1〜22 は変更しない。設計に関わる裁定は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) へ移管した（各行の → 印）。

23. 第 2 弾は 1 つの計画の中を 3 段に分ける。2a＝I2V（先頭フレーム）のセルフテスト、2b＝バックエンド側の全部とテスト、2c＝操作パネルと Gradio と MCP。最後にオーナーの目視を行う。GPU を使う実測の了承は段ごとに取る（長時間 GPU が空くときはオーナーから声をかける）。このセッションの到達点は、オーナーの目視と main への merge まで。
24. ステップ数と shift（映像・音声）の 3 本のスライダーの既定値は 20／12／3。範囲は、ステップ数が 1〜50 の整数、shift（映像）が 1〜20、shift（音声）が 1〜20 で、shift の刻みは 0.1。既定値を決めるための 50 ステップとの比較は行わない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§9 へ移管
25. 第 1 弾のレビューで先送りした技術課題 3 点のうち、大きな画角での復号の VRAM の直し（復号結果を窓ごとに uint8 にして CPU へ移す）は第 2 弾に入れる。テキストエンコーダを層ごとにファイルから GPU へ直接読む根本策と、T2V で映像 VAE の符号化器を読まない省略は α版の後に回し、台帳 §1-98・§1-99 に起票した。 → 2 点とも裁定 32・33 で第 2 弾へ戻した（§1-98・§1-99 は削除）
26. 常駐（生成のあいだに読んだ重みを次の生成まで捨てずに持つこと）は α版から有りで進める。本製品の推奨スペックは VRAM 16GB・主メモリ 64GB であり、これに足りない機械への対応は後から考える。常駐の on／off は Settings 画面の「Acceleration／生成の高速化」の区画に出す。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§4 へ移管
27. SageAttention（高速な attention の実装）の既定は、本製品の現状の既定に従わせる（H3 専用の既定は作らない）。
28. I2V の先頭画像の VAE 符号化は、分布の平均を使う（サンプルしない）。調査の結果（ComfyUI と DiffSynth も平均を取り、diffusers だけが固定の種でサンプルする）のとおり。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
29. サンプラは Euler 固定とし、選択肢は出さない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
30. 操作パネルの方針。LTX に固有の機能（IC-LoRA、Edit の AlphaGen、Chained の Stage-2〔upscale pass〕の clip length など）は H3 では非表示にする。動画生成の基本機能（中間キーフレームの指定、Chained、Inpainting、Outpainting など）は H3 でも将来実装する予定なので、グレーアウトして操作できない状態で配置だけ決めておく。H3 には「Ref2V」（参照素材からの生成）という目玉機能があり、LTX とまったく違う UI が要る。その UI／UX は Ref2V の実装時に考える（予告）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
31. 計画の敵対的レビューは第 1 弾と同じく Opus で行う。
32. 常駐の方式。§1-98（テキストエンコーダを層ごとにファイルから GPU へ直接読む）を第 2 弾に戻し、常駐を α版から有りにする。理由: DiT と VAE（約 26GiB）を常駐させたまま現行の方式でテキストエンコーダ（約 29GiB）を主メモリに読むと、主メモリの山が約 60GiB になり、64GB 機でもページファイルに溢れて常駐の目的（速くする）と逆になる見込みだから。直接読みにすればテキストエンコーダは主メモリをほぼ使わず、常駐しても山は約 30GiB に収まる見込み。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §4 へ移管
33. §1-99（T2V で映像 VAE の符号化器を読まない）は、I2V のために映像 VAE を符号化器と復号器に分けて組む構造の変更で自然に片づくので、第 2 弾へ吸収する（裁定 25 の小さな変更。第 2 弾の計画の了承で確定）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3 へ移管
34. 操作パネルの Edit タブは、H3 では 4 つのサブタブ（Retake・Outpainting・Inpainting・Alpha Gen）が全部閉じるため、既存の規則によりタブごと灰色になる。裁定 30 の「グレーアウトして配置だけ決めておく」はこの形で満たす（第 2 弾の計画の了承で確定）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
35. NAG のアコーディオンは H3 では非表示でなく灰色にし、要求に NAG の項目を載せない（非表示にすると有効にしたまま解除できず全ジョブが 422 になるため）。Single の A2V カードは連結要求そのものなので `chain` で閉じる（灰色）。IC-LoRA の参照動画のアコーディオンは非表示（第 2 弾の計画の了承で確定）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
   > **訂正（2026-10-18）**: Single の A2V カードとバッチ A2V は非表示にする（2026-10-18 の目視の所感。実装は台帳 §1-100）。
   > **更新（2026-10-21）**: 非表示は取り下げ（A2V を近日実装するため灰色のまま。設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 の裁定 35 の訂正行・[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §171）。

### 2026-10-18 の裁定（第 3 弾の着手時・追記）

第 3 弾（`install-MiniMaxH3.bat` と README）の計画のセッションで、オーナーが答えた裁定と、計画の了承で確定した提案である。裁定 1〜35 は変更しない。設計に関わる裁定は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) へ移管した（各行の → 印）。

36. ライセンスは README に明記する（名称・適用除外地域・取得者の自己責任）。インストーラは取得前に注意文とライセンスの URL を表示するだけで、同意の入力は作らない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §10 へ移管
37. README で H3 を「α版」と明示し、できること・できないこと・快適上限が未較正で大きな画角では遅くなること・推奨（VRAM 16GB・主メモリ 64GB）を書く。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §10 へ移管
   > 訂正（2026-10-19）: 「大きな画角では遅くなる」の README の文は、VRAM 16GB mode の実装で現行化した（記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169）。
38. GPU 世代の文言は正直に書く。「ドライバ R580 以上・RTX 20 以降（GTX 10 以前は動かない）。確かめたのは RTX 40 だけ。RTX 30・50 は未確認。RTX 20 は int8 が遅い経路になり SageAttention も使えず、動作も未確認」。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §10 へ移管
   > 訂正（2026-10-18）: 文言は裁定 43 で更新した。
39. サブマシンでの検証を門にする。構築と取得と UI の導入済み表示までが必須、常駐なしの T2V 640×384 が 1 本通れば合格（VRAM が足りなければ前者まで）。
40. README の凍結ゾーン（1〜133 行目。境界は 134 行目の見出し `# Nz-Videomniの詳細`）は触らない。H3 の話題は詳細側に書く。
41. インストーラが `MiniMaxAI/MiniMax-H3` の `LICENSE` も取得して `models/MiniMaxH3/LICENSE` に置く。本製品はモデルを再配布しないが、ライセンスは第三者へ渡すときに写しと NOTICE を求めている。写しを手元に置くのは安全側で費用はほぼ無い。NOTICE は作らない（計画の了承で確定）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §10 へ移管
42. サブマシンの T2V が主メモリ 32GB に起因して失敗した場合（ページファイルの不足など）も「構築と取得と UI まで」で合格とし、失敗の事実を記録簿に書く。記録上サブマシンの VRAM は 16GB なので、落ちるとすれば主メモリのほう（裁定 39 の補足。計画の了承で確定）。
43. README の GPU 世代と主メモリの文言を、サブマシンの結果を受けて更新（RTX 30 で確認・32GB はページファイル前提で短い生成を確認済み・推奨 64GB）。裁定 38 の文言の更新（オーナー了承 2026-10-18）。

### 2026-10-18 の裁定（VRAM 16GB mode の着手時・追記）

台帳 §1-101 の第 2 段（設計と実装）の計画のセッションで、オーナーが答えた裁定と、計画の了承で確定した提案である。裁定 1〜43 は変更しない。設計の規則にあたる 44〜47・51・53 は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) へ移管した（各行の → 印）。進め方の 48〜50・52 は本節に残す。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169。

44. 名称は「VRAM 16GB mode」。Settings の「Acceleration／生成の高速化」のトグルで、H3 では既定 on、off も選べる。Gradio と MCP にも出す。値はジョブごとに毎回送る。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§6.5 へ移管
45. 同じワーカーの中の切り替えにする。off は現行の経路に一切触れず（基準の指紋 `0da6111c…` の不変で担保する）、on が新しい経路。ワーカーや仮想環境は複製しない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
46. 数値の約束。目標はビット一致。門で最終桁の差が出たときは、ブロックごとの最大差がゆらぎ級であることとオーナーの目視で品質差が無いことの 2 つで採否を決める。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§6.5 へ移管
47. OOM は、ジョブを失敗させて「解像度かフレーム数を下げるか、VRAM 16GB mode を切る」と案内する（「画角」と書かない）。黙って退避を許す経路は作らない。既知の限界: 上限はこのプロセスの割り当てだけを縛るので、ジョブ中に他のプロセスが VRAM を増やした場合の退避は防げない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§6.5 へ移管
48. 目標の尺度は台帳 §1-101 のコマ数のままにする。生成時間は本テーマの対象外。
49. 実装担当は、設計に疑問があれば作業を止めて監督へエスカレーションする権利を持つ。設計の変更は禁止。
50. 計画の敵対的レビューは、「過剰設計やスパゲッティ化の危険」と「調査で裏付けが取れるか」の 2 点で採否を決める。
51. 分割の単位（ヘッド群・行の塊）と常駐数は、形状だけから決める決定的な規則にする。割り当て器の上限だけは、実行時の値から導いてよい。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§6.5 へ移管
52. PinkCherry Hybrid が置くだけで動いた事実を、本書 §13.3 と設計正本 §13 に 1 行ずつ記録する（オーナー報告 2026-10-18）。
53. sage のヘッド数依存の差（50 ブロック中 2 ブロック・相対 ≤ 6.6e-3）は、B-G5 のオーナーの目視で合格し採用する。sdpa はビット一致。裁定 46 の目視条件を満たしたものとして扱う（2026-10-18）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§6.5 へ移管

### 2026-10-19 の裁定（キーフレームの着手時・追記）

台帳 §1-97（末尾キーフレームと中間キーフレーム）の計画のセッションで、オーナーが答えた裁定と、計画の了承で確定した提案である。裁定 1〜53 は変更しない。設計の規則にあたる 56・57・59〜61 は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md)へ移管した（各行の → 印）。進め方の 54・55・58 と、裁定 61 のうち了承の取り方は本節に残す。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §170。

54. 到達点は、計画の了承の後の実装まで（計画と同じセッション）。
55. 末尾を第 1 弾、中間を第 2 弾とする。ComfyUI は論理の参考に留め、コードは写さない。
56. キーフレームを置ける位置は、既存の系統別の規則表 `RequestRules` に項目を足して系統別にする。全系統に共通の config の 2 値（`conditioning_frame_idx_multiple`／`conditioning_keyframe_grid_offset`）と、系統が分かる前に走る pydantic（API の入力の型を検査する仕組み）の 8n+1 の丸めは廃止し、系統別の検査へ移す。UI の表 FEATURE_UI は使い回さない（機能の有無の表と数値の規則は別物）。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§3 へ移管
57. 強度は操作パネルで非表示にする（灰色ではない）。記録には「強度が効いていない」と分かる形にする（1.0 にしない）。仕組みは「強度を省略できるようにし、系統の規則表が既定を書き戻す」。LTX は 0.8 を書き戻し、H3 は書き戻さずに null のまま。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
58. バッチは本項で触らない（H3 ではバッチの枠が丸ごと灰色になるため）。
59. 中間のフレームの画像は画像読み取り部（vision tower）に渡さない（条件行だけ）。足りなければ「渡す」を比較の腕として後で検討する。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2・§3 へ移管
60. 公式のプロンプトの指南の指示文（「Picture 1 は 0.00 秒、Picture 2 は S.SS 秒」）は README に書くだけで、製品は入れない。 → 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管
61. GPU を使う実測は、弾ごとにまとめて了承を得る。T2V と先頭だけの I2V の結果は変えない（指紋で回帰を確かめる）。 → 後半を設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 へ移管（了承の取り方は本節に残す）

### 2026-10-24 の裁定（LoRA の着手時・追記）

台帳 §1-104（Style LoRA と Turbo LoRA の同時実装）の計画のセッションで、オーナーが答えた 11 件の裁定（62〜72）と、計画の了承の後に決めた文言の裁定（73）である。裁定 1〜61 は変更しない（裁定 10 の「LoRA は後回し」を実装に進めた）。設計の規則は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §6.6 にある。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §176。

62. 形式は「ComfyUI が読める名前」とする。点区切り `diffusion_model.` と下線区切りの kohya `lora_unet_` の両方、接尾辞 `lora_A／lora_B` と `lora_down／lora_up` の両方を受ける。`.alpha` は任意で、無ければ倍率 1.0、有れば alpha ÷ rank とする。
63. 判定は、一覧で「非対応」と表示するが、選択も実行もできるようにする。生成ボタンの下に黄色い注意文を出す。一部のモジュールだけを狙うファイル（token_refiner が無いなど）は許容する。
64. 同時に使う本数に制限は設けない。
65. 強度は既存のタグとチップ（0.05〜2.0・既定 1.0）を流用する。H3 ではチップの音声トグルと絵文字を隠し、`audio_strength` は送らない（422 にはしない）。動画と音声に個別の強度を持たせる件は台帳 §3 に起票する。
66. ステップ数と shift は自動では変えない。
67. LoRA の一覧は選択中の系統で切り替える（H3 は `models/MiniMaxH3/LoRA`、LTX は `models/LTX23/StyleLoRA`）。系統とディレクトリの対応はサーバー側の 1 か所に置き、操作パネルは系統を知らない。
68. `install-MiniMaxH3.bat` で空の `LoRA` フォルダと案内ファイルを作る（ダウンロードは無し）。
69. Gradio と MCP も同時に対応する。
70. LTX 側のコードは触らない（`engine/` は凍結のまま）。
71. 試験に使う LoRA ファイル 3 本の移動は、計画の了承の後の最初の一手として監督が行う。
72. CPU の正本は保存 dtype のままにし、GPU で使うときに FP32 へ持ち上げる（可逆・数値は同一）。
73. 注意文は原文のまま「LoRAが非対応の形式です。効果がない可能性があります。」とする。LTX でも同じ条件で同じ文言を出す（LTX では送信時に 422 のエラーカードも並ぶ）。応答に旗は足さない。

#### 設計上の判断（裁定ではない）

計画の敵対的レビューの問いに監督が答えた判断である。オーナーは了承のときに覆せた。

1. 「非対応」は、H3 の対象に写らないキーが 1 つでもあれば非対応とする（AdaLN・norm など）。対象の一部が無いだけ（token_refiner が無いなど）は対応とする。
2. `config.yaml` の IC-LoRA の登録は H3 の一覧に出さない（記述子が `lora_dir` を持つ系統は設定の登録を含めない）。
3. off と VRAM 16GB mode の一致は、1024 列の塊を両経路で使うことで形として保証する。測って判断する分岐は作らない。
4. 費用は v1 では受け入れ、高速化は台帳 §3-253 に起票する。ジョブ間の LoRA キャッシュは作らない。
5. Library のカードを押したときのタグは `<lora:名前:1.0:1.0>` のままにする（UI の例外を増やさない）。H3 では 3 つ目は送られず、チップにも出ない。

### 2026-10-25 の裁定（A2V の着手時・追記）

台帳 §1-103（A2V〔音声から映像〕）の裁定 74〜80 は、すべて設計の規則なので、本節には番号の範囲だけを置き、本文は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §2 に書いた。設計正本の裁定 62・63 は、本ノートの 62・63 とは別物である。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §178。

## §11 未検証事項と次段の調査候補

> **更新（2026-10-16）**: 各項目の進み具合と新しい項目は §13.9。

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

## §13 2026-10-16 の更新（鮮度の確認と本製品の現物との照合）

2026-10-16 に、調査担当 2 体（公式と周辺の鮮度・コミュニティのファインチューン）が調べ直し、監督が本製品の現物と突き合わせた結果の記録である。§1〜§9 の本文は書き換えず、古くなった箇所には訂正の引用を添えて本節を指している。

### 13.1 要旨（9 月から変わった点）

- 裁定が変わった。DiT は枝刈り GGUF Q4 から**公式系 FL2VA の枝刈り int8 ConvRot** へ、テキストエンコーダは **int8 ConvRot 27.14GB** へ（§10 の追記）。
- コミュニティ配布の主流は**枝刈り＋int8 ConvRot の safetensors（約 20.97GB）**になった。GGUF は少数派で、残っている（13.3）。
- Comfy-Org の int8 ConvRot は DiT・テキストエンコーダ・映像 VAE の 3 本とも同じ量子化記述子で、記述子は ComfyUI の `QUANTIZATION.md` で公開された仕様である（13.2）。
- コミュニティ版はヘッダの書き方が 8 点で揺れており、「置くだけで動く」にはその吸収が要る（13.3）。
- DiffSynth-Studio は 2026-08-20 の時点で、Comfy-Org の枝刈り int8 ConvRot を comfy-kitchen 経由で直読みしていた（9 月の見落とし）。diffusers には単一ファイル読み込みが入ったが、枝刈り版は拒否する（13.4）。
- comfy-kitchen の CUDA 拡張は CUDA ランタイム 13.0 以上・ドライバ R580 以上が要る。cu129 から torch 2.13+cu130 にして 2.17 倍速くなった第三者の実測がある（torch の版も変わっている）。本製品の torch 2.9.1+cu128 はこの条件に当たる（13.4・13.5）。
- 16GB の Ada 世代の第三者の実測が出てきた。Ampere 16GB はまだ見つからない（13.5）。
- FL2VA と Ref2VA の重みは、ヘッダが同じでもデータが違うことが確定した（13.2）。
- MiniMax 公式の重み・ライセンス・推奨設定には更新がない（13.6）。

### 13.2 Comfy-Org の int8 ConvRot 形式の実測

2026-10-16 に `https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/` の下の各ファイルのヘッダだけを HTTP Range で取得して解析した（実測）。`comfy_quant` テンソルの中身（JSON のバイト列）は全件読んだ。

**DiT 枝刈り版 `minimax_h3_fl2va_pruned_int8_convrot`（テンソル 932 本・`__metadata__` なし）**

- テンソル名に接頭辞はない。`blocks.{0..49}.{attn.qkv_proj, attn.out_proj, attn.q_norm, attn.k_norm, mlp.fc1, mlp.fc2, norm1, norm2, adaln_proj.linear}`、`token_refiner.blocks.{0,1}.*`、`token_refiner.final_norm`、`condition_proj`、`video_patch_proj`、`audio_patch_proj`、`final_layer.{norm, adaln_proj.linear, video_out, audio_out}`、`rope.inv_freq`、`adaln_t_table`。
- 量子化されているのは、本体 50 ブロックの qkv_proj [21504,5376]・out_proj [5376,7168]・fc1 [28672,5376]・fc2 [5376,14336] の **200 本だけ**（I8＋`weight_scale` F32 [N,1]＋`comfy_quant` U8 [72]）。token_refiner の 2 ブロックは bf16 のまま（約 1.54GB）。patch_proj と final out は F32、condition_proj と token_refiner は BF16。
- 200 本すべての記述子が `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`。名前は「tensorwise（テンソル単位）」だが、実際の重みスケールは出力チャネルごと（`weight_scale` の形状は `[出力次元, 1]`）。
- dtype の内訳は BF16 220・F32 210・I8 200・U8 200・F16 102。容量は blocks の I8 が 19.27GB、refiner の BF16 が 1.54GB。
- 枝刈りの印は `adaln_t_table` F32 [1025,8] と、`blocks.N.adaln_proj.linear.weight` **F16** [96768,8]（bias も F16）。`time_embedder.*` はない。
- Ref2VA 枝刈り int8 版のヘッダは FL2VA 版とバイト単位で同じ。ただし sha256（HF tree API の lfs oid）は異なり、重みのデータは違うことが確定した。公式の枝刈り版は FL2VA と Ref2VA で `adaln_t_table` の値（32KB）が違う（SHA1 不一致。bf16 版と int8 版では同じ値）ので、公式品どうしはこの値で区別できる。

**DiT 非枝刈り版 `minimax_h3_fl2va_int8_convrot`（テンソル 1035 本）**

- `__metadata__.config` があり、`hidden_size 5376, num_layers 50, token_refiner_num_layers 2, heads 56×128, ffn 14336, latents_dim 24, audio_latents_dim 32, patch_size [1,2,2], text_dim 5120, time_embed_dim 2688, adaln_out_features 96768` が入っている。
- AdaLN 射影 [96768,2688] も I8 だが、記述子は `convrot_groupsize: 64`（2688 は 256 で割り切れないため）。groupsize 256 と 64 が 1 ファイル内で混在する。`time_embedder.proj_in/out` は F32。

**テキストエンコーダ `qwen3vl_32b_minimax_h3_int8_convrot`（テンソル 1602 本）**

- `__metadata__.minimax_h3_te = {"num_hidden_layers":50,"output":"unnormalized_hidden_after_layer_50"}`。ファイルの時点で 50 層に切り詰め済み。
- `model.layers.0〜49` の q/k/v/o・gate/up/down の 350 本が I8（groupsize 256・合計 24.38GB）。
- 量子化から外れているのは次の 3 つ: `model.embed_tokens` bf16 [151936,5120]（1.56GB）／画像読み取り部（vision tower） `visual.*`（27 ブロック・merger・deepstack_merger 3 本・patch_embed・pos_embed。すべて bf16、計 1.19GB）／ノルム類。
- `lm_head` と `model.norm` は入っていない。

**映像 VAE `minimax_h3_video_vae_int8_convrot`**

- int8 になっているのは、ViT 型デコーダの 36 ブロック（to_qkv・to_out・ff.w1・ff.w2 の 144 本）だけ。エンコーダの 3D 畳み込みは F16。
- メタデータに `vae_ratio 16, vae_ratio_t 4, vae_clip_length 17, vae_token_drop 3, z_channels 24`。

**参考: `pruned_w6a8`**

- 記述子が `{"format":"w6a8_int8","group_size":32,"convrot":true,...}` の 201 本と、int8_tensorwise の 8 本（6bit を I8 に詰めたもの＋`weight_s_channel`/`weight_s_rel` F8_E4M3）。

**§6 の「直読みには逆回転の自前実装が要る」について**: 半分だけ正しい（推測）。行列積の経路では、活性化のほうを読み込み時にアダマール回転すれば済む。逆回転が要るのは bf16 に逆量子化するときだけで、13.4 の comfy-kitchen の実装と同じである。

### 13.3 コミュニティ配布の実態

**人気の H3 系モデル（CivitAI と CivitAI Red。数値は 2026-10-16 時点）**

出典は CivitAI API（`https://civitai.com/api/v1/models/<ID>`、一覧は `…/api/v1/models?baseModels=MiniMax%20H3&sort=Most%20Downloaded&nsfw=true&types=Checkpoint`）。RedCraft のダウンロード数はページ全体の合計。表の「Hybrid」は、FL2VA と Ref2VA の重みを混ぜて 1 本にしたもの（下の「Hybrid の実態」）。「Turbo」は少ないステップ数で生成するための蒸留を統合したもの。「VSA」は Video Sparse Attention の略とみられ、注意の計算を間引いて速くする派生実装を指す（推測。`to_gate_compress` というテンソルを持つ）。

| モデル（ID）・作者 | DL／👍 | 種別 | 配布形式（最新版の主なもの） | 枝刈り | 系統 | サイズ | 推奨設定ほか | ライセンス表記 |
|---|---|---|---|---|---|---|---|---|
| DaSiWa MiniMax H3（2877206）Darksidewalker | 58,406／1,718 | マージ・Hybrid・Turbo 版あり | int8 ConvRot（per_row）＋int4（2 種） | 枝刈り（ヘッダ実測） | Hybrid（「REF2VA + FL2VA compatible」） | 20.97GB／12.5・16.8GB | 非 Turbo は 20〜25 ステップ・shift 映像 10〜12／音声 3〜5。Turbo は 4 か 8 ステップ。作者の workflow を使う | H3 Community License を明記（§III・§IV を引用）。ファイルの メタデータ にもライセンス名と量子化道具を記載 |
| H3 Eros Max（2851079）tenstrip（Red） | 49,959／2,024 | マージ（参照差分を全ブロックに適用・Viggle・Turbo 統合・LoRA 混合） | int8 ConvRot（per_row・一部ブロックは bf16）＋bf16 | 枝刈り | 参照型＋T2V | 21.7GB／40.2GB | 8〜12 ステップ・shift 10/3・FastH3 の雛形。VSA 推奨。「小さいカードには向かない」 | 「H3 community license の条項が適用」（beta2） |
| Minimax H3 INT8/INT6/INT4 ConvRot（2830065）tsolful | 47,295／2,132 | 量子化のみ | int8 ConvRot・W6A8・AsymW4A8 | 枝刈り | FL2VA／REF2VA の別版 | 20.97／15.98／12.5GB | W4A8 は ComfyUI v0.31 以降。I/O・変調・埋め込み・出力ヘッドは BF16 のまま | NOTICE なし |
| PinkCherry MiniMax H3 Hybrid（2838593）sexgod1979（Red） | 13,653／559 | フルファインチューン。v1.8 は Hybrid | int8 ConvRot＋bf16。旧版は fp16／bf16 の非枝刈り | 枝刈り（「adaln を rank16 に」） | v1.8 は Hybrid、v1.5 以前は FL2VA | 21.06GB／40.3GB | Euler/simple・lightx2v Turbo で 12 ステップ、なしで 30 ステップ・0.7MP 以上 | NOTICE なし |
| MiniMax H3（2821932）CivitaiOfficial | 12,853／1,124 | 公式の転載 | bf16・int8・nvfp4 ほか | 両方 | 両方 | — | — | — |
| RedCraft Hybrid H3（958009）AiMetatron（Red） | ページ全体 379k | H3 で下書きし、LTX 2.5 で 2K に上げる 2 段の仕組み | H3 は int8（枝刈り版） | 枝刈り | FL2V Lightning（記載） | 21.98GB ほか | H3 は約 6 ステップ。LTX 側は 8＋3 ステップ。16GB 以上 | H3 Community License と LTX-2.x License を併記 |
| MiniMax H3 Remix（2879272）FX_FeiHou | 8,277／316 | 2 本の H3 を組み替え＋LoRA を固定融合 | bf16（非枝刈り）・int8 ConvRot（34.0GB＝非枝刈り） | 非枝刈り | 不明 | 66.2／34.0GB | 8〜12 ステップ・res_multistep・Turbo LoRA 0.75。専用ノード Easy-H3 | 旧 int8 は壊れていて「再ダウンロードを」と告知 |
| MiniMax H3 SparseRef15 Hybrid（2900456）Aki7777777 | 8,019／355 | Hybrid（FL2VA 土台。散らばった 15 ブロックの AdaLN が Ref2VA。LoRA のマージなし） | int8 ConvRot・Partial-INT8（AdaLN と時間条件は FP32）・int4・bf16 | 枝刈り | Hybrid | 20.97／12.5／40.3GB | 推奨値の記載なし。LoRA は FL2VA 用を使う | NOTICE なし |
| 10eros_Max_Int8_ref2va（2868392）Stuubzzz（Red） | 4,968／205 | 量子化と強度の焼き込み | int8 ConvRot | 枝刈り | Ref2VA | 20.97GB | — | — |
| MiniMax H3 INT4 ConvRot（2830162）MM744 | 4,448／115 | 量子化 | int4 | 両方 | 両方 | 11.3／17.9GB | — | — |
| H3 Eros MAX GGUF（2902153） | 2,134／76 | GGUF 化 | Q8_0／Q6_K／Q5_0 | 両方 | — | 16〜37GB | — | — |
| PinkCherry fl2va v1_final（2938352）extraltodeus | 1,581／71 | Turbo 統合 | int8 ConvRot | 枝刈り | FL2VA | 20.97GB | — | — |
| Base MinimaxH3 Turbo Merge 8-Step（2951893）Felldude | 1,274／52 | Turbo 統合 | int8・nvfp4・mxfp8・bf16 | 両方 | 両方 | 20.95GB ほか | 8 ステップ。「ConvRot のグループサイズは 64（公式は 256）」 | 派生物の作成を禁止 |
| MINIMAX-H3-hybrid-int8（2930277）mystifying | 659／56 | Hybrid（B25〜49 を REF2VA に） | int8 ConvRot | 枝刈り | Hybrid | 20.97GB | — | 転載品（作者は hoidhxd） |
| MiniMax H3 QuantFunc A4W4（2977438） | 403／10 | 量子化と Turbo | int4（SVDQuant）。専用ノードが要る | — | FL2VA 4 ステップ／REF2VA 8 ステップ | 12.1GB | — | NOTICE の再配布義務を明記 |

- ライセンスの表示: NOTICE を明示して付けているのは QuantFunc だけだった。DaSiWa は本文に条項を引用している。残りの大半は CivitAI の権限欄だけ。QuantFunc と AiAngel の説明文に「H3 Community License は EU・英国・韓国・米国を標準の許諾から除外」という記述がある。

**配布形式の分布**

- 上位 30 本の集計（事実）: safetensors の int8 が 22 本、bf16 が 8 本、fp8 が 6 本、GGUF が 6 本、int4 が 5 本、nvfp4 が 4 本。bf16 だけを配る作品は見つからなかった。GGUF は少数派だが、10 月にも新版が出ている（Eros MAX GGUF 10-10）ので、廃れてはいない。
- ヘッダを実測した int8 ファイル 14 本（公式 3・コミュニティ 11）は、すべて `int8_tensorwise`＋`convrot:true` だった（実測）。ただし中身の揺れは大きい。

**ヘッダの揺れ 8 点（本製品のローダが吸収すべきもの）**

1. 接頭辞: 無しと `model.diffusion_model.` の 2 通り（Singularity と Cobijada が後者）。
2. `comfy_quant` に余分なキーが付く（`per_row`／`per_channel`／`orig_dtype`。DaSiWa は `orig_dtype`・`per_row`、Eros beta6 は `per_row`、MATLOWAI は `per_channel`）。`convrot_groupsize` は 256 と 64 が 1 ファイル内で混在する。
3. 量子化を `comfy_quant` ではなく旧式の `__metadata__._quantization_metadata` で記述する版がある（Cobijada と FastH3。道具名は「Star Ultimate Model Converter」）。
4. 量子化から外す層が配布物ごとに違う。4〜5 ブロックを bf16 で残す版（Eros beta6 は 4 ブロック）、token_refiner・condition_proj・final_layer まで int8 にする版（Eros beta6）がある。
5. AdaLN の dtype が F16／F32／BF16 で揺れる（DaSiWa は AdaLN と patch_proj が BF16、PinkCherry は AdaLN が F32）。`adaln_t_table` に BF16 の版がある（Eros beta4）。
6. 枝刈りの形が揺れる。ランクは 8 と 16、格子は 1025 と 4097（PinkCherry v1.5 は rank16・格子 4097。v1.8 は メタデータ が rank16 なのにテンソルは [1025,8]）。
7. 想定外の追加テンソル: `adaln_basis`／`adaln_mean`（MATLOWAI）／`blocks.N.attn.to_gate_compress.weight`（VSA）／`h3.*` の U32 キー。
8. `__metadata__` は無い・config 付き・学習ログ付きの 3 通り。

ヘッダの取得は、Hugging Face 上の同一ファイル（CivitAI と同じバイト数）から HTTP Range で行った。**CivitAI からの直接取得は 401 で拒否された。** また CivitAI の「fp」欄は当てにならない（Eros beta5 の「fp8」は w4a8、tsolful の W6A8 は「int8」表示）。

**変換の道具**: dasiwa-quant-station・Star Ultimate Model Converter・h3ddle の `convert-turbo-package.py`・PinkCherry の自前ツール（`int8_convrot_mse_clip` を記録）・Kijai（tsolful の int4）。量子化の書式は ComfyUI の `QUANTIZATION.md` で公開仕様である（https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md ）。

**Hybrid の実態**

- 作り方（事実）: 重みの再学習はなく、テンソル単位で差し替えている。解説記事の説明は「FL2VA を土台に、後段ブロックの `adaln_proj` だけを Ref2VA から取る」。既定は 25〜49 で、30〜49 は画質寄り、15〜49 は参照寄り（https://comfyui-wiki.com/en/news/2026-08-11-minimax-h3-fl2va-ref2va-hybrid ）。読み込み時に合成する独自ノードが ComfyUI_MinimaxH3HybridLoader（https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader ）。合成済みファイルの例は sandpies（メタデータ に `hybrid_preset: block_range_adaln_25-49`）と mystifying／hoidhxd（B25〜49）。
- 別方式もある: xmarre と MATLOWAI は Ref2VA の差分をランク 1024 で融合、SparseRef15 は 15 ブロックの AdaLN を差し替え、Eros は参照差分を全ブロックに適用、XUELUO は 25〜49 層をマージ、PulpCut は差し替え用 AdaLN だけを独自の補助ファイル（`h3.hybrid_adaln.*`）に入れている。45〜49 だけにするべきだという異論もある。
- 推論への影響（事実）: ComfyUI 本体のモデルコードは FL2VA／Ref2VA を同じ構造として扱う。違いは列の並びで、FL2VA は `[text|cond rows|audio|video]`、Ref2VA は `[text|reference blocks|audio|video]`。参照入力は `MiniMaxH3ReferenceToVideo` 側で組む（`comfy/ldm/minimax/model.py` の冒頭コメント・Issue #15563 の再現グラフ）。したがって FL2VA 用のローダで Hybrid を読んでも、読み込みは成功する。参照経路を使うかどうかは条件付けの組み方で決まる。FL2VA ローダで読んで不具合が出たという報告は見つからなかった。
- Hybrid 独自ノードの不具合報告: 非枝刈り int8 で画面がちらつく（#5）、w4a8 ではキーの集合が違って合成できない（#2）、bf16 でアクセス違反（#9）、細部が膨らむ（#11）。
- 事実: 公式の枝刈り版は FL2VA と Ref2VA で `adaln_t_table` の値が違う（SHA1 不一致。bf16 版と int8 版では同じ値）。
- 推測: 枝刈り版同士で `adaln_proj` を差し替えると rank8 の基底がずれる可能性がある。DaSiWa の Hybrid は独自の `adaln_t_table` を持つので、密な重みで合成してから枝刈りし直したとみられる。

**Turbo 統合版**

- 上位 30 本のうち約 9 本が Turbo 統合版を持つ。DaSiWa（4／8 ステップ）・Eros（8＋4 を融合）・Felldude（8 ステップ）・extraltodeus・Heptagram・RedCraft（Lightning）・QuantFunc・molbal（GGUF）。Hugging Face には MATLOWAI（約 55 万 DL）・PulpCut・10Eros 8stepTURBO もある。
- safetensors で配られているものはすべて int8 ConvRot（一部 int4／nvfp4 を併置）で、「置くだけ」の対象になり得る。
- 推奨 shift はばらつく（映像 6〜12、Heptagram は 6〜16）。値はモデルごとに既定を持たせる必要がある（推測）。

**動作の確認（2026-10-18）**: PinkCherry Hybrid は、本製品のフォルダに置くだけで読めて生成が通った（オーナー報告。裁定 52）。構造が同じことの根拠は、上の「推論への影響」の ComfyUI のコードの読解（FL2VA 用のローダで Hybrid を読んでも読み込みは成功する）であり、実測ではない。対応の保証は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §12 のとおり後回し。

### 13.4 実装ベースと周辺の現状

**diffusers（Apache-2.0）**

- 単一ファイル読み込みが入った（PR #14839、2026-10-01 マージ）。Comfy-Org の bf16 版を対象にしている。判別キーは `token_refiner.blocks.0.attn.qkv_proj.weight`。変換表は blocks→transformer_blocks、qkv の分割（Comfy 系は `[q;k;v]` の積み重ね、MiniMax 公式の分割ファイルはヘッドごとの交互配置）、fc1 の `[gate;value]` を `[value;gate]` に並べ替え。**`adaln_t_table` があると ValueError で拒否する**（枝刈りは非対応のまま）。`comfy_quant`／`weight_scale` を扱う処理もない。
- 「partitions」は、transformer の 2 つのチェックポイント（`transformer/`＝t2va・fl2va、`transformer_ref/`＝ref2va）を指す。
- attention は `dispatch_attention_fn` に mask なしで渡される。そのため `flash`/`_flash_3_hub`/`sage`/`sage_hub`/`_sage_qk_int8_pv_fp16_cuda` などの全バックエンドが使える。
- 尺の上限: 15 秒ちょうど（362 フレーム）を通す PR #14742 はマージされずに自動で閉じた。現在の diffusers で作れる最長は **345 フレーム（14.375 秒）**。
- 低 VRAM レシピの組み合わせ（torchao int8 v2 ＋ group offload の `use_stream=True` ＋ `low_cpu_mem_usage`）は、pin_memory で落ちる。issue #14433 は未解決。また diffusers の低 VRAM の手順は 24〜32GB 向けで、16GB は想定されていない（監督の査定）。
- 推測: Comfy 形式の int8 ConvRot を diffusers で読むには、int8 Linear の差し替え（DiffSynth の `ComfyKitchenLinear` と同じ形）か bf16 への逆量子化、枝刈り用の時刻埋め込みの差し替え、qkv・fc1 の並べ替えの 3 つが必要になる。

**DiffSynth-Studio（Apache-2.0）**

- 例 `examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py` がある（#1620、2026-08-20）。Comfy-Org の `minimax_h3_fl2va_pruned_int8_convrot` と `qwen3vl_32b_minimax_h3_int8_convrot` をそのまま指定している。VAE は MiniMax 公式のものを使い、ディスクオフロード・`vram_limit`・832×480・124 フレーム・50 ステップ。
- 仕組み: `diffsynth/core/quant/backends/comfy_kitchen.py` が `comfy_quant` の印を読み、comfy-kitchen の `QuantizedTensor` を載せた Linear を組み立てる（comfy-kitchen は optional-deps の `quant`）。`diffsynth/models/minimax_h3_dit_comfy.py`（3KB）は、枝刈り版の時刻埋め込みを `adaln_t_table` の線形補間で再現する。`MiniMaxH3TextEncoderComfyOrgStateDictConverter` が Comfy 形式の名前を Qwen3-VL の名前に写す。
- attention は今も区間ごとの Python ループ（`_sdpa_varlen_attention`）。10 月の追加は VDN（ハイブリッド attention）対応（#1743、2026-10-13）。

**comfy-kitchen（Apache-2.0、v0.2.37、2026-10-02 公開）**

- `TensorWiseINT8Layout` の仕組み: 重みは保存前にグループごとのアダマール行列 H で回転してある（W·Hᵀ、H は 4×4 のクロネッカー積で作る「regular」型。サイズは 4 のべき乗に限る）。実行時は活性化に x·H をかけ、行ごとに動的に int8 化し、int8×int8 の行列積（int32 で累積）を取って、スケールを戻す。ConvRot は重み側のときだけ許され、出力チャネルごとのスケールが必須。
- 行列積の実装は 3 段構え: eager は `torch._int_mm`（cuBLASLt）／triton カーネル／CUDA 拡張（SM≥8.0 なら CUTLASS の int8 GEMM と逆量子化の融合、駄目なら cuBLAS int8。groupsize 256 の時は回転と量子化を融合したカーネルがある）。
- Windows x64 用の CUDA wheel があり、Windows の既定アーキテクチャは `75-virtual;80;89;120f`（sm_89 を含む）。要件は PyTorch ≥2.7・**CUDA ランタイム ≥13.0・ドライバ R580 以上**。`dependencies = []` なので単体で import できる（CUDA 拡張が読めない時は eager/triton だけで登録される）。
- 未解決の issue #136: triton の int8 GEMM が出力位置を int32 で計算しており、m×n ≥ 2³¹ で不正メモリアクセスになる（H3 の高解像度×長尺で発生）。
- 推測: torch 2.9.1+cu128 の環境では CUDA 拡張が使えず、eager 経路（`torch._int_mm`）になって数倍遅くなる見込み。

**ComfyUI 本体（GPL-3.0。コードは移植せず、仕様書として読むだけ。§7 の方針どおり）**

- PyTorch の CUDA 版が 13 未満（`torch.version.cuda < 13`）だと comfy-kitchen の CUDA バックエンドを無効にし、triton も既定で無効にする（`comfy/quant_ops.py`）。
- ヘッダの揺れの吸収方法（コードで確認）: 判定は `video_patch_proj`＋`audio_patch_proj` があれば H3。層数・幅・ヘッド数はすべて形状から読み、`adaln_t_table` の形状から格子数とランクを取る（rank16 も通る）。`to_gate_compress` があれば VSA 版。`__metadata__["config"]` があれば上書き（`comfy/model_detection.py`）。接頭辞は剥がす（`unet_prefix_from_state_dict`）。旧式の量子化記述 `_quantization_metadata` や `scaled_fp8` は変換する（`convert_old_quants`）。`comfy_quant` は `format`・`convrot`・`convrot_groupsize` だけを読み、未知のキーは無視。`params` の下に入れ子にした書き方も受け付ける。`comfy_quant` が無い層は通常の浮動小数点で読む（`comfy/ops.py`）。9 月 20 日以降 `blocks.N.attn.comfy_attention.config` という注意機構の指定が増えた（PR #16419）。つまり ComfyUI は 13.3 の揺れ 8 点を、おおむね「形状から構成を導く・知らないキーは無視する」で吸収している。
- 不具合の事例: Remix の旧 int8 は壊れていた（作者が差し替え）／公式の bf16 版を ComfyUI v0.32 で動かすと真っ黒になるが int8 は正常（Issue #15563・未解決）／4bit パックの形状を読み誤る問題の修正提案（PR #15620）／fp16 で推論すると活性値があふれる（PR #16508）／h3ddle の `input-major-v1` 配置は独自形式。

**ライセンス一覧**: comfy-kitchen Apache-2.0・comfy-quants GPL-3.0・diffusers Apache-2.0・DiffSynth-Studio Apache-2.0（ComfyUI 本体は §7 のとおり GPL-3.0）。comfy-quants GPL-3.0 は GitHub の API の license 欄・2026-10-16 確認（comfy-kitchen Apache-2.0・diffusers Apache-2.0・DiffSynth-Studio Apache-2.0 も同じ方法で確認）。

### 13.5 16GB・Ada 世代の実測

すべて第三者による報告で、ComfyUI・Windows のもの（二次情報。本製品での実測ではない）。

- **RTX 4060 Ti 16GB**・RAM 64GB・int8 ConvRot・864×480・124 フレーム・20 ステップ: cu129 で 12 分 50 秒、torch 2.13+cu130 で 5 分 55 秒。https://note.com/tank_ai/n/nab26edd4ab96
- **RTX 4080 16GB**・torch 2.13+cu130・FlashAttention 2.8.3・int8 ConvRot・18 ステップ: 5 秒の動画が 382〜427 秒、10 秒が 19〜22 分（1 ステップあたり約 58〜61 秒）、VRAM 約 14.6GB。解像度・RAM・起動オプションは記載なし。https://note.com/aibijyolab/n/n062ec8f871d9
- **RTX 4080**（ComfyUI issue #15261 の報告。まとめ記事経由の孫引き）: Ref2VA・1280×736・5 秒で 690 秒、RAM 64GB。
- **RTX 4070 Ti SUPER**（まとめ記事の中の第三者報告）・RAM 32GB・Turbo LoRA・8 ステップ・1280×720: 1 ステップあたり 47 秒。Turbo の 4 ステップでは破綻したという報告もある。https://smeltcore.com/recipes/minimax-h3-on-rtx-4070-ti-super-ada-at-16-gb-and-the-turbo-lora-question
- 失敗例: #15628: RTX 4070 **12GB**・Windows で DynamicVRAM が初期化中に止まる／#15665: 5070 Ti 16GB で、v0.32 以降は高解像度で約 4 倍遅い／#17017（2026-10-14）: Windows で単一 GPU モードが強制される。
- `--novram` や block swap を使った 16GB Ada の報告は見つからなかった。

### 13.6 公式の更新

- `MiniMaxAI/MiniMax-H3` の最終コミットは 2026-08-13 のまま。LICENSE にも QA にも変更はない。同じ組織の新しい H3 系の重みもない。GitHub `MiniMax-AI/MiniMax-H3` の最終 push は 2026-08-15。公式モデルカードの記述は現在も同じ（一次資料で確認済み）。
- H3-Regenerate-2K は「not yet open-sourced」。Context-IR と Regenerate-2K は Open Platform の API でだけ使える。ライセンスの適用除外地域（米国・EU・英国・韓国）向けには申請フォームの記載がある。
- 公式エコシステム索引（awesome-minimax-h3-integration、最終 2026-09-17）は、Context-IR の形式を真似るコミュニティ実装 OpenH3-IR（Apache-2.0）と、fal が後学習した重み非公開の「H3 Max」系を新しく掲載した。
- 推奨設定（ステップ数 20・shift 12/3）にも変更の形跡はない（推測）。
- フレーム規約と解像度（事実）: 公式モデルカードは今も 4〜15 秒・24 FPS・短辺 768。diffusers は 5〜15 秒・`17n+5` への切り上げ・32 の倍数。VAE は空間 16 倍・時間 4 倍・潜在 24ch（VAE のメタデータで再確認）。DiT は patch [1,2,2]（config で再確認）。1 トークンあたり 32px 四方に相当する。

### 13.7 本製品の現物との照合

外部の α版ガイド（13.8）は本製品の実装を知らずに書かれている。監督が本製品の実在する資産と突き合わせ、次の 6 点を補正した（監督の査定）。

1. **int8 ConvRot の実行方式は既にある。** LTX 2.5 で使っている int8 ConvRot の読み込みが `engine/sft_quant` にある。重みは int8 のまま保持し、Linear の呼び出しごとに bf16 へ戻す方式で、外部ガイドの §9 が「最終実装にしないこと」と言う方式にあたる。W8A8 の行列積（重みも活性化も 8bit のまま掛け算する方式）は本製品に無い。α版は既存の方式で通し、W8A8 にするかどうかは計測の後で切り分ける。また既存の読み込み（`engine/sft_quant/dequant.py` の `CONVROT_GROUPSIZE = 256`）は groupsize 256 だけを受け付ける。公式の枝刈り版は全部 256 なので α版は通るが、非枝刈り版の AdaLN 射影（64）やコミュニティの 64 の配布物（13.3）は通らない。

   > **訂正（2026-10-17）**: この査定のうち「α版は既存の方式で通し、W8A8 にするかどうかは計測の後で切り分ける」は、オーナー裁定 16 で誤りとされた。α版の int8 の実行は comfy-kitchen の CUDA 拡張に任せる。既存の読み込みは、ヘッダの判定には使えても H3 の実行経路には使わない。
2. **ブロック転送は実装済み。** 1 ブロックずつ・ページロックメモリ・非同期・先読み・二重バッファの転送は、実装済みである。block swap の本体は `engine/transformer/block_swap_service.py`、先読み・ページロックメモリ・非同期転送は `engine/transformer/block_swap_prefetch.py`（LTX 2.3 と 2.5 で共通。2.5 は `engine25/gguf_transformer.py` がこれを読み込んで使う）。量子化したままブロックが載る（`block_swap_prefetch.py` の説明文と `engine/sft_quant/quant_service.py` の説明文末尾「ride block swap / prefetch」）。H3 では新しく作るのではなく、H3 のブロック構造への接続になる。
3. **テキストエンコーダの層オフロードにも前例がある。** LTX 2.3 の `--te-offload` と LTX 2.5 の `te_layers_on_gpu`。

   > **更新（2026-10-18）**: H3 では既存の層オフロードを使わず、層ごとにファイルから直接読む（裁定 32。設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §4）。
4. **diffusers の低 VRAM の手順は 24〜32GB 向け**で、16GB は想定外である（13.4）。
5. **時間の見積もりが無い。** H3 は full attention（全トークンどうしで注意を取る方式）で、外部ガイドの 1344×768・345 フレームはトークン数が 9 月の 16GB 実測の条件（§5 の 640×384・226 フレーム）の約 6 倍になる。
6. **製品への組み込みが書かれていない。** 記述子（manifest）・`install-H3.bat`・専用の仮想環境・アダプタ・フレーム規約 17n+5・`featureScope.ts`・快適上限・Gradio と MCP。仕分けの考え方は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §9（「そのまま使い回すもの」と「エンジン系統に固有のもの」に仕分けるだけ）のとおりで、**フレーム規約 17n+5 への対応が本丸**であること（本書 §9）は今も有効である。

   > **訂正（2026-10-17）**: インストーラの名前は `install-MiniMaxH3.bat`（裁定 15）。

### 13.8 α版の範囲（外部ガイドの要点と裁定後の形）

**これは設計ではなく、範囲のメモである。** 設計正本は [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md)。外部ガイドは、オーナーが 2026-10-16 に別のエージェントとの議論の結果として共有した「Nz-Videomni MiniMax H3 対応 α版 スピードガイド」で、著者は 9 月 6 日の調査と本製品の実装を知らない。ここではその要点を、§10 の 2026-10-16 の裁定（6〜12）で修正した形で記す。

> **更新（2026-10-17）**: α版の範囲と完結条件は §10 の裁定 13〜21 で確定した。第 1 弾はワーカーのセルフテストで 1 本、α版の完結は操作パネルからの T2V と I2V（先頭フレームのみ）と `install-MiniMaxH3.bat` と README まで。快適上限マーカーは非表示（較正は台帳 §1-95）、ステップ数と shift は第 1 弾では定数（画面に出すのは第 2 弾。裁定 22）、末尾と中間キーフレームは後回し（同 §1-97）。下の「監督が賛同した点」の 3 つはそのまま計画の材料とし、実装の土台の修正点 4 つと論点 4 つは裁定 17 にある。

> **更新（2026-10-19）**: 末尾と中間のキーフレームは実装済み（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-241、設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3）。

α版で対応するもの:

- DiT は公式系 FL2VA の枝刈り int8 ConvRot（`minimax_h3_fl2va_pruned_int8_convrot.safetensors`。Comfy-Org/MiniMax-H3 の diffusion_models）1 本。
- タスクは T2VA と FL2VA（同じ重み）。音声も生成する。
- テキストエンコーダは int8 ConvRot 27.14GB（`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`。同 text_encoders）。
- CPU オフロードは、既存の block swap を H3 のブロック構造に接続する（13.7 の 2）。
- 解像度と尺は快適上限の計測で決める。1344×768・345 フレームを先に固定しない（裁定 9）。
- 置くだけで動くことを理想とする（裁定 6）。揺れ（13.3）を吸収する範囲は設計正本 §7。
- 配布は公開リポジトリから直接ダウンロードする（裁定 7）。

α版では後回しにするもの:

- LoRA と Turbo（裁定 10。Style LoRA に対応するときに一緒に進める）。
- Ref2VA・Hybrid・他の形式（fp8・w6a8・GGUF・int4・nvfp4・部分 int8 などの混在精度）・1088p と 2K・sparse attention。

監督が賛同した点（オーナーの裁定ではない）:

- **3 段の推論**: 条件付け（テキストエンコーダで条件を作り、終わったら解放）→ ノイズ除去（DiT をブロックごとに GPU へ送る）→ デコード（DiT を解放してから VAE を載せる）。
- **判定はヘッダで**: ファイル名やファイルサイズで判定せず、safetensors のメタデータ・テンソル名・形状・dtype・量子化の記述子で判定する。テンソルごとに dtype と量子化を持てる内部表現が望ましい。
- **差し替えられる境界**: α版の実装は 1 種類だけでも、重みの形式・AdaLN の形（枝刈り／非枝刈り）・条件付け（FL2VA／Ref2VA／Hybrid）を後から足せる境界を残す。

外部ガイドから採用しないもの:

- §18 の配布方針（Nz-Videomni 用の Hugging Face リポジトリから取得する）。裁定 7 で不採用。
- 「1344×768・345 フレームを正式目標」。裁定 9 で不採用。

### 13.9 未検証事項の更新

§11 の各項目の進み具合:

- テキストエンコーダ 3 案（§8）の実測比較: 裁定 8 でまず int8 ConvRot 27.14GB を使うことになり、3 案は将来の選択肢に回った。
- 量子化の実際の出力品質: 新しい一次情報はない。
- bitsandbytes の Windows 動作: 新しい一次情報はない（③NF4 案は将来の選択肢に回った）。
- VAE タイル処理の品質: 新しい一次情報はない。
- Ada 世代・Ampere 世代での初回実測: Ada 世代は第三者の値が出た（13.5）。**Ampere 16GB の実測はまだ見つからない。**
- FL2VA と Ref2VA の重み内容が異なること: 確定（ヘッダは同じ・sha256 は違う。13.2）。
- DiffSynth の attention の差し替え: 区間ループのままなのが確定。diffusers のほうは mask なしで全バックエンドが使えるので、こちらが参考になる（13.4）。

新しい項目:

- **ガイダンス蒸留（negative prompt なし）の真偽**: ComfyUI の Issue #15563 の再現グラフは cfg 4、PinkCherry の メタデータ は `guidance_scale 4.0` だった。真偽は未確認。
- **comfy-kitchen の triton の issue #136**: m×n ≥ 2³¹ で不正メモリアクセスになる。高解像度×長尺で当たるため、どの行列積の経路を使うかに関わる。
- **SageAttention の cu130 向け wheel の有無**: H3 の仮想環境を cu130 にした場合、既存 2 エンジンと同じ wheel は使えない（推測）。有無は未確認。
- **H3 専用の仮想環境を cu130 にしたときの利用者側の要件**: ドライバ R580 以上と、ディスクの増加（裁定 11 で README に書く）。どれだけ増えるかは未測定。
- **枝刈り Hybrid の基底のずれ**: 枝刈り版どうしで `adaln_proj` を差し替えると rank8 の基底がずれる可能性がある（推測。13.3）。
- **Ampere 16GB の実測なし**（上の再掲。サブマシンで自前に測ることになる）。

> **更新（2026-10-19）**: 上の項目のうち次のものは片づいた。H3 の仮想環境の利用者側の要件（ドライバ R580 以上・ディスクの増加）は README に書いた（裁定 11）。Ampere 16GB は、サブマシン（RTX 3080 Laptop 16GB）で導入から T2V 1 本まで通した（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §167.8）。SageAttention は H3 の仮想環境（torch cu130）に 2.2.0 post6 が入り、製品で使っている（版の正本は `engine_h3/venv-engine-h3.freeze.txt`。同 §168.8 の環境）。comfy-kitchen の triton の issue #136 は、triton の経路を切って CUDA の経路だけを使うので当たらない（`engine_h3/quant_linear.py` の `require_cuda_int8`。m×n ≥ 2³¹ の CUDA 経路の実測は台帳 §1-95 に預けた。VRAM 16GB mode の on では届かず、off でだけ届きうる＝設計正本 §13）。ノイズ除去の段の VRAM の溢れは VRAM 16GB mode で解消した（台帳 CLOSED 3-240・同 §169）。

### 13.10 出典（2026-10-16 の追加分。§12 と重なるものは省く）

一次資料:

- [Comfy-Org/MiniMax-H3 のファイル一式（2026-10-16 にヘッダを実測）](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)
- [ComfyUI `QUANTIZATION.md`（量子化の書式の公開仕様）](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md)
- [ComfyUI issue #15563](https://github.com/Comfy-Org/ComfyUI/issues/15563) / [PR #15620](https://github.com/Comfy-Org/ComfyUI/pull/15620) / [PR #16508](https://github.com/Comfy-Org/ComfyUI/pull/16508) / [PR #16419](https://github.com/Comfy-Org/ComfyUI/pull/16419)
- [diffusers PR #14839（単一ファイル読み込み）](https://github.com/huggingface/diffusers/pull/14839) / [diffusers の MiniMax-H3 文書（ソース）](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_h3.md)
- [comfy-kitchen（GitHub）](https://github.com/Comfy-Org/comfy-kitchen) / [comfy-kitchen（PyPI）](https://pypi.org/project/comfy-kitchen/)
- [ComfyUI_MinimaxH3HybridLoader](https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader)
- CivitAI API（`https://civitai.com/api/v1/models/<ID>`。一覧は `https://civitai.com/api/v1/models?baseModels=MiniMax%20H3&sort=Most%20Downloaded&nsfw=true&types=Checkpoint`。数値は 2026-10-16 時点）

二次情報:

- [ComfyUI Wiki: FL2VA と Ref2VA の Hybrid](https://comfyui-wiki.com/en/news/2026-08-11-minimax-h3-fl2va-ref2va-hybrid)
- [note: RTX 4060 Ti 16GB の実測（tank_ai）](https://note.com/tank_ai/n/nab26edd4ab96) / [note: RTX 4080 16GB の実測（aibijyolab）](https://note.com/aibijyolab/n/n062ec8f871d9)
- [smeltcore: RTX 4070 Ti SUPER と Turbo LoRA](https://smeltcore.com/recipes/minimax-h3-on-rtx-4070-ti-super-ada-at-16-gb-and-the-turbo-lora-question)
- ComfyUI issue #15261・#15628・#15665・#17017、comfy-kitchen issue #136、diffusers PR #14742・issue #14433、DiffSynth-Studio #1620・#1743（いずれも 13.4・13.5 で言及したもの）

## §14 推論高速化の事前調査と台帳への配置（オーナー共有・2026-10-21）

2026-10-21 にオーナーが共有した事前調査の記録である。台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) の §1-105 と、§3 に新しく起票した 4 件（§3-245〜§3-248）の材料として置く。本製品の内部実測の数値は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3（A-G5）と `engine_h3/constants.py` で裏取りした。外部のベンチマークは裏取りしておらず、Nz-Videomni の実測値として扱ってはならない。

### 14.1 §1-105（第 1 弾：数値等価）の 4 件の材料

- **QKV の活性化量子化の共有**: VRAM 16GB mode の経路では、Q／K／V の射影を 7 つのヘッド群に分けて計算している。同じ入力に対する INT8 ConvRot の前処理を共有し、重複した計算を減らす。
- **Cyclic prefetch（ステップ間の先読み）**: 次のステップの冒頭で使うブロックの重みを先読みする。節約できる GPU の待ち時間が小さい可能性があるので、20 ステップの生成で 5 秒程度以上の改善の余地があるかを事前に確かめる。
- **CPU から pinned staging（固定メモリの中継領域）への先読み**: CPU 側の重みのコピーを GPU の計算と重ねる。評価するのはコピーの処理時間ではなく、GPU の待ち時間がどれだけ減るかである。
- **QKV のコピーとレイアウト変換の削減（条件付き）**: SageAttention を呼ぶ前の `contiguous()` などが対象である。既存の設計は strided な Q／K の渡し方を採っていない（設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §6.5 の「採らないもの」）ので、数値の一致と VRAM の使用量を検証し直してから扱う。

> **更新（2026-10-25）**: 4 件を調べて実測し、結論を出した（数値はすべて [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §177）。**QKV の活性化量子化の共有は採用**し、出力を変えずに 1 ステップが短くなった（設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §5 の例外・§6.5）。**Cyclic prefetch と pinned staging への先読みは見送った**。GPU が待つのはステップの頭だけで、20 ステップでの短縮が 14.6 の「原則として見送る」の範囲にとどまると実測で分かったため。**QKV のコピーとレイアウト変換の削減**は、台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-255 に起票し直した。台帳 §1-105 はクローズした（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-254）。

参考資料:

- https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine_h3/quant_linear.py
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine_h3/dit.py
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine/transformer/block_swap_prefetch.py
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/Outputs-archive/h3-roadmap-2026-10-19/OWNER_memo_speedup_longvideo.md
- https://github.com/hao-ai-lab/FastVideo/blob/main/scripts/benchmarks/minimax_h3_4090/README.md

### 14.2 ヘッド群（Head group）の拡大

既存の内部実測（RTX 4070 Ti SUPER・I2V 896×1152・124 フレーム・SageAttention・2 ステップ。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3 の A-G5。現在の値 `VRAM16_HEAD_GROUP = 8` を定める `engine_h3/constants.py` のコメントにも同じ数値がある）:

| ヘッド群 | 1 ステップ | 割り当て（Allocated） | 予約（Reserved） |
|---|---|---|---|
| 8 | 21.74 秒 | 4,411 MiB | 7,262 MiB |
| 14 | 20.99 秒 | 4,965 MiB | 7,264 MiB |
| 28 | 20.54 秒 | 6,257 MiB | 8,354 MiB |

- 20 ステップに換算すると、14 で約 15 秒、28 で約 24 秒の短縮になる。ただし実測は 124 フレーム・2 ステップのもので、20 ステップを通して測ったものではない。
- VRAM の使用量が増えると、快適上限のフレーム数が下がる可能性がある。124 フレームではヘッド群 14 の予約はほとんど増えていないが、長尺の生成での挙動は確かめていない。既存の実測から線形に外挿すると、8→14 で快適上限が約 16% 下がるという見積もりも出るが、粗い推定であって実際の上限ではない。
- 採否の判断で重く見る点:
  1. 台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-95 で、現行のヘッド群 8 の快適上限の較正が済んでいること。
  2. ヘッド群を変えたことで快適上限が 20% 以上下がるなら、採用を考え直す。
  3. LTX 2.3〜2.5 と同じ水準のフレーム数を保つという設計目標を優先する（オーナーの基準。設計正本には書かれていない）。
  4. 数値の差・品質・OOM〔VRAM 不足による停止〕の危険と、速度の向上とを費用対効果で比べる（基準は 14.6）。

> **更新（2026-10-25）**: 上の表の速さの差の大部分は、ヘッド群が大きいほど qkv の入力の量子化の回数が減ること（群 8 で 1 ブロック 21 回・14 で 12 回・28 で 6 回）で説明できる見込みだった。その量子化を 1 ブロック 1 回にした（14.1 の更新）ので、**ヘッド群を広げる残りの得は、上の表よりかなり小さくなる見込み**である（推測。新しい群 8 の速さは [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §177。14／28 は測り直していない）。

参考資料:

- https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine_h3/constants.py
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/MINIMAX_H3_DESIGN.md
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/Outputs-archive/h3-vram16-2026-10-18/gates/G5_summary.jsonl
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/Outputs-archive/h3-vram16-2026-10-18/gates/G4_summary.jsonl
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/COMFORT_LIMIT_TABLE.md
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/PENDING_TASKS.md

### 14.3 INT8 ConvRot のカーネルの高速化

- **Bit-exact Triton fusion**: FastVideo には、RoPE・AdaLN・SwiGLU でビット一致を目指した融合カーネル〔複数の演算を 1 回の GPU 呼び出しにまとめたもの〕がある。Nz-Videomni の出力と一致するかは確かめていない。
- **通常の kernel fusion**: 演算をまとめて、メモリへの読み書きとカーネルの起動を減らす。丸めの順序が変わるので数値の差が出うる（設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §6.5 の「採らないもの」と、裁定 46 の門の対象）。
- **INT8 GEMM epilogue fusion**: 行列積（GEMM）の直後の後処理（scale・bias など）を行列積に融合する。comfy-kitchen の既存の最適化と重なっていないかの調査が要る。
- **AdaLN〔条件づけ用の正規化〕の計算の再利用**: 重複している AdaLN の計算を減らす。枝刈り版の H3 では AdaLN の rank が 8 なので、効果は限られる可能性がある。
- **Regional Compile〔部分的なコンパイル〕／CUDA Graph〔GPU の呼び出し列を記録して再生する仕組み〕**: コンパイルや実行グラフで、呼び出しのオーバーヘッドを減らす。動的な block swap との整合は確かめていない。
- 「INT8 ConvRot の回転の整数化」は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-53 に既にあるので、重ねて起票しない。

参考資料:

- https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/models/dits/minimax_h3_fusions/exact.py
- https://github.com/NVlabs/Sana/tree/sol-engine
- https://github.com/Comfy-Org/comfy-kitchen
- https://docs.pytorch.org/docs/stable/torch.compiler.html
- https://github.com/Rootport-AI/Nz-Videomni/blob/main/Docs/ACCELERATION_RESEARCH_NOTES.md

### 14.4 Sparse Attention（注意の疎化）

- **Sol-Attn**: 学習済みの gate〔どこを間引くかを決める小さな網〕が要らない。MiniMax H3 の INT8 ConvRot での外部実測があり、RTX 5080 16GB・8 ステップで 58.2 秒→45.1 秒。ただしこの改善には、メモリ転送との競合が和らいだことも寄与している。
- **VSA（Video Sparse Attention）**: 学習済みの gate で Attention を疎にする。FastVideo に実装があり、Turbo と併用した例もある。ファインチューンしたモデルで gate が使えるか（互換性）が課題である。
- **SpargeAttn／Cube／SubBlock Sparse**: その他の疎化の方式。H3・VRAM 16GB mode・現行のヘッド分割に合うかと、実装の費用は確かめていない。

参考資料:

- https://github.com/quzopl/ComfyUI-SolAttn-H3
- https://github.com/NVlabs/Sana/tree/sol-engine
- https://github.com/hao-ai-lab/FastVideo
- https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- https://arxiv.org/abs/2505.13389
- https://github.com/thu-ml/SpargeAttn

### 14.5 Cross-step Cache（ステップ間の再利用）

- **TeaCache／FirstBlockCache（FBC）／Cache-DiT／EasyCache**: 過去のステップの残差や中間結果を再利用して、DiT の計算を省く。FBC には、RTX 5090・INT8 ConvRot・20 ステップ・SageAttention2 で 57.96 秒→40.26 秒という外部実測がある。
- 生成の軌道が変わりやすく、4 ステップの Turbo との相性に懸念がある。PSNR・SSIM〔画質の客観指標〕だけでは、主観で見た映像と音声の品質を保証できない。このため優先度は低い。

参考資料:

- https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache
- https://github.com/NVlabs/Sana/tree/sol-engine
- https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md

### 14.6 費用対効果の共通の基準

20 ステップの生成での壁時計の時間（wall time。実際に待つ時間）の短縮を、主な判断材料にする。

| 短縮時間 | 暫定の判断 |
|---|---|
| 1〜2 秒 | 原則として見送る |
| 3〜5 秒 | 小さな改修なら検討する |
| 5〜15 秒 | 実装の負担と比べる |
| 15 秒以上 | 有力な候補 |

- 将来の 4 ステップの Turbo での絶対の短縮時間も考える。
- 外部のベンチマークを Nz-Videomni の実測値として扱わない。
- VRAM の消費が増える方式では、速度の向上だけでなく快適上限への影響も採用の条件に入れる。
- 対象外: FP8・GGUF・NVFP4 への対応。RoPE のテーブルの事前計算は実装済み（`engine_h3/pipeline_h3.py` が生成の冒頭で `rope_tables` を 1 回組む）なので起票しない。

### 14.7 技術の分類と台帳への配置

- **A 数値等価型**: 計算結果を変えずに速くする。
- **B 演算最適化型**: 計算の構造を保ち、ごく小さな数値の差を許す。
- **C Attention 近似型**: Attention を間引く。Turbo と原理的に併用できる。
- **D ステップ間近似型**: 過去のステップの結果を流用する。品質の危険が大きく、Turbo との相性に懸念がある。

台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) への配置:

- §1-105 = A 群の具体的な改修（14.1。3-254 でクローズ）。
- §3 の改修項目 = ヘッド群の拡大（§3-245。14.2）・INT8 ConvRot のカーネルの高速化（§3-246。14.3）。
- §3 の研究課題 = Sparse Attention（§3-247。14.4）・Cross-step Cache（§3-248。14.5）。
- ヘッド群の拡大は速度の向上が実測されているが、快適上限を損なう可能性があるので、§1-95 の較正が済むまで着手しない。§1 への新しい起票はしない。


## §15 Ref2VA の一次資料調査（2026-10-28）

台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-107（Ref2VA〔参照素材からの映像と音声の生成〕）の計画のために、2026-10-28 に行った一次資料の調査の記録である。ファイルはダウンロードせず、ヘッダと表だけを HTTP Range で取得した。本製品がどう作ったかは設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3.2（と裁定 82）にあり、本節と食い違えば設計正本を採る。行番号は下の版に対するもの。

### 15.1 参照した版

| 出典 | 版 |
|---|---|
| ComfyUI（Comfy-Org/ComfyUI） | commit `08cf0284ff45723aa67841c80cabb0e3cb50c9d7` |
| MiniMax 公式 GitHub（MiniMax-AI/MiniMax-H3） | commit `d21241f0a4b3acbb34c97dae47fa417b7065e438` |
| SGLang（sgl-project/sglang） | commit `54be2010d310577c62b23e4b3857f45a5b833d5f` |
| DiffSynth-Studio（modelscope/DiffSynth-Studio） | commit `e5a44c42776978457b61d51d595a1f6746c3d87a` |
| ComfyUI の文書（Comfy-Org/docs） | commit `491d3262bee90683b02821551f5bcd1f2939b1cc` |
| ComfyUI の雛形（Comfy-Org/workflow_templates） | main（2026-10-28 取得） |

URL の書式は `https://github.com/<repo>/blob/<commit>/<path>#L<n>`。ComfyUI の略称は、te＝`comfy/text_encoders/minimax.py`、nodes＝`comfy_extras/nodes_minimax_h3.py`、model＝`comfy/ldm/minimax/model.py`、base＝`comfy/model_base.py`、vae＝`comfy/ldm/minimax/vae.py`、avae＝`comfy/ldm/minimax/audio_vae.py`。SGLang のファイルは `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/` の下、DiffSynth は `diffsynth/pipelines/minimax_h3_audio_video.py`。

### 15.2 配布物と SHA-256（一次資料で確認済み）

出典: `https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/diffusion_models?expand=true`（sha256 は各要素の `lfs.oid`、バイト数は `size`）。全ファイルの最終コミットは `3f57e829…`「Initial commit」（2026-07-30）。

| ファイル名 | バイト数 | sha256 |
|---|---:|---|
| minimax_h3_fl2va_bf16.safetensors | 66,280,487,368 | 907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182 |
| minimax_h3_fl2va_int8_convrot.safetensors | 34,038,892,334 | 7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5 |
| minimax_h3_fl2va_pruned_bf16.safetensors | 40,225,724,176 | a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4 |
| minimax_h3_fl2va_pruned_fp8_scaled.safetensors | 20,958,205,608 | 12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993 |
| **minimax_h3_fl2va_pruned_int8_convrot.safetensors** | **20,970,379,616** | **e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a** |
| minimax_h3_fl2va_pruned_w6a8.safetensors | 15,983,746,636 | ac746a2e41628ab25afd44d2b22a7fab8d7e66cd07a01bf89ed9d74b0d3f0c35 |
| minimax_h3_ref2va_bf16.safetensors | 66,280,487,368 | e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026 |
| minimax_h3_ref2va_int8_convrot.safetensors | 34,038,894,550 | 9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9 |
| minimax_h3_ref2va_pruned_bf16.safetensors | 40,225,724,176 | 37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe |
| minimax_h3_ref2va_pruned_fp8_scaled.safetensors | 20,958,205,608 | f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9 |
| **minimax_h3_ref2va_pruned_int8_convrot.safetensors** | **20,970,379,616** | **9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779** |
| minimax_h3_ref2va_pruned_w6a8.safetensors | 15,983,746,636 | ece96bbbce76670ec782de84acc888fd9d9b210bd2ab97c1957b39896734f9f1 |

- 太字の 2 本が本製品の取得するもの（値の正本は記述子 `scripts/manifests/40-minimaxh3.json` の `known_weights`）。バイト数は同じで、sha256 は違う。
- **ヘッダの比較（実測）**: 両者のヘッダ長は 95,416 B で、**ヘッダ JSON はバイト単位で一致**した（テンソル 932 個。`__metadata__` は無い）。`adaln_t_table`（F32・`[1025, 8]`）は両方にあり、8,200 要素のすべてで値が違う（差の最大 0.01385）。ヘッダだけで両者を区別できる項目は無く、区別にはファイル名・sha256・テンソルの値のどれかが要る。§11 の「FL2VA と Ref2VA の重み内容が実際に異なることの確認」はこれで確かめた。
- Hugging Face の README（https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/README.md）には使い分けや推奨設定の記述は無く、Ref2VA 用の 4 ステップ Turbo（`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`）が LoRA の一覧にある。

### 15.3 推奨設定（一次資料で確認済み）

出典: ComfyUI の文書 https://github.com/Comfy-Org/docs/blob/491d3262bee90683b02821551f5bcd1f2939b1cc/tutorials/video/minimax/minimax-h3.mdx と同じフォルダの `minimax-h3-native.mdx`。

- Turbo なしの既定は 20 ステップ。Turbo ありは T2V／I2V が 8、R2V（参照からの生成）・Multiframe・Fun ControlNet は 4（minimax-h3.mdx L141）。
- 「参照の行は毎ステップ効くので、4 ステップの Turbo では参照がほとんど効かない（姿勢や顔がずれる）。参照に強く合わせるなら Turbo を切るか 25 に近づける」（L143）。native の文書も「参照を保ちたいときは 20〜25 ステップ」（L221）。
- shift は映像 12・音声 3（L164）。学習の解像度は短辺 768・面積の上限 768×1344。参照動画はこの枠に縮める。参照画像は `ref_image_size` に従い、`match` は生成の画素数へ、`max` は短辺 2048 へ縮める（L104・L106）。既定は `match`（native L219）。R2V の重みは `ref2va`（native L220）。
- 雛形 `video_minimax_h3_r2v.json` の値: `UNETLoader` は `minimax_h3_ref2va_pruned_int8_convrot.safetensors`、サンプラーは `res_multistep`、スケジューラーは `simple, 20, 1.0`、`MiniMaxH3ReferenceToVideo` は `1344×768, length 124, "match"`。
- Ref2VA の Turbo LoRA（ModelTC/Minimax-H3-Turbo の README、commit 02e26d5 の L73-L81・L128-L137）は 544p・shift 12／3・4 ステップ・参照画像 `match` で学習。公式の Ref2VA の手順（`scripts/readme/reproducible-768p-ref2va-request.sh`）は SGLang の既定（50 ステップ・shift 12／3）。

### 15.4 参照 0 件のときの挙動（実装ごとに違う。一次資料で確認済み）

| 実装 | 参照 0 件（文章だけ） | 参照 0 件で先頭フレームだけ |
|---|---|---|
| MiniMax 公式（README の手順の SGLang 配信） | 拒否 | 拒否 |
| 公式 API（platform.minimax.io） | 「文章から動画」として受ける（どの重みで処理するかは書かれていない） | 「画像から動画」として受ける。参照と先頭・末尾のフレームは同時に使えない |
| ComfyUI `MiniMaxH3ReferenceToVideo` | 通る（列は文章だけの生成と同じ） | ノード単体には先頭フレームの入力が無い。`MiniMaxH3AddGuide` を足せば通る |
| DiffSynth-Studio | 通る（FL2VA と同じ組み方で、条件の行なし） | 通る |

- SGLang: Ref2VA の `model_index.json` は `"partition": "ref2va", "tasks": ["ref2va"]`（https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/Ref2VA/model_index.json#L29-L35）。`release_metadata.py` L161-L176 は区分に無いタスクを拒否し（https://github.com/sgl-project/sglang/blob/54be2010d310577c62b23e4b3857f45a5b833d5f/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py#L161-L176）、`task_profiles.py` L186-L188 の `conditions_required=True` と `request_validation.py` L150-L153 で条件 0 件の ref2va を拒否する（https://github.com/sgl-project/sglang/blob/54be2010d310577c62b23e4b3857f45a5b833d5f/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py#L150-L153）。
- ComfyUI: nodes L294-L364 は参照が無ければ `minimax_refs` を付けない（https://github.com/Comfy-Org/ComfyUI/blob/08cf0284ff45723aa67841c80cabb0e3cb50c9d7/comfy_extras/nodes_minimax_h3.py#L294-L364）。model の `PackedLayout`（L356-L455）は参照もキーフレームも無ければ `[text | audio | video]`（https://github.com/Comfy-Org/ComfyUI/blob/08cf0284ff45723aa67841c80cabb0e3cb50c9d7/comfy/ldm/minimax/model.py#L356-L455）。
- DiffSynth: L547-L549 の `if not references: return {}`、L842-L845 で FL2VA の組み方へ（https://github.com/modelscope/DiffSynth-Studio/blob/e5a44c42776978457b61d51d595a1f6746c3d87a/diffsynth/pipelines/minimax_h3_audio_video.py#L838-L845）。
- **「Ref2VA の重みで参照なしは FL2VA に劣る」という記述は、どの一次資料にも無い**（公式 README・API・ComfyUI の文書・SGLang・DiffSynth を検索）。公式は文章だけの生成を FL2VA に割り当てている（README L84・L188）ので、品質の保証の外だと考えられる（推測）。

### 15.5 キーフレームとの併用（実装ごとに違う。一次資料で確認済み）

- **ComfyUI: できる。** 公式の雛形「Multiframe Reference」（`video_minimax_h3_multiframe_reference.json`）が Ref2VA の重み＋ AddGuide 3 つ（フレーム 36・72・120）である。文書 minimax-h3-multiframe.mdx L74 は「Add Guide にだけつないだ画像はテキストエンコーダーから見えない」と書く。並びは `[text | 条件の行 | 参照 | 目的の音声 | 目的の映像]` で、条件の行の時刻は全参照の後ろの `cursor` を原点にする（model L370-L455）。
- **SGLang: 参照が 1 件以上あり、フレームが先頭か末尾のときだけできる**（`task_profiles.py` L8-L11・L72-L76、`packed_sequence.py` L316-L317・L517-L520）。
- **公式 API: できない**（「Image-to-video and reference-to-video are mutually exclusive」。https://platform.minimax.io/docs/api-reference/video-generation-v2-create）。
- **DiffSynth: 実質できない**（参照があると先頭・末尾のフレームの行を使わない。L998・L842-L843）。
- 公式のプロンプトの指針（`skills/h3-prompt-writing/references/ref-en.txt` L62-L66）は、Ref2VA で画像を先頭のフレームにする方法として、条件の行ではなく参照画像とプロンプトの文（「<Picture 2> is the first frame of [Shot 1] …」）を示している。

### 15.6 入力の組み方（3 実装の差。一次資料で確認済み）

- **並び**: ComfyUI は画像 → 動画 → 単独の音声の固定順（動画に音声が付けば、その `<Audio j>` を `<Video k>` の直前に出す。nodes L246-L249・L297-L358）。SGLang は要求の `conditions` の順（presentation.py L229）、DiffSynth は渡した順（L146・L264-L287）。種類ごとに並べ直すのは ComfyUI だけ。
- **時刻と cursor**（model L95-L130・L398-L452）: 文章のトークンは 0〜text_len−1、参照の `cursor` は text_len から。画像は全行が `cursor` で、その後 `cursor += 1.0`。単独の音声は `cursor + arange(rt)` で `cursor += rt`、横の座標は目的の映像の格子の両端（L418）。動画は `_video_t_grid(vt, cursor)`（1 トークンあたり 5/3×(1,4,4,4,4)）で、`cursor += max(rt, 映像の時間幅)`。目的の音声と映像はすべての参照の後ろの `cursor` から。SGLang（packed_sequence.py L433-L506）と DiffSynth（L744-L809）も同じ計算。
- **ノイズ時刻**（model L32-L33・L674-L683）: `VISUAL_COND_TIMESTEP = 0.999`、`AUDIO_COND_TIMESTEP = 1.0`。区画の t は ref_img `max(t_v, 0.999)`・ref_audio `max(t_a, 1.0)`・cond `max(t_v, 0.999)`。SGLang（denoise_loop.py L30・L32）と DiffSynth（L38・L1020）も同じ値。
- **映像の参照のノイズ**（model L535-L560）: `aug·r + (1−aug)·noise`（aug 0.999）、`torch.Generator("cpu").manual_seed(seed)`（サンプラーの seed。base L2252）。条件ごとに同じ乱数列を最初から引く。毎 forward で作り直すが値は同じ。音声は seed+1・aug 1.0（混ぜない）。
- **画像の大きさ**: ComfyUI は `match`（既定。`scale = min(1, √(W×H/(w×h)))`）と `max`（短辺 2048）。どちらも縦横比を保ち、各辺を 32 の倍数に丸め、拡大しない（nodes L267-L268・L300-L308）。SGLang の既定は `diffusers`（短辺 2048 に拡大もする。`configs/sample/minimax_h3.py` L57）、DiffSynth も短辺 2048（拡大あり。L121・L482-L484）。SGLang の注記によると `match` は ModelTC の Ref2VA Turbo LoRA の学習の前処理（constants.py L73-L74）。参照動画は ComfyUI・DiffSynth とも短辺 768・面積の上限 768×1344（元が小さければそのまま。nodes L321-L325・DiffSynth L486-L488）。
- **参照動画の音声**: ComfyUI は別の入力 `ref_video_audio_N` で組にする（L314-L319・L345-L350）。SGLang は `video` の種類でも音声を自動で取り出す（task_profiles.py L203-L212）。DiffSynth は `video` を無音として扱う（L532-L533）。
- **切り詰めと 15 秒**: 公式の制約は 1 本 2〜15 秒・合計 15 秒・全部で 12 個（README L85・API の表）。ComfyUI に合計 15 秒の検査も切り詰めも無い。参照動画は目的の長さで切ってから 17k+5 に切り下げ（L326-L333）、5 フレーム未満は拒否する。参照音声は切らない（`_encode_ref_audio` L75-L82）。SGLang は参照動画と参照音声を目的の長さで切る（reference_encoding.py L781-L796・L811、stages/audio_encoding.py L147-L152・L191）。DiffSynth は参照動画を目的のフレーム数で切って 17k+5 に切り下げ、音声は切らない（L490-L496・L508）。
- **札**: `<Picture n>`・`<Video n>`・`<Audio n>` は普通の文字列で、特殊トークンではない（te L3-L4・L175・L178・L187。追加の特殊トークンは `<d>`・`</d>`・`<|cutoff|>`・歌詞とキャプションの札だけ〔L32-L34〕）。SGLang も `add_special_tokens=False` で普通の文字列として扱う（presentation.py L31・L270-L286）。

### 15.7 テキストエンコーダーへの提示（ComfyUI。一次資料で確認済み）

- te L169-L197: 素材ごとに `counters[kind] += 1`。画像は `"<Picture %d>: "` の後に視覚ブロック、音声は `"<Audio %d>: "` だけ、動画は `"<Video %d>: "` の後に 2 フレームごとに `"<%.1f seconds>"` と 2 フレームの視覚ブロック（`video_block=True`）。最後に利用者のプロンプト。区切りの改行や空白は入れず、各文字列を別々にトークン化し、開始・終了のトークンは付けない（q3 L150-L151）。番号は種類ごとに 1 から。参照が 0 件なら FL2VA と同じ経路（L192-L195）。
- 視覚ブロックは `VISION_START`（151652）・埋め込みの差し込み口・`VISION_END`（151653）（L164-L167）。種別のタグは視覚ブロック（前後の 1 トークンずつを含む）が 0、札と時刻の文字が 1（L75-L82）。DiT の中の `seg_tag` は text 1・video 0・audio 2・cond 0・ref_img 0・cond_audio 2・ref_audio 2（model L713）。
- 画像の前処理: テキストエンコーダーに渡すのは `match` で縮めた後の画像そのもの（nodes L308-L309。VAE と同じテンソル）。`process_qwen2vl_images(..., patch_size=16, image_mean/std=0.5)`（`qwen3vl.py` L62-L66）、min_pixels 3136・max_pixels 12,845,056 は ComfyUI の既定（`qwen_vl.py` L9-L19）。視覚トークン数は H×W÷1024（1344×768 で 1008）。本製品の tokenizer フォルダの `preprocessor_config.json` は公式の値（`shortest_edge 65536`・`longest_edge 16777216`）で、ComfyUI の既定と違う。
- 参照動画のフレーム（nodes L326-L341・te L180-L191）: 目的の長さで切り 17k+5 に切り下げた後、`range(0, n, 12)`（0.5 秒ごと・枚数の上限なし）。奇数なら最後を複製し、2 枚ずつ `process_video_block`（Qwen-VL の時間パッチ 2 を 2 フレームで埋める。grid_t=1）。時刻の札は組の 2 枚の平均。15 秒（362 フレーム）で 16 組（`<0.2 seconds>`, `<1.2 seconds>`, …, `<14.2 seconds>`, `<15.0 seconds>`）、1344×768 なら合計 16,128 トークン。5 秒（124 フレーム）なら 6 組。
- AddGuide（キーフレーム）の画像はテキストエンコーダーを通らない（nodes L191-L240。FL2VA の `MiniMaxH3ImageToVideo` は先頭・末尾の画像をテキストエンコーダーに渡す〔L156〕）。参照は positive の条件にだけ付き、公式の雛形はネガティブを使わない（L286・L362-L363）。

### 15.8 潜在の符号化（ComfyUI。一次資料で確認済み）

- **参照画像**（nodes L311・vae L792-L809）: 映像 VAE で 1 フレームとして符号化（`T==1` なら 1 回）。分布の平均だけを `(mean − LATENTS_MEAN)/LATENTS_STD` で正規化。形 `[1, 24, 1, H/16, W/16]`。空間の格子は `_frame_grid(h, w)`（model L70-L74・L88-L92。面積の平方根で正規化）で、参照は自身の大きさ（L403）、目的の映像は目的の大きさ（L357）。
- **参照動画**（vae L664-L677・sd L1051）: 1 回の呼び出しの中で 17 フレームずつ符号化し、各塊 5 潜在、最後の 3 を捨てて `vt = 5k+2`。時刻は `_video_t_grid(vt, cursor)`（model L95-L102。24fps×5/3＝毎秒 40 で音声の潜在のレートにそろう）。15 秒（362 フレーム）で vt＝107、1344×768 なら 107,856 行。
- **参照音声**（nodes L75-L82・avae L377-L443）: 32kHz に変換し、800 サンプルの倍数まで 0 で埋め、左右を別々に符号化して平均を正規化。潜在 `[1, 32, 2, rt]`、`rt = ceil(L/800)`（毎秒 40）。15 秒で rt＝600・1,200 行。
- **配置と出力**: 参照の潜在はキーフレームの潜在の後ろに並べ（base L2238-L2247）、配置はサンプリングの開始時に 1 度だけ組む（base L2260-L2266）。注意はマスクなし（model L200）。`final_layer` に渡すのは目的の映像と音声の区画だけで、参照と条件の行の予測は捨てる（model L775-L786）。

### 15.9 未確認

1. 公式 API が文章だけ・先頭フレームだけの要求を、FL2VA と Ref2VA のどちらの重みで処理しているか。
2. Ref2VA の重みで参照 0 件のとき、FL2VA より品質が劣るかどうか（一次資料に記述が無い）。
3. SGLang に参照の合計 15 秒の検査があるか（調べた範囲には見当たらない）。
4. Multiframe の雛形で、`ref_images` に入っていない `<Picture 2>`〜`<Picture 4>` をプロンプトに書いていることの効果。
5. ヘッダを比べたのは pruned int8 ConvRot の 2 本だけで、他の量子化版のヘッダが一致するか。
6. ComfyUI の SDTokenizer が各文字列をどう分けるか・モノラルの参照音声の正確な扱い（`output_channels=2, pad_channel_value="replicate"` から複製と推測）・SGLang の画像読み取り部の min／max pixels。


## §16 Chained（潜在の持ち越し）の一次資料調査（2026-10-29）

台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-106（MiniMax H3 の Chained）の計測の計画のために、2026-10-29 に行った一次資料の調査の記録である。コードは実行していない。ソースは浅く複製して読んだ。本製品のエンジンの事実は設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3.1（行ごとの連続値マスク）と本製品のコードで確かめた。計測の結果は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §183 が正本で、本節には数値を書かない。本節と設計正本が食い違えば設計正本を採る。

### 16.1 結論

- 台帳 §1-106 の方式（前のクリップの映像と音声の潜在の末尾を次のクリップの先頭に置き、その行を固定して残りだけノイズ除去し、重なりを切り落として連結する。復号して再符号化はしない）は、一次資料で裏取りできた。
- コミュニティの 3 実装（H3 Continuum・LongMedia・vLLM-Omni）は、固定の仕方は違っても同じ骨格に収斂している。
  - 重なりは 17k+5 フレームに限る。既定は 22 フレーム（映像の潜在 7 個）。
  - 前のクリップの潜在の末尾 5k+2 個を、符号化し直さずに使う。置き方は、次のクリップの先頭に固定する方式（H3 Continuum・LongMedia）と、条件行に足す方式（vLLM-Omni）がある。
- 本製品の `_denoise` は、固定に要る `video_fix`・`audio_fix` をすでに持つ。製品の経路から映像のマスクを渡す所だけが無い（16.7）。
- 継ぎ目の品質を数値で比べた報告は、どの一次資料にも無い。比較は本製品で測る（結果は記録簿 §183）。

### 16.2 映像の符号化器〔VAE〕の時間構造（訂正を含む）

- 映像 VAE〔映像と潜在を相互に変換する圧縮器〕は、動画を**先頭から 17 フレームずつの塊に区切り、塊ごとに独立に**符号化する。塊の中の時間方向の畳み込みは因果（過去側だけを 0 で埋める）で、塊をまたぐ状態は持たない。
- 1 つの塊 → 潜在 5 個。担当するフレームは (1, 4, 4, 4, 4) の順（塊の先頭 1 フレーム、残りを 4 フレームずつ）。
- 17n+5 フレームの動画は、17 フレームの塊が n 個（潜在 5n 個）と、**末尾の端数 5 フレーム**からなる。末尾の塊は 17 フレームになるまで最後のフレームを繰り返して符号化し、潜在 5 個のうち末尾の 3 個を捨てるので、端数 5 フレーム → 潜在 2 個（担当は 1, 4）。合計は 5n+2 個。
- **訂正**: 2026-10-19 の [`R4_fl2va_ref2va_a2v.md`](Outputs-archive/h3-roadmap-2026-10-19/R4_fl2va_ref2va_a2v.md) 133 行目の推測「先頭 5 フレーム → 潜在 2 個、その後 17 フレームごとに潜在 5 個」は誤りである。**端数 5 フレーム → 潜在 2 個は末尾**にある。ComfyUI 文書の旧版にも「Frames pack from frame 0 in 17-frame blocks, so the leftover 5 frames are the last block, not the first.」とある。
- 帰結（推測。コードからの導出）:
  - 前のクリップの末尾 17k+5 フレームの区間は、塊の境目から始まる。その潜在は、前のクリップの潜在の末尾 5k+2 個とまったく同じになる。
  - 末尾 5k+2 個を切り出して次のクリップの先頭 5k+2 個に置くと、各潜在の担当フレーム数 (1, 4, 4, 4, 4, …, 1, 4) がそのまま一致する。先頭の 2 個 (1, 4) は、どの塊の先頭 2 個とも同じ形である。
  - 個数が 5k+2 でない（たとえば 5 個）と、位置がずれて「1 フレーム担当」の潜在が間違った場所に来る。採用している実装は無い。
- 復号器は 7 潜在（5＋重なり 2）の窓ごとに復号する。窓の境目では 5 フレームを混ぜ合わせる。窓の中は全結合の注意で、因果ではない。
- 音声の復号器（BigVGAN〔音声波形を作る復号網〕）は、注意が無く、畳み込みの詰め物が左右対称で、因果ではない。窓も無く、潜在全体を一度に通す。因果の注意があるのは音声の符号化器だけで、潜在を持ち越す連結では符号化器を通らない。
- 音声の潜在は 1 秒あたり 40 個で、映像の時間座標（5/3 × 24 フレーム＝毎秒 40）と物差しが同じである。ただし 5 フレームの重なりは音声の潜在 8.33 個にあたり、割り切れない。
- 出典:
  - ComfyUI `comfy/ldm/minimax/vae.py` L798-L812（符号化）・L905-L950（復号）・L557・L559・L581。
  - 本製品 `engine_h3/video_vae.py` L436-L439・L612-L619・L645-L655・L743-L790、`engine_h3/constants.py` L31-L33・L79-L96、`engine_h3/audio_vae.py` L206-L243・L273-L282・L370-L402。
  - vLLM-Omni `vae_temporal.py` L45-L89 も同じ。

### 16.3 H3 Continuum の「Masked AV」（MIT）

版は 3.10.2（commit `d04977f1a06e840d95a1787c2d46d33ddf377b69`・2026-10-27）。

- **「Native Masked」という語はリポジトリのどこにも無い。** 実際の呼び名は「Masked AV Continuation」（内部の識別子 `masked_av_prefix_22_v1`）と「Standard」。状態の検査のエラー文にだけ「Native Continuity」がある（`state.py:108`）。
- **重なり**（Continuity の設定。`constants.py:46-59`・`temporal.py:71-80`）:

  | 設定 | フレーム数 | 映像の潜在 | 音声の潜在 |
  |---|---:|---:|---:|
  | Fast | 5 | 2 | 8 |
  | Balanced（既定・推奨） | 22 | 7 | 37 |
  | Strong（原典で Experimental） | 39 | 12 | 65 |

  映像と音声の両方を固定する Masked AV は、Balanced 22 のときだけ使う（`driving_nodes.py:350-376`）。ほかの設定では旧方式の Reference Context に落とす。音声の継続を切ると、映像だけを固定する。
- **持ち越し**: 前のクリップのサンプリング済みの潜在の末尾をそのまま切り出す（映像 `session.py:241`・音声 `:242`）。切り出しの開始位置が 5 の倍数であることを要求する（`temporal.py:104-107`）。次のクリップの先頭へそのまま上書きする（`v3/masked_continuation.py:100-104`）。復号も再符号化もしない。
- **固定**: ComfyUI 本体の `noise_mask` をそのまま使う。値は 0（固定）と 1（生成）だけで、連続値は使わない（同 `:106-135`）。ComfyUI 本体は、固定した行の時刻を映像 0.999・音声 1.0 に止め、毎ステップ元の値を入れ直す（映像は `0.999·clean + 0.001·noise`、音声は clean のまま。`comfy/ldm/minimax/model.py:32-33`・`:687-704`、`comfy/model_base.py:2303-2326`）。サンプリングの後に固定区間を元の潜在で書き戻し、ビット単位で一致させる（`restore_masked_*`、同 `:169-207`）。
- **時間座標**: 各クリップを時刻 0 から生成する。前のクリップから続く時刻のずらしは与えない（`v2/sequence.py:1148`）。旧方式の Reference Context は、持ち越しの行の座標を目標の先頭の座標に重ねる（`layout_adapter.py:610-643`）。
- **音声の端数**: 音声の潜在の個数は `round(フレーム数 × 40/24)`（22 フレームなら 37 個）。前のクリップの端数のずれ（`audio_grid_offset`）は、Masked 方式では報告文に書くだけで、置く位置の補正には使っていない（`v2/sequence.py:1077-1081`）。最大 1/3 個（約 8 ミリ秒）ずれたまま置いていることになる（推測）。
- **最初のフレームの条件**: Masked のとき、2 本目以降には First Image のキーフレームを付けない（`v2/sequence.py:1057`）。Last Frame は最後のクリップにだけ付ける。
- **継ぎ目**: 重なりは切り落とす（`trim_frames`・`v2/sequence.py:1103`）。復号は**クリップごと**で、各クリップの `[trim_frames:]` だけをつなぐ（`v3/assembly_v35.py:396`、`docs/ARCHITECTURE.md:70-78`）。切り落としを選んだ理由の原文は無い。後処理（Finalize）に、音声は継ぎ目の探索と 10〜60 ミリ秒のクロスフェード、映像は主に解析がある。
- **品質の報告**:
  - 課題 #13（未解決）: クリップを重ねるたびにコントラスト・鋭さが強まり焼ける。作者は Standard に「no active production mitigation」と書く。Masked 方式でも 60 秒（6 クリップ）で劣化した報告がある。
  - 課題 #24（閉）: ComfyUI 本体の版（0.34.0 と 0.34.6 以降）で継ぎ目の滑らかさが変わった報告。作者は「0.35 の denoise mask の変更」を疑い、継ぎ目の約 5 フレーム後に最大の変化があると分析した。修正のコミットは見つからない。
  - 課題 #19（閉）・#33（未解決）: 継ぎ目が切れる・開始画像に戻る、2 本目でプロンプトに従わない。
  - 数値で比べた報告は無い。
- **「FL2VA で検証済み」の確認結果**（CHANGELOG の 3.6.0 の節 L331-L332、Release v3.6.0＝2026-08-27）:
  - 原文は「GPU-accepted T2VA, I2VA, Reference Image + Reference Audio, Balanced 22f/37T Joint AV, and 3×5-second FL2VA Long Terminal Merge Masked continuation.」で、固定した先頭がビット一致で確定したこと、音声を聴いて合格したことも書く。
  - この「FL2VA」（先頭と末尾のフレームと音声から作る仕組み）は、主に FL2VA の重みの意味で使われ、最後の 2 チャンクを 1 回のサンプルにまとめる「Long Terminal Merge」（Last Frame を使うとき）を指す。
  - 継ぎ目の品質を比べた記述は無い。**Ref2VA の重みで Masked 方式を検証したという記述は見つからなかった**（未確認）。
- **プロンプトとステップ**: 同梱の文書は、2 本目以降の本文を「Continuation of Chunk 1.」で始めることを勧めるが、「not a verified quality guarantee」とする。同梱のワークフローは simple・20 ステップ。shift の推奨は見つからなかった。Turbo のサンプルは 4step・6 ステップで「not a verified quality preset」と書く。

### 16.4 LongMedia（Apache-2.0）

版は 0.6.63（commit `409e4cb03c85cf455bf195d5185d106c0d821487`・2026-09-29）。

- **0.6.40 でキーフレーム方式から、denoise mask で固定した先頭へ変えた。** 理由はリリースノート `docs/RELEASE_NOTES_0.6.40_EN.md` L23 に書かれている。複製した動きの文脈とノイズの位相が、短い動きの繰り返しの周期を生んでいた。`nodes.py` L24247 のコメントは、先頭を新しく生成してキーフレームで約 22 フレームを案内する方式では、案内の区間が約 0.92 秒（24fps）の動きの模様として全体に繰り返されうる、と書く。
- **固定**: 前の結果の映像の潜在の末尾 `video_latent_t(overlap)` 個を、新しい潜在の先頭にそのままコピーする（`nodes.py:24263`）。音声は末尾 `round(overlap/24×40)` 個（同 `:24236`）。その区間の denoise mask を 0 にする（映像 `:24364`・音声 `:24368`）。映像のマスクは潜在の時刻ごとの値を持つ形 `(1,1,T,1,1)`。
- **キーフレームや参照の併用はしない**（`_clone_guider_with_segment_audio`・`nodes.py:17915-17926`）。同じ末尾をキーフレームに入れ直すと、時間の権威が二重になり周期的な動きの種になりうる、と書く。
- **重なりの長さ**: 既定 22 フレーム、17 刻み（`docs/PARAMETER_REFERENCE_EN.md` L23・L148）。個数は `video_latent_t(f) = 2 + 5×((f−5)//17)`（`latent_ops.py:24-27`）。
- **時間座標**: 窓の `context_start`（重なりを含む始まり）× 40/24 を、RoPE〔位置の埋め込み〕の時間座標に足す（`nodes.py:17947`、`temporal_positioning.py:15-43`）。テキストと静止画の参照は動かさない。H3 Continuum（各クリップ 0 から）とは違う。
- **重なりの除去は潜在で行い、累積した潜在を復号する。** 映像は `next_video[:, :, overlap_video_t:]`、音声は `next_audio[..., overlap_audio_t:]` を連結する（`latent_ops.py:693-850`）。復号は連続して生成された並びごと（`docs/ARCHITECTURE_EN.md` L83）。H3 Continuum（クリップごとに復号）とは違う。
- **半々の混ぜ合わせの報告**: 重なり区間を smoothstep で混ぜる機能があるが、既定は無効（`blend_video_overlap`）。見える継ぎ目をまたいで混ぜることを、わざと禁じている。コメントは「a 50/50 mix decodes as several ghosted/stalled pixel frames followed by a jump」（`latent_ops.py:783-789`）。ぼやけて止まった後に跳ぶ、という報告である。
- 先行調査 R4 の「末尾を次の単位のキーフレームとして渡す」という記述は、0.6.40 より前の方式で、現在は当てはまらない。
- 外部動画から続ける場合は、末尾の画素を VAE で符号化し直し、フレーム 0 のガイドにする（`nodes.py:24518-24714`）。リリースノート 0.6.61 は、公開の検証時に GPU が無かったと書く（L13）。

### 16.5 vLLM-Omni（Apache-2.0）と ttulttul

版は vllm-project/vllm-omni commit `8d4b2789acd7e5cf0ac68a0f3f219327622bbdfe`（2026-10-29）。元になった実装は ttulttul/ComfyUI-Minimax-H3-Continuation（MIT・commit `e1768d5f`・README のみ読んだ）。

- **Ref2VA〔参照素材からの生成〕専用の guide 方式**（`vllm_omni/diffusion/models/minimax_h3/continuation.py`）:
  - 窓 277 フレーム・重なり 22 フレーム（L59-L60）。どちらも 17n+5 で、窓は 107〜345（L62-L65）。次の窓の始まりは `前の end − 22`（L69-L80）。
  - 累積した映像の潜在の末尾 7 個をそのまま切り出し（L123-L125）、条件行の `latent_guide` として足す（L127-L144）。目的の映像はマスクなしで全部新しく生成し、隠れた重なりを捨てて新しい部分だけを足す（レシピ L931-L936）。
  - 時間位置は目的の映像と同じ起点。窓の始まり × 40/24 を `temporal_offset` として加える（L116、`packed_sequence.py:689-694`）。
  - `task != "ref2va"` なら例外（L57-L58）。`latent_guide` を受け付けるのは Ref2VA の配置の関数だけ。理由の文章による説明は無い。
  - 「継ぎ目が滑らかな保証はない」と書き、数値の報告は無い（レシピ L964-L966）。
- これは §1-109 の領分で、§1-106（FL2VA）の範囲外である。
- **音声の重なりは全体の時間軸で丸める**: `overlap_audio_t = round((start+22)×5/3) − round(start×5/3)`（L30-L31・L49-L50）。窓ごとに丸めて誤差が積み重なるのを避ける作法で、境目は累積の 24fps／40Hz の時間軸の上で四捨五入する（L95-L96）。
- 累積した潜在は、全部の窓が終わってから 1 回で復号する（レシピ L961-L962）。

### 16.6 ComfyUI 本体（GPL-3.0）と DiffSynth（Apache-2.0）

ComfyUI は commit `0df1646a9cd71d4c990a87014fb37797a2956ead`（2026-10-28）。GPL-3.0 なので、仕様として読むだけでコードは写さない（裁定 17）。

- **denoise mask の粒度**（`comfy/model_base.py:2270-2287`）: 映像は時間方向が潜在 1 個単位、空間方向が 2×2 パッチ単位。音声は潜在 1 個単位。値は連続値で 1/256 刻み。行の時刻は `1 − m×σ` を映像 0.999・音声 1.0 で頭打ちにし、毎ステップ元の値を入れ直す（`model.py:687-704`）。本製品の行ごとの連続値マスクと同じ意味である。
- **LatentCut／LatentConcat は H3 の潜在に使えない**（推測。実行はしていない）: H3 の潜在は `NestedTensor` で、`narrow`／`cat` を受ける仕組みが無く、音声の形 `[B,32,2,T]` に時間軸の指定を当てると違う軸を切る。H3 用の切り出しや連結のノードは本体に見当たらない（`comfy_extras/nodes_latent.py:140-215`、`comfy/nested_tensor.py:3-6`）。
- **AddGuide は画素の入力だけ**を受け、VAE で符号化して条件行にする（`nodes_minimax_h3.py:166-240`）。潜在を直接受ける入力は無い。通常の使い方は「末尾の画素を符号化し直す」方式である。
- **公式の I2V 継続の雛形**: 文書の旧版（commit `161b02726831525bc3ee8689694194d818100138`）は、各クリップの最後のフレームから I2V をつなぐと誤差が積み重なる（顔のきめが先に失われる。Turbo は焼けが早い）と書いていた。この節は 2026-10-28 に削除された（commit `710728e`・「Remove Desktop-only i2v_continuation」）。雛形 `video_minimax_h3_i2v_continuation.json` の中身は普通の I2V に見え、前の動画の末尾を取り出すノードは無い。
- **DiffSynth の Retake**: 設計正本 §3.1 のとおり、固定した行を clean のまま注入し（ノイズを足さない）、時刻も映像で 1.0 にする。連結の固定行の注入の形は設計正本の裁定 84 で決定した（16.8）。

### 16.7 本製品との対応

- **固定の仕組みはある。** `H3Pipeline._denoise` の `video_fix`（映像）と `audio_fix`（音声）が、設計正本 §3.1 の行ごとの連続値マスクにあたる。意味は H3 Continuum・LongMedia・ComfyUI 本体の固定と一致する（m＝0 の行は時刻が上限に止まり、毎ステップ注入され、最後に clean へビット一致で置き換わる）。製品の経路が渡しているのは `audio_fix` だけで、`video_fix` を渡す所は無い（呼ぶのはテストだけ）。H3 Continuum と LongMedia は 0/1 の硬い固定だが、本製品は連続値を持つので、固定の強さを変えて比べられる。
- **潜在を返す口は無い。** `generate` は必ず `_decode` まで進む。private な `_denoise` の戻り値を受ければ、計測の台本は製品コードを変えずに潜在の持ち越しを組める。
- **映像の重なりは 17k+5 フレーム**（潜在 5k+2 個）に限る。これだと、本製品の VAE の構造でも担当フレーム数が一致する（16.2）。
- **音声の継ぎ目が潜在の境界に乗る条件**（推測。算術からの導出）: 継ぎ目の開始フレーム S の音声の位置 S×5/3 が整数になるのは S が 3 の倍数のときである。クリップが 17a+5 フレーム、重なりが 17m+5 フレームなら S＝17(a−m) なので、条件は **a−m ≡ 0 (mod 3)**。満たさなければ、継ぎ目の位置に最大 1/3 潜在（約 8 ミリ秒）の置き場所の誤差が出る（ka は全体の時間軸で丸めた整数）。固定した末尾と映像のずれは最大 2/3 潜在。この条件はクリップの長さと重なりの組で決まり、重なりだけでは決まらない。長さの違うクリップをつなぐときは、累積の開始フレーム S で判定する。重なりそのものが音声の潜在で割り切れるか（39 フレームは 65 個）とは別の話である。H3 Continuum はこの誤差を補正せずに置いている（16.3）。
- **音声の重なり ka は全体の時間軸で丸める**（vLLM-Omni の作法。16.5）。窓ごとに丸めない。
- **時間座標**: H3 Continuum は各クリップを 0 から生成し、LongMedia と vLLM-Omni は窓の始まりを足す。本製品の台本は 0 から始める（時間を連続させる案は §1-109 の領分）。
- **復号**: 映像の復号器は 1 窓ぶんしか GPU に載せないので、VRAM は長さで増えない。連結した潜在 1 本の復号は、窓の重なりの混ぜ合わせが継ぎ目にも効く（推測）。クリップごとに復号して画素でつなぐ案は H3 Continuum の方式で、両方を同じ潜在から作って比べられる。音声の復号器は窓が無く、長いほど VRAM が増える。実測は記録簿 [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §183.4。
- **乱数**: LTX 2.5 の Chained は区間 i に `seed+i` を使う先例がある（`engine25/chain25.py`）。同じ種を繰り返すと、新しく作る部分が毎回同じ乱数から始まる（推測）。

### 16.8 固定行の注入の形と時刻の比較

固定行に DiT〔拡散の本体〕が見る値と、その行の時刻の比較である。計測の数値は書かない（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §183 が正本）。

| 実装 | 固定行に見せる値 | 固定行の時刻 | 出典 |
|---|---|---|---|
| ComfyUI 本体 | `0.999·clean + 0.001·noise`（音声は clean） | 映像 0.999・音声 1.0 | `comfy/ldm/minimax/model.py:32-33`・`:687-704`、`comfy/model_base.py:2303-2326`（16.3）。粒度は `model_base.py:2270-2287`（16.6） |
| H3 Continuum・LongMedia | ComfyUI 本体と同じ（本体の経路を使う） | ComfyUI 本体と同じ | 16.3・16.4 |
| DiffSynth の Retake | clean そのもの（ノイズを足さない） | 映像も 1.0 | 設計正本 [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md) §3.1（DiffSynth の commit は 16.11 の `e5a44c4`） |
| 本製品の連結 | clean そのもの（映像・音声とも） | 映像 0.999・音声 1.0 | 設計正本の裁定 84。**選択の理由は裁定 84 を参照**（ここには書かない） |

### 16.9 ライセンスの扱い

| 出典 | ライセンス | 扱い |
|---|---|---|
| H3 Continuum | MIT | 参考にできる |
| LongMedia | Apache-2.0 | 参考にできる |
| vLLM-Omni | Apache-2.0 | 参考にできる |
| ttulttul/ComfyUI-Minimax-H3-Continuation | MIT | 参考にできる（README と `docs/LEARNINGS.md` を読んだ） |
| DiffSynth-Studio | Apache-2.0 | 移植できる（設計正本 裁定 17） |
| ComfyUI 本体 | GPL-3.0 | 仕様として読むだけ。コードは写さない |

### 16.10 未確認

1. Ref2VA の重みでの Masked 方式（H3 Continuum に検証の記述が無い）。→ 本製品で計測した。VERIFICATION_LOG §185。
2. 継ぎ目の品質の数値の報告（どの実装にも無い）。
3. H3 Continuum が重なりの切り落としを選んだ理由（原文が無い）。
4. 焼け（課題 #13）が Masked 方式の何に起因するか。
5. vLLM-Omni の guide 方式が Ref2VA 専用である理由（文章による説明は無く、実装範囲の制限と見られる。推測。§16.12）。
6. 公式 I2V 継続の雛形の実体（Desktop 版だけの機能らしい。推測）。

### 16.11 出典の一覧

| 出典 | 版 | 読んだ場所 |
|---|---|---|
| ukr8b3g-cmyk/ComfyUI-H3-Continuum（MIT） | 3.10.2・`d04977f1a06e840d95a1787c2d46d33ddf377b69` | `CHANGELOG.md` L331-L332・`constants.py`・`temporal.py`・`state.py`・`v2/session.py`・`v2/sequence.py`・`v3/masked_continuation.py`・`v3/driving_nodes.py`・`v3/assembly_v35.py`・`layout_adapter.py`・`docs/ARCHITECTURE.md`・`docs/USER_GUIDE.md`。課題 #13・#19・#24・#33。Release v3.6.0。追補（§16.12）は `0cc94207e70a98f2d560b9f888864b5687ec4415`（2026-10-29。差は Run のロックの修正）で、`docs/AB_TEST_LOG_JA.md`・`docs/VALIDATION_RESULTS.md`・`docs/V39_REFERENCE_INLINE.md`・`docs/README_HISTORY.md`・`docs/USER_GUIDE.md`・`examples/*.json`・課題 #15・#21 も読んだ |
| vizart-vj/ComfyUI-MiniMax-H3-LongMedia（Apache-2.0） | 0.6.63・`409e4cb03c85cf455bf195d5185d106c0d821487` | `nodes.py`・`latent_ops.py`・`temporal_positioning.py`・`docs/RELEASE_NOTES_0.6.40_EN.md`・`docs/PARAMETER_REFERENCE_EN.md`・`docs/ARCHITECTURE_EN.md`。追補（§16.12）で `CHANGELOG.md`・`docs/MODES_GUIDE_EN.md`・`workflows/README_EN.md`・`docs/RELEASE_NOTES_0.6.61_EN.md`・`0.6.62`・`workflows/*.json` も読んだ |
| vllm-project/vllm-omni（Apache-2.0） | `8d4b2789acd7e5cf0ac68a0f3f219327622bbdfe`。追補（§16.12）は `1c8d26d03fc27d1c104010489fcd6fda309908b5`（PR #7838 `139a47a5`・#8460 `198c2105`・#9342 `b5a6fd83`） | `vllm_omni/diffusion/models/minimax_h3/` の `continuation.py`・`packed_sequence.py`・`vae_temporal.py`・`pipeline_minimax_h3.py`、`recipes/MiniMaxAI/MiniMax-H3.md`。追補（§16.12）で `docs/serving/videos_api.md`・`vllm_omni/model_executor/models/minimax_h3/long_video.py`・`vllm_omni/diffusion/models/minimax_h3/latent_mask.py` も読んだ |
| ttulttul/ComfyUI-Minimax-H3-Continuation（MIT） | `e1768d5f`（フルは `e1768d5fdfc6f9519d2090dcf78458c2d9625f80`） | README。追補（§16.12）で `docs/LEARNINGS.md`・`example_workflows/guided_continuation.json` も読んだ |
| Comfy-Org/ComfyUI（GPL-3.0） | `0df1646a9cd71d4c990a87014fb37797a2956ead` | `comfy/ldm/minimax/model.py`・`vae.py`・`comfy/model_base.py`・`comfy_extras/nodes_minimax_h3.py`・`nodes_latent.py`・`comfy/nested_tensor.py`。追補（§16.12）は `b5a432157de6b8884f81104cd02926adf4d597e4`（2026-10-29）で、`comfy_extras/nodes_minimax_h3.py` の履歴・PR #15375・#15988・課題 #16589・#16673 を仕様として読んだ |
| Comfy-Org/docs | `161b02726831525bc3ee8689694194d818100138`（旧版）・`710728e` | `minimax-h3.mdx`。追補（§16.12）は main（2026-10-31 取得）の `tutorials/video/minimax/minimax-h3-multiframe.mdx` |
| Comfy-Org/workflow_templates | main（最終コミット `e7cd011d`） | `templates/video_minimax_h3_i2v_continuation.json`。追補（§16.12）は main（2026-10-31 取得）の `templates/video_minimax_h3_multiframe_reference.json` |
| modelscope/DiffSynth-Studio（Apache-2.0） | `e5a44c42776978457b61d51d595a1f6746c3d87a`（§15.1 と同じ） | Retake の記述は設計正本 §3.1 による。追補（§16.12）は `diffsynth/pipelines/minimax_h3_audio_video.py` の最終コミット `5c3abf5c3e918baa94458036df6dd994beedc0a5`（main は `543bf843411a32e1520301a5f8a09e2c0ae27a1d`） |
| 本製品 | dev（2026-10-29） | `engine_h3/` の `pipeline_h3.py`・`denoise_mask.py`・`rope.py`・`video_vae.py`・`audio_vae.py`・`constants.py`・`media_out.py`・`worker.py` |

URL の書式は `https://github.com/<repo>/blob/<commit>/<path>#L<n>`。調査の報告の原本は、作業場の `h3chain\R1_continuum.md`・`R2_others.md`・`R3_engine.md`。

### 16.12 追補（2026-10-31）

台帳 §1-109（Ref2VA の重みでの Chained と参照の併用）の計測の前に、Ref2VA の重みで「固定行の連結」と「参照行」が同時に効くかを、一次資料で追加調査した。コードは浅く複製して読んだだけで、実行していない。ComfyUI 本体（GPL-3.0）は仕様として読み、コードは写していない。**本製品の計測の結果は VERIFICATION_LOG §185 を参照。本節には本製品の数値を書かない。**

**結論**

1. **仮説（Ref2VA の重みでも、固定行の連結と参照行が同時に効く）を直接裏付ける記録は、見つからなかった。** 「その形で作れる」前例はあるが、「その形で品質が出る」ことを示す記録は無い。
2. 支持する材料は LongMedia の主力の雛形（Ref2VA の重み＋クリップごとの参照＋固定行）。懸念する材料は vLLM-Omni の実測（Ref2VA の重みは、条件行として渡した前の末尾を緩くしか追わない）。
3. したがって判定は GPU の実験でしか付かず、本製品で計測した（§185）。

**vLLM-Omni（Apache-2.0）**

- **PR #9342**「Continue a previous video from cached latents across requests」（2026-10-30。https://github.com/vllm-project/vllm-omni/pull/9342 ）。別の要求で終わったクリップの潜在を保存し、次の要求の先頭に置いて続ける機能。置き方は**条件行**（フレーム 0 の映像＋音声のタイムラインガイド）で、マスクした目的の行ではない。
  - PR 本文の Known limits（原文）: 「The Ref2VA checkpoint follows the tail loosely (seam about 11 dB), so T2VA/FL2VA continuations are recommended.」。Ref2VA で同じクリップを続けた例の継ぎ目の PSNR〔画素の一致度。大きいほど似ている〕は約 11 dB で、「a visible cut」と書かれている。FL2VA の重みでの続きは 19.8〜32.9 dB。
  - 利用者向けの文書 `docs/serving/videos_api.md` L466-L470 にも同じ内容がある。
  - **測ったのは条件行の方式で、本製品の固定行の方式ではない。** 固定行なら重なりの区間は前のクリップとビット一致する。問題になるのは、固定した区間の直後の、新しく作る最初のフレームがつながるかである。
  - 条件は 4×H200・1344×768・124 フレーム・seed 42。動画は添付されていない。
- **PR #7838**（Ref2VA 専用の長尺の継続。2026-09-20。https://github.com/vllm-project/vllm-omni/pull/7838 ）と **PR #8460**（タイムラインガイド。2026-10-11。https://github.com/vllm-project/vllm-omni/pull/8460 ）の経緯:
  - PR #7838 の本文には Ref2VA 専用という**制約だけがあり、理由の文章は無い**。検証は CPU の回帰試験だけで、画質や継ぎ目の測定は無い。
  - 当時のガイドの区画（`latent_guide`）は、Ref2VA 系の配置関数にしか受け口が無かった。PR #8460 で、ガイドが全タスクに使えるようになった。
  - PR #9342 は、この全タスク向けの経路で継続を作り、FL2VA のほうがよく追うと実測した。
  - **判断: Ref2VA 専用は、最初の実装が Ref2VA の配置関数にだけ受け口を作ったことによる実装範囲の制限で、モデルの性質による理由は書かれていない（推測）。** 要求の中の長尺の継続（`long_video`）は今も Ref2VA 専用のままである。
  - 継続の元になった ttulttul の実装は FL2VA の重みで作られている。

**ttulttul/ComfyUI-Minimax-H3-Continuation（MIT。vLLM-Omni の方式の元）**

- 雛形の重みは FL2VA（`example_workflows/guided_continuation.json`）。条件行の方式は FL2VA で手で確かめて成功した、と書く（`docs/LEARNINGS.md`）。
- **固定行（マスクした重なり）の方式は、試したうえで捨てている**（`docs/LEARNINGS.md` L3-L11）。潜在を硬くつなぐと復号後に継ぎ目でちらつきや崩れが出ることがある、線形の混ぜ合わせには「visual confetti」が残った、と書く（同 L43-L70）。
  - **時期に注意。** この実験は 2026-08-16〜17。ComfyUI 本体が H3 の行ごとの時刻のマスクを入れたのは PR #15375（2026-08-17）で、その後に PR #15988（2026-09-09）でマスクの速度の換算を直した。ttulttul の結論は、行ごとの時刻が無い時期の外側からのマスクに対するもので、本製品の固定行（行の時刻を止め〔映像は 0.999・音声は 1.0〕て毎ステップ注入する）とは仕組みが違う（推測。日付と記述からの判断）。

**LongMedia（vizart-vj/ComfyUI-MiniMax-H3-LongMedia、Apache-2.0。版は §16.11）**

- **主力の雛形が Ref2VA の重み＋クリップごとの連結である。** `workflows/MiniMax-H3-LongMedia-Director-Unified.json` の重みは Ref2VA、Setup の値は `ref2va`・`multiclip`・重なり 22。`workflows/README_EN.md` が「Primary LongMedia Director workflow」と呼ぶ。推奨の組み方は `docs/MODES_GUIDE_EN.md` L268-L281。
- **連結は固定行。** 2 本目以降は、前のクリップの生成済みの潜在の末尾を目的の先頭に写し、そこのマスクを 0 にする（`nodes.py` L24241-L24246 付近）。コメント（L24237-L24240）に、**条件行だけ（新しい先頭＋ガイド）では 2 本目以降で動きが強まったので、固定行に移った**という記録がある。
- 違い: `segmented` は固定行だけ。`multiclip` は固定行に加えて、同じ末尾を条件行のキーフレームとしても足す。
- **参照はクリップをまたいで効かせる設計。** CHANGELOG 0.6.62 に、人物の参照が MultiClip の境目をまたいで効くよう、対象の映像の問い合わせだけ、参照の注意の配分を、持ち越した長い参照の流れと釣り合わせた、とある。持ち越した動きの文脈に人物の参照が負ける問題を、作者が認めて手当てしたと読める（推測）。
- 継ぎ目の品質の数値は無い。0.6.61・0.6.62 のリリースノートは「No live ComfyUI/GPU render was available」と書く。

**H3 Continuum（ukr8b3g-cmyk/ComfyUI-H3-Continuum、MIT）**

- **「Reference Context」の正体は、前のクリップの末尾を、参照と同じ種類の条件行の塊として足す旧方式である。** チャンクごとに利用者の参照を入れ直す機能ではない（`CHANGELOG.md` 3.6.0・`docs/README_HISTORY.md` L215）。現在の振り分けでは、Balanced の 22 フレーム＋音声の継続は Masked AV、それ以外は Reference Context に落ちる（`docs/USER_GUIDE.md` L321）。
- **Masked AV＋参照画像＋参照音声の GPU の合格の記録はある**（`CHANGELOG.md` 3.6.0）が、重みの名前は書かれていない。合格の観点は音声の聴感が中心で、映像の継ぎ目と参照の効きは書かれていない。
- Ref2VA の重みでの連結の記録は、Masked AV より前の版（V3.2.1・V3.3.0）のものだけで、持ち越しは旧方式（推測）。`docs/AB_TEST_LOG_JA.md` の Ref2VA＋参照 2 枚の連結は「技術 PASS、実用可能」と目視で判定された。継ぎ目の評価の記述は無い。
- **V3.9 のチャンクごとの参照**（`docs/V39_REFERENCE_INLINE.md`）: `Reference Use = All chunks`（全チャンク同じ）と `Per chunk`（チャンクごとに違う）がある。参照を外しても、持ち越した映像から人物が消えるとは限らない、という注意がある。**GPU の合格の前**であり、雛形の重みは FL2VA（Continuum は FL2VA の重みにも参照を入れる）。
- 課題 #33（V3.9 でチャンク 2 の指示に従わない）は「No root cause confirmed」。**Ref2VA＋Masked で継ぎ目・焼け・参照が効かない、と名指しした報告は見当たらなかった。**

**ComfyUI の公式の雛形と文書**

- 公式の雛形「Multiframe Reference」（`Comfy-Org/workflow_templates` の `templates/video_minimax_h3_multiframe_reference.json`）: 重みは Ref2VA。参照画像は 1 枚。`MiniMaxH3AddGuide` を 3 つ直列につなぎ、位置は **1.5 秒・3 秒・5 秒**（フレーム 36・72・120）。**フレーム 0 に AddGuide は置かない。** 先頭フレームは `<Picture 1>` を「the first frame of [Shot 1]」とプロンプトの文で指定する。公式の文書 `minimax-h3-multiframe.mdx` も、`<Picture 1>` を「the identity lock and the first frame」と説明する。
- したがって、この雛形は「Ref2VA＋AddGuide の併用」の例だが、**先頭フレームを条件行で固定する例ではない。**
- ComfyUI 本体の課題（「ref2va」で検索）に、「Ref2VA で先頭フレームのガイドが効かない」「参照とマスクの併用」「継続」を扱う題名は見当たらなかった。

**MiniMax 公式と DiffSynth-Studio**

- 公式 README（`MiniMax-AI/MiniMax-H3`）: FL2VA は先頭・末尾の画像 0〜2 枚を受ける。Ref2VA はマルチモーダルな参照（画像 9・動画 3・音声 3、合計 12）を受ける。**Ref2VA の行に先頭・末尾のフレームの入力は無い。** 2 つの重みの DiT の設定ファイルは同一で、構造は同じ、値だけが違う。Ref2VA が先頭フレームの条件を学習していないという記述は、公式には見当たらない。技術報告は見つからなかった。
- **公式の Ref2VA の連結の作法は、前の動画を参照動画 `<Video N>` として渡すこと**である（`skills/h3-prompt-writing/references/ref-en.txt`。`<Video N>` は「an editing source, continuation starting point, or whole-video temporal structure」）。条件行や固定行ではない。本製品の固定行とは別の道だが、Ref2VA の重みが学習した形に近い可能性がある（推測）。
- 公式 API は、参照を渡したときに先頭・末尾のフレームを同時に受けない（「Image-to-video and reference-to-video are mutually exclusive」）。
- **二次資料**（公式ではない。出典の記載が無い）: 混合重みの配布ページ（smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models）は、FL2VA と Ref2VA の差が、ブロックごとの `adaln_proj`（時刻の変調の重み）にほぼ集中していると書く。事実なら、時刻 0.999 で固定した行の扱いが 2 つの重みで違うことの説明がつくが、**裏付けは無い**（推測）。
- DiffSynth-Studio: Retake（区間の描き直し）と参照は、コードの上では同時に渡せる（併用を拒む検査は見当たらない）。ただし**併用の例と記述は無い**。Retake の例は FL2VA の区画の重みである。

**§1-115（Ref2VA で参照とキーフレームを併用すると先頭フレームが固定されない）との関係**

- 公式の雛形がフレーム 0 を固定していないこと、vLLM-Omni の実測で Ref2VA が条件行の末尾を緩くしか追わないことから、§1-115 は本製品の不具合でなく**重みの性質である可能性が高まった**（推測）。

**追補の出典**（§16.11 に無いもの。それ以外の版は §16.11 の表に足した）

| 出典 | 版 | 読んだ場所 |
|---|---|---|
| MiniMax-AI/MiniMax-H3 | `199b7d0e1fd95e7b3eafa8c15eca692534142427`（2026-10-28） | `README.md`・`FL2VA/`・`Ref2VA/` の `transformer/config.json`・`transformer_ref/config.json`・`Ref2VA/model_index.json`・`skills/h3-prompt-writing/references/ref-en.txt` |
| MiniMax 公式 API | 2026-10-31 取得 | https://platform.minimax.io/docs/api-reference/video-generation-v2-create |
| 二次資料: smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models | 2026-10-31 取得 | モデルカード（公式ではない。出典の記載なし） |

調査の報告の原本は、作業場の `h3refchain\R3_primary_sources.md`（セッション限り）。
