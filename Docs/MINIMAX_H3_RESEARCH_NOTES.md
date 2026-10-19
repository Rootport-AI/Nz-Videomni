# MiniMax H3 事前調査ノート（参考資料）

## 本書の位置づけ

**2026-09-06に実施した、MiniMax H3を第3のベースモデルとして追加するための事前調査の記録である。設計資料ではなく参考資料として収蔵する。**

- マルチエンジン機構（ドロップダウンによるベースモデル切り替え）の設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)である。MiniMax H3固有の設計正本は本書執筆時点では未執筆であり、執筆後に本書と記述が食い違った場合は設計正本を採ること。
- 台帳（[`PENDING_TASKS.md`](PENDING_TASKS.md)）への起票は未実施（設計段で行う）。

  > **更新（2026-10-16）**: 起票済み（§1-93）。

- 2026-10-16 に鮮度の確認と裁定の更新を行った。更新内容は §13、裁定の更新は §10 の追記にある。§1〜§9 の本文は 2026-09-06 時点のままで、古くなった箇所には訂正の引用を添えてある。
- 2026-10-17 に計画のセッションを開始し、オーナーの裁定 13〜21 を §10 に追記した。§13.7 の 1 と §13.8 に訂正と更新の引用を添えてある。
- 設計正本は [`MINIMAX_H3_DESIGN.md`](MINIMAX_H3_DESIGN.md)（2026-10-17 新設）。
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
