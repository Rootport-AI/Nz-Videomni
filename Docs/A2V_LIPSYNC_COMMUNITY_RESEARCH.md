# A2V / リップシンクの使い勝手 — コミュニティ先行報告の統合レポート

- 作成: 2026-07-05（A2V設計セッションの裏取り資料）
- 目的: 我々のA2V（アップロード音声のlatentを全長凍結→動画のみdenoise＝公式 `A2VidPipelineTwoStage` と同型）の観測を、LTX-2 / LTX-2.3 のコミュニティ先行報告と突き合わせ、我々の仮説①②③が「モデルの性質」なのかを判定する。
- 手法: リサーチ統括エージェント＋観点別サブエージェント5本（英日）でReddit/GitHub/HuggingFace/ComfyUI/X/note.com/技術ブログを横断収集。各所見に**出典URL＋信頼度**を付す。
- 信頼度の凡例: **[公式]** Lightricks一次資料 ＞ **[複数]** 独立した複数报告 ＞ **[単発]** 単一の実務者/バグ報告 ＞ **[一般分野]** LTX非限定の同種モデル知見（転用は要注意） ＞ **[未確認]** マーケ/SEO調・スニペットのみ。

---

## 0. 要約（先に結論）

| 我々の仮説 | コミュニティ裏取りの判定 | 一言 |
|---|---|---|
| **① 女声の方が得意** | **✗ 支持されない（むしろ逆の単発報告あり）** | 唯一の性別別報告は「**男声の方が崩れにくい**」。我々のZira>Davidは声質ではなく**話速・明瞭度・波形品質の交絡**の可能性が高い。 |
| **② 特定音源の波形が苦手** | **○ 支持（複数＋一般分野）** | 「濁った/圧縮された/単調な/破裂音が潰れた音声」は同期が滑る、という報告が一貫。ただし要因は"合成/実録"ではなく**音声の明瞭さ・プロソディ・感情整合**。 |
| **③ 背景の書き込み・人物の動きが増えると弱くなる** | **○ 支持（動き/複数話者は公式明言・"背景"単独は未確認）** | arXiv Limitationsが複数話者混同を明言・公式prompt guideが「過負荷シーンは明瞭さ低下」「極端な視野角で口の維持が困難」。**我々の"賑やかな町×歩行×男声=弱い"は"動き＋広い画角＋ナレ化"の合算で既知プロファイル通り**。"複雑背景"単独を名指す一次資料は無い。 |

**改善ノブ上位3（期待度つき）**
1. **画角を寄せる（正面・トーキングヘッド・単一話者・動きを抑える）＋プロンプトに "lip sync" を入れる** — 期待度: 高（[公式]＋[複数]）
2. **音声前処理: クリーンな発話・先頭に0.2–0.5s無音・10秒以上の尺・破裂音が潰れない明瞭なTTS・プロンプトの感情と声のトーンを一致** — 期待度: 中〜高（[単発]LTX＋[一般分野]）
3. **ComfyUI: 音声リップシンク時はモーション制御ガイド（IC-LoRA/AddGuideMulti）をoffに**（口不動化の主因の一つ・2ユーザー実証）。加えてネイティブ解像度厳守（非標準で audio VAE整合崩れ）。※24fps→30-48fpsのfps主張はSEO調で実スレ未確認＝要自己検証 — 期待度: 中（[複数]HF#66）

**重要な前提**: 我々の実装（音声凍結→動画のみdenoise）は**公式パイプライン `A2VidPipelineTwoStage` と同一機構**（stage1=半解像度で音声凍結・動画のみdenoise、stage2=distilled LoRAで2倍アップサンプル・音声固定）。したがって弱いリップシンクは"実装バグ"ではなく**モデル自体の既知の脆さ（fragile）**を踏んでいる可能性が高い。出典: `github.com/Lightricks/LTX-2` README（`packages/ltx-pipelines/README.md`）[公式]。

---

## 観点1: リップシンク品質の一般評（声質・話速・言語）

**総評（[複数]）**: LTX-2/2.3のリップシンクは旧来の後付けリップシンクより本質的に良い、と広く評価される一方、**"fragile（脆い）"** という言葉で繰り返し語られる。入力次第で「完璧」にも「口が全く動かない」にもなり、明確な予測ルールがないのが共通認識。

- 「LTX2.3のリップシンクは**非常に脆い**。長尺で入力音声に同期させ続けるのは難しい」「モーション制御/IC-LoRAガイドが音声リップシンクと"喧嘩"し、体は動くのに口の動きが完全に殺される」 — 複数寄稿者（gpundt, Foolsjoker）: https://huggingface.co/RuneXX/LTX-2.3-Workflows/discussions/66 **[複数]**
- 「**どの画像でもリップシンクが出るわけではない**。うまくいく画像もあれば、変なズームイン動画にしかならない画像もある。特定の写真は何をしてもダメで、別の写真は毎回うまくいく」 — Kijai/LTXV2_comfy discussion #35: https://huggingface.co/Kijai/LTXV2_comfy/discussions/35 **[複数]**
- 「Ltx 2.3 image audio 2 video が動かない・**口が動かない**」（無音口不動のバグ、未解決・root cause未確定） — https://github.com/Lightricks/ComfyUI-LTXVideo/issues/438 **[単発]**
- 「LTX 2 Dev FP8 の image+audio モードで**発話リップシンクが機能しない**」（tensor-mismatch。回避後も音声が"背景に流れるだけ"で口不動） — https://github.com/Comfy-Org/ComfyUI/issues/12161 **[単発]**
- 失敗モードの記述（第三者ブログ・SEO調）: 「ロボットが水でうがいするような声」「1980年代の吹替のようなリップシンク」「フレーム30で1秒遅れる音声ドリフト」「顎が開かない腹話術状態」「密な音声で全音節に合わせようとして口が不自然な形に伸びる」 — https://ltx-23.org/blog/social-viral-platforms **[未確認]**（一次報告と内容整合的だが検証不可）
- 日本語（LUTA@AI）: 「**音声が5〜6秒未満だとリップシンクが反応せず**、10秒以上で安定動作」「FFLF（始終点）と外部音声リップシンクは同一WFで排他」 — https://note.com/luta_ai/n/n6cfb06fb6b69 **[単発]**
- 日本語（てくあ）: 「約30秒のimage+audio→videoが"かなり上手く"同期」「弱点=**音声が多言語ミックスになる**」「音声クローンの再現度は高いが効果音生成は弱い」 — https://note.com/ai_tec/n/nc38f6b36d0d8 **[単発]**

### 仮説① 女声の方が得意 — **✗ 支持されない（唯一の性別別報告は逆）**
- ワークフロー作者RuneXXの明言: 「**男声入力の方が、モデルは遥かに苦労が少なかった**」「リップシンクの動きは**男声（画像と一致する）でより良く働くようだった**」。逆に**女声入力ではモデルが"ナレーション/ボイスオーバー"読みにデフォルトしやすく**、「動画の人物が喋っている」とプロンプトで明示する必要があった、とも。 — https://huggingface.co/RuneXX/LTX-2-Workflows/discussions/10 **[単発]**
- これは我々の観測（Zira女声>David男声）と**逆方向**。ピッチ（高/低）を単独で切り分けた報告は皆無で、上記も性別と「ナレーション化しやすさ・プロンプト挙動」を混同している。
- **判定**: 「女声が本質的に得意」を支持する外部証拠は無い。むしろ単発だが逆の報告がある。**我々のZira>Davidは声質そのものではなく交絡因子（話速・明瞭度・波形の圧縮/破裂音・感情の平板さ）で説明する方が妥当**。→ 観点5・観点2参照。

### 仮説② 特定音源の波形が苦手 — **○ 支持**
- 「濁った(mushy)ポッドキャスト音源は同期が滑る／軽くde-ess・ゲートしたVOはLTXに"背骨"を与えた」（LTX公式ブログのスニペット、full fetch失敗のため未検証引用） **[未確認→公式寄り]**。ロボット/ノイズ音声は"badly dubbed"化。
- **判定**: 波形の質（明瞭さ・破裂音の明瞭度・ノイズ・単調さ）が同期品質を左右する、は一貫支持。ただし「合成か実録か」ではなく**音声のクオリティ属性**が本質（観点5で定量あり）。

### 言語（英語 vs 他）
- 生成言語は**公式に"英語"のみ**（観点4）。「175+言語」等はUI翻訳/マーケ表現で、品質報告ではない。
- 独立した言語別品質比較は**ほぼ皆無**。唯一の関連は上記"多言語ミックス音声で弱い"という日本語単発のみ。**この小領域はコミュニティデータが実質存在しない。**

### データが存在しない小領域
- 高ピッチ vs 低ピッチの単独比較: **無し**（男女の単発が代理になるだけ）。
- 速い発話 vs 遅い発話の体系的テスト: **無し**（下記の推論的支持のみ）。
- **Reddit（r/StableDiffusion, r/comfyui, r/LocalLLaMA）: 該当スレッドがヒットせず**。本トピックの議論は現状HuggingFaceのDiscussionsタブ・GitHub issues・X・note.comに集中。

---

## 観点2: シーン複雑度の影響（クローズアップ vs 引き・動き・複数人・背景）

### 仮説③ 背景の書き込み＋人物の動きでリップシンクが弱くなる — **○ 支持（一部は公式が明言・ただし"背景"単独は未確認）**

**公式一次資料（最重要）**
- **arXiv「LTX-2: Efficient Joint Audio-Visual Foundation Model」（HaCohen et al., Lightricks）** — https://arxiv.org/abs/2601.03233 / html: https://arxiv.org/html/2601.03233v1 **[公式]**
  - Limitations（逐語）: 「**複数話者シナリオでは、モデルは発話内容をキャラクターに不整合に割り当て、どのキャラが特定の台詞を喋るべきかを混同することがある**」＝**複数話者で劣化**を公式が明言。
  - 「~20秒を超えると temporal drift・同期劣化・シーン多様性低下」（尺効果）。
  - Figure 3: cross-attentionが「話者間で注意を動的に移し…**クローズアップ発話では口の領域にフォーカス**」＝**クローズアップ発話はアーキ的に優遇されるケース**と位置づけ。
- **公式 LTX-2.3 Prompt Guide** — https://ltx.io/model/model-blog/ltx-2-3-prompt-guide **[公式]**: 「詳細度をショットスケールに合わせよ—クローズアップは引きより多くの詳細が要る」。強み=「感情的な人間の瞬間」「単一被写体の強い感情表現・微細なジェスチャー・表情のニュアンス」。警告=「**過負荷なシーン（多すぎるキャラ/アクションは明瞭さを下げる）**」。
- **公式 LTX-2 prompting guide** — https://ltx.io/model/model-blog/prompting-guide-for-ltx-2 **[公式]**: 「アクション/キャラ/指示を足すほど、一部が出力に現れない確率が上がる」。
- **公式「How To Create Good Lip Syncing With AI」** — https://ltx.io/blog/lip-sync-ai **[公式]**: 「**正面〜3/4の顔角度が最も信頼できる**」「**視野角が極端になるほど、現実的な口の動きの維持は難しくなる**」「複数話者は処理難度を上げる（各顔を別々に追跡する必要）」。**背景の複雑さ単独への言及は無し**。

**コミュニティ（直接fetch確認）**
- GitHub ComfyUI-LTXVideo #395「複数キャラで誰が喋っているか制御できない」: 「キャラA/Bが台詞の一部にランダムにリップシンクし、正しい部分には合わない」「順序付けしても両者が同時に喋るように動く」。未解決。 — https://github.com/Lightricks/ComfyUI-LTXVideo/issues/395 **[公式repo・単発]**＝上記arXiv制限の具体的失敗モード。
- モーション制御条件付け（IC-LoRA/AddGuideMulti）が音声リップシンクを"剥がす/上書き"して口不動化（gpundt/Foolsjoker、2ユーザー） — https://huggingface.co/RuneXX/LTX-2.3-Workflows/discussions/66 **[複数]**＝**人物の動き（motion-control経由）がリップシンクを壊す**実証。
- LipDub側でも公式に「**単一話者に留めよ**」 — https://www.runcomfy.com/comfyui-workflows/ltx-2-3-iclora-lipdub-in-comfyui-precise-lip-sync-video-creation **[公式寄り]**
- 日本語(一創/note): 「LTX-2.0では**外部音声リップシンクで口の動きに引っ張られて顔全体が崩壊**、特に長尺。LTX-2.3で一貫性がかなり改善」 — https://note.com/ai_tec/n/nc38f6b36d0d8, https://www.issoh.co.jp/tech/details/11290/ **[複数(日)]**＝動き/長尺が顔・口を崩す方向は2.0で顕著、2.3で緩和も傾向は残る。
- 反例(宣伝寄り): SI2V全身演技WFは「喋る全身演技付きリップシンク」を謳う — https://note.com/ai_hakase/n/n12f518d3f059 **[単発・宣伝]**。"できる"であって"クローズアップと同等品質"ではない。
- 集約(未検証・複数二次): 「極端な角度と賑やかな背景を避けよ・顔を中央・大きなカメラ移動やアクションはリップシンク信号と競合」 — 方向性は公式と整合だが各ページ逐語未確認 **[未確認・中]**。

- **判定（精緻化）**:
  - **人物の動き・複数話者・アクション過多で劣化 = ○ 公式が明言**（arXiv Limitations＋公式prompt guide＋GitHub #395＋HF #66）。
  - **クローズアップ/正面が優位 = ○ 公式**（arXiv Fig.3＋lip-sync-ai＋prompt guide）。
  - **"背景の書き込み"単独で劣化 = △ 一次資料では未確認**（公式は"キャラ数/アクション/視野角"を言うが"複雑背景"を名指ししない）。二次集約のみ＝真のギャップ。
  - **我々の「賑やかな町×歩行×男声ナレ(720p)=弱い」は、"歩行(動き)＋広い画角＋ナレ化しやすさ"の合算で説明可能**。＝モデルの既知プロファイル通り。"背景の賑やかさ"だけを主犯にするのは一次証拠不足。

### プロンプト/画角/解像度の実践ノブ（観点2発）
- **プロンプトに "lip sync" を明示的に含める**（口の動きを優先させる） — [公式] ltx.io/blog + comfy.org。期待度: 中〜高。
- 女声などで"ナレーション読み"に落ちる場合、「**動画の人物が喋っている**」と明示 — RuneXX #10 **[単発]**。期待度: 中。
- 正面・口が隠れない・顔中央・必要ならタイトクロップ — [公式寄り]。期待度: 高。
- 微細モーション（瞬き・小さな首振り）を記述して静止画化を防ぐ — comfy.org **[公式寄り]**。期待度: 中（静止化回避であって同期強化ではない）。

---

## 観点3: ガイダンス設定・蒸留 vs dev

- **`a2v_guidance_scale`（= modality CFG scale）既定 3.0**、`1.0` で無効化。「非同期な音声/動画予測から遠ざけるようステアリングする」機構（`SKIP_A2V_CROSS_ATTN`/`SKIP_V2A_CROSS_ATTN` を跳ばした予測から離す）。 — DeepWiki（Lightricks/LTX-2 コード構造反映、AI生成wiki）: https://deepwiki.com/Lightricks/LTX-2/4.11-multimodal-guidance ＋ 独立に同値3.0を持つ community実装 github.com/techfreakworm/LTX2.3-ImageAudioToVideo **[複数・medium]**。介入余地は 1.0(off) の**上側**のみ。
- 公式READMEの音声ガイダンス群: `--audio-cfg-guidance-scale` / `--audio-stg-guidance-scale` / `--audio-stg-blocks` / `--audio-rescale-scale` / `--audio-skip-step`（既定値は `constants.py`、READMEに数値は非掲載）。一般域: `cfg_scale` 2.0–5.0 / `stg_scale` 0.5–1.5 / `rescale_scale` 0.5–0.7。 — https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md **[公式（名称）／low（数値）]**
- コミュニティ実測: `video_cfg_scale=3.5`, `video_rescale_scale=0.7`。**stage1で `num_inference_steps>10` にするとリップシンクが劣化**（MPS/Apple Silicon特有の可能性・過denoise感度の示唆） — techfreakworm 実装 **[単発・MPS特有]**。
- 別ブログの実測ノブ（SEO調）: **Image Strength 0.65 / Audio Guidance 1.5**。「画像強度が高すぎると顎が開かない」「音声強度がテキスト制約なしで過大だと口が怪奇な形に」 — https://ltx-23.org/blog/social-viral-platforms **[未確認]**。
- 実運用config例（GitHub issue #126）: 動画/音声とも `cfg=3.5`・40 steps・LoRA strength 0.8・1088×1920/24fps/97frames — https://github.com/Lightricks/LTX-2/issues/126 **[単発]**。

### 蒸留 vs dev のリップシンク品質差 — **明確なデータ無し（矛盾）**
- HFモデルカード: 蒸留=「8 steps, CFG=1」、dev=「full model・bf16で柔軟/学習可」。リップシンク固有の主張なし。 — https://huggingface.co/Lightricks/LTX-2.3 **[公式]**
- 「蒸留は速いが**motion/coherenceが低品質**、devの方が高stepで良い結果」 — unsloth GGUF discussion **[単発]**。
- 逆に「**蒸留の方が概ね良品質**、devはhit or missでblurry/fuzzy、二段devはノイズ増幅・音声もmuddy」 — GitHub issue #126 **[単発]**。
- **判定**: 蒸留/devのリップシンク品質を単独で比較した信頼データは**存在しない**。方向性の主張が同一repoユーザー間でも矛盾＝seed/設定/パイプライン依存が大きい。どちらが上とも断定不可。
- 付随: distilled 1.1 で**音声バグ**（呼吸時の"歪んだ鼻声"、seed散発、旧版に無し） — https://github.com/Lightricks/LTX-2/issues/197 **[公式repo/単発]**。sync自体でなく音質だが試聴印象に影響。
- 別軸の品質レバー: OmniNFT RL LoRA が音声-動画同期を狙い「DeSync 0.569→0.269（JavisBench）」を主張 — https://ltxworkflow.com/models **[ベンダー主張・未検証]**。

---

## 観点4: 公式情報

- **arXiv 2601.03233「LTX-2: Efficient Joint Audio-Visual Foundation Model」（Lightricks, [公式]）**: Limitationsで**複数話者の発話割り当て混同**を明言・~20秒超で同期劣化/temporal drift・Figure 3でクローズアップ発話時に口領域へ注意集中。 — https://arxiv.org/abs/2601.03233 （sync精度の数値ベンチは本パスでは未読・技術詳細の追加一次候補）
- **公式プロンプトガイド群（[公式]）**: LTX-2.3 Prompt Guide（https://ltx.io/model/model-blog/ltx-2-3-prompt-guide 「過負荷シーンは明瞭さ低下」「詳細はショットスケールに合わせよ」）／LTX-2 prompting guide（https://ltx.io/model/model-blog/prompting-guide-for-ltx-2 ）／「How To Create Good Lip Syncing」（https://ltx.io/blog/lip-sync-ai 「正面〜3/4が最も信頼できる」「極端な視野角で口の維持が困難」「複数話者は処理難度を上げる」「クリーンな発話音声で口形が正確」）。
- **モデルカード（HF, [公式]）**: 「DiTベースのaudio-videoファウンデーションモデル、単一モデルで**同期した動画と音声**を生成」。生成**言語=英語のみ**（他所の"9言語"はUI翻訳）。明記された制限: 「**発話でない音声を生成する場合、音質が低下しうる**」＝speechが強いケース、環境音/非発話は弱い。技術制約: 幅×高は32で割り切れ・フレーム数は8n+1。 — https://huggingface.co/Lightricks/LTX-2.3
- **公式パイプ README（[公式]）**: `A2VidPipelineTwoStage`＝「入力音声で駆動する動画生成」。stage1「半解像度で**音声条件付き（音声を凍結し動画のみdenoise）**」、stage2「2倍アップサンプル＋distilled LoRAで動画のみrefine、音声固定」。「入力音声は audio VAE でエンコードされ初期音声latentとして使用」。CLI: `--audio-path`（必須）,`--audio-start-time`,`--audio-max-duration`。 — https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md ＝**我々の実装と同型を公式が定義**。
- **公式ブログ ltx.io/blog/lipsync-vs-lipdub（[公式]、reader proxy経由で取得）**:
  - Lipsync=「音声入力に口の動きを同期させ**新規動画を生成**（標準のA2Vパイプライン）」。LipDub=「**既存フッテージを再吹替する専用IC-LoRA**、話者の見た目とシーンを保ちつつ台詞を差替」。
  - **公式プロンプトTip: 「プロンプトに "lip sync" を含めると、モデルが口の動きを優先しやすい」**。
  - LipDub制限: 「新音声は参照動画の音声特性に条件付けられる。**元トラックはパススルーしない**」。台詞は**テキスト駆動**（参照/ターゲット音声を取らない＝feature request #242で確認）。
- **LipDub IC-LoRA モデルカード**: Control Type「Video & Audio」・参照解像度=出力と同じ・"lip-dubbing dataset"で学習・gated。 — https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub **[公式]**
- **公式repoの既知issue（[公式repo]）**:
  - #242 「LipDubに参照音声を渡したい」（LipDubは現状テキスト駆動のみ・音声再吹替不可）
  - #197 「新distilled 1.1で音声バグ（歪んだ鼻声）」
  - #200 「audio-to-videoのvocoder/バンドルcheckpoint未公開」
  - #187 「ローカル音声生成が動かない」
  - #208(closed) 「音声生成をoffにして推論高速化」（音声はトグル可＝品質/速度トレードオフ）
  - ComfyUI-LTXVideo#438 「image+audio→videoで口が動かない」（未解決）
- **公式に明記が無い（ドキュメント欠落）**: 声質/アクセント/歌唱等の可否、音声前処理推奨（sample rate/format/ノイズ低減）、sync精度の数値ベンチ。arXiv `2601.03233`「LTX-2: Efficient Joint Audio-Visual Foundation Model」が検索に出たが本パスでは未fetch（さらなる技術詳細が要るなら次の一次資料候補）。

---

## 観点5: TTS音声 vs 実録音声の相性

### LTX固有
- **LTX固有で「TTSが実録より同期が悪い」という報告/ベンチは一切見つからず**。公式資料はTTSを"通常の入力パス"として扱い、劣化パスとは位置づけていない。 **[LTX固有・medium]**
- LTXの既知失敗モードは**声の合成/実録とは直交**: 解像度/latent整合バグ、低FPS、プロンプト感情と音声感情の不一致（「声は怒っているのにプロンプトが"happy"だとズレる」）、複数話者diarization。 — https://ltx-23.org/blog/social-viral-platforms **[LTX固有・fetch済]**
- LTX音声前処理Tip（WF作者推奨）: **ステレオに変換・先頭に0.2–0.5s無音**（monoは"ナレーション的"、frozen-first-frame/ズレ回避） — https://huggingface.co/RuneXX/LTX-2.3-Workflows/discussions/39 **[LTX固有・fetch済]**
- **結論(LTX)**: TTS（Zira/David/ElevenLabs等）が実録より劣る、という証拠も反証も**LTXには無い**。ただしWeb上のLTX-2.3音声評価は薄くマーケ/how-to中心＝"報告が無い"は弱い証拠であって同等の証明ではない。

### 一般分野（LTX非限定・転用注意）
- **Wav2Lip論文がTTS vs 吹替 vs 実録を直接測定し、TTSが有意に難しい**（objective/subjective両方）: Overall Experience TTS 4.05 / Dubbed 4.13 / Random 4.15、sync精度 TTS 3.85 / Dubbed 4.08 / Random 4.18、モデル出力への選好票 TTS 51.2% / Dubbed 60.2% / Random 64.5%。著者自ら「TTS生成音声のリップシンクには改善余地」。 — https://ar5iv.labs.arxiv.org/html/2008.10010 **[一般分野・high]**。ただし2020年のGAN系で、LTXのdiffusion/DiT同時生成に機構が転用できる保証はなし（相対で約3–8%の低下）。
- SadTalker #868: **TTS由来の問題はしばしば言語/音素カバレッジ問題**（中国語TTSで悪化）＝"合成 vs 実録"ではなく音素側。 — https://github.com/OpenTalker/SadTalker/issues/868 **[一般分野・medium]**
- OmniHuman系: 「p/b/mで唇が閉じ切らないと途端に合成感、ノイズ/クリップ音声は子音をぼかし口形が曖昧に」＝**音声のアーティファクト**が要因（TTS由来そのものではない）。 **[一般分野・medium]**
- Sync等ツール共通の品質ドライバ: クリーン/単一話者/低ノイズ/自然な話速/尺一致。**"合成音声"は独立リスク要因として挙げられていない**。 — https://sync.so/docs/compatibility-and-tips/improving-lip-sync-quality **[一般分野・high]**。推奨仕様: 48kHz/16bit+・WAV/MP3・先頭末尾の無音トリム（無音区間にリップを合わせないため・"≥300ms無音時の口の安定"が評価軸）。

### 我々のZira/Davidテストへの含意
- **TTSは"確定した不利"でも"確定した同等"でもない**。field証拠が指す真のリスクは、TTSが持ちうる相関属性＝**平板なプロソディ（プロンプト感情と不一致=LTX明記の失敗モード）・破裂音のクリップ/ロボ調・不自然な話速・自然な間の欠如**。クリーンで自然な間・適切な句読点のTTS（現代のZira/David/ElevenLabs妥当設定）なら実録に近づき、低品質/単調/クリップTTSは劣るが、それは"合成だから"ではなく"劣化音声一般"の理由。
- **推奨**: 同一台本で **TTS vs 実録の並行テスト**を1本行い、Zira/Davidが実際に劣るなら、まず**プロソディ/話速/感情整合**を疑う（"TTSだから頭打ち"と決めつけない）。**特にDavid男声"早口"は仮説③(速い発話)＋観点2の負荷に該当**し、我々の弱化を最もよく説明する。

---

## 我々の仮説の最終判定

- **① 女声の方が得意 → ✗ モデルの性質としては支持されない。** 性別を扱った唯一の外部報告（RuneXX）は**逆（男声の方が崩れにくい）**。ピッチ単独の証拠は皆無。我々のZira>Davidは**声の性別ではなく、Davidの早口・明瞭度・波形品質という交絡**で説明する方が整合的。→ 声質仮説は棄却寄り、要因は②③に再配分。
- **② 特定音源の波形が苦手 → ○ 支持（複数＋一般分野）。** ただし本質は「合成/実録」ではなく**音声の明瞭さ・破裂音の明瞭度・単調さ/感情整合・ノイズ**。Wav2LipではTTSが実測で数%不利だが旧世代GAN。
- **③ 背景/動きが増えると弱くなる → ○ 支持（一部は公式明言）だが"背景"単独は未確認。** **人物の動き・複数話者・アクション過多での劣化＝arXiv Limitations等の公式が明言**。クローズアップ/正面優位も公式。ただし**"複雑背景"を単独要因として名指しする一次資料は無い**（二次集約のみ＝ギャップ）。**我々の"町×歩行×男声ナレ"の弱化は"動き＋広い画角＋ナレ化"の合算で説明でき、モデルの既知プロファイル通り**＝実装欠陥ではない可能性が高い。

---

## 実践的改善ノブ候補リスト（期待度つき）

| # | ノブ | 内容 | 期待度 | 根拠(信頼度) |
|---|---|---|---|---|
| 1 | 画角を寄せる | 正面・トーキングヘッド・**単一話者**・動き/背景を抑制 | **高** | [複数]／[公式寄り] |
| 2 | プロンプトに "lip sync" | 口の動きを優先させる公式Tip | 中〜高 | [公式] ltx.io/comfy.org |
| 3 | フレームレート 30–48fps | 24fpsはP/B/M等の口形フレームを落とす（**Redditコンセンサスと称するが実スレッド未確認・SEO調の可能性＝鵜呑み禁物**） | 低（要自己検証） | [未確認] ltx-23.org |
| 4 | ネイティブ解像度厳守 | 1280×720/480×832等。非標準で audio VAE整合崩れ→音声遅延 | 中 | [未確認]＋[単発] |
| 5 | 音声: 先頭0.2–0.5s無音＋≥10s尺 | frozen-first-frame回避・短尺(<5–6s)は同期不発 | 中 | [単発]LTX(HF#39, LUTA) |
| 6 | 音声: クリーン/明瞭/de-ess・破裂音を潰さない | 濁り/クリップ/単調は同期が滑る | 中〜高 | [複数]＋[一般分野] |
| 7 | 感情整合 | プロンプトの感情と音声のトーンを一致（平板TTS注意） | 中 | [LTX固有] ltx-23.org |
| 8 | ガイダンス調整 | `a2v_guidance_scale`≈3.0起点。画像強度を上げ過ぎない(顎が開かない)・音声強度を上げ過ぎない(口が歪む)。community値: img strength≈0.65 / audio guidance≈1.5 | 中 | [公式(名称)]＋[未確認(数値)] |
| 9 | ステップ数を過大にしない | stage1で過denoiseするとsyncが壊れる報告（MPS特有の可能性） | 低〜中 | [単発・MPS] |
| 10 | 精密な再吹替は LipDub IC-LoRA | crop guidesで口領域を安定クロップ・単一話者 | 中 | [公式寄り] runcomfy |
| 11 | "ナレーション化"回避 | 「動画の人物が喋っている」と明示（女声で顕著） | 中 | [単発] RuneXX#10 |
| 12 | ComfyUIモーション制御ガイドをoff | IC-LoRA/AddGuideMultiが音声リップシンク条件を剥がす | 中 | [複数] HF#66 |
| 13 | 単一話者に限定 | 複数話者は発話割当を混同（公式が既知制限と明言・確定回避策なし） | N/A(制限) | [公式] arXiv 2601.03233 ＋ GitHub#395 |

---

## 情報が存在しない/未成熟な領域（無理に結論しない）

- **ピッチ（高/低）単独の品質差**: データ無し（男女の単発が代理になるだけ）。
- **速い発話 vs 遅い発話の体系テスト**: 無し（推論的支持のみ／我々のDavid早口が唯一の内部証跡）。
- **言語別の品質比較（英 vs 日 等）**: ほぼ皆無。生成言語は公式に英語のみ、多言語はUI表現。
- **TTS vs 実録の LTX固有比較**: ベンチ・ユーザー報告ともゼロ。field証拠(Wav2Lip)は旧世代で転用不確実。
- **蒸留 vs dev のリップシンク単独比較**: 信頼データ無し（矛盾する単発のみ）。
- **公式の音声前処理仕様/sync数値ベンチ**: モデルカード/README/ブログには未掲載（Lightricks側のドキュメント欠落）。arXiv 2601.03233 はLimitations/Fig3を確認済みだが、**sync精度の定量ベンチ節は本調査で未抽出**＝深掘りする場合の残タスク。
- **"複雑背景"単独の劣化**: 一次資料に名指しの記述なし（公式は"キャラ数/アクション/視野角"のみ）。二次集約のみ＝未確認のギャップ。
- **Reddit上の議論**: 事実上不在。議論はHF Discussions/GitHub/X/note.comに集中。

---

## 主要出典一覧（実アクセス確認済み中心）

- 公式: https://arxiv.org/abs/2601.03233 (LTX-2 論文・Limitations) ／ https://huggingface.co/Lightricks/LTX-2.3 ／ https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub ／ https://github.com/Lightricks/LTX-2 (README `packages/ltx-pipelines/README.md`) ／ https://ltx.io/blog/lipsync-vs-lipdub ／ https://ltx.io/blog/lip-sync-ai ／ https://ltx.io/model/model-blog/ltx-2-3-prompt-guide ／ https://ltx.io/model/model-blog/prompting-guide-for-ltx-2 ／ https://comfy.org/workflows/video_ltx2_3_ia2v-adca306765ce/
- 公式repo issues: https://github.com/Lightricks/LTX-2/issues/197, /242, /200, /187, /126, /208 ／ https://github.com/Lightricks/ComfyUI-LTXVideo/issues/395, /438 ／ https://github.com/Comfy-Org/ComfyUI/issues/12161
- コミュニティ(英): https://huggingface.co/RuneXX/LTX-2-Workflows/discussions/10 ／ https://huggingface.co/RuneXX/LTX-2.3-Workflows/discussions/66, /39, /141 ／ https://huggingface.co/Kijai/LTXV2_comfy/discussions/35 ／ https://www.runcomfy.com/comfyui-workflows/ltx-2-3-iclora-lipdub-in-comfyui-precise-lip-sync-video-creation ／ https://deepwiki.com/Lightricks/LTX-2/4.11-multimodal-guidance
- コミュニティ(日): https://note.com/ai_tec/n/nc38f6b36d0d8 ／ https://note.com/luta_ai/n/n6cfb06fb6b69 ／ https://note.com/ai_hakase/n/n12f518d3f059 ／ https://www.issoh.co.jp/tech/details/11290/
- 一般分野: https://ar5iv.labs.arxiv.org/html/2008.10010 (Wav2Lip) ／ https://github.com/OpenTalker/SadTalker/issues/868 ／ https://sync.so/docs/compatibility-and-tips/improving-lip-sync-quality
- 未検証寄り(SEO/マーケ): https://ltx-23.org/blog/social-viral-platforms ／ https://ltxworkflow.com/models ／ https://introl.com/blog/ltx-2-audiovisual-diffusion-synchronized-video-audio-2026
