# F: H3 の快適上限マーカーを重みの形式ごとに切り替える仕組みの棚卸し（2026-11-22）

調べた版: `Nz-Videomni` の作業ツリー（dev・HEAD 54f0724f 前提）。読むだけで、編集・テスト・GPU は使っていない。
電卓として使った台本: 同じフォルダの `marker.py`（印の位置の計算）・`hdr.py`（手元の重み 9 本のヘッダを標準ライブラリだけで読み、ブロックの大きな線形層に量子化の印が無いものを数えた）。
「推測」と書いたもの以外は、コード・文書・ヘッダで確かめた事実。

---

## 1. バックエンドの判定

### 1.1 関数と規則
- 種類の名前の一覧: `h3_format.py:616` `H3_QUANT_CLASSES = ("int8", "fp8", "w4a8")`。
- 規則: `h3_format.py:619-633` `quant_class(cfg)`。`cfg.quant`（印のある層 → 形式）の形式の集合だけを見て、w4a8 があれば `"w4a8"`、なければ fp8 があれば `"fp8"`、それ以外は `"int8"`。**量子化の層が 1 つも無い DiT も `"int8"`**（docstring と `tests/test_h3_quant_class.py:53` の `"unquantized": "int8"`）。
- 入力は `inspect_h3_dit` の返り値 `(header, cfg)` の `cfg` だけ（`services/pipeline_manager.py:707-708`）。

### 1.2 決める時点と保存先
- 読み込み成功時: `PipelineManager.load` の末尾 `pipeline_manager.py:471`、`reload` の末尾 `:524`（どちらも `_identify_transformer()` の直後）。
- 起動時: `api/context.py:129`（PipelineManager を作った直後に 1 回。前回の選択の復元に対して）。
- 保存先: **メモリの属性 1 つだけ** `self.transformer_quant_class`（`pipeline_manager.py:253-259`、初期値 None）。設定ファイル・state.json には書かない。解放（unload）では消えない。
- 判定の中身 `identify_quant_class`（`pipeline_manager.py:685-710`）: 先に None にし、H3 でない・登録簿が無い・パスが解決できないなら None のまま。`inspect_h3_dit` → `quant_class`。例外は WARNING を出して None（読み込みは失敗させない）。

### 1.3 「ブロックの大きな線形層に印の無いものが 1 つでもある」を今の情報で判定できるか
**できる。足りないものは無い。**
- `cfg.num_layers`（本体ブロック数。`h3_format.py:217`・`:525`）と `cfg.quant`（接頭辞を外した層名 → 形式。`:229`）がある。
- 本体ブロックは `_count_contiguous` で 0 から連続、かつ `_BLOCK_REQUIRED`（`:253-256`）で `attn.qkv_proj.weight`・`attn.out_proj.weight`・`mlp.fc1.weight`・`mlp.fc2.weight` が全ブロックにあることが保証済み。
- したがって `for i in range(cfg.num_layers)` で `blocks.{i}.attn.qkv_proj` など 4 層のどれかが `cfg.quant` に無ければ「印の無い大きな線形層がある」と言える。
- 印の無い層の重みが I8・F8 なら検査器が断る（`_refuse_unmarked`、`:366-384`）ので、**印が無い＝浮動小数（BF16／F16／F32）**。dtype まで区別したいなら、同時に返る `header.tensors[prefix + "blocks.N.….weight"].dtype` で取れる（cfg には dtype を残していない）。
- **注意点（事実）**:
  - adaln_proj は数えてはいけない。公式の重みは adaln_proj が F16 で印が無い（B_header_survey §4。R・F・PulpCut は F16、pinkcherry は F32、h3ErosMax・10Eros は BF16）。
  - token_refiner（2 ブロック）も数えてはいけない。手元の 9 本すべてで BF16・印なし（`hdr.py` の結果で refiner の印は 0）。block swap の対象は `dit.blocks` だけ（`engine_h3/pipeline_h3.py:1652`）で、refiner は常駐側なので置き場の大きさに効かない。
- 手元の 9 本を同じ規則で数えた結果（`hdr.py`）:

| 重み | 印の無い大きな線形層を持つブロック | 今の種類 |
|---|---|---|
| Sulphur-3 keepB0-1-49 | 0・1・49（BF16 12 層） | int8 |
| 10Eros int8HQ | 0・1・47・48・49（BF16 20 層） | int8 |
| 公式 FL2VA・公式 Ref2VA・PulpCut・pinkcherry v18・h3ErosMax・s3_raw_int8 | なし | int8 |
| 10Eros w4a8 14gb | なし（w4a8 180 層＋int8 20 層で 50 ブロック×4 層を埋める） | w4a8 |

→ 規則を足すと、種類が変わるのは **Sulphur-3 keepB0-1-49 と 10Eros int8HQ の 2 本だけ**。公式・PinkCherry・h3ErosMax・PulpCut・s3_raw_int8・10Eros w4a8 は誤って入らない。

---

## 2. 配り方

### 2.1 `GET /models`
- `api/models_registry.py:206-210`: `base_models[]` の各要素に `transformer_quant_class` を必ず入れる。有効なベースモデルだけ属性を読み、他は None。ファイルは読まない。
- 応答の型を縛るスキーマ（pydantic の Literal など）は無い（`transformer_quant_class` を参照する .py は `pipeline_manager.py`・`models_registry.py`・`mcp_server/tools/system.py` の 3 つだけ）。**新しい文字列はそのまま流れる。**
- 値: `"int8"`｜`"fp8"`｜`"w4a8"`｜`null`。null は LTX・有効でないベースモデル・ファイルが無い・ヘッダが読めないとき（仕様書 `Videomni_Backend_Specification.md:1757`）。

### 2.2 記述子
- `scripts/manifests/40-minimaxh3.json:46`: `"comfort": { "single_rows_budget": 219516, "single_rows_budget_by_quant_class": { "w4a8": 203092 } }`（同じファイル 7 行目の `_note` にも説明がある）。
- 型: `config.py:342-363` の `EngineComfortProfile.single_rows_budget_by_quant_class: dict[str, PositiveInt]`（既定 `{}`）。
- 起動時の検査: `api/context.py:245-251` が鍵を `H3_QUANT_CLASSES` と照合し、違えば RuntimeError で起動を止める。**新しい鍵は `H3_QUANT_CLASSES` に足さないと起動できない**（逆に言えば、足せば検査は自動で通る）。
- サーバーは線を読まない（助言専用。`config.py:347`）。

### 2.3 文書・周辺
- 仕様書: 最新は v0.5.123（`Videomni_Backend_Specification.md:179`、§6.7 `:1591`、§6.9(c) の例 `:1742`・説明 `:1757`）。値の列挙と「量子化の層が無い DiT も int8」がここに書かれている。
- 操作パネルの API 文書: `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md` §3.2（200 行）・§3.5（257 行）に値の列挙。
- MCP: `mcp_server/tools/system.py:66-74`（get_config）・`:109-115`（list_models）の docstring に `int8 / fp8 / w4a8 / null` の列挙。中身は素通し（`client.get_json`）なので動きは変わらない。
- Gradio: `gradio_ui/constraints.py:246-262` は `single_rows_budget` が null でないかだけを見て LTX の警告を隠す。種類も種類ごとの線も読まない（H3 の印は作らない）。影響なし。
- 操作パネルのモック: `webui/src/bridge/mockBridge.ts:196`（h3 のプロファイル `{ w4a8: 203092 }`）・`:1610`（有効な H3 は常に `"int8"`）。予備の設定 `webui/src/modes/single/defaultConfig.ts:151`。

### 2.4 種類の値を 1 つ足すときに変更が要る場所
必須（これが無いと起動しない・線が出ない）:
1. `h3_format.py:616` `H3_QUANT_CLASSES` に名前を足す。
2. `h3_format.py:619-633` `quant_class` に新しい分岐（優先順は §5）。
3. `scripts/manifests/40-minimaxh3.json:46` の `single_rows_budget_by_quant_class` に鍵と値。`_note`（7 行目）も。

テストの更新（§6.7）と文書の更新（§7）が続く。

---

## 3. 操作パネル側

### 3.1 経路
1. 型: `webui/src/api/types.ts:200` `export type TransformerQuantClass = "int8" | "fp8" | "w4a8";`、`BaseModelBlock.transformer_quant_class?: TransformerQuantClass | null`（`:1312`）、`EngineComfortProfile.single_rows_budget_by_quant_class?: Record<string, number> | null`（`:1137`。鍵は文字列）。
2. 受け取り: `webui/src/shell/useBaseModels.ts:229` `quantClass: block.transformer_quant_class ?? null`（値の検査なし）、`activeQuantClass`（`:341-343`）。一覧は読み込みのたびに取り直す（`refresh`、`:265`）。
3. 線を選ぶ 1 か所: `webui/src/shell/comfortTable.ts:270-285` `resolveSequenceRowsBudget(limits, engineFamily, quantClass)`。`hasOwnProperty.call(byClass, quantClass)` で辞書を引き、正の有限数ならそれ、そうでなければ `single_rows_budget`。
4. 印の位置: `webui/src/shell/h3SequenceRows.ts:178-191` `h3ComfortFrames`（17n+5 の梯子を上から下り、S ≤ 線 の最初の段）。
5. 呼び出し元（すべて同じ解決関数）:
   - Single: `modes/single/useGenerationForm.ts:734`、`modes/single/SingleScreen.tsx:185`
   - Chained（クリップごと）: `modes/chained/ChainedScreen.tsx:681`・`:1157`
   - Edit の Outpainting: `modes/edit/useOutpaintForm.ts:602`
   - Edit の Inpainting: `modes/edit/useInpaintForm.ts:554`
   - 右クリックの種: `timeline/prefillSeed.ts:503`（AppShell から `quantClass` を渡す。`shell/AppShell.tsx:1388`）
   - カーソルの項目の初期値（AppShell の #9/#10）: `shell/AppShell.tsx:1070-1074`
   - 受け渡し: `AppShell.tsx:1876`・`:1923`・`:1985`、`EditScreen.tsx:265`・`:437`

### 3.2 不透明な鍵として引くだけか
- **実行時は不透明な文字列の鍵として辞書を引くだけ。** 値ごとの分岐は無い（`"int8"`／`"fp8"`／`"w4a8"` を src の中で比べている箇所は、型定義・コメント・テスト・モックだけ）。`useBaseModels` も値を検査しない。
- したがって **バックエンドとマニフェストに新しい鍵を足せば、いま配置済みの操作パネル（1-286 以後の版）は、コードを変えず再ビルドもせずに新しい線で印を出す**（事実の組み合わせからの結論。実機で確かめてはいない）。
- 型 `TransformerQuantClass` は実際に届く値より狭くなる（型の上の嘘）が、実行には影響しない。型を正しく直すなら `types.ts:200` の一行と説明のコメント（`types.ts:197-199`・`:1129-1136`・`:1305` 付近、`useBaseModels.ts:74-75`）で、これは再ビルドと配布（aux2 の埋め込み・配布先 2 か所）を伴う。

### 3.3 テスト（vitest）で値を固定している箇所
- `shell/comfortTable.test.ts:400-439`（w4a8 → 203092、int8・fp8・null → 219516、不正値・鍵なし）
- `modes/chained/ChainedScreen.h3Comfort.test.tsx:172-193`（w4a8 で 328、int8 で 345）
- `shell/useBaseModels.test.ts:419-461`（`"int8" | "fp8" | "w4a8" | null | undefined` を型で固定）
- `bridge/mockBridge.test.ts:708-719`（モックの h3 プロファイルを `toEqual` で固定）、`bridge/mockBridge.h3.test.ts:430-439`（有効な H3 は `"int8"`）
- 新しい鍵を足しても、これらは壊れない（モックと予備の設定を変えない限り。どれも実際のマニフェストとは突き合わせていない）。

---

## 4. 線の値の作り方と暫定の線

### 4.1 既存の 2 本の導き方
- **既定の線 219,516**: 快適上限表 §17.5 の (a)。参照なしの点で「快適の最大」219,516（Ref2VA・参照なし・1920×1088×362・K=0 の S。プロンプト 30 トークン込み）と「OOM の最小」223,604（1920×1088×362・K=1）の間に線が引け、**快適の最大の点の S そのもの**を線にした（`COMFORT_LIMIT_TABLE.md:1127`、§1.1 の `:29`、設計正本 裁定 94「線は安全側」`MINIMAX_H3_DESIGN.md:266`）。
- **w4a8 の線 203,092**: 快適上限表 §21.4・§21.5 (b)。Kijai の w4a8 の 1920×1088・K=1 の c（快適の上端）328 フレームの S（`COMFORT_LIMIT_TABLE.md:1374`・`:1396`）。U（1 段上の OOM）は 345 の 213,348。
- **決まり（事実）**: 「境界の 1 段上の OOM 点と快適な点の間」のどこかに引くのではなく、**実測で快適だった最も大きい S（プロンプト 30 トークン込み）にちょうど置く**。LTX の線の規則も同じ向き（「既知の溢れ点より下の最も高い段」§1.4 `:89`）。
- 操作パネルの S はプロンプトを数えない（`h3SequenceRows.ts:22-23`）ので、較正点そのもの（S−30）は線より 30 低く、印はちょうどその点に乗る（例: 1920×1088×328・K=1 は操作パネルの S で 203,062 ≤ 203,092）。

### 4.2 K の数え方（文書の定義）
- K＝送るキーフレームの枚数。条件行 cond＝K×(高さ÷32)×(幅÷32)。参照が 0 件のときだけ、text に最大 2 枚分の（格子＋8）を足す（`COMFORT_LIMIT_TABLE.md:35`、`h3SequenceRows.ts:146-150`）。参照つきの印は本棚卸しでは計算していない。

### 4.3 D_vram_fermi.md の 3 つの前提の違い
- 固定分の増加はどれも +2,204MiB（環 6 本が BF16 のブロックの大きさ 738.35MiB になる。コードで決まる）。
- 違うのは 3 つ: 1 行あたりの上乗せ（楽観 +5,116 バイト＝w4a8 の実測値／中央 +5,376 バイト＝bf16 の入力 1 本分の差をコードから／悲観 +7,228 バイト＝否定済みの w4a8 の予約側の読みを上限の目安に流用）、断片化の余裕（795／900／939MiB）、上限（14,025／13,999／13,999MiB）。
- 結果の行の予算（プロンプト 30 込みの形）: 楽観 163,657・中央 160,472・悲観 154,676 行。

### 4.4 暫定の線の候補と印の位置（`marker.py` の計算。フレーム数の上限は 481。24fps）
同じ決まり（「快適と見込む最大の S にちょうど置く」）を概算に当てると、候補は (A) 予算そのもの、(B) 予算に収まる 1920×1088・K=1 の最大の段の S（＋30）。

(A) 予算そのものを線にした場合:

| 線（行） | 1920×1088 K=0 | 1920×1088 K=1 | 2560×1472 K=0 | 2560×1472 K=1 | 1280×768 K=0／K=1 |
|---|---|---|---|---|---|
| 今の既定 219,516 | 362 | 345 | 192 | 192 | 481／481 |
| 今の w4a8 203,092 | 328 | 328 | 175 | 175 | 481／481 |
| 楽観 163,657 | 260（10.8 秒） | 260 | 141（5.9 秒） | 141 | 481／481 |
| 中央 160,472 | 260 | 243（10.1 秒） | 141 | 124（5.2 秒） | 481／481 |
| 悲観 154,676 | 243 | 243 | 124 | 124 | 481／481 |

(B) 梯子の段の S に寄せた場合:
- 中央・悲観の見込みの c（1920×1088・K=1）は 243 → 線 151,808（＝151,778＋30）。印は 1920×1088 が K=0・K=1 とも 243、2560×1472 が K=0・K=1 とも 124、1280×768 は 481。**(A) の悲観と同じ印**になる。
- 楽観の見込みの c は 260 → 線 162,064。印は 1920×1088 が 260／260、2560×1472 が K=0 で 141・K=1 で 124、1280×768 は 481。

参考の S（操作パネルの数え方。プロンプトなし）:
- 1920×1088・K=0: 243＝147,690、260＝157,946、277＝168,204
- 1920×1088・K=1: 243＝151,778、260＝162,034、277＝172,292
- 2560×1472・K=0: 124＝136,574、141＝155,030、158＝173,486
- 2560×1472・K=1: 124＝143,942、141＝162,398、158＝180,854
- 1280×768 は 481（梯子の上限）で K=0 が 137,924・K=1 が 139,852。どの候補でも上限の 481 に印が出る（印は動かない）。

- 楽観の 1920×1088・K=1・260 は、D の見積もりで余裕が 65MiB しかない（D §5.2）。
- 「線は安全側」（裁定 94）に合わせるなら悲観（154,676）か (B) の 151,808 が素直な対応（推測を含む判断材料で、決定ではない）。

---

## 5. 優先の規則の判断材料

- **w4a8 の線が低い理由**（記録簿 §212.6、快適上限表 §21.6・D §1.2）: w4a8 の線形層は qkv の入力を共有しない経路（`forward_rows`）なので bf16 の入力を注意の段の間持ち続ける（割り当てで +5,116 バイト／行）＋呼び出しごとの大きな int8 の一時テンソル 2 本による予約側の断片化（予測より +360〜+1,037MiB）。Kijai は環が int8 より小さい（−920MiB）ので一部打ち消される。
- **BF16 混在の線が低い理由**（記録簿 §221.9、D §2〜§4）: 環 6 本が最大のブロック（BF16）の大きさになる固定分 +2,204MiB ＋ 素の qkv を行で切り出す経路で bf16 の入力を持ち続ける +約 5,400 バイト／行。
- **重なり方（推測）**:
  - 1 行あたりの上乗せは同じ仕組み（bf16 の入力を注意の段の間持ち続ける）で、山は全ブロックの最大なので、両方あっても二重にはならない見込み。
  - 環の大きさは最大のブロックで決まるので、BF16 のブロックがあれば w4a8 の「環が小さい」打ち消し（−920MiB）は消え、+2,204MiB が乗る。
  - w4a8 の断片化の上乗せは w4a8 の層がある限り別に乗る。Eros（w4a8＋int8 の端のブロック。環は int8 と同じ大きさ）が Kijai より下で OOM した事実（§21.5 (c)）は、環の大きさと w4a8 の効果が足し算で効くことと合う。
  - よって w4a8 の層と BF16 のブロックの両方を含む重みは、BF16 混在の線（約 15.5〜16.4 万行）よりさらに低い可能性がある。少なくとも w4a8 の線（203,092）より BF16 混在の線のほうが低いので、**低い線を選ぶなら BF16 混在の判定を w4a8 より先に置く**必要がある（今の `quant_class` は w4a8 を最初に判定している）。
- **fp8 と BF16 のブロック（推測）**: fp8 のブロックのバイト数は int8 と同じ（1 バイト／要素）で、fp8 は int8 と同じ境界だった（§21.5 (a)）。BF16 のブロックがあれば環は BF16 の大きさになるので、int8＋BF16 と同じ線でよい見込み。
- **実在の重み**: `models\MiniMaxH3\Weights` の 9 本のうち、w4a8＋BF16・fp8＋BF16 は 1 本も無い（`hdr.py`・B_header_survey §1・§8）。fp8・Kijai の w4a8 の重み（§21 で使ったもの）はいまこのフォルダに無い。BF16 混在は Sulphur-3 keepB0-1-49 と 10Eros int8HQ の 2 本だけ。
- **量子化の層が無い DiT（全部 BF16）**: 新しい規則では BF16 混在の種類に入る（今は int8）。環は BF16 混在と同じ大きさなので、int8 の線より新しい線のほうが近い見込み（推測）。F16・F32 のまま残したブロックは BF16 より大きい（F32 なら環が約 2 倍）が、該当する重みは無い。

---

## 6. 暫定の線を載せたときに起きうる問題（影響の棚卸し）

### 6.1 文書の正本との整合
- 快適上限表の題は「実測較正版」（`COMFORT_LIMIT_TABLE.md:1`）で、冒頭（`:4`）に「表の性格は『フェルミ推定』から『実測較正済みの式による計算値』へ変わった」とある。概算の線はこの性格づけと食い違う。
- 前例はある: §9.5 の Alpha Gen の `alpha_gen_budget` は「暫定値」として置かれた（2026-10-07 オーナー裁定、`:443`）。同じ書き方（節の見出しに「暫定値」、根拠の記録へのリンク）を取れる。
- 第1.1節の留意 10（`:46`）は「種類が int8 と判定されて既定の線が当たる」と書いており、書き換えが要る。第1.1節の「配信の形」（`:29`）の鍵の列挙も。
- 設計正本 裁定 105 ⑤（`MINIMAX_H3_DESIGN.md:384`）は「それまでは、この種の重みにも int8 の線の印が出る（裁定 102 の C1 の規則のまま）」、裁定 102 の C1（`:358`）は「w4a8 の層が 1 層でもあれば w4a8、それ以外は既定」。どちらも新しい裁定か更新注記が要る。
- 台帳 §1-291（`PENDING_TASKS.md:18-22`）の概要「int8 の線の快適上限の印が出る」も変わる。
- 小さな既存の食い違い（今回とは別）: 快適上限表 §1.1（`:30`）は「Outpainting 以外の Edit 系……には出さない」と書くが、Inpainting にも印がある（`useInpaintForm.ts:545-566`、CLOSED 1-286「Single・Chained・Inpaint・Outpaint の 4 つの印」）。

### 6.2 既存の利用者への見え方の変化
- 印が動くのは Sulphur-3 keepB0-1-49 と 10Eros int8HQ を選んだときだけ（§1.3 の表）。公式・PinkCherry・h3ErosMax・PulpCut・s3_raw_int8・10Eros w4a8 は今と同じ。
- 動く量: 1920×1088 で 345〜362 → 243〜260 フレーム（2〜6 段。約 4〜5 秒短い）、2560×1472 で 192 → 124〜141（3〜4 段）。1280×768 は 481 のままで動かない。
- 概算が外れた場合: 楽観側に外れれば印の内側で OOM（今と同じ種類の失敗が、位置を変えて残る）、悲観側に外れれば作れる長さを短く見せる（印は助言だけでサーバーは止めないので、生成そのものは妨げない）。

### 6.3 Chained・Outpainting・Inpainting・参照つきの共用
- 5 か所すべてが `resolveSequenceRowsBudget` を通るので、新しい線は同時にすべてへ出る（§3.1）。個別に分ける仕組みは無い。
- 共用の根拠は int8 の実測だけ（§19・§20、裁定 102 の留保 `:363` で w4a8 も「推定」）。BF16 混在の固定分 +2,204MiB はノイズ除去の段の環なので、Chained の各クリップ・Outpainting のノイズ除去にも同じく乗る見込み（推測）。環は復号の前に返す（`block_swap_prefetch.py:93-96`、D §2）ので、復号の段の山には効かない（推測）。
- 参照つき: 参照の行も S に重さ 1 で入る。1 行あたりの上乗せは bf16 の入力 [S, 5376] に比例するので参照の行にも乗る見込み（推測）。参照つきの BF16 混在は概算もしていない。

### 6.4 右クリックの種
- `prefillSeed.ts:503` と `AppShell.tsx:1070-1074` も同じ関数なので、新しいオブジェクトの初期フレーム数も新しい線で決まる（BF16 混在の重みを選んでいれば、初期値が短くなる）。参照 0 件・K は意図から決める（快適上限表 §1.1 留意 3）。

### 6.5 API の互換
- 古い操作パネル（1-286 より前）: `transformer_quant_class` も種類ごとの線も知らないので、既定の線 219,516 のまま（今と同じ）。
- 今の操作パネル（1-286 以後）: 知らない値でも検査せず辞書を引くので、新しい線が出る（§3.2）。
- 新しい操作パネル＋古いバックエンド: 値が来ないか鍵が無いので既定の線。
- MCP の利用者（AI エージェント）: 値は素通しで、docstring の列挙だけが古くなる。
- Gradio: 影響なし。
- 仕様書の §6 は「凍結 API 契約への加算」の扱いで、既存の欄に新しい値を 1 つ足すことになる。版を上げる（v0.5.124）記載が要る。厳密に列挙を検査するクライアントは手元には無い（事実の範囲）。

### 6.6 起動時の判定のキャッシュ
- 種類は永続化しておらず、起動時に復元した選択のヘッダから毎回判定する（`context.py:129`）。規則を変えたサーバーを起動した時点で、前回選んでいた重みの種類も新しい規則で出る。古い値が残る仕組みは無い。
- 追加の判定はヘッダの層名の集合を見るだけで、読み込み量は今と同じ（ヘッダだけ）。
- 既存の性質のまま: MCP・Gradio・生成時の自動読込で重みが替わったとき、操作パネルの印は次の取り直しまで古い（裁定 102 の留保）。
- `services/weight_identity.py`（SHA-256 のキャッシュ）・`services/weight_class.py`（LTX の種別とそのキャッシュ）には関係しない。

### 6.7 テストの固定値
- バックエンド:
  - `tests/test_h3_quant_class.py:48-54` の `DITS`。特に `"unquantized": "int8"`（`:53`）は、新しい規則では新しい種類になる。新しい種類（BF16 のブロック混在）の項目も足す。
  - `tests/test_comfort_budgets.py:85`（`== {"w4a8": 203092}` の完全一致）、`:256-267`（記述子の写し）、`:279-282`（不正な鍵の検査。鍵の一覧に新しい名前が入る）。
  - `tests/test_model_swap_load.py:367`（記述子の `comfort` の生の dict の完全一致）。
  - `tests/test_h3_format.py` の混在の検査（`test_bf16_blocks_and_an_int8_adaln_proj`、記録簿 §221.7）は受理だけを見ているので壊れない見込み。
- 操作パネル: 何もしなくても壊れない（§3.3）。モック・予備の設定・型を新しい値に合わせるなら、`mockBridge.test.ts:708-719` などを一緒に直す。

### 6.8 その他
- 鍵の名前（例: BF16 のブロックを含むことが分かる名前）は、`transformer_quant_class` の他の値（量子化の形式の名前）と性格が違う（形式ではなく「量子化していないブロックがある」こと）。説明文で区別が要る。
- 今の docstring・文書の「`"int8"` は w4a8 でも fp8 でもないこと」「量子化の層が無い DiT も int8」（`h3_format.py:624-626`、仕様書 `:1757`、API_REFERENCE §3.5）は書き換えが要る。

---

## 7. 変更が要るファイルの一覧と規模

### 7.1 バックエンドだけで済ませる場合（操作パネルは配置済みのまま動く）
コード（合計で数十行）:
- `h3_format.py`: `H3_QUANT_CLASSES` に 1 語、`quant_class` に分岐（本体ブロックの 4 層の印の有無を数える 5〜10 行）と docstring。
- `scripts/manifests/40-minimaxh3.json`: `comfort` に鍵と値、`_note`。
- docstring・コメントの値の列挙: `api/models_registry.py:43-49`、`config.py:342-348`、`services/pipeline_manager.py:253-259`、`mcp_server/tools/system.py:66-74`・`:109-115`。

テスト（数十行）:
- `tests/test_h3_quant_class.py`（新しい項目と `unquantized` の期待値）、`tests/test_comfort_budgets.py`、`tests/test_model_swap_load.py`。

文書:
- `Docs/COMFORT_LIMIT_TABLE.md`（改訂行・§1.1 の配信の形と留意 10・暫定値の節〔§9.5 の前例に倣う〕）
- `Videomni_Backend_Specification.md`（§0.1 の版 v0.5.124・§6.7・§6.9(c)）
- `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md`（§3.2・§3.5 の値の列挙。操作パネルのリポジトリ側の文書だがコードではない）
- `Docs/MINIMAX_H3_DESIGN.md`（新しい裁定、裁定 102 の C1・裁定 105 ⑤への更新注記）
- `Docs/VERIFICATION_LOG.md`（新しい節）、`Docs/PENDING_TASKS.md`（§1-291 の概要）
- 再起動だけで反映（マニフェストは起動時に読む）。操作パネルの再ビルド・aux2 の配布は不要。

### 7.2 操作パネルも合わせて直す場合（型の整合とモック）
- `webui/src/api/types.ts`（`TransformerQuantClass` に 1 語と説明のコメント）、`webui/src/shell/useBaseModels.ts:74-75` のコメント。
- 任意: `webui/src/bridge/mockBridge.ts:196`・`webui/src/modes/single/defaultConfig.ts:151`（新しい鍵の写し）と `bridge/mockBridge.test.ts:708-719`、`shell/comfortTable.test.ts` に新しい鍵の 1 件、`shell/useBaseModels.test.ts:423` の型。
- 規模は 10〜30 行程度だが、型検査（`npm run typecheck`）・vitest・ビルド・aux2 の埋め込み（`build.ps1`／`deploy.ps1`）・配布先 2 か所への配置と、DEVLOG の追記が要る。
