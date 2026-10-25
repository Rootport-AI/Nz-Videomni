# P1 計画のレビュー（台帳 §1-105 ①「QKV の活性化量子化の共有」）

レビュー日 2026-10-25。対象は `outputs/h3-speedup-2026-10-24/P1_plan_qkv_share.md`（全 402 行）。読んだのは計画書・R1 の ① の節・R2 の全文・`engine_h3/quant_linear.py`・`lowvram.py`・`lora.py`・`pipeline_h3.py`（`require_cuda_int8` の呼び出し・`_vram16_begin`）・`dit.py`（読むだけ）・`tests/test_h3_engine.py`（偽の `_ck`・VRAM 16GB mode と LoRA のテスト・GPU テスト）・`engine/transformer/block_swap_service.py`・comfy-kitchen 0.2.37 の `backends/cuda/__init__.py`（1588〜1621 行・2035〜2289 行と補助関数）・`__init__.py`・`registry.py`・`_activations.py`・設計正本 §5／§6.5／§6.6・VERIFICATION_LOG §176・LoRA の門の `GATES.md` と `gates/*.json`・`g2_block_equal.py` と前回の JSON・台帳 §1-105／§3-245〜3-253・仕様書の該当行・README の H3 の記述。

実行した確認は CPU だけ: `.venv-engine-h3\Scripts\python.exe -I` で comfy_kitchen を import し、版・`__version__` の有無・`_C` の関数の docstring・Python の補助関数の署名・`DTYPE_TO_CODE` を表示した（GPU の計算はしていない）。コードとファイルは本書以外に変えていない。

---

## 総評

- **実装に進んでよい。ただし下の Major 2 件を計画書に反映してから。** Critical 0・Major 2・minor 13。
- 数値等価の論法（§3）は、写す範囲と写さない分岐の両方を自分で行を追って確かめ、成り立つと判断した。`shares_input` の条件のもとで Turing の融合核・m ≤ 2 の GEMV・m＝1 の専用核・残差の融合・Turing の詰め物はどれも到達しない（下の「良いと確認できた点」）。
- Major は、(1) 前提で決まっている「comfy-kitchen の版が 0.2.37 であること」の検査が計画の本文（§2.4）に入っておらず、§11 の 4 で未決のまま残っていること、(2) 門 A-6 の事前確認が書かれた順序のままでは成り立たず、また A-5〜A-7 のどれも「共有の経路を実際に通ったこと」を直接には示さないこと、の 2 つ。どちらも計画書の数行の直しで済む。

---

## 指摘（重さ順）

### 1.【Major】版の一致（0.2.37）の検査が計画の本文に無い

- **該当箇所**: 計画書 §2.4（検査の手順 1〜5）・§9 の 1 行目・§11 の 4・§5.2（ログの文言）・§8 の設計正本 §5 の加筆文。コードは `quant_linear.py:174-211`（`require_cuda_int8`）。
- **事実**:
  - 前提（監督の決定）は「起動時に記号の存在・引数の並び・**comfy-kitchen の版が 0.2.37 であること**を検査し、外れたらログに残して現行の経路に戻す」。計画の §2.4 の手順 1〜5 に版の検査は無く、§11 の 4 で「足すかどうか」を問うている。§9 の 1 行目は「名前と並びが同じまま中の分岐が変わった場合は検査で拾えない」と書いており、これは版の検査を入れれば解消する危険である。
  - 版の取り方を確かめた（CPU だけの実行）: `importlib.metadata.version("comfy-kitchen")` と `importlib.metadata.version("comfy_kitchen")` はどちらも `0.2.37` を返す。**`comfy_kitchen.__version__` は存在しない**（`getattr(comfy_kitchen, "__version__", None)` が None。`comfy_kitchen/__init__.py` にも定義が無い）。したがって `importlib.metadata` を使うしかない。
  - 固定の正本は `engine_h3/venv-engine-h3.freeze.txt:50`（`comfy-kitchen==0.2.37`。確認済み）。
- **影響**: 前提と違う形で実装される。版を上げたときに、写した補助関数が古いまま共有の経路に乗る危険（§9 自身が挙げているもの）が残る。
- **最小の対処案**: §2.4 の手順の先頭に「0. `importlib.metadata.version("comfy-kitchen") == "0.2.37"`（`PackageNotFoundError` なら不合格）」を足す。置き場所は CUDA に触れない `_shared_quant_missing(cb)` の中にして、C1（実物で None を返す）と C2（版を変えた偽物で理由に版が出る）でそのまま固定する。§9 の 1 行目は「版が変われば共有は止まる。版を上げるときは補助関数の元の行番号と見比べ、G1 と g3 を回してから定数の版を書き換える」に直す。§11 の 4 は削除（決定済み）。§5.2 の理由の例に `"comfy-kitchen 0.2.38 != 0.2.37"` の形を足す。§8 の設計正本 §5 の加筆文の「起動時に記号と引数の並びを調べ」を「起動時に版・記号・引数の並びを調べ」に。
- **代替案**: なし（前提で決まっている）。

### 2.【Major】門 A-6 の事前確認は書かれた順序では成り立たない。A-5〜A-7 は共有の経路を通った証拠を持たない

- **該当箇所**: 計画書 §7.1 の A-6 の「台本を回す前に `shared_quant_status()` が `(True, None)` であることを、…`runpy` で g2 を呼ぶ前に表示と assert をする」。A-5・A-7 の合格条件（`qkv_quant_shared` が `true`）。
- **事実**:
  - `g2_block_equal.py` は `require_cuda_int8()` を `main()` の中（77 行）で呼ぶ。計画では検査 `_probe_shared_quant` はその末尾で走り、それまでは `_SHARED_REASON = "not probed"`（計画 §1.1）。したがって **`runpy` で g2 を呼ぶ「前」に `shared_quant_status()` を見ると `(False, "not probed")` で、assert は必ず落ちる**。g2 の末尾は `sys.exit(main())`（確認済み）なので、`runpy` の後に見るなら `SystemExit` を受ける必要がある。
  - `qkv_quant_shared` はロード時の検査の合否で、ブロックごとの判定（`shares_input`）の結果ではない（計画 §5.1 の「意味の限界」が自ら書いている）。共有の経路に乗らなくても、戻り先は現行と同じ計算なので、**A-5（指紋）・A-6（50 ブロック）・A-7（LoRA 6 本）はすべて素通りで合格する**。共有の経路を通ったことを示すのは A-8 の速度だけで、A-8 には LoRA が無い。`lora.apply(x=None)` の経路が GPU で実際に走ったことは、A-7 の結果からは直接には言えない（`shares_input` が LoRA の有無に依存しないので、A-8 からの間接の推論にはなる）。
- **影響**: A-6 の包みの台本が書かれたとおりなら落ち、回避して書き直すと「経路を通ったか」の確認が抜けたまま門が通る。
- **最小の対処案**: 包みの台本（git 管理外）の中だけで、(a) `require_cuda_int8()` を先に 1 回呼んでから `shared_quant_status() == (True, None)` を assert する（g2 がもう一度呼んでも、`disable_backend` は何度呼んでも同じで、検査もやり直すだけ）、(b) `H3Int8Linear.forward_rows_q` と `forward_rows` を数える包みを差し込み、終了後に「on の回では `forward_rows` が 0 回・`forward_rows_q` が 21 × ブロック数 × 2（sage と sdpa）回」を確かめる。A-7 の LoRA の門は selftest のワーカー経由なので数を取りにくい。A-7 の主張は「LoRA ありでも全桁一致」にとどめ、「共有の経路で LoRA が走った」の証拠は CPU テスト C5 と、(b) を足した A-6 に LoRA を付けない形で十分とするか、監督が判断する。製品のコードには計数の仕組みを足さない（計画 §5.1 の方針どおり）。
- **代替案**: (b) の代わりに、A-6 の包みで `quant_linear._SHARED_CB` を None にした回（戻り先）も 1 本回し、「共有あり」と「共有なし」の時間差（50 ブロックで約 1.45 秒 × 50/50 の比）で通ったことを見る。数える包みのほうが単純で確実。

### 3.【minor】§11 の 1〜3 は前提で決まっている。計画書の本文を確定の形に

- **該当箇所**: 結論の 6・§1.3 の「監督の確認が要る」・§5.1 の「推奨」・§2.4 の「中から呼ぶ理由…§11 の 3」・§11 の 1〜4。
- **事実**: 前提で「LoRA の行数は `xa` から取る（新しい引数を足さない）」「記録 `h3.phases.denoise.vram16.qkv_quant_shared` を足す」「検査は `require_cuda_int8` の末尾で呼ぶ」「版を検査する」が決まっている。計画は 4 件とも未決として書いている（内容は 3 件が前提と一致し、版の 1 件が指摘 1）。
- **影響**: 実装の担当が未決として扱い、監督に再び問う手間が出る。
- **最小の対処案**: §11 の 1〜4 を削り、本文の該当箇所の「確認が要る」「推奨」を確定の書き方にする。§11 の 2 の後半（仕様書の版を上げるか）だけは残す。

### 4.【minor】`_C` の引数名は確認済み。名前が無いときの退避の分岐は要らない

- **該当箇所**: §2.4 の 4（「`cublas_gemm_int8` と `dequantize_int8_linear` の引数名は未確認」「`arg0` などの場合は個数だけを比べる」）・§10 の最後の箇条。
- **事実**（CPU だけの import で docstring の 1 行目を表示）:
  - `cublas_gemm_int8(a: ndarray[dtype=int8, shape=(*, *), device='cuda'], b: ndarray[dtype=int8, shape=(*, *), device='cuda'], c: ndarray[dtype=int32, shape=(*, *), device='cuda'], workspace: ndarray[device='cuda'], stream_ptr: int) -> None`
  - `dequantize_int8_linear(input: ndarray[dtype=int32, shape=(*, *), device='cuda'], x_scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], weight_scales: ndarray[dtype=float32, device='cuda'], bias: ndarray[device='cuda'], output: ndarray[shape=(*, *), device='cuda'], output_dtype_code: int, stream_ptr: int) -> None`
  - `quantize_int8_rowwise_convrot64` と `cutlass_int8_dequant` も R1 の記述どおりの名前と並びだった。Python 側の 5 関数の `inspect.signature` も計画 §2.4 の 4 のとおり（`quantize_int8_rowwise_convrot64 (weight_2d, group_size, stochastic_rounding=0, input_act=None, input_act_weight=None, input_act_eps=0.0)`・`_convrot_fused_shared_memory_fits (x, k, group_size)`・`_prefer_cublas_int8_fallback (m, n, k, device_index)`・`_int8_weight_scale_arg (weight_scale, device)`・`_gemm_vector_arg (v, device, dtype)`）。
- **影響**: 「名前が無ければ個数だけ」の分岐は 0.2.37 では通らない死んだ枝になる。
- **最小の対処案**: `_C` の 4 関数は docstring の 1 行目を**文字列のまま**固定値と比べる（引数名を取り出す解析を書かない）。版を 0.2.37 に固定するので、1 行目全体の一致で足りる。個数だけの退避の分岐は書かない。§10 の「実装の担当が最初にすること」は不要になる。
- **代替案**: 引数名の並びだけを正規表現で取り出して比べる（型の表記の揺れに強いが、版を固定する以上その強さは要らない）。

### 5.【minor】検査の状態を毎回やり直す形にする。版の検査を CPU テストの届く関数に置く

- **該当箇所**: §1.1（`_SHARED_CB`・`_SHARED_REASON`）・§2.4。
- **事実**: ワーカーはロードのたびに `require_cuda_int8` を呼ぶ（`pipeline_h3.py:517-519`）。GPU テストも `_gpu_int8_linear` のたびに呼ぶ（`tests/test_h3_engine.py:1859-1861`）。計画は `_probe_shared_quant` が「設定する」とだけ書き、入口で既定に戻すかは書いていない。
- **影響**: 2 回目の検査が途中で不合格になったとき、前回の `_SHARED_CB` が残る書き方をすると、記録と実際が食い違う。
- **最小の対処案**: `_probe_shared_quant` の最初の行で `_SHARED_CB, _SHARED_REASON = None, "not probed"` に戻してから調べる、と §2.4 に 1 行足す。

### 6.【minor】`shares_input` の条件のうち、既存の値から常に真のものがある

- **該当箇所**: §2.2 の #3（「`x` が 2 次元で、K が `self.in_features` に等しく」）と #7（「重みが int8・スケールが f32」）。§1.1 の `quantize_input`（2 次元化をする）。
- **事実**:
  - 呼び出し側は `lowvram.block_forward` の 1 か所だけで、`h` は `torch.empty_like(n1)` から作る 2 次元の `[S, hidden]`（`lowvram.py:127-129`）。`hidden` は qkv の `in_features` と同じ（`dit.py:229`）。2 次元の条件を課すなら、`quantize_input` の 2 次元化（`reshape(-1, …)`）は通らない行になる。
  - 重みと scale の dtype は `H3Int8Linear` の作りで int8／f32 に決まっている（`quant_linear.py:84-92`）。
  - 一方、**デバイスの一致（#7 の前半）は意味がある**。共有の経路は登録の制約（`cb:4464-4486`。`registry.get_implementation` を通らない）を経ないので、重みが GPU に無い呼び出しを戻り先へ回して `NoCapableBackendError` で止める、という計画 §2.2 の理由は正しい。
- **影響**: 無害だが、条件の表と CPU テスト C3 の行が増える。
- **最小の対処案**: #3 は「K % 256 == 0 かつ 256 ≤ K ≤ `_CONVROT_FUSED_MAX_K`」だけ、#7 は「重みとスケールが `x` と同じデバイス」だけにする。2 次元化は `quantize_input` に残す（`forward_rows` と同じ 4 行を書くという計画の方針どおり）。
- **代替案**: そのまま残す（害は無い）。

### 7.【minor】`forward_rows_q` の「`lora_xa` が None なら RuntimeError」は要らない

- **該当箇所**: §1.1 の `forward_rows_q` の行。
- **事実**: LoRA が付いていて `lora_xa` が None なら、変更後の `lora.apply` は `x.reshape(...)`（`x` は None）で `AttributeError` を投げて止まる（`lora.py:295` の現行の行を計画 §1.3 のとおり `xa is None` の側に残すため）。`lora.py` の方針は「`apply` に except は無い。失敗はジョブの失敗」（`lora.py:36`）。
- **影響**: 分岐が 1 つ増えるだけ。
- **最小の対処案**: 書かない。配線の誤りは `AttributeError` で十分に止まる。
- **代替案**: 文言を分かりやすくしたいなら残す（2 行）。

### 8.【minor】CPU テスト C4／C5 の偽物の作り方

- **該当箇所**: §6.1 の C4・C5。
- **事実**:
  - C4 は `shares_input`・`quantize_input`・`_int8_gemm_prequantized` の 3 つを差し替える。本物の `quantize_input`（整形と `_SHARED_CB.quantize_int8_rowwise_convrot64` の呼び出し）は CPU テストで一度も走らない。`forward_rows_q` の本物（切り方・bias・`lora.apply(x=None)`）は走るので、配線の確認としては足りる。
  - 偽の行列積を「`(xq * xs) @ (w * ws)ᵀ` を float64 で」と別の式で書くと、off 側の偽物 `_int8_linear_as_float64`（`tests/test_h3_engine.py:1551-1559`。`F.linear` を float64 で）と同じ算術である保証が式の上で読みにくい。
  - weakref で `h` の解放を見る形は CPU で成り立つ（同じ作法の先例が `tests/test_h3_engine.py:1689-1707`）。ただし偽の `quantize_input`・`shares_input` が引数を保持すると（`unittest.mock` の呼び出しの記録など）、`h` が解放されずテストが落ちる。
- **影響**: テストの意図は満たすが、読みにくさと書き損じの余地がある。
- **最小の対処案**: (a) 偽の行列積は `_int8_linear_as_float64(xq, weight, weight_scale, bias, out_dtype=out_dtype, convrot=False, convrot_groupsize=256)` をそのまま呼ぶ（偽の量子化が返す `xq` は float64 の `h`、`xs` は使わない）。(b) `quantize_input` は差し替えずに、`_SHARED_CB` に `quantize_int8_rowwise_convrot64=lambda x2d, g: (x2d.to(torch.float64), torch.ones(...))` を持つ `SimpleNamespace` を入れる。これで本物の `quantize_input` も CPU で走り、差し替えは `shares_input` と行列積の 2 つに減る。(c) 偽物は引数を保持しない、と §6.1 に 1 行書く。

### 9.【minor】GPU テストのあとに `_SHARED_CB` が残る。C2・C6 の作り

- **該当箇所**: §6.1 の C2・C6・「モジュールの状態は `monkeypatch.setattr` で差し替え」。
- **事実**:
  - `H3_GPU_TESTS=1` で全件を回すと（門 A-4）、GPU テスト（`tests/test_h3_engine.py:787-789`・`1858-1861`）が `require_cuda_int8()` を呼び、変更後は検査が `_SHARED_CB` をモジュールの状態として設定する。後に並ぶ CPU テスト（2015 行以降など）は `_SHARED_CB` が設定された状態で走る。CPU の小さなモデルは `convrot=False`（`_int8_linear_as_float64` が `assert not convrot`）で `shares_input` が偽になるので、既存のテストは落ちない。
  - C6 は「`_SHARED_CB` が None（既定）のとき」を前提にしているが、A-4 の回し方では既定ではない。
  - C2 の「`_probe_shared_quant` は例外を投げず `_SHARED_CB` が None」を偽の `cb` で確かめる方法が書かれていない。計画の検査は自分で `comfy_kitchen.backends.cuda` を import する（§2.4 の 1）ので、偽の `cb` を渡す口が無い。
- **影響**: C6 が A-4 の回し方で意図した分岐を見ないまま合格する。C2 は実装の担当が口（引数）を足しかねない。
- **最小の対処案**: C6 は `monkeypatch.setattr(quant_linear, "_SHARED_CB", None)` を明示する。C2 の検査の部分は、`quant_linear._shared_quant_missing` を「理由の文字列を返す偽物」に差し替えてから `_probe_shared_quant()` を呼び、`shared_quant_status() == (False, 理由)` を見る（実物の `cb` の import は CPU で通ることを確認済み。手順 5 のデバイスの確認には届かない）。`_probe_shared_quant` に引数は足さない。

### 10.【minor】G2 は G1 に含まれる

- **該当箇所**: §6.2 の G2。
- **事実**: G1 は「共有した量子化で 21 切片を結合したもの＝`lin(x)`（中で別に量子化する）」を `torch.equal` で見る。量子化が非決定的なら G1 は落ちるので、G1 の合格は同じ入力に対する決定性を含む。
- **影響**: テストが 1 本多いだけ。
- **最小の対処案**: G2 を書かない。
- **代替案**: 「行列積を通さない確認」として残す（費用は小さい）。

### 11.【minor】門の重複と、門の中の矛盾

- **該当箇所**: §7.1 の A-5 と A-7・§11 の 6。
- **事実**:
  - A-5 の on（`outputs/h3-vram16-gates/g3.sh` の on）と A-7 の g2（`Docs/Outputs-archive/h3-lora-2026-10-24/gates/gates.sh` の `g2) base g2_nolora_on --vram16`）は、同じ selftest・同じ 4 本の重み・同じ G5 のプロンプト・同じ既定で、違いは出力先だけ（両台本を読んで確認）。前回の sha256 も同じ `0da6111c…`（`gates/g2_nolora_on.json`）。
  - A-5 は off も回すが、A-7 は「off の側は経路を変えていないので回さない」としている。off の経路（`dit.py`・`forward`）は変わらず、変わるのは `require_cuda_int8` の末尾に検査が足されることだけで、off の計算には効かない。
- **影響**: GPU の時間が約 5 分（off と重複の on）余分にかかる。判断の基準が門の間でそろっていない。
- **最小の対処案**: A-5 を A-7 の g2 に統合し（合格条件は同じ `0da6111c…` の全桁一致）、off は回さない。off も回すなら、A-7 の off（§11 の 6）と同じ扱いにそろえる。

### 12.【minor】A-8 の「前」の作り方と取り直しの文言

- **該当箇所**: §7.1 の A-8・§11 の 5。
- **事実**: 「前」を `shares_input` を常に偽にして作ると、検査は合格のままなので、「前」の回の記録 `qkv_quant_shared` は `true`、ロードのログも「shared」になる。また「外したせいで前後それぞれ 2 ステップ未満になったら、その回を取り直す」は、1 回あたり 2 ステップ（2・3 ステップ目）なので、「各回」なのか「前後の合計」なのかが読み分けにくい。
- **影響**: 「前」の回の run.json を後で読むと、共有したように見える。
- **最小の対処案**: 計測台の中の差し替えを `quant_linear._probe_shared_quant` を「`_SHARED_CB=None`・理由 `"disabled by --no-qkv-share"`」にする偽物に替える（ロードの前に差し込む）。記録とログが「前」の実態どおりになる。取り直しの条件は「前と後のそれぞれで、負荷の無い 2・3 ステップ目が合計 2 個未満なら、その側の回を足す」のように書き分ける。
- **補足（妥当性）**: 合格の線「1.0 秒以上速い」は、上限 1,447 ms（R2 §3 (a)）に対し約 0.45 秒の余裕がある。負荷の無い回どうしのばらつきは 0.01〜0.10 秒（R2 §2.1 の E／N3／A-G5）で、線は十分に区別できる。CPU が GPU の先を走っている（R2 §2.3 の起動の呼び出しの待ち 15.7 秒）ので、GPU の時間の短縮はそのまま壁時計に出ると見てよい。`g2_block_equal.py` の `--report` の注意（A-6・§9 の最後の行）は計画に入っている。

### 13.【minor】文書の更新一覧の抜け

- **該当箇所**: §8。
- **事実と対処案**（どれも 1 行程度）:
  - 設計正本 §6.5「上限と OOM」の「**記録**」の箇条（`done` の `h3.phases.denoise.vram16` に入る項目の列挙）に `qkv_quant_shared` が無い。仕様書 1323 行を直すなら、正本のこちらも直す（足すのは名前と意味の 1 句）。
  - 設計正本 §6.5「数値の約束」の「確かめた範囲」は §169.3 だけを指す。共有の経路の結果の参照として §177 を足す。
  - `quant_linear.py` の冒頭の docstring の「Fail loud, never slow」の段（16〜23 行）に、共有の経路は登録を通らず、デバイスの一致で戻り先に回して fail loud を保つ、の 1 文を足す（計画 §1.1 の追加の段落と同じ場所でまとめてもよい）。
  - 「1 ステップ約 22 秒」の記述が 2 か所ある（設計正本 §6.5 の結論の段落、README 1003 行の「I2V 896×1152・124 フレーム・20 ステップで、1 ステップ約 22 秒、全体で約 9 分」）。約 1.4 秒の短縮で約 20.2 秒になる見込み。直すか据え置くかを §8 に書いておく（README を直すなら記憶の「README 編集前に必読」の規則に従う。1003 行は凍結の範囲〔1〜131 行〕の外）。
  - 台帳 §3-245（ヘッド群の拡大の再評価）: R1 は「① を入れるとヘッド群の拡大の残りの得が小さくなる（推測）」と書いている（R1 ① の「見積もり」の最後の箇条）。§3-245 の判断材料に §177 への参照を 1 つ足すかを決めておく（台帳に歴史は書かず、参照だけ）。
  - フロントエンドの DEVLOG・MCP・Gradio は、今回の変更が届かないので不要（確認: 計画 §5.1 のとおり、アプリ側に新しい欄を読むコードは無い）。

### 14.【minor】細かな記述の誤り

- §6.2 の G1 のメモリの見積もり（「入力は約 427 MB、重みは約 115 MB」）は出力を数えていない。m＝39,764・N＝21,504 の `lin(x)` の出力は bf16 で約 1.71 GB、21 切片の結合でもう約 1.71 GB。合計で約 4 GB で、16 GB には収まる。見積もりの文だけ直す。
- §7.1 の A-1 の件数の基準（§176.3 の 215 件合格・14 件スキップ）は、変更の前に同じ作業ツリーで一度回して取り直すほうが確実（§176 の後のコミットで件数が動いていても、比べる相手がずれない）。

---

## 未確認の懸念（根拠が足りないもの）

- **CUTLASS の核の選び方がポインタの揃え方に依存するか**: 核のソースは `.pyd` だけで読めない。現行の `x_qdata` も共有の `xq` も PyTorch の割り当て器の新しい確保で、同じ揃え方の種類になるはず（割り当て器は 512 バイト単位）。重みの切片の先頭は n0 × 5,376 バイトずれ、1,024 × 5,376 は 16 の倍数。G1 と A-6 で確かめられる。
- **`importlib.metadata` が import 中の comfy_kitchen と別の配布を指す可能性**: 同じ仮想環境に 2 つの配布がある場合だけ。今回の仮想環境では起きない見込み。対処は要らない（足すなら `distribution(...).locate_file` と `comfy_kitchen.__file__` の比較だが、過剰）。
- **複数 GPU**: §2.4 の手順 5 は `torch.cuda.current_device()` を見る。パイプラインの `dev` が別の番号のときに記録の値がずれうる。各呼び出しの #6 はテンソルのデバイスで見るので、計算の正しさには影響しない。製品は 1 GPU 前提と見られる（未確認）。
- **CPU テスト C1 で comfy_kitchen を import すること**: `_ck` の docstring（`quant_linear.py:45-50`）は「CPU テストでは import しない（拡張の探索が雑音になる）」としている。開発機では import は通った。CUDA の拡張が無い機械でエンジンのテストを回す運用があるなら、C1 に `pytest.importorskip` 相当が要る（運用の有無は未確認）。
- **割り当てのピーク**: FFN の区間と注意の区間のどちらが実際の山かは測られていない（計画 §4.2 のとおり）。A-9 の「上回らない」で受けるのは妥当。

---

## 計画で良いと確認できた点

- **写す範囲（2190〜2289 行）と写さない分岐の到達不能**（自分で行を追った。`cb`＝`.venv-engine-h3/Lib/site-packages/comfy_kitchen/backends/cuda/__init__.py`）:
  - m＝1 の専用核（2091〜2126 行）は `m == 1` が条件。`shares_input` の m > 2 で届かない。
  - m ≤ 2 の GEMV（2169〜2188 行）は `(m == 1 and k % 4 == 0) or (m == 2 and k % 16 == 0)`。m > 2 で届かない。
  - Turing の融合核（2195〜2209 行）は `_cuda_device_should_use_turing_kernels`（`cb:310-313`）が `_cuda_device_is_turing`＝compute capability が **ちょうど (7, 5)**（`cb:289-295`）を要る。`shares_input` の #6 `_cuda_device_supports_cutlass_int8_dequant`（major ≥ 8。`cb:321-331`）で届かない。
  - Turing の詰め物（2258〜2263 行）も `_cuda_device_is_turing` が条件。届かず、`padded_n = n` の枝だけになる。
  - 残差の融合（2224〜2244 行）は `residual is not None` のときだけ `resid_2d` が作られる。共有の経路は残差を渡さないので届かない。`_finish` は `residual` が None なら `out` をそのまま返す（`_activations.py:70-79` の `apply_residual`）。入力は 2 次元なので形も変わらない。
  - ConvRot の別の量子化（2150〜2156 行）は `_fused_convrot_ok` が偽のときだけ。`shares_input` の #2〜#4 が `_fused_convrot_ok` の 5 条件（2061〜2067 行）と同じなので届かない。
- **公開の `comfy_kitchen.int8_linear` は引数を変えずに CUDA 実装を呼ぶだけ**（`comfy_kitchen/__init__.py:997-1050`。`out_dtype` が None なら bf16 にするだけで、H3 は bf16 を明示している）。`registry.get_implementation` は包みを付けずにモジュールの関数そのものを返す（`registry.py:256-270`）。CUDA 実装の `int8_linear` を後から包み直す行も無い（`cb` を検索して 0 件）。したがって現行の経路の中身は `cb.int8_linear` の本文そのもので、写しの比較の相手として正しい。
- **量子化の引数の 1 対 1 の対応**（計画 §3.1 の 3 の表）: 包み（`cb:1588-1621`）と `int8_linear` の中（2131〜2149 行）を突き合わせ、`stochastic`（False）・`act_code`（`_input_act_code(None)`）・`seed`（0）・`act_weight`（`_empty_cuda_tensor(device, x.dtype)`。`cb:2025-2031`）・`act_eps`（0.0）・ストリーム（入力のデバイスの現在のストリーム）・出力の形（`_input_act_width(None)`＝1。`_activations.py:39-41`）が一致した。
- **行列積の側の引数**: `_int8_weight_scale_arg`（`cb:528-531`）は「同じデバイス・f32・連続」ならそのまま返す。`[N, 1]` の連続なテンソルの行の切片 `[1024, 1]` は連続なので、現行と同じテンソルが渡る。`ws_cutlass` は `numel == n` で同じテンソル。`_prefer_cublas_int8_fallback` は 0.2.37 では常に False（`cb:534-540`）。`_DISABLE_CUTLASS_INT8` はモジュールの変数で、`int8_linear` も呼ぶたびに読む（`cb:1635`）。G1 の `monkeypatch.setattr(cb, "_DISABLE_CUTLASS_INT8", True)` は両方に効く。
- **数値等価の論法を補強する事実**: int8 × int8 の行列積は int32 で足し合わせる（K＝5,376 では最大でも 127² × 5,376 ≈ 8.7 × 10⁷ で int32 に収まる）ので、タイルの切り方や N の違いで和の順序が変わっても結果は変わらない。浮動小数の演算は後処理（スケールを掛けて bf16 に丸める）だけで、要素ごと。したがって一致に効くのは「CUTLASS か cuBLAS＋逆量子化か」という経路の選び方だけで、補助関数はそれを `int8_linear` と同じ関数・同じ値で決める。
- **`h` を捨てる時点と LoRA**: 計画 §1.2 の形では `xa_for(qkv, h)` が `quantize_input(h)` より前にあり、`h` の最後の使用の後で `del h` になる。`block_forward` の中の入れ子の関数（129・159・165・167・172 行のラムダ）はどれも `h` を参照しないので、`h` はセルの変数ではなく、`del h` で参照が切れる。153 行より後に `h` を参照する行は無い（`h2` は別の名前）。`lora.apply` は `xa` があると `x2d` を使わない（`lora.py:302-309`。使うのは行数 `x2d.shape[0]` の 299・300・304 行だけ）。`xa_for` は LoRA の付いたモジュールでは spec ごとに `[M, r]` の f32 を返す非空のリスト（`lora.py:265-280`）なので、`xa[0].shape[0]` は行数として正しい。Style・Motion・Turbo・複数本の重ねがけのどれでも `xa` は spec の数だけの要素を持ち、算術は変わらない。
- **block swap との整合**: 重みの差し替えはブロックの forward の入口だけで起きる（同期の経路は `blk.to(device)`・`p.data = p.data.to(device)`。`block_swap_service.py:239-289`。先読みの経路は `engine.on_block_forward(idx)`。同 355-361 行）。ブロックの途中では差し替わらない。`xq`・`xs` は活性化で、先読みの受け皿（重みの区画）とは別の通常の確保なので、受け皿の輪とは関わらない。`forward_rows_q` が毎回 `self.weight[n0:n1]` から切る方針は `forward_rows` と同じで正しい。
- **検査の位置**: `require_cuda_int8` の末尾では `_ck()` が comfy_kitchen を import 済みで、CUDA の拡張 `_C` も読み込み済み（`list_backends()["cuda"]["available"]` の確認が先にある）。`g2_block_equal.py`（77 行）・GPU テスト・ワーカーのロードがどれも `require_cuda_int8` を通るので、台本を変えずに共有の経路が門にかかる、という計画の理由は正しい。
- **記録の追加の影響**: `_vram16_begin` は staticmethod で、記録の辞書に純粋な Python の値を 1 つ足すだけなら、`torch.cuda` の呼び出しの順を固定したテスト（`tests/test_h3_engine.py:1936-1942`）に影響しない。鍵の一覧を固定したテストは無い（計画 §5.1 のとおり）。
- **門の数値**: A-6 の sage の差の値（ブロック 22 の `max_abs` 256.0・`elements_differ` 115,388,441・`rel_out` 0.00659228097488087、ブロック 40 の 1792.0・6,974,901・0.0030670819750614436）は `outputs/h3-vram16-gates/g2_block_equal.json` と一致。A-7 の 6 本の sha256 の先頭 16 桁は `Docs/Outputs-archive/h3-lora-2026-10-24/gates/*.json`（`g2_nolora_on`・`g4_turbo_640_on`・`g5_style_on`・`g5_motion_on`・`g5_mixed_on`・`g5_style_on_noprefetch`）と一致。A-8 の md5 `eb2a53ae…` は R2 §2.1 の 5 回の値。
- **計画書の行番号**: `quant_linear.py`（41・123〜164・166・174〜211 行）・`lowvram.py`（137〜153 行）・`lora.py`（283〜316・300・304・306 行）・`pipeline_h3.py`（517〜519・861〜864 行）・`cb` の各行（2050〜2289 行の表、`cb:321-331`・`455`・`519`・`528`・`534-540`・`1588-1595`・`1631`・`1635`・`4464-4486`）は、すべて今の作業ツリーと一致した。
- **台帳の番号**: CLOSED の最新は 3-251、台帳 §3 に 3-252・3-253 が生きているので、次の CLOSED は 3-254（`PENDING_TASKS_CLOSED.md:3093`・`PENDING_TASKS.md:134,141`）。VERIFICATION_LOG の次は §177（§176.9）。計画の記述どおり。
