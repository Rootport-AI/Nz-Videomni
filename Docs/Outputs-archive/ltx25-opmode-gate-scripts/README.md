> **このファイルは実機の `outputs/ltx25-opmode-gate/scripts/README.md`（git追跡外）のスナップショットである（2026-09-02複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# §3-124 実機ゲート G-B の常設資産

**結論から**: このフォルダは「画角拡張ジョブの直後に通常生成が落ちる」不具合（§3-124）の
修正を、実機で確かめるための道具一式です。**中身を実行するには実GPUが要ります。**
2026-08-30 の作成時点では**1本も実行していません**（CPUだけで済む健全性確認、
`op_gate.py check` と `canvas_precheck.py` は実行済み・全一致）。

- `op_gate.py` — 連続投入ドライバ（G-B(1) 回帰証明。3腕）
- `canvas_precheck.py` — キャンバスの先行照合（G-B(2) の第1段。CPUのみ）
- `SHA_ARMS.md` — O1／O5／O8 のバイト同一性照合の手順書（G-B(2)）

---

## 1. 何を証明する道具か

修正前は、画角拡張（アウトペインティング）のジョブだけが `torch.inference_mode()` で
走っていました。そのジョブ中に**ページ固定ステージングプール**（`PinnedStagingPool`）が
新しく確保したバッファには「推論専用」の印が残り、**同じ常駐ワーカーで次に走った通常生成**が
そのバッファへ書き戻そうとして
`RuntimeError: Inplace update to inference tensor outside InferenceMode`
（`engine/transformer/block_swap_prefetch.py:574`）で落ちました。
失敗の一次記録は `outputs/ltx25-nagvsf-gate/manifest.jsonl` の **15〜16行目**です。

修正（§3-124）は引き算——デコレータ1行の削除で、engine25 は全経路が
`torch.no_grad()` になりました。`op_gate.py` は**壊れていた並び順そのもの**を
実際に投げ直して、後続ジョブが完走することを確かめます。

## 2. 腕が「無効」になりうるという罠

「推論専用」の印が付くのは、プールが**成長して確保し直した瞬間だけ**です
（`block_swap_prefetch.py:259-269`）。成長させているのは画角拡張ジョブが付ける
**IC-LoRA** で、最大ブロックが 207.9 MB から 233.9 MB へ太ります。
つまり**プールが成長しなかった腕は、修正前でも汚染されなかった**ので、
緑（合格）でも何も証明しません。

そのため各腕の合格条件は3つあります。

| 条件 | 内容 | 外れたときの意味 |
|---|---|---|
| (a) | 画角拡張ジョブのワーカーログに `pinned staging pool = 2 x 233.9 MB` が出る | **腕は無効**。合格ではなく作り直し |
| (b) | 画角拡張のあとのジョブが例外なく完走する（`completed` かつ `error` なし） | 不合格。ロールバック判断へ |
| (c) | 全ジョブの `metadata.json` の `block_swap_prefetch_used` が `"on"` | **腕は無効**。`"on->off"` に縮退した腕は :574 を通らないので空振り |

プールは grow-only で、BlockSwapService（＝ワーカープロセス）と寿命を共にします。
**成長ログは1プロセスにつき1回しか出ません。**
そこで `op_gate.py` は**腕ごとにサーバーを立て直します**（既定動作）。
`--no-restart` は調査用で、ゲート本番では使いません。

## 3. 3つの腕

| 腕 | 並び | 狙い |
|---|---|---|
| `m1` | 通常 → 画角拡張 → 通常 | 落ちていた形そのもの |
| `m2` | 通常 → 画角拡張 → 連結（chain） | 別のドライバから同じプールへ書き戻す経路 |
| `m3` | 通常 → 画角拡張 → 画角拡張 → 通常 | 画角拡張自身も自分の残り物を踏めること |

**どの腕も先頭が通常ジョブ**です。これがプールを 207.9 MB の基準値で確保し、
次の画角拡張ジョブが**成長させる側**になるからです。
（不具合は「画角拡張が最初のジョブのときだけ」ではありません。実際に見つけた a7 の腕の前には
通常ジョブが何本も走っています。）

通常ジョブと連結ジョブには**凍結SHAの照合**も載せてあります
（通常 512×320／121フレーム＝`65c9eb53…`、連結 2×73フレーム＝`42f82ae2…`）。
落ちないことに加えて**バイトまで従来どおり**であることが、ついでに取れます。

## 4. 依存チェーン（消してはいけないもの）

```
outputs/ltx25-chain-g5/g5.py            機構の出どころ。一時サーバー（ポート18699）・
  │                                      HTTP・ログ追尾・manifest 追記の実装
  ├─ outputs/ltx25-nagvsf-gate/nv2.py         前例1: g5 のパスを自分の木へ差し替える作法
  ├─ outputs/ltx25-nagvsf-gate/a7_outpaint.py 前例2: 画角拡張ジョブのHTTP投入と動画アップロード
  └─ outputs/ltx25-opmode-gate/scripts/op_gate.py  ← 本ドライバ（g5 を import する）

outputs/ltx25-outpaint-gate/evidence/_material/src_audio_full.mp4   素材（リポジトリ内）
outputs/ltx25-outpaint-gate/scripts/g2_run.py                       SHA腕のドライバ
models/LTX23/IC-LoRA/in-outpainting/…-0.9.safetensors               画角拡張のIC-LoRA
```

**`g5.py` を消すと本ドライバは動きません。** 前例2本（nv2.py・a7_outpaint.py）は
読み物としての依存で、import はしていません。

**素材はリポジトリ内の `evidence/_material/` を指しています。**
`S:\OriginalApps\12_Nz-LTX23-AviUtl2\_outpaint_g2_tmp\` の同名ファイルとは
バイト同一（2026-08-30 照合）ですが、あちらは画角拡張ゲートの RESULTS.md で
「Disposable（消してよい）」と宣言済みなので、**常設資産の参照先にはしません**。

## 5. 実行手順（実GPU）

前提: **オーナーへ事前報告し、画像生成が止まっていることを確認してから**。
GPUジョブは厳格に直列。ポート18699 に**他のゲートのサーバーが残っていないこと**を
先に確かめてください（`op_gate.py stop` は 18699 の一時サーバーを止めます）。

```
py = S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/.venv/Scripts/python.exe
cd  S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/ltx25-opmode-gate/scripts

$py op_gate.py check              # CPUのみ。素材と証拠コピーのSHA照合
$py op_gate.py envsnap before     # CPUのみ。オーナー環境のスナップショット
$py op_gate.py run --arms m1,m2,m3
$py op_gate.py stop
$py op_gate.py envsnap after      # before と全項目一致であること
```

- 一時サーバーは**ポート18699**、設定・状態・ログ・出力・アップロードをすべて
  `outputs/ltx25-opmode-gate/_rt/` へ隔離します。オーナーの `config.yaml` /
  `state.json` / `logs/` / `outputs/` / `uploads/` には触れません。
- `envsnap before` は**このフォルダを作ったあとに**取ってください。
  スナップショットは `outputs/` の要素数を数えるので、途中でフォルダが増えると
  before と after が食い違います。
- 記録は追記専用の `manifest.jsonl`（ジョブ1本＝1行＋腕の判定＝1行）、
  ジョブごとの詳細は `runs/<腕>_<ジョブ>/`（`run.json` / `request.json` /
  `metadata.json` / `worker_tail.csv` / `output.mp4`）、
  腕の判定は `runs/<腕>_verdict.json` にも書きます。
- 途中で止めても再開できます（`arm_pass` が真の腕は飛ばします。`--force` で再走）。

## 6. 判定の読み方

`runs/<腕>_verdict.json` の `verdict` が

- `PASS` — (a)(b)(c) すべて満たし、クラッシュの署名も出ていない
- `FAIL` — 条件(b)が崩れた。**そこで停止**し、オーナーへ報告（ロールバック判断）
- `INVALID` — (a) か (c) が崩れた。**合格として読まないこと。** 腕を作り直す

`crash_signature_seen` が真なら、`Inplace update to inference tensor outside
InferenceMode` がログかジョブのエラーに出たということです。修正が効いていれば
1回も出ません。

## 7. バイト同一性（無害証明）は別紙

`SHA_ARMS.md` を参照してください。O1・O5・O8 の3本を回し、
`outputs/ltx25-outpaint-gate/evidence/` の恒久コピーとバイト比較します。
