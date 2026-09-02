"""単発生成＋チェーン生成ツール（``submit_generate`` / ``submit_chain``、2本）。

``api/models.py`` の ``GenerateRequest`` の表面をほぼそのまま公開するが、隠し
フィールド（``pipeline`` / ``num_inference_steps`` / ``guidance_scale`` /
``crf``）は計画D8により出さない -- ``two_stage_hq`` は現状モックのみで、
distilledパイプラインの固定値（8ステップ・CFG=1.0）を変える意味がないため
（``two_stage_hq`` 系の非公開理由はそのまま存置する）。一方でAcceleration機能は
**6項目すべて**を公開する --
``attention_backend``（既定 ``"sdpa"``、``"sage"`` も選べる）と
``block_swap_prefetch``（既定on。backend §44、実装は先読み block swap。
offにすると従来の同期スワップになる。S4, 2026-08-01: 実機ゲートG1〜G7全PASS
を条件にオーナーが確定した既定反転）、``keep_resident``（既定off。ジョブ間の
CPU骨格キャッシュ。LTX 2.5 では常駐するのがテキストエンコーダだけで、
同じ名前でも中身が違う。§76）、``keep_resident_embeddings``（既定off。**LTX 2.5
専用**で、埋め込み処理器のジョブ間常駐。台帳 §3-114。ここまでの5項目と違い、
LTX 2.3 を選んでいるときに既定以外にすると422になる向きである）、
``fused_gguf_dequant_kernel``（既定on。GGUF逆量子化の
Triton 1カーネル化。出力はビット単位で不変。§51, 2026-08-04: 実機ゲート
G1〜G8全PASSを条件にオーナーが確定した既定反転）、``vae_mode``（既定
``"default"``、``"prune_vaed"`` で枝刈り版デコーダ。Docs/PENDING_TASKS_CLOSED.md
§3-66（起票当時は§3-50）、2026-08-05 にモックから実機能へ転換）を公開する。
``vae_mode`` はかつて「現状モック（受理のみで効果が無い）なので出さない」
（計画D1）として除外していたが、2026-08-05 のオーナー裁定で公開へ転じた
（実装と実機ゲートG1〜G7の合格を待ってからの最終ステップ。
``PRUNAVAED_WORKORDER.md`` §0-8・§6.3）。

ネストオブジェクトを取る2機能は 2026-09-01 に公開した（``Docs/PENDING_TASKS_CLOSED.md``
§3-115 / §3-122、2026-09-01にクローズ済み。
**ツール本数は22本のまま不変**で、引数が増えただけ）: ``submit_chain`` の
撮り直し（``retake_*`` 5引数 → ``retake`` ネスト）と、``submit_generate`` の
画角拡張（``outpaint_*`` 6引数 → ``outpaint`` ネスト）。どちらもAPI側
（``RetakeSpec`` / ``OutpaintSpec``）は前から実装済みで、MCPは引数を素通し
するだけである。

送信ボディは「Noneまたは空は送らない」を徹底する（計画のペイロード契約）。
``crop_width`` / ``crop_height`` は両方指定 or 両方省略のみを許す（片側だけの
指定はサーバーへ投げる前にここで弾く）。NAG系フィールドは ``nag_enabled=False``
のとき丸ごと省略する（サーバーの ``negative_prompt`` 必須チェックは
``nag_enabled=True`` のときだけ働くため、無効時に既定値を送っても実害は無い
が、パネルの「素の生成」との差分を最小化する意味でも送らない）。
"""

from __future__ import annotations

import math
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from api.models import (
    BLOCK_SWAP_PREFETCH_DEFAULT,
    FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    KEEP_RESIDENT_DEFAULT,
    KEEP_RESIDENT_EMBEDDINGS_DEFAULT,
)
from mcp_server.client import get_client
from mcp_server.params import ChainClipArg, ConditioningImageArg, LoraArg

_SUBMIT_TIMEOUT = 30.0
_CHAIN_SUBMIT_TIMEOUT = 30.0

# 画角拡張（outpainting, ``Docs/PENDING_TASKS_CLOSED.md`` §3-70（起票当時は
# §1-13）/ ``Docs/PENDING_TASKS_CLOSED.md`` §3-122、
# 2026-09-01にクローズ済み）の既定値。正本は
# ``api/models.py`` の ``OutpaintSpec``（stage1=5 / stage2=2）と、操作パネル側の
# ``webui/src/modes/edit/outpaintGeometry.ts``（``DEFAULT_BLEND_DILATION_STAGE1``）。
_OUTPAINT_BLEND_DILATION_STAGE1_DEFAULT = 5
# 画角拡張が必ず1本だけ要求する control 系アダプタ名。操作パネルの
# ``controlLoras.ts`` の ``OUTPAINT_LORA_NAME`` と同一。
_OUTPAINT_LORA_NAME = "in-outpainting"
_OUTPAINT_LORA_STRENGTH = 1.0

# 撮り直し（Retake, ``Docs/PENDING_TASKS_CLOSED.md`` §3-115、2026-09-01に
# クローズ済み）の補助引数の既定値。正本は ``api/models.py`` の
# ``RetakeSpec``（head_px=25 / tail_px=24 / regenerate_audio=True）。
_RETAKE_HEAD_PX_DEFAULT = 25
_RETAKE_TAIL_PX_DEFAULT = 24
_RETAKE_REGENERATE_AUDIO_DEFAULT = True


def _outpaint_stage2_from_stage1(r: int) -> int:
    """stage 1 の膨張段数から stage 2 の段数を導く（公式ワークフローの 5:2）。

    操作パネル側の正本 ``webui/src/modes/edit/outpaintGeometry.ts`` の
    ``stage2FromStage1`` の移植: ``r <= 0`` だけ特例で 0（「膨張なし」を選んだ
    のに stage 2 だけ膨らむのを防ぐ）、それ以外は ``max(1, round(r * 2 / 5))``。

    Pythonの ``round`` は偶数丸め（banker's rounding）だがJavaScriptの
    ``Math.round`` は 0.5 切り上げ -- ここでは差が出ない。``r`` は整数なので
    ``r * 2 / 5`` の小数部は 0 / 0.2 / 0.4 / 0.6 / 0.8 のいずれかにしかならず、
    ちょうど .5 になる ``r`` が存在しないためである。
    """
    if r <= 0:
        return 0
    return max(1, round(r * 2 / 5))


def _snap_frame_rate(value: float) -> float:
    """フレームレートを整数へ丸める（サーバーが受理する範囲内のときだけ）。

    ``1.0 <= value <= 60.0`` かつ有限のときだけ ``floor(value + 0.5)`` で丸めて
    ``float`` で返し、それ以外（範囲外・NaN・±inf）は**素通し**する。

    **なぜ整数か（台帳 §3-71 / §3-72）.** 非整数fpsが生成リクエストに乗ると、
    ①チェーン生成のstage-2音声タイル再組立検算が丸め誤差で落ち、ごく普通の
    クリップ構成が422になる（§3-71。29.97 の ``[257, 257]`` が代表例。24や30
    なら通る）、②mp4書き出しでfpsが整数へ切り捨てられ、長尺ほど音声が
    ずれていく（§3-72）。サーバー側の恒久修正は凍結領域に広く及ぶため、
    **クライアントの全入口で整数へスナップして**両方の再現経路を塞ぐ。

    **なぜ範囲外を素通しするか.** 範囲の正本はサーバーの
    ``Field(ge=1.0, le=60.0)``（``api/models.py``）であり、範囲外は422で
    弾かれるべき入力である。ここで黙って既定値（24）などに差し替えると、
    利用者が誤った引数を渡したことに気づけないままGPUジョブが走ってしまう。
    整数化はUI方針だが、範囲はAPIの制約——という役割の違いをそのまま
    実装に写している。

    **なぜ ``round()`` を使わないか.** Pythonの ``round()`` は偶数丸め
    （banker's rounding）なので ``round(0.5) == 0`` / ``round(2.5) == 2`` と
    なり、JavaScriptの ``Math.round``（常に0.5切り上げ）と食い違う。
    ``math.floor(value + 0.5)`` なら ``Math.round`` と同じ結果になる。

    **3実装の関係（写経パリティテストを書く前に読むこと）.** 丸め規則
    ``floor(x + 0.5)`` 相当は3つの実装で共通である:
    ``webui/src/modes/single/paramUtils.ts`` の ``snapFrameRate``（操作パネル）、
    本関数（MCP）、``gradio_ui/handlers.py`` の ``_snap_frame_rate``（Gradio）。
    ただし**範囲外の扱いは正反対**で、webuiは[1, 60]へクランプするのに対し、
    MCPとGradioは素通ししてサーバーの422へ委譲する。「3つとも同じ関数」と
    思い込んだパリティテストを書くと必ず落ちるので注意。
    """
    if not math.isfinite(value) or value < 1.0 or value > 60.0:
        return value
    return float(math.floor(value + 0.5))


async def submit_generate(
    prompt: str,
    negative_prompt: str = "",
    nag_enabled: bool = False,
    nag_scale: float = 11.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
    width: int = 512,
    height: int = 320,
    crop_width: int | None = None,
    crop_height: int | None = None,
    num_frames: int = 49,
    frame_rate: float = 24.0,
    seed: int = -1,
    conditioning_images: list[ConditioningImageArg] | None = None,
    loras: list[LoraArg] | None = None,
    reference_video_id: str | None = None,
    conditioning_attention_strength: float | None = None,
    reference_video_strength: float | None = None,
    neg_method: str = "nag",
    vsf_scale: float = 1.5,
    attention_backend: str = "sdpa",
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT,
    keep_resident: bool = KEEP_RESIDENT_DEFAULT,
    keep_resident_embeddings: bool = KEEP_RESIDENT_EMBEDDINGS_DEFAULT,
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    vae_mode: Literal["default", "prune_vaed"] = "default",
    outpaint_pad_left: int = 0,
    outpaint_pad_right: int = 0,
    outpaint_pad_top: int = 0,
    outpaint_pad_bottom: int = 0,
    outpaint_blend_dilation_stage1: int = _OUTPAINT_BLEND_DILATION_STAGE1_DEFAULT,
    outpaint_freeze_source_audio: bool = True,
) -> dict[str, Any]:
    """1本の動画生成ジョブを登録します（POST /generate、単発のT2V/I2V）。

    現在選択中のベースモデル（LTX 2.3 / LTX 2.5 など）が対応していない機能を
    使うと 422 FEATURE_UNSUPPORTED になります。どの機能が使えないかは
    ``list_models`` の ``base_models[].unsupported_features`` を見てください。
    LTX 2.5 では、このツールで使えないのは ``vae_mode``
    （既定値以外にした場合）だけです。**``nag_enabled``（ネガティブプロンプト。
    NAG と VSF）は 2026-08-30 から LTX 2.5 でも使えます**——``negative_prompt``
    / ``nag_scale`` / ``nag_tau`` / ``nag_alpha`` / ``neg_method`` /
    ``vsf_scale`` も同時に効くようになりました。**ただし ``nag_alpha=0`` に
    しても「NAG なし」とビット単位で同じ絵にはなりません**（負のプロンプトを
    1件足すぶん内部のバッチが増え、行列積のカーネル選択が変わりうるため）。
    **無効化したいときは ``nag_enabled`` を ``False`` にしてください。**
    ``loras``（スタイルLoRA・制御系IC-LoRA）と
    ``reference_video_id``、``conditioning_attention_strength`` /
    ``reference_video_strength`` は LTX 2.5 でも使えます。
    ``keep_resident`` も 2026-08-25 から LTX 2.5 で使えます（既定off のまま。
    2.5 が常駐させるのはテキストエンコーダの重み1つだけで約7.7GiBです）。
    ``keep_resident_embeddings``（埋め込み処理器のジョブ間常駐）は
    **LTX 2.5 専用**です（既定off）。**これは向きが逆で、LTX 2.3 を選んで
    いるときに true を送ると 422 FEATURE_UNSUPPORTED になる初めての引数です**
    ——「LTX 2.5 で使えない機能」ではなく「LTX 2.3 で使えない機能」です。
    ``attention_backend``（SageAttention）も 2026-08-25 から LTX 2.5 で
    使えます（既定 ``"sdpa"`` のまま）。**これだけは他の高速化と違い、
    ``"sage"`` にすると同じシードでも生成結果の細部が変わります**。
    速さは動画の大きさに強く依存し、1280x768 の連結生成で約1.10倍、
    512x320 級では効かないか、かえって遅くなることがあります。
    sageattention が入っていない環境では 422 にはならず自動的に ``"sdpa"``
    へ降格して完走し、``metadata.json`` の ``attention_used`` が
    ``"sage->sdpa"`` になります（降格の規律は LTX 2.3 と同じです）。
    ``submit_chain``（連結生成）は本体そのものは使えますが、素材の指定など
    一部の引数が使えません（``submit_chain`` の説明を見てください）。

    同時に実行できるジョブは1本だけです（Phase 1の制約）。既にジョブが
    進行中の場合は 409 JOB_BUSY のエラーになります -- 先に ``job_status`` /
    ``wait_for_job`` で完了を確認してください。このツール自体はジョブを
    「登録」するだけで完了を待ちません。結果は返ってきた ``job_id`` を
    ``wait_for_job`` に渡して確認してください。

    サイズ・尺の制約:
      * ``width`` / ``height`` は必ず64の倍数（two-stage distilledパイプラ
        インの制約）。``reference_video_id`` を使う場合は128の倍数が必要
        （制御系IC-LoRAは参照動画を半分解像度で消費するため）。
      * 最終的な表示サイズを64の倍数以外にしたい場合は ``crop_width`` /
        ``crop_height`` を使ってください（生成後にクロップされます）。両方
        セットするか、両方省略するかのどちらかにしてください。
      * ``num_frames`` は 8n+1（9, 17, 25, ... 481）である必要があります。

    IC-LoRA:
      * ``conditioning_images`` を1件以上指定するとI2V（画像条件付け）に
        なります。空ならT2V。
      * ``loras`` で指定するアダプタ名は ``list_loras`` で確認してください。
        参照動画が必須な制御系（control）アダプタを使う場合は、必ず
        ``loras`` の先頭（``loras[0]``）に置き、``reference_video_id`` も
        指定してください（スタイル系/character系アダプタは参照動画不要）。

    画角拡張（outpainting、元動画の外側を描き足して画角を広げる）:
      * ``outpaint_pad_*``（左右上下のパディング、単位px）を1つ以上0より大きく
        すると画角拡張になります。4辺すべて0なら通常の生成です。
      * **``width`` / ``height`` は「拡張後の最終キャンバス」であり、パディング
        はその内側から切り出されます。** 元動画の解像度と一致しなければならない
        のは「残す領域」＝ ``width - outpaint_pad_left - outpaint_pad_right``
        × ``height - outpaint_pad_top - outpaint_pad_bottom`` のほうです
        （サーバーがffprobeで実測して照合し、食い違えば422で拒否します）。
        元動画がリサイズされることはありません。
      * 満たすべき幾何条件は4つです。(1) ``width`` / ``height`` は**128の倍数**
        （参照動画を使うため64ではなく128）。(2) 残す領域が元動画の解像度と
        完全一致。(3) 残す領域は**縦横とも256px以上**（なじみ処理がキャンバス
        長辺の約1/10まで内側に及ぶため）。(4) 元動画のフレーム数が
        ``num_frames`` 以上。
      * ``reference_video_id`` が**必須**です（画角を広げる対象の元動画。
        ``upload_video`` で取得）。``conditioning_images`` と
        ``crop_width`` / ``crop_height`` とは排他です。
      * **``in-outpainting`` という制御系LoRAが1本だけ必要で、このツールが
        自動で ``loras`` へ追加します**（``{"name": "in-outpainting",
        "strength": 1.0}``。あなたが ``loras`` に同名のものを既に入れていれば
        何もしません）。**``in-outpainting`` が導入されていない環境ではLoRAの
        解決に失敗して404になります**——事前に ``list_loras`` で存在を確認して
        ください。
      * **アップロード前に、ローカルで ffprobe 等により元動画の解像度と
        フレーム数を確認し、キャンバス（128の倍数）を逆算してください。
        ``upload_video`` の応答は解像度を返しません**（``max_frames`` を渡せば
        フレーム数と ``fps`` だけは返ります）。
      * ``outpaint_blend_dilation_stage1`` は継ぎ目のなじみ幅（マスクぼかし）の
        段数です。実寸のピクセル幅は ``段数 × キャンバス長辺 ÷ 64`` で、
        1920px幅・既定5段なら約150pxになります。stage 2 の段数は公式ワーク
        フローと同じ5:2の比で**自動追従**するので、指定するのはstage 1だけです。
        既定（5）のままなら ``blend_dilation_stage1`` / ``_stage2`` の
        両キーとも送りません（このフィールドを知らない古いサーバーでも通る
        ようにするため）。

    Args:
        prompt: 生成プロンプト（必須）。
        negative_prompt: ネガティブプロンプト。``nag_enabled=True`` のときは
            空文字列不可。
        nag_enabled: NAG（Normalized Attention Guidance、CFGを使わないネガ
            ティブプロンプト手法）を有効にするか。
        nag_scale, nag_tau, nag_alpha: NAGのパラメータ（``nag_enabled=True``
            のときのみ意味を持つ）。
        neg_method: 非CFGネガティブプロンプトの方式（``"nag"`` または
            ``"vsf"``、``nag_enabled=True`` のときのみ意味を持つ）。VSF
            （Value Sign Flip, arXiv:2508.10931）は正負のコンテキストを連結
            し1回のattentionで処理する方式で、NAGより排除力が強い一方、正
            プロンプトへの忠実度はNAGが上（性格が異なるため選択式）。
        vsf_scale: VSFの負側V（value）への乗算係数α（既定1.5、0〜10、
            ``neg_method="vsf"`` のときのみ意味を持つ。0でも無効化にはなら
            ない）。
        width, height: 生成解像度（64の倍数、参照動画使用時は128の倍数）。
        crop_width, crop_height: 最終出力のクロップサイズ（両方指定 or 両方
            省略）。
        num_frames: フレーム数（8n+1、9〜481）。
        frame_rate: フレームレート。1〜60の範囲内なら**整数へ四捨五入して**
            送る（29.97→30、23.976→24。台帳 §3-71 / §3-72、
            ``_snap_frame_rate`` 参照）。範囲外の値は丸めずそのまま送るので
            サーバーが422で弾く。
        seed: 乱数シード（-1でランダム）。
        conditioning_images: I2V用のキーフレーム画像（最大5件、``upload_image``
            で得た ``image_id`` を使う）。
        loras: 適用するIC-LoRAアダプタのリスト。各要素の ``audio_strength``
            は映像軸（``strength``）とは独立した音声軸の適用強度（省略可、
            0〜2）。省略時は音声側も ``strength`` に追従（従来と同一）。0は
            音声側の重みを一切適用しない（style LoRAが生成音声を壊す事例
            への対処）。
        reference_video_id: 制御系IC-LoRA用の参照動画ID（``upload_video`` で
            取得）。
        conditioning_attention_strength: 制御系IC-LoRAの追従の強さ（0〜1、
            ``loras`` 指定時のみ）。
        reference_video_strength: 参照動画の条件付け強度（0〜1、``loras``
            指定時のみ）。
        attention_backend: attentionの実装（``"sdpa"``（既定）または
            ``"sage"``）。``"sage"`` はエンジンに sageattention が導入済みの
            場合のみ有効で、未導入時はサーバーが自動的に ``"sdpa"`` へ降格
            して完走します。**``"sage"`` 有効時は同一シードでも生成結果の
            細部が変わります**（数値精度が異なるため）。利用可否は
            ``backend_status`` の ``acceleration.sage_available`` で確認で
            きます。実際に使われた方式はジョブ完了後のメタデータの
            ``attention_used`` に記録されます。
        block_swap_prefetch: block swap（VRAM節約のためtransformerのブロックを
            CPUとGPUのあいだで出し入れする仕組み）の転送を、計算の裏に先読み
            で隠します（既定on。offにすると従来の同期スワップになります）。
            **``attention_backend`` と違い、生成結果は変わりません**（転送
            方式だけが変わるので、同一シードならビット単位で同一になります）。
            block swapが無効な設定では黙って無効になります。利用可否は
            ``backend_status`` の ``acceleration.block_swap_prefetch_available``
            で確認できます。実際に効いたかはジョブ完了後のメタデータの
            ``block_swap_prefetch_used`` に記録されます。
        keep_resident: モデルのCPU側「骨格」をジョブ間で常駐させ、2回目以降の
            生成の前処理を大幅に短縮します（実測 約70秒→約10秒）。**既定off**
            （``block_swap_prefetch`` とは既定の向きが逆）。**メインメモリを
            約20GB常駐で使うため、64GB以上を推奨**します。``attention_backend``
            と違い**生成結果は変わりません**（同一シードでビット単位で同一）。
            offに戻すとキャッシュを解放し、再度onにすると作り直しで50〜70秒を
            1回だけ払い直します。``gguf_per_layer_quant=0`` のモデル構成では
            エラーになり、``dit_cpu_load=0`` または ``block_swap_prefetch=false``
            との併用では自動的にoffへ降格します（メインメモリ二重化の回避）。
            実際に効いたかはジョブ完了後のメタデータの ``keep_resident_used``
            （``"off"`` / ``"on"`` / ``"on->off"``）に記録されます。
            **上の数値と併用制限はすべて LTX 2.3 のものです。** LTX 2.5 でも
            2026-08-25 から使えますが、**契約が同じだけで中身は別物**で、
            常駐するのは Gemma 4 テキストエンコーダの重み1つだけ（実測
            7.68GiB）です。2本目以降の生成が 27.6秒→20.5秒（約25%短縮）に
            なり、LTX 2.5 側には併用の制限も自動降格も無いため、エコーは
            ``"on"`` / ``"off"`` の2値しか出ません。
        keep_resident_embeddings: **LTX 2.5 専用**。2.5 の埋め込み処理器
            （プロンプトを読み取ったあとの内部表現を整える部品）のCPU側の
            重み1本をジョブ間で常駐させ、毎ジョブのGGUF読み直しを省きます
            （**既定off**）。``keep_resident`` と同じく**生成結果は1バイトも
            変わりません**（構築が速くなるだけです）。代償はメインメモリで、
            約4.6GiB が常駐します。``keep_resident`` とは別のスイッチなので、
            両方onにするとメモリ増分は加算になります。実際に効いたかはジョブ
            完了後のメタデータの ``keep_resident_embeddings_used``（``"on"`` /
            ``"off"`` の2値。降格経路が無いので ``"on->off"`` は出ません）に
            記録されます。**LTX 2.3 を選んでいるときに true を送ると 422
            FEATURE_UNSUPPORTED になります**（2.3 にはこの部品がありません）。
        fused_gguf_dequant_kernel: GGUF（K量子化 Q4_K/Q5_K/Q6_K）の逆量子化を
            Tritonの1カーネルにまとめて高速化します（**既定on**。実機で
            約17.5%短縮）。
            ``attention_backend`` と違い**生成結果は変わりません**（現行実装
            とのビット一致を必須要件として検証しています。同一シードならビット
            単位で同一）。Tritonが無い・カーネルが例外を出した・起動時の自己
            検証で不一致だった、のいずれでも黙って従来実装へ降格し、生成は
            落としません。実際に効いたかはジョブ完了後のメタデータの
            ``fused_gguf_dequant_kernel_used``（``"off"`` / ``"on"`` /
            ``"on->off"``。``"on->off"`` は「要求したが実際には適用されな
            かった」）に記録されます。
        vae_mode: 映像VAE（潜在表現と映像を相互変換する部品）の**デコーダ**の
            実装選択。``"default"``（既定）または ``"prune_vaed"``。
            ``"prune_vaed"`` は枝刈り版（PrunaVAED）で、映像の復元が速くなる
            代わりに**出力品質がわずかに低下する可能性があります**
            （Acceleration のうち唯一「絵が変わる」つまみです）。
            **既定は ``"default"`` であり、既定のままなら従来と完全に同じ
            です**（キーはサーバーへ送られません）。枝刈り版の重みが導入され
            ていない環境では黙って既定デコーダへ降格し、生成は落としません。
            実際に何で復元したかはジョブ完了後のメタデータの ``vae_mode_used``
            （``"off"`` / ``"on"`` / ``"on->off"``。``"on->off"`` は「頼んだが
            重みが無かったので既定で完走した」）に記録されます。
            **LTX 2.5では本引数の指定自体が422になりますが、``vae_mode_used``
            は引き続き記録され、値は載っているデコーダの実名（現行の構成では
            ``"conv"``）に変わります**（台帳 §3-131。語彙の正本はバックエンド
            ``Videomni_Backend_Specification.md`` §6.6）。
        outpaint_pad_left, outpaint_pad_right, outpaint_pad_top,
        outpaint_pad_bottom: 画角拡張で描き足す幅（px、0〜4096、既定0）。
            **4辺すべて0なら画角拡張は行いません**（``outpaint`` はサーバーへ
            送られません）。1つでも0より大きければ画角拡張になり、
            ``reference_video_id`` が必須になります。上の「画角拡張」の節に
            ある幾何条件4つを必ず満たしてください。
        outpaint_blend_dilation_stage1: 継ぎ目のなじみ幅の段数（0〜15、既定5＝
            公式ワークフローの値）。stage 2 の段数は5:2の比で自動追従します。
            **既定のままなら送信しません。** 画角拡張をしない（4辺すべて0）のに
            既定以外にすると、黙って無視される代わりにエラーになります。
        outpaint_freeze_source_audio: 元動画の音声を両ステージで凍結するか
            （既定True＝公式ワークフローと同じ。広がった画角に合わせて音声を
            作り直させたい場合だけFalseにします）。**既定のままなら送信
            しません。** 画角拡張をしない（4辺すべて0）のにFalseにすると、
            黙って無視される代わりにエラーになります。

    Returns:
        job_id, status, created_at, next（次に呼ぶべきツールの案内文）。
    """
    total_outpaint_pad = (
        outpaint_pad_left + outpaint_pad_right + outpaint_pad_top + outpaint_pad_bottom
    )
    # 「指定が黙って消える」型のPOST前チェック（既存のCROP_SIZE_INCOMPLETEと
    # 同じカテゴリ）。幾何条件・排他はサーバーの422に任せ、ここでは二重に
    # 持たない。
    if total_outpaint_pad == 0 and (
        outpaint_blend_dilation_stage1 != _OUTPAINT_BLEND_DILATION_STAGE1_DEFAULT
        or not outpaint_freeze_source_audio
    ):
        raise ToolError(
            "OUTPAINT_PADS_ALL_ZERO: outpaint_blend_dilation_stage1 / "
            "outpaint_freeze_source_audio を指定するには、outpaint_pad_left / "
            "_right / _top / _bottom のいずれかを0より大きくしてください"
            "（4辺すべて0だと画角拡張そのものが行われず、指定が無視されます）"
        )
    if (crop_width is None) != (crop_height is None):
        raise ToolError(
            "CROP_SIZE_INCOMPLETE: crop_width と crop_height は両方指定するか、"
            "両方省略してください"
        )

    payload: dict[str, Any] = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    payload["width"] = width
    payload["height"] = height
    if crop_width is not None and crop_height is not None:
        payload["crop_output"] = {"width": crop_width, "height": crop_height}
    payload["num_frames"] = num_frames
    payload["frame_rate"] = _snap_frame_rate(frame_rate)
    payload["seed"] = seed

    if nag_enabled:
        payload["nag_enabled"] = True
        payload["nag_scale"] = nag_scale
        payload["nag_tau"] = nag_tau
        payload["nag_alpha"] = nag_alpha
        payload["neg_method"] = neg_method
        payload["vsf_scale"] = vsf_scale

    if attention_backend != "sdpa":
        payload["attention_backend"] = attention_backend
    # block_swap_prefetch: sent ONLY when it differs from the server's own
    # default (BLOCK_SWAP_PREFETCH_DEFAULT, S4 2026-08-01 = True) — mirrors
    # gradio_ui/handlers.py's discipline. "send only when True" would have
    # gone unsafe the moment the server's default flipped to True: an
    # explicit "off" call would then never reach the wire and the server's
    # own default would silently turn it back on.
    if block_swap_prefetch != BLOCK_SWAP_PREFETCH_DEFAULT:
        payload["block_swap_prefetch"] = block_swap_prefetch
    # keep_resident: same "differs from the server's own default" rule — but
    # KEEP_RESIDENT_DEFAULT is False, so in practice the key rides only on an
    # explicit True. Do not fold the two into one "send when True" branch: they
    # are the same RULE with opposite defaults, and the rule is what survives a
    # future default flip.
    if keep_resident != KEEP_RESIDENT_DEFAULT:
        payload["keep_resident"] = keep_resident
    # keep_resident_embeddings: the same rule once more, against its own
    # constant. Its default is False today, so in practice the key rides only on
    # an explicit True — but the comparison, not a "send when True" branch, is
    # what would survive a future default flip, exactly as for keep_resident
    # above. Placed beside it because they are the same KIND of switch (two
    # objects kept resident across jobs), not because they travel together.
    if keep_resident_embeddings != KEEP_RESIDENT_EMBEDDINGS_DEFAULT:
        payload["keep_resident_embeddings"] = keep_resident_embeddings
    # fused_gguf_dequant_kernel: same rule again (FUSED_GGUF_DEQUANT_KERNEL_
    # DEFAULT flipped to True on 2026-08-04, so the key now rides only on an
    # explicit False -- the rule is unchanged, which is exactly why it was
    # written as a comparison rather than a "send when True" branch), and
    # appended last so the default payload's key order is untouched.
    if fused_gguf_dequant_kernel != FUSED_GGUF_DEQUANT_KERNEL_DEFAULT:
        payload["fused_gguf_dequant_kernel"] = fused_gguf_dequant_kernel
    # vae_mode (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50
    # at the time; 2026-08-05): same "differs from the server's own
    # default" rule, expressed against the "default" literal because the
    # server declares it inline (api/models.py:213) rather than through a
    # shared constant. Appended after fused_gguf_dequant_kernel so the default
    # body's key set and order stay frozen. Mirrors gradio_ui/handlers.py:908.
    if vae_mode != "default":
        payload["vae_mode"] = vae_mode

    if conditioning_images:
        payload["conditioning_images"] = [ci.model_dump() for ci in conditioning_images]
    if loras:
        payload["loras"] = [lora.model_dump(exclude_none=True) for lora in loras]
    if reference_video_id:
        payload["reference_video_id"] = reference_video_id
    if conditioning_attention_strength is not None:
        payload["conditioning_attention_strength"] = conditioning_attention_strength
    if reference_video_strength is not None:
        payload["reference_video_strength"] = reference_video_strength

    # 画角拡張（outpainting, ``Docs/PENDING_TASKS_CLOSED.md`` §3-70（起票当時は
    # §1-13）/ ``Docs/PENDING_TASKS_CLOSED.md`` §3-122、
    # 2026-09-01にクローズ済み）。ネストオブジェクトは
    # end_source と同じく「使うときだけ丸ごと足す」（4辺すべて0＝使わない）。
    # 最後に追加しているので、既定値だけの呼び出しのボディのキー集合と順序は
    # これまでと1バイトも変わらない。
    if total_outpaint_pad > 0:
        outpaint: dict[str, Any] = {
            "pad_left": outpaint_pad_left,
            "pad_right": outpaint_pad_right,
            "pad_top": outpaint_pad_top,
            "pad_bottom": outpaint_pad_bottom,
        }
        # 既定（5）のときは両キーごと省略する -- 操作パネル
        # （webui/src/modes/edit/useOutpaintForm.ts）と同じ後方互換の規律。
        # 非既定のときだけ、stage 2 を5:2で追従させて2キーとも送る。
        if outpaint_blend_dilation_stage1 != _OUTPAINT_BLEND_DILATION_STAGE1_DEFAULT:
            outpaint["blend_dilation_stage1"] = outpaint_blend_dilation_stage1
            outpaint["blend_dilation_stage2"] = _outpaint_stage2_from_stage1(
                outpaint_blend_dilation_stage1
            )
        if not outpaint_freeze_source_audio:
            outpaint["freeze_source_audio"] = False
        payload["outpaint"] = outpaint

        # `in-outpainting` の自動注入（2026-09-01 オーナー裁定）。操作パネルは
        # このLoRAを固定でピン留めしており、サーバーも outpaint には
        # preprocess の無い control 系アダプタをちょうど1本要求する。
        # エージェントに毎回書かせると忘れて422になるだけなので、パネルと
        # 同じ振る舞いをここで再現する。重複判定は **name の一致だけ** で行う
        # -- 呼び出し側が strength 0.7 で明示していたらその指定を尊重する。
        existing_loras: list[dict[str, Any]] = payload.get("loras", [])
        if not any(lora.get("name") == _OUTPAINT_LORA_NAME for lora in existing_loras):
            payload["loras"] = [
                *existing_loras,
                {"name": _OUTPAINT_LORA_NAME, "strength": _OUTPAINT_LORA_STRENGTH},
            ]

    client = get_client()
    result = await client.post_json("/generate", json=payload, timeout=_SUBMIT_TIMEOUT)
    return {
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "created_at": result.get("created_at"),
        "next": "wait_for_job(job_id) を呼ぶ",
    }


def _serialize_chain_clip(clip: ChainClipArg) -> dict[str, Any]:
    """``ChainClipArg`` を送信用dictへ変換する。``prompt`` が None（=チェーン
    基準プロンプトを継承）と ``conditioning_images`` が None/空（=クリップ0
    以外、または画像なし）の場合はキーごと省略する（計画のペイロード契約）。"""
    d: dict[str, Any] = {"num_frames": clip.num_frames}
    if clip.prompt is not None:
        d["prompt"] = clip.prompt
    if clip.conditioning_images:
        d["conditioning_images"] = [ci.model_dump() for ci in clip.conditioning_images]
    return d


async def submit_chain(
    prompt: str,
    clips: list[ChainClipArg],
    negative_prompt: str = "",
    nag_enabled: bool = False,
    nag_scale: float = 11.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
    width: int = 512,
    height: int = 320,
    crop_width: int | None = None,
    crop_height: int | None = None,
    frame_rate: float = 24.0,
    seed: int = -1,
    overlap_frames: int = 3,
    overlap_strength: float = 0.5,
    source_video_id: str | None = None,
    source_video_context_frames: int = 73,
    source_audio_id: str | None = None,
    loras: list[LoraArg] | None = None,
    reference_video_id: str | None = None,
    conditioning_attention_strength: float | None = None,
    reference_video_strength: float | None = None,
    chunked_upsample: bool = True,
    neg_method: str = "nag",
    vsf_scale: float = 1.5,
    attention_backend: str = "sdpa",
    block_swap_prefetch: bool = BLOCK_SWAP_PREFETCH_DEFAULT,
    keep_resident: bool = KEEP_RESIDENT_DEFAULT,
    keep_resident_embeddings: bool = KEEP_RESIDENT_EMBEDDINGS_DEFAULT,
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    vae_mode: Literal["default", "prune_vaed"] = "default",
    end_source_video_id: str | None = None,
    end_source_image_id: str | None = None,
    end_source_context_frames: int = 72,
    end_source_strength: float = 1.0,
    retake_video_id: str | None = None,
    retake_window_start_sec: float | None = None,
    retake_head_px: int = _RETAKE_HEAD_PX_DEFAULT,
    retake_tail_px: int = _RETAKE_TAIL_PX_DEFAULT,
    retake_regenerate_audio: bool = _RETAKE_REGENERATE_AUDIO_DEFAULT,
) -> dict[str, Any]:
    """クリップチェーン生成ジョブを登録します（POST /generate/chain）。

    現在選択中のベースモデル（LTX 2.3 / LTX 2.5 など）が対応していない機能を
    使うと 422 FEATURE_UNSUPPORTED になります。どの機能が使えないかは
    ``list_models`` の ``base_models[].unsupported_features`` を見てください。
    **LTX 2.5 では、連結生成そのもの（複数クリップを1本に繋ぐ本体の動作）に
    加えて、``source_video_id``（V2V継続）と ``source_audio_id``（A2V。長尺
    A2V＝複数クリップにまたがる音声も含みます）、``loras``（スタイルLoRA・
    制御系IC-LoRA）と ``reference_video_id``（参照動画。長尺IC-LoRA＝複数
    クリップにまたがる参照動画も含みます）も使えます**。
    **``end_source_video_id`` / ``end_source_image_id``（素材（末尾））も
    2026-08-26 から LTX 2.5 で使えます**——この日に撮り直し（Retake）と
    素材（末尾）が開通し、``submit_chain`` が投げられるモードは LTX 2.5 でも
    全部通るようになりました（幾何・受理範囲・``metadata.json`` の
    ``end_source`` ブロックはいずれも LTX 2.3 と同一です）。
    **``nag_enabled``（ネガティブプロンプト。NAG と VSF）も 2026-08-30 から
    LTX 2.5 の連結生成で使えます**（``negative_prompt`` / ``nag_scale`` /
    ``nag_tau`` / ``nag_alpha`` / ``neg_method`` / ``vsf_scale`` も同時に。
    連結生成では1本の ``negative_prompt`` が全クリップ・全ステージに効きます）。
    ただし次の引数は使えず、既定値以外にすると 422 FEATURE_UNSUPPORTED に
    なります: ``vae_mode``。これを使いたい場合は ``load_pipeline``
    で LTX 2.3 に切り替えてください。
    **``keep_resident`` は 2026-08-25 から LTX 2.5 の連結生成でも使えます**
    （既定off のまま。2.5 が常駐させるのはテキストエンコーダの重み1つだけ
    です）。``keep_resident_embeddings``（埋め込み処理器のジョブ間常駐）は
    **LTX 2.5 専用**で、連結生成でも使えます（既定off）。**これは向きが逆で、
    LTX 2.3 を選んでいるときに true を送ると 422 FEATURE_UNSUPPORTED になる
    初めての引数です。****``attention_backend``（SageAttention）も 2026-08-25 から
    LTX 2.5 の連結生成で使えます**（既定 ``"sdpa"`` のまま。チェーンの全
    クリップ・全ステージへ一律に効き、実測は 1280x768・2クリップ×121
    フレームで約1.10倍）。**ただし ``"sage"`` にすると同じシードでも生成
    結果の細部が変わります。** エコーはチェーン全体の畳み込みで、どこか
    1回でも降格すれば ``"sage->sdpa"`` になります。

    複数クリップを1本の連続した動画に合成します（クリップ間はlatentレベルで
    継ぎ目なく繋がります――ピクセル領域での結合ではありません）。同時に実行
    できるジョブは1本だけです（Phase 1の制約）。既にジョブが進行中の場合は
    409 JOB_BUSY のエラーになります。このツールもジョブを「登録」するだけで
    完了を待ちません（``wait_for_job`` で確認してください）。

    クリップ数の下限:
      * ``source_video_id`` も ``source_audio_id`` も ``reference_video_id``
        も指定しない通常のチェーンは、``clips`` が2件以上必要です（1クリップ
        なら submit_generate と同じなので）。
      * A2V（``source_audio_id`` 指定）は ``clips`` が **1〜24件** に対応
        します（長尺A2V）。アップロードした音声1本がチェーン全体の時間軸を
        受け持ち、サーバー側が各クリップのstage-1区間へ自分の窓を割り当てます
        （クリップごとに音声を分けてアップロードする方式ではありません）。
      * V2V継続（``source_video_id`` 指定）は1クリップ以上（アップロード動画
        の末尾がクリップ0の先頭として凍結されるため、1クリップでも成立します）。
      * ``reference_video_id``（制御系IC-LoRA）を使うチェーンは、**1〜24
        クリップ** に対応します（2026-08-11のオーナー判断で、旧alpha版の
        「ちょうど1クリップ」制限は撤廃）。長い参照動画1本をサーバー側が
        各クリップのstage-1区間へ自動スライスして注入し、参照が足りない
        区間は参照なしで生成されます。``source_video_id`` とは排他です。
        例外として、深度（depth）系の制御アダプタは2クリップ以上の
        チェーンでは非対応（422 ``LORA_DEPTH_CHAIN_UNSUPPORTED``。depthの
        前処理はチェーン全体分の参照を一括処理する設計のため）。
      * 撮り直し（``retake_video_id`` 指定）は **``clips`` がちょうど1件**
        です（そのクリップ自身が作り直す窓だからです）。

    V2V / A2V:
      * ``source_video_id`` と ``source_audio_id`` は同時に指定できません
        （このツールがPOST前にエラーにします）。
      * ``source_video_id`` を使う場合、アップロード動画の末尾
        ``source_video_context_frames`` フレーム（8n+1、既定73）がクリップ0の
        先頭として凍結されます。クリップ0には ``conditioning_images`` を
        同時指定できません（先頭が既に元動画で埋まっているため）。
      * ``source_audio_id`` を使う場合、アップロード音声に映像が同期するよう
        生成されます。クリップ0の ``conditioning_images``（開始フレーム画像）
        とは併用できます。
      * ``conditioning_images`` はどのモードでもクリップ0のみ有効です
        （クリップ1以降は前クリップの潜在表現を引き継ぐ後続セグメントのため）。
      * ``end_source_video_id`` / ``end_source_image_id``（素材（末尾））は
        **クリップが何件でも使えます**。指定した素材の
        先頭 ``end_source_context_frames`` フレーム（8の倍数。**8を推奨**
        します——実機比較で最良で、錨が長いほど窓を素材の再現に費やし
        生成の創造性が下がります。**既定の72は契約上の既定値であって
        推奨値ではありません**）が最後のクリップの末尾フレームとして
        凍結されます。素材は ``end_source_context_frames + 1`` フレーム以上
        必要です（因果VAEのプライマ1枚分）。
        **クリップが1件のときは窓内モードで、生成は1つのデノイズ窓の中で
        その素材へ到達するように行われます（推奨）**。**2件以上のときは
        逆順Chainedで、最後のクリップから順に生成し、各クリップは1つ後ろの
        クリップの冒頭を自分の末尾として引き継ぎます（継ぎ目の強さは
        ``overlap_strength`` が正順と同じように効きます。受理されますが
        推奨外です**——実機ゲート後のオーナー目視・試聴で、クリップの境目・
        末尾（錨直前）に映像のモーフや音楽の不統一といった品質劣化が出る
        ことを確認しており、これは仕様として許容しています。品質を重視して
        複数クリップを終端付きで繋ぎたい場合は、end_source をクリップ1件
        ずつ使い、生成物を次の素材にして過去へ遡って生成する手動リレーが
        実用的な回避策です。**この推奨外の逆順Chainedは、LTX 2.5 では音声の
        継ぎ目がさらに悪くなります**（同一の依頼・素材・解析器で、LTX 2.3 は
        6箇所とも連続、LTX 2.5 は 4/6 が不連続。映像側は 2.5 でも 6/6 連続
        です）。**推奨のクリップ1件（窓内モード）には影響しません。**
        **出力の長さはどちらの場合もクリップの合計**であって、素材の分だけ
        伸びることはありません。
        ``retake_video_id`` / ``source_audio_id`` / ``reference_video_id``
        とは排他です。``source_video_id`` との併用は**クリップ1件のときだけ**受理
        されます（頭と尾の両方を固定して補間する構図）。2件以上との併用は
        422で拒否されます。
      * ``end_source_strength``（既定1.0）は1.0で素材どおりに終わります
        （既定）。下げるとStage-1での素材へのなじみ方が緩くなりますが、
        Stage-2で改めて固定されるため最終フレームは常に素材どおりになり
        ます。

    撮り直し（Retake、既存クリップの「まん中」だけ作り直す時間方向の
    inpainting）:
      * ``retake_video_id``（``upload_video`` で取得したID）を指定すると
        撮り直しになります。サーバーは元動画から窓
        ``[retake_window_start_sec, retake_window_start_sec + 窓長)`` を
        フレーム単位で切り出し、その窓**だけ**をエンジンへ渡します。
      * **窓長を決めるのは ``clips[0].num_frames`` ただ1つです**——長さを表す
        第2の引数は意図的に存在しません（食い違いを作らないため）。
        受理される範囲は既定のstage-2窓で **[73, 169] フレーム**であり、
        ``get_config`` の ``limits.retake_window_min_frames`` /
        ``retake_window_max_frames`` で確認できます。``clips`` はちょうど1件
        にしてください。
      * ``retake_window_start_sec`` は**必須**です（0以上の秒数）。
        ``retake_video_id`` を指定して ``retake_window_start_sec`` を省略すると、
        このツールがPOST前にエラーにします。
      * **窓の開始秒を決める下調べには ``upload_video(file_path, max_frames=...)``
        を使ってください**——``max_frames`` を渡したときだけ、保存された動画の
        ``frame_count`` と ``fps`` が実測されて返ります（``max_frames`` は
        「先頭Nフレームだけ残す」引数でもあるので、測るだけのときは元の尺より
        確実に大きい値を渡してください）。
      * ``retake_head_px``（既定25）と ``retake_tail_px``（既定24）は、窓の
        前後で凍結したまま残す「糊しろ」のフレーム数です。**``retake_head_px``
        は 8n+1、``retake_tail_px`` は 8の倍数でなければなりません**——映像VAEが
        因果的で、窓の両端が別の潜在グリッドに乗るためです。
        **既定の25/24は較正済みの推奨値であり、広くすれば良いというものでは
        ありません**（49/48は音声の継ぎ目をむしろ弱めることが実測されています）。
      * 納品されるmp4は**窓まるごと**で、糊しろはトリムされていません。この
        糊しろが、タイムライン上で元動画へ重ね直すための「のりしろ」であり、
        品質の継ぎ目を編集点ではなく窓の外側の縁へ追い出す仕組みです。
      * ``retake_regenerate_audio`` を False にすると、窓の**元の波形**を
        そのまま使い戻します（ボコーダをスキップします）。この場合、元動画に
        音声トラックが実際に無いと422で拒否されます。
      * ``source_video_id`` / ``source_audio_id`` / ``reference_video_id`` /
        ``end_source_video_id`` / ``end_source_image_id`` /
        ``clips[0].conditioning_images`` とは**すべて排他**です（いずれも1本の
        クリップの端を取り合うため）。

    その他:
      * ``loras`` はチェーン全体・全ステージに一律で効きます（クリップごとの
        強度指定はv1では未対応）。
      * ``chunked_upsample``（既定True、操作パネルの既定と同じ）は、アップ
        サンプル処理をチャンク分割してVRAM使用量を平坦化します（長尺・高解像
        度チェーンでのOOM回避）。常に明示的に送信します。

    Args:
        prompt: チェーン全体の基準プロンプト（各クリップの ``prompt`` 省略時
            に使われる）。
        clips: クリップごとの指定（``num_frames`` は必須、``prompt`` /
            ``conditioning_images`` は省略可）。
        negative_prompt, nag_enabled, nag_scale, nag_tau, nag_alpha:
            submit_generate と同じ意味（チェーン全体・全ステージ共通）。
        neg_method, vsf_scale: submit_generate と同じ意味
            （``nag_enabled=True`` のときのみ意味を持ち、チェーン全体・全
            ステージ共通で効く）。
        width, height: 生成解像度（64の倍数、参照動画使用時は128の倍数）。
        crop_width, crop_height: 最終出力のクロップサイズ（両方指定 or 両方
            省略）。
        frame_rate: フレームレート。1〜60の範囲内なら**整数へ四捨五入して**
            送る（29.97→30、23.976→24。台帳 §3-71 / §3-72、
            ``_snap_frame_rate`` 参照）。範囲外の値は丸めずそのまま送るので
            サーバーが422で弾く。
        seed: 乱数シード（-1でランダム）。
        overlap_frames: クリップ間の重なり潜在フレーム数（1〜8）。
        overlap_strength: 重なり部分の強度（0〜1）。
        source_video_id: V2V継続元の動画ID（``upload_video`` で取得）。
        source_video_context_frames: 元動画から凍結するフレーム数（8n+1）。
        source_audio_id: A2V駆動音声のID（``upload_audio`` で取得）。
        loras: 適用するIC-LoRAアダプタのリスト。各要素の ``audio_strength``
            は映像軸（``strength``）とは独立した音声軸の適用強度（省略可、
            0〜2）。省略時は音声側も ``strength`` に追従（従来と同一）。0は
            音声側の重みを一切適用しない（style LoRAが生成音声を壊す事例
            への対処）。
        reference_video_id: 制御系IC-LoRA用の参照動画ID（1クリップ限定）。
        conditioning_attention_strength: 制御系IC-LoRAの追従の強さ（0〜1）。
        reference_video_strength: 参照動画の条件付け強度（0〜1）。
        chunked_upsample: アップサンプルのチャンク分割（既定True）。
        attention_backend: attentionの実装（``"sdpa"``（既定）または
            ``"sage"``）。``"sage"`` はエンジンに sageattention が導入済みの
            場合のみ有効で、未導入時はサーバーが自動的に ``"sdpa"`` へ降格
            して完走します。**``"sage"`` 有効時は同一シードでも生成結果の
            細部が変わります**（数値精度が異なるため）。利用可否は
            ``backend_status`` の ``acceleration.sage_available`` で確認で
            きます。実際に使われた方式はジョブ完了後のメタデータの
            ``attention_used`` に記録されます（チェーン全体・全ステージ共通）。
        block_swap_prefetch: submit_generate と同じ意味（既定on。offにすると
            従来の同期スワップになります。チェーン全体・全ステージ共通で
            効きます。生成結果はオン/オフどちらでも同一です）。
        keep_resident: submit_generate と同じ意味（**既定off・メモリ64GB以上
            推奨**。生成結果は変わりません。チェーンでも1つの設定がチェーン
            全体に効きます——骨格キャッシュはジョブ単位ではなくワーカー単位で
            持つためです）。LTX 2.5 では常駐するのがテキストエンコーダの重み
            1つだけ（約7.7GiB）で、その構築はジョブあたり1回です（実測で
            2本目の連結生成が 35.17秒→30.13秒）。
        keep_resident_embeddings: submit_generate と同じ意味（**LTX 2.5 専用・
            既定off**。生成結果は変わりません。チェーンでも1つの設定がチェーン
            全体に効き、埋め込み処理器の構築はジョブあたり1回です。**LTX 2.3 で
            true にすると 422 になります**）。
        fused_gguf_dequant_kernel: submit_generate と同じ意味（**既定on**。
            生成結果は変わりません＝現行実装とビット一致。実行できない環境では
            黙って従来実装へ降格します。チェーン全体・全ステージ共通で効きます）。
        vae_mode: submit_generate と同じ意味（``"default"``（既定）または
            ``"prune_vaed"``。枝刈り版は映像の復元が速くなる代わりに**出力品質
            がわずかに低下する可能性があります**。**既定は ``"default"`` で、
            既定のままなら従来と完全に同じです**。チェーン全体・全クリップ・
            全ステージ共通で効きます。結果は ``vae_mode_used`` に記録されます）。
            **LTX 2.5では本引数の指定自体が422になりますが、``vae_mode_used``
            は引き続き記録され、値は載っているデコーダの実名（現行の構成では
            ``"conv"``）に変わります**（台帳 §3-131。語彙の正本はバックエンド
            ``Videomni_Backend_Specification.md`` §6.6）。
        end_source_video_id: 末尾を凍結する素材の動画ID（``upload_video`` で
            取得）。``end_source_image_id`` とは同時に指定できません。
        end_source_image_id: 末尾を凍結する素材の画像ID（``upload_image`` で
            取得）。``end_source_video_id`` とは同時に指定できません。
        end_source_context_frames: 素材の先頭から凍結するフレーム数（8の
            倍数）。**8を推奨**（実機比較で最良。錨が長いほど素材の再現に
            窓を費やし、創造性が下がる）。**既定の72は契約上の既定値で
            あって推奨値ではない**。
        end_source_strength: 素材（末尾）の固定強度（0〜1、既定1.0）。
            1.0＝素材どおりに終わる（既定）。下げるとStage-1での素材への
            なじみ方が緩くなりますが、Stage-2で改めて固定されるため最終
            フレームは常に素材どおりになります。
        retake_video_id: 撮り直す元動画のID（``upload_video`` で取得）。これを
            指定すると撮り直しになり、``clips`` はちょうど1件・窓長は
            ``clips[0].num_frames`` になります。
        retake_window_start_sec: 作り直す窓の開始秒（0以上）。**``retake_video_id``
            を指定するなら必須**です（省略するとPOST前にエラーになります）。
            ``upload_video(max_frames=...)`` が返す ``frame_count`` / ``fps``
            から算出してください。
        retake_head_px: 窓の先頭で凍結したまま残す糊しろのフレーム数
            （**8n+1**、既定25）。
        retake_tail_px: 窓の末尾で凍結したまま残す糊しろのフレーム数
            （**8の倍数**、既定24）。**既定の25/24は較正済みの推奨値で、
            広げれば良いというものではありません。**
        retake_regenerate_audio: 窓の音声を作り直すか（既定True）。Falseにすると
            元の波形をそのまま使い戻します（元動画に音声トラックが無いと422）。

    Returns:
        job_id, status, created_at, num_clips, next（次に呼ぶべきツールの案内文）。
    """
    # 撮り直しのPOST前チェック2本。どちらも「指定が黙って消える / 意図しない値に
    # 化ける」型（既存の CROP_SIZE_INCOMPLETE と同じカテゴリ）に限っている。
    # 幾何条件（8n+1・8の倍数・窓長の範囲）と排他はサーバーの422が正本で、
    # ここに二重の判断を置かない。
    if retake_video_id and retake_window_start_sec is None:
        raise ToolError(
            "RETAKE_WINDOW_START_REQUIRED: retake_video_id を指定する場合は "
            "retake_window_start_sec（作り直す窓の開始秒、0以上）も指定して"
            "ください（既定値はありません）"
        )
    if not retake_video_id and (
        retake_window_start_sec is not None
        or retake_head_px != _RETAKE_HEAD_PX_DEFAULT
        or retake_tail_px != _RETAKE_TAIL_PX_DEFAULT
        or retake_regenerate_audio != _RETAKE_REGENERATE_AUDIO_DEFAULT
    ):
        raise ToolError(
            "RETAKE_ARGS_WITHOUT_VIDEO_ID: retake_window_start_sec / "
            "retake_head_px / retake_tail_px / retake_regenerate_audio を指定する"
            "には retake_video_id も指定してください（撮り直しでなければ"
            "これらの指定は無視されます）"
        )
    if source_video_id is not None and source_audio_id is not None:
        raise ToolError(
            "SOURCE_XOR_VIOLATION: source_video_id と source_audio_id は同時に"
            "指定できません（V2V継続とA2Vはv1では併用できません）"
        )
    if end_source_video_id is not None and end_source_image_id is not None:
        raise ToolError(
            "END_SOURCE_XOR_VIOLATION: end_source_video_id と "
            "end_source_image_id は同時に指定できません"
        )
    if (crop_width is None) != (crop_height is None):
        raise ToolError(
            "CROP_SIZE_INCOMPLETE: crop_width と crop_height は両方指定するか、"
            "両方省略してください"
        )

    payload: dict[str, Any] = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    payload["width"] = width
    payload["height"] = height
    if crop_width is not None and crop_height is not None:
        payload["crop_output"] = {"width": crop_width, "height": crop_height}
    payload["frame_rate"] = _snap_frame_rate(frame_rate)
    payload["seed"] = seed

    if nag_enabled:
        payload["nag_enabled"] = True
        payload["nag_scale"] = nag_scale
        payload["nag_tau"] = nag_tau
        payload["nag_alpha"] = nag_alpha
        payload["neg_method"] = neg_method
        payload["vsf_scale"] = vsf_scale

    if attention_backend != "sdpa":
        payload["attention_backend"] = attention_backend
    # block_swap_prefetch: same "differs from BLOCK_SWAP_PREFETCH_DEFAULT"
    # discipline as submit_generate above — see that comment for why "only
    # when True" is unsafe now that the server's own default is True.
    if block_swap_prefetch != BLOCK_SWAP_PREFETCH_DEFAULT:
        payload["block_swap_prefetch"] = block_swap_prefetch
    # keep_resident: same rule as submit_generate (default off -> sent only on
    # an explicit True).
    if keep_resident != KEEP_RESIDENT_DEFAULT:
        payload["keep_resident"] = keep_resident
    # keep_resident_embeddings: same rule as submit_generate, beside it for the
    # same reason (default off -> sent only on an explicit True).
    if keep_resident_embeddings != KEEP_RESIDENT_EMBEDDINGS_DEFAULT:
        payload["keep_resident_embeddings"] = keep_resident_embeddings
    # fused_gguf_dequant_kernel: same rule as submit_generate, appended last.
    if fused_gguf_dequant_kernel != FUSED_GGUF_DEQUANT_KERNEL_DEFAULT:
        payload["fused_gguf_dequant_kernel"] = fused_gguf_dequant_kernel
    # vae_mode: same rule as submit_generate (compared against the "default"
    # literal), appended after it. Mirrors gradio_ui/handlers.py:537.
    if vae_mode != "default":
        payload["vae_mode"] = vae_mode

    payload["overlap_frames"] = overlap_frames
    payload["overlap_strength"] = overlap_strength
    payload["clips"] = [_serialize_chain_clip(c) for c in clips]

    if source_video_id:
        payload["source_video"] = {
            "video_id": source_video_id,
            "context_frames": source_video_context_frames,
        }
    if source_audio_id:
        payload["source_audio"] = {"audio_id": source_audio_id}
    if end_source_video_id:
        payload["end_source"] = {
            "video_id": end_source_video_id,
            "context_frames": end_source_context_frames,
            "strength": end_source_strength,
        }
    elif end_source_image_id:
        payload["end_source"] = {
            "image_id": end_source_image_id,
            "context_frames": end_source_context_frames,
            "strength": end_source_strength,
        }
    # 撮り直し（``Docs/PENDING_TASKS_CLOSED.md`` §3-115、2026-09-01にクローズ済み）。
    # end_source と同型に「使うときだけネストを丸ごと足す」。
    # 5キーは既定値のままでも全部入れる -- RetakeSpec 側の既定と同じ値なので
    # 送信しても意味は変わらず、ネストの形が呼び出しごとに揺れないほうが
    # metadata の突き合わせが読みやすいため（end_source と同じ作法）。
    # ``window_start_sec`` が None のまま来ることは無い（上のPOST前チェックで
    # 弾いている）。
    if retake_video_id:
        payload["retake"] = {
            "video_id": retake_video_id,
            "window_start_sec": retake_window_start_sec,
            "head_px": retake_head_px,
            "tail_px": retake_tail_px,
            "regenerate_audio": retake_regenerate_audio,
        }

    if loras:
        payload["loras"] = [lora.model_dump(exclude_none=True) for lora in loras]
    if reference_video_id:
        payload["reference_video_id"] = reference_video_id
    if conditioning_attention_strength is not None:
        payload["conditioning_attention_strength"] = conditioning_attention_strength
    if reference_video_strength is not None:
        payload["reference_video_strength"] = reference_video_strength

    # 常に明示送信（操作パネルの既定値と揃える、計画のペイロード契約の例外）。
    payload["chunked_upsample"] = chunked_upsample

    client = get_client()
    result = await client.post_json("/generate/chain", json=payload, timeout=_CHAIN_SUBMIT_TIMEOUT)
    return {
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "created_at": result.get("created_at"),
        "num_clips": result.get("num_clips"),
        "next": "wait_for_job(job_id) を呼ぶ",
    }


def register(mcp: FastMCP) -> None:
    mcp.tool()(submit_generate)
    mcp.tool()(submit_chain)
