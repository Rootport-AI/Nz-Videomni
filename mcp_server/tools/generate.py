"""単発生成＋チェーン生成ツール（``submit_generate`` / ``submit_chain``、2本）。

``api/models.py`` の ``GenerateRequest`` の表面をほぼそのまま公開するが、隠し
フィールド（``pipeline`` / ``num_inference_steps`` / ``guidance_scale`` /
``crf``）は計画D8により出さない -- ``two_stage_hq`` は現状モックのみで、
distilledパイプラインの固定値（8ステップ・CFG=1.0）を変える意味がないため
（``two_stage_hq`` 系の非公開理由はそのまま存置する）。一方でAcceleration機能は
**5項目すべて**を公開する --
``attention_backend``（既定 ``"sdpa"``、``"sage"`` も選べる）と
``block_swap_prefetch``（既定on。backend §44、実装は先読み block swap。
offにすると従来の同期スワップになる。S4, 2026-08-01: 実機ゲートG1〜G7全PASS
を条件にオーナーが確定した既定反転）、``keep_resident``（既定off。ジョブ間の
CPU骨格キャッシュ。LTX 2.5 では常駐するのがテキストエンコーダだけで、
同じ名前でも中身が違う。§76）、``fused_gguf_dequant_kernel``（既定on。GGUF逆量子化の
Triton 1カーネル化。出力はビット単位で不変。§51, 2026-08-04: 実機ゲート
G1〜G8全PASSを条件にオーナーが確定した既定反転）、``vae_mode``（既定
``"default"``、``"prune_vaed"`` で枝刈り版デコーダ。§3-50, 2026-08-05 に
モックから実機能へ転換）を公開する。
``vae_mode`` はかつて「現状モック（受理のみで効果が無い）なので出さない」
（計画D1）として除外していたが、2026-08-05 のオーナー裁定で公開へ転じた
（実装と実機ゲートG1〜G7の合格を待ってからの最終ステップ。
``PRUNAVAED_WORKORDER.md`` §0-8・§6.3）。

送信ボディは「Noneまたは空は送らない」を徹底する（計画のペイロード契約）。
``crop_width`` / ``crop_height`` は両方指定 or 両方省略のみを許す（片側だけの
指定はサーバーへ投げる前にここで弾く）。NAG系フィールドは ``nag_enabled=False``
のとき丸ごと省略する（サーバーの ``negative_prompt`` 必須チェックは
``nag_enabled=True`` のときだけ働くため、無効時に既定値を送っても実害は無い
が、パネルの「素の生成」との差分を最小化する意味でも送らない）。
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from api.models import (
    BLOCK_SWAP_PREFETCH_DEFAULT,
    FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    KEEP_RESIDENT_DEFAULT,
)
from mcp_server.client import get_client
from mcp_server.params import ChainClipArg, ConditioningImageArg, LoraArg

_SUBMIT_TIMEOUT = 30.0
_CHAIN_SUBMIT_TIMEOUT = 30.0


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
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    vae_mode: Literal["default", "prune_vaed"] = "default",
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
        frame_rate: フレームレート。
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

    Returns:
        job_id, status, created_at, next（次に呼ぶべきツールの案内文）。
    """
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
    payload["frame_rate"] = frame_rate
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
    # fused_gguf_dequant_kernel: same rule again (FUSED_GGUF_DEQUANT_KERNEL_
    # DEFAULT flipped to True on 2026-08-04, so the key now rides only on an
    # explicit False -- the rule is unchanged, which is exactly why it was
    # written as a comparison rather than a "send when True" branch), and
    # appended last so the default payload's key order is untouched.
    if fused_gguf_dequant_kernel != FUSED_GGUF_DEQUANT_KERNEL_DEFAULT:
        payload["fused_gguf_dequant_kernel"] = fused_gguf_dequant_kernel
    # vae_mode (§3-50, 2026-08-05): same "differs from the server's own
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
    fused_gguf_dequant_kernel: bool = FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    vae_mode: Literal["default", "prune_vaed"] = "default",
    end_source_video_id: str | None = None,
    end_source_image_id: str | None = None,
    end_source_context_frames: int = 72,
    end_source_strength: float = 1.0,
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
    です）。**``attention_backend``（SageAttention）も 2026-08-25 から
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
        ``retake`` / ``source_audio_id`` / ``reference_video_id`` とは排他
        です。``source_video_id`` との併用は**クリップ1件のときだけ**受理
        されます（頭と尾の両方を固定して補間する構図）。2件以上との併用は
        422で拒否されます。
      * ``end_source_strength``（既定1.0）は1.0で素材どおりに終わります
        （既定）。下げるとStage-1での素材へのなじみ方が緩くなりますが、
        Stage-2で改めて固定されるため最終フレームは常に素材どおりになり
        ます。

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
        frame_rate: フレームレート。
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
        fused_gguf_dequant_kernel: submit_generate と同じ意味（**既定on**。
            生成結果は変わりません＝現行実装とビット一致。実行できない環境では
            黙って従来実装へ降格します。チェーン全体・全ステージ共通で効きます）。
        vae_mode: submit_generate と同じ意味（``"default"``（既定）または
            ``"prune_vaed"``。枝刈り版は映像の復元が速くなる代わりに**出力品質
            がわずかに低下する可能性があります**。**既定は ``"default"`` で、
            既定のままなら従来と完全に同じです**。チェーン全体・全クリップ・
            全ステージ共通で効きます。結果は ``vae_mode_used`` に記録されます）。
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

    Returns:
        job_id, status, created_at, num_clips, next（次に呼ぶべきツールの案内文）。
    """
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
    payload["frame_rate"] = frame_rate
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
