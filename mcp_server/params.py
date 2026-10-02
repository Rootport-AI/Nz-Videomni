"""MCPツールが受け取る引数のPydanticモデル群。

サーバー側（``api/models.py``）が既に全ての範囲・整合性チェックを持っている
（422がそちらの正本）ので、ここには意図的にバリデータを置かない。フィールド
名・既定値だけを ``api/models.py`` の対応するモデル（``ConditioningImage`` /
``LoraSpec`` / ``ChainClip``）に合わせる。ここでの型はあくまで
「MCPクライアントに見せる引数の形」であり、``submit_generate`` / ``submit_chain``
が実際のPOSTボディを組み立てる際に送信用の辞書へ変換される
（``ConditioningImageArg`` / ``LoraArg`` は ``model_dump()``、``ChainClipArg`` は
``tools/generate.py`` の ``_serialize_chain_clip``）。
"""

from __future__ import annotations

from pydantic import BaseModel


class ConditioningImageArg(BaseModel):
    """1枚のキーフレーム画像条件付け（``submit_generate`` / ``ChainClipArg`` 共通）。

    ``image_id`` は ``upload_image`` が返す ID。``crf`` はサーバー側の隠し
    フィールド（two_stage_hq専用のモック残骸。``Docs/MCP_SERVER_DESIGN.md``
    §2 の D8 で非公開）なので、ここには出さない。``frame_idx`` はサーバー側で
    0 または 8n+1 グリッドへスナップ・クランプされる（0 なら開始フレーム扱い）。
    """

    image_id: str
    frame_idx: int = 0
    strength: float = 0.8


class LoraArg(BaseModel):
    """IC-LoRAアダプタ1件の指定（``submit_generate`` / ``submit_chain`` 共通）。

    ``name`` は ``list_loras`` で得られる登録済みアダプタ名（ファイルパスでは
    ない）。参照動画を要する制御系（control）アダプタも ``loras`` の先頭に
    置く必要は無い（サーバーは制御系かどうかを並び順ではなく各アダプタの
    種別〔``list_loras`` の ``kind``〕で見分ける）。``audio_strength`` は映像軸
    （``strength``）とは独立した音声軸の適用強度（省略時は ``strength`` に
    追従、0は音声側の重みを無効化）。ここにはバリデータを置かない（範囲・
    整合性チェックはサーバー側422が正本）。
    """

    name: str
    strength: float = 1.0
    audio_strength: float | None = None


class ChainClipArg(BaseModel):
    """チェーンの1クリップ分の指定（``submit_chain`` の ``clips`` の1要素）。

    ``prompt`` を省略すると、チェーン全体の基準プロンプトを継承する
    （プロンプト伝播）。``conditioning_images`` はクリップ0のみ有効
    （クリップ1以降は前クリップの潜在表現を引き継ぐ後続セグメントのため）。
    """

    prompt: str | None = None
    num_frames: int = 49
    conditioning_images: list[ConditioningImageArg] | None = None
