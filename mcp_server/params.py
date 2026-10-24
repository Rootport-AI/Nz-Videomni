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
    ベースモデルの規則に寄せられる（0 は開始フレーム。LTX はそれ以外を 8n+1
    へスナップ・クランプ、MiniMax H3 は 0〜``num_frames``−1 の任意の
    フレームをそのまま使い、それより後ろは最後のフレームに寄せる。どちらも
    最大 10 枚）。``strength`` は省略
    （``None``）でベースモデルの既定になる（LTX は 0.8）。MiniMax H3 には
    強度が無く、値を指定すると 422 FEATURE_UNSUPPORTED（``keyframe_strength``）
    になる。
    """

    image_id: str
    frame_idx: int = 0
    strength: float | None = None


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

    ``prompt`` は省略・空・空白だけなら、チェーン全体の共通プロンプトだけが
    使われる。指定すると、共通プロンプトの後ろに半角スペース1つで追記される
    （Stage-1 用）。Stage-2（全体の仕上げ）は共通プロンプトだけを使う。
    ``conditioning_images`` はクリップ0のみ有効（クリップ1以降は前クリップの
    潜在表現を引き継ぐ後続セグメントのため）。
    """

    prompt: str | None = None
    num_frames: int = 49
    conditioning_images: list[ConditioningImageArg] | None = None
