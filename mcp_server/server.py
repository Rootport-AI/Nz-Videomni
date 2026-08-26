"""FastMCP server construction + stdio entrypoint.

NEVER ``print()`` anywhere under ``mcp_server/`` (here or in any ``tools/*``
module): the stdio transport carries JSON-RPC on stdout, so any stray byte on
that stream corrupts every message after it. All logging in this process goes
to stderr instead (``logging.basicConfig(stream=sys.stderr, ...)`` in
:func:`main`).
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import register_all

SERVER_NAME = "nz-videomni"

INSTRUCTIONS = """\
Nz-Videomni バックエンド（LTX 2.3 / LTX 2.5 動画生成）を操作するためのツール群です。

■ ベースモデル（LTX 2.3 / LTX 2.5）の切り替え
  バックエンドは複数のベースモデル（推論エンジンごと入れ替わる大枠）を持ち、
  一度に1つだけが選択されています。list_models の base_models[] で、導入済みか
  （installed）・現在選択中か（active）・そのベースモデルで使えない機能
  （unsupported_features）を確認してください。切り替えは
  load_pipeline(base_model="LTX25") のように id を渡します。切り替えは
  ワーカーの載せ替えを伴い数秒〜十数秒かかるので、backend_status の
  status.state が ready になったことを確認してから生成を投げてください。
  切り替え直後の1本目の生成はキャッシュが冷えていて通常の約2倍かかります
  （wait_for_job がタイムアウトしても失敗ではないので呼び直してください）。
  LTX 2.5 では submit_generate の nag_enabled / vae_mode（いずれも既定値以外）
  が使えません（422 FEATURE_UNSUPPORTED）。
  keep_resident（モデル骨格の常駐）は 2026-08-25 から LTX 2.5 でも使えます
  （既定 off のまま。ただし LTX 2.3 とは常駐する中身が違い、2.5 が抱えるのは
  テキストエンコーダの重みだけで約7.7GiBです）。
  attention_backend（SageAttention）も 2026-08-25 から LTX 2.5 で使えます
  （既定 "sdpa" のまま）。ただしこれは他の高速化と違い、"sage" にすると
  同じシードでも生成結果の細部が変わります。速さは動画の大きさに強く依存し、
  1280x768 の連結生成で約1.10倍、512x320 級では効かないか、かえって遅く
  なることがあります。sageattention が入っていない環境では 422 にはならず
  自動的に "sdpa" へ降格して完走します（この降格の規律は 2.3 と同じです）。
  loras（スタイルLoRA・制御系IC-LoRA）と reference_video_id、および
  conditioning_attention_strength / reference_video_strength は LTX 2.5 でも
  使えます。submit_chain は連結生成そのものに加えて
  source_video_id（V2V継続）と source_audio_id（A2V。複数クリップにまたがる
  長尺A2Vも含みます）、loras と reference_video_id（複数クリップにまたがる
  長尺IC-LoRAも含みます）が使えます。end_source_video_id /
  end_source_image_id（素材（末尾））も 2026-08-26 から LTX 2.5 で使えます
  ——この日に撮り直し（Retake）と素材（末尾）が開通し、submit_chain が
  投げられるモードは LTX 2.5 でも全部通るようになりました。submit_chain で
  まだ使えないのは nag_enabled / vae_mode の2つだけです
  （keep_resident と attention_backend は submit_chain でも使えます）。
  なお撮り直しは、そもそも submit_chain に引数がありません（LTX 2.3 でも
  同じで、2.5 で失われた機能ではありません）。

■ 同時実行は1ジョブまで
  バックエンドは Phase 1 の制約として、生成ジョブを同時に1本しか実行できません。
  ジョブが進行中に新しい submit_generate / submit_chain を呼ぶと 409 JOB_BUSY
  エラーになります。先に job_status か wait_for_job で完了を確認してから次を
  投げてください。pipeline の読み込み（load_pipeline）も同様にジョブ実行中は
  できません。

■ 基本の流れは「投げて→待つ」（submit + poll）
  生成は時間がかかるため、submit_generate / submit_chain はジョブを登録して
  すぐ job_id を返すだけです。結果は wait_for_job（または job_status を繰り
  返し呼ぶ）で確認してください。wait_for_job はタイムアウトしてもエラーには
  せず、その時点の進捗を timed_out フラグ付きで返します。

■ 動画はローカルの絶対パスで返る
  生成された動画は base64 などで埋め込まれず、常にサーバー機上の絶対パスで
  返されます（save_job_video で任意のフォルダへコピーも可能）。パスは
  MCPサーバーを動かしているマシン上のものです。

■ end_source（素材（末尾））はクリップ何件でも使えますが、1件を推奨します
  submit_chain の end_source_video_id / end_source_image_id を指定すると、
  clips が**1件のときは窓内モード**（素材がそのクリップ自身の末尾として凍結され、
  1つのデノイズ窓の中で素材へ到達するように生成されます。**推奨**）、
  **2件以上のときは逆順Chained**（最後のクリップから順に生成し、各クリップは
  1つ後ろのクリップの冒頭を自分の末尾として引き継ぎます。**受理されますが
  推奨外**——実機ゲート後のオーナー目視・試聴で、クリップの境目・末尾（錨直前）に
  映像のモーフや音楽の不統一といった品質劣化が出ることを確認しており、これは
  仕様として許容しています）になります。出力の長さはどちらの場合もクリップの
  合計であって、素材の分だけ伸びることはありません。品質を重視して複数クリップを
  終端付きで繋ぎたい場合は、end_source をクリップ1件ずつ使い、生成物を次の
  素材にして過去へ遡って生成する手動リレー（AviUtl2タイムライン側で組み合わせる）
  が実用的な回避策です。
  end_source_context_frames は 8 を推奨します——実機比較で最良で、錨が長い
  ほど窓を素材の再現に費やし、生成の創造性が下がります。既定の 72 は契約上の
  既定値であって推奨値ではありません。
  2件以上のときは source_video（素材（冒頭））との併用が拒否されます（422）。
  併用したい場合はクリップを1件にしてください。

■ AviUtl2 のタイムライン連携は対象外
  このツール群はバックエンドの生成・ジョブ管理・バッチ計画のみを扱います。
  AviUtl2 側のタイムラインへの配置や編集操作は行いません。
"""


def build_server() -> FastMCP:
    """Construct the FastMCP server with every tool registered."""
    mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    register_all(mcp)
    return mcp


def main() -> None:
    # stream=sys.stderr is load-bearing: the default StreamHandler target is
    # stderr already, but this is spelled out explicitly so nobody "fixes" it
    # to stdout later and silently kills the JSON-RPC stream.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    build_server().run()  # stdio transport (FastMCP default)


if __name__ == "__main__":
    main()
