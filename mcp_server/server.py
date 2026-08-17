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

SERVER_NAME = "nz-ltx23"

INSTRUCTIONS = """\
Nz-LTX23 バックエンド（LTX 2.3 動画生成）を操作するためのツール群です。

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

■ end_source（素材（末尾））はクリップ1件のときのみ推奨
  submit_chain の end_source_video_id / end_source_image_id は、clips が1件
  （窓内モード）のときに使ってください。指定した素材の先頭フレームがクリップ
  自身の末尾として凍結され、1つのデノイズ窓の中でその素材へ到達するように
  生成されます（出力の長さは変わりません）。end_source_context_frames は
  8 を推奨します——実機比較で最良で、錨が長いほど窓を素材の再現に費やし、
  生成の創造性が下がります。既定の 72 は契約上の既定値であって推奨値では
  ありません。clips が2件以上のチェーンでの
  end_source は非推奨です（旧方式: 帯を後ろに継ぎ足すため、生成結果が素材へ
  クロスフェードしてしまう既知の問題があります）。指定自体は受理されますが、
  実験用途以外には使わないでください。

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
