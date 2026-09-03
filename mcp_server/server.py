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
  LTX 2.5 で submit_generate が使えないのは vae_mode（既定値以外）だけです
  （422 FEATURE_UNSUPPORTED）。nag_enabled（ネガティブプロンプト。NAG と VSF）
  は 2026-08-30 から LTX 2.5 でも使えます——negative_prompt / nag_scale /
  nag_tau / nag_alpha / neg_method / vsf_scale も同時に効くようになりました。
  ただし nag_alpha=0 にしても「NAG なし」とビット単位で同じ絵にはならないので、
  無効化したいときは nag_enabled を false にしてください。
  keep_resident（モデル骨格の常駐）は 2026-08-25 から LTX 2.5 でも使えます
  （既定 off のまま。ただし LTX 2.3 とは常駐する中身が違い、2.5 が抱えるのは
  テキストエンコーダの重みだけで約7.7GiBです）。
  keep_resident_embeddings（埋め込み処理器の常駐）は LTX 2.5 専用です
  （既定 off。実測4.66GiB。keep_resident とは別のスイッチで、両方 on にすると
  メモリ増分は加算されます）。これだけは向きが逆で、LTX 2.3 を選んでいるとき
  に true を送ると 422 FEATURE_UNSUPPORTED になります——「LTX 2.5 で使えない
  機能」ではなく「LTX 2.3 で使えない機能」の1つ目です。
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
  まだ使えないのは vae_mode の1つだけです（keep_resident・attention_backend・
  nag_enabled は submit_chain でも使えます。nag_enabled は 2026-08-30 から
  LTX 2.5 でも使えるようになりました）。keep_resident_embeddings は
  submit_chain でも LTX 2.5 専用で、LTX 2.3 では 422 になります。
  撮り直し（Retake）も 2026-09-01 から submit_chain の引数として使えます
  （retake_video_id ほか5引数。LTX 2.3 / LTX 2.5 のどちらでも使えます）。
  同じ日に、画角拡張（Outpainting）も submit_generate の引数として使える
  ようになりました（outpaint_pad_* ほか6引数）。どちらも下の専用の節を
  読んでください。

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

■ 撮り直し（Retake）— submit_chain の retake_video_id ほか5引数
  既に手元にある動画の「まん中」だけを作り直す機能です（時間方向の
  inpainting）。retake_video_id（upload_video で取得）と
  retake_window_start_sec（作り直す窓の開始秒）を指定すると、サーバーが
  その窓だけを切り出して作り直し、窓まるごとを返します。
  **窓の長さを決めるのは clips[0].num_frames ただ1つで、長さを表す第2の
  引数は存在しません。** clips はちょうど1件・窓長は既定のstage-2窓で
  [73, 169] フレーム（get_config の limits.retake_window_min_frames /
  retake_window_max_frames）です。
  **retake_window_start_sec には既定値がありません**——retake_video_id を
  指定して省略すると、submit_chain がPOST前にエラーにします。
  **窓の開始秒を決める下調べには upload_video(file_path, max_frames=...)
  を使ってください**——max_frames を渡したときだけ、保存された動画の
  frame_count と fps が実測されて返ります（max_frames は「先頭Nフレームだけ
  残す」引数でもあるので、測るだけのときは元の尺より確実に大きい値を
  渡してください）。
  糊しろの retake_head_px（8n+1・既定25）と retake_tail_px（8の倍数・
  既定24）は較正済みの推奨値で、**広げれば良いというものではありません**。
  source_video_id / source_audio_id / reference_video_id / end_source_* /
  clips[0].conditioning_images とはすべて排他です。

■ 画角拡張（Outpainting）— submit_generate の outpaint_pad_* ほか6引数
  手元の動画の外側を描き足して画角を広げる機能です。
  outpaint_pad_left / _right / _top / _bottom（px）のいずれかを0より大きく
  すると有効になり、4辺すべて0なら通常の生成のままです。
  **width / height は「拡張後の最終キャンバス」であり、パディングはその
  内側から切り出されます。** 元動画の解像度と一致しなければならないのは
  「残す領域」（width - pad_left - pad_right × height - pad_top -
  pad_bottom）のほうで、サーバーが実測して照合します。
  幾何条件は4つです: width/height は128の倍数・残す領域が元動画の解像度と
  一致・残す領域は縦横とも256px以上・元動画のフレーム数が num_frames 以上。
  reference_video_id（広げる対象の元動画）が必須で、conditioning_images と
  crop_width/crop_height とは排他です。
  **in-outpainting という制御系LoRAが1本だけ必要で、submit_generate が自動で
  loras へ追加します**（既に同名を入れていれば何もしません）。
  **in-outpainting が導入されていない環境では404になる**ので、事前に
  list_loras で存在を確認してください。
  **upload_video の応答は解像度を返しません。** キャンバス（128の倍数）を
  逆算するには、アップロード前にローカルで ffprobe 等により元動画の解像度を
  確認してください。
  outpaint_blend_dilation_stage1（なじみ幅の段数、既定5）は stage 2 が5:2の
  比で自動追従するので、指定するのは stage 1 だけです。

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
