# Spark-H3 の ComfyUI 版のバックエンド（作者のファイルの写し）

このフォルダーのファイルは、Spark-H3 の作者が公開した配布物から、**1 バイトも変えずに**写したものです。製品の側の糊は `engine_h3/spark_h3.py` にあり、このフォルダーのファイルは、その糊からだけ読み込まれます。

## 出どころ

- 作者のリポジトリ: GitHub `zechengtang/Spark-H3`（https://github.com/zechengtang/Spark-H3）
- リリースのタグ: `comfyui-v0.2.0`
- 写した元の ZIP: `ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm89-cu130.zip`（SHA-256 `d610d19859fffe0eda69f0075749c4b4c1f5cd0ff0bc4967cbfac3466e007391`。作者の `SHA256SUMS.txt` の値と一致）。ZIP の中の `ComfyUI-Spark-H3/` の下から写しました。
- RTX 30 系向け（`…-sm86-cu130.zip`）と RTX 50 系向け（`…-sm120-cu130.zip`）の ZIP の同じファイルも、下の表と SHA-256 が一致します（3 つの ZIP で同じファイルです）。

## ライセンス

Apache License 2.0。全文は、このフォルダーの `LICENSE`（作者の配布物の `LICENSE` そのもの）にあります。作者の `pyproject.toml` も `license = {file = "LICENSE"}`・`License :: OSI Approved :: Apache Software License` としています。

## 変えていないこと

- 下の表のファイルは、中身・改行・名前を変えていません。リポジトリの `.gitattributes` で `engine_h3/third_party/spark_h3/** -text` とし、git の改行の変換でも変わらないようにしています。
- 製品が足したのは、説明文だけの `__init__.py`（このフォルダーと、1 つ上の `engine_h3/third_party/`）と、この `README.md` だけです。
- 写したファイルと表の SHA-256 が一致することは、自動テスト `tests/test_spark_h3_vendored.py` が確かめます。

## ファイルごとの SHA-256

| ファイル | 大きさ（バイト） | SHA-256 |
|---|---|---|
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `comfyui_backend.py` | 11,340 | `19ffe8a3bb6a38190d0efc206e54a5b534544d2f56ff749dad979d7f6fe4d346` |
| `comfyui_reblock_plan.py` | 18,033 | `317e51dc30b1ec713bf12ee93ca27bdbde220d2a2979b8fd131008de7e475c12` |
| `h3_sparse_attention/__init__.py` | 65 | `488b033d926aecc778b04937866ff4766ea5f262097bb99c0afe78e77fffcdab` |
| `h3_sparse_attention/landmark_direction.py` | 6,942 | `ae9420f9fc068077f0bd1dae58f7d76969a5fce4a9eaa95a06e50114bbf3a2f2` |
| `h3_sparse_attention/landmark_initial_order.py` | 2,766 | `9bb6fb7d93071e908c7ecfc3ebf4ffc157274fe69bfd6527ac1d2ed377f8be75` |
| `h3_sparse_attention/landmark_lazy_k.py` | 9,563 | `d2b5f80d01991b55e4458751694046d2cbb7d5e8384db3993b8378a1271b7011` |
| `h3_sparse_attention/landmark_projection.py` | 5,162 | `a5a813c1d37a842be3dc519094581a78bb8b8aabe5725b478096aa96dbeaad49` |
| `h3_sparse_attention/landmark_tree_clustering.py` | 23,739 | `7826a0e8b23b8f3509f7edd43d886008aebfffd94aaa659e94b706e210894ba9` |
| `h3_sparse_attention/landmark_tree_triton.py` | 52,933 | `9245fd89817b10bd3c578b97f5adc386169cc77ae687f60d7b6eeb0355d00f87` |
| `h3_sparse_attention/landmark_tree_v2.py` | 54,866 | `9668ecdd8caf9945e72e7cf2124be9b3a094890ffd3f54801601f663f5e76183` |
| `h3_sparse_attention/landmark_tree_v2_triton.py` | 21,449 | `da09bf4bc7412ca1d98a316bebf1f24680e39721773eef521312e2c5139e8e0f` |
| `h3_sparse_attention/landmark_v2_cosine.py` | 9,014 | `9c3e1f98883fc4cc2a5ab929f0bd0941f3470f09cb95109d724ebf2b38a263bb` |
| `h3_sparse_attention/landmark_v2_cosine_fast.py` | 15,408 | `e953dadcc7bfaee66c7dd181ad90831243ce221eff74c48b2fbc08cea21af65d` |
| `h3_sparse_attention/landmark_v2_cosine_triton.py` | 7,214 | `b0ecff71afef9bd999afe3c07e13debb982b27a70cb6f970406f0a9f56896870` |
| `h3_sparse_attention/landmark_v2_euclidean.py` | 4,416 | `0acb54fc48d49903ec6efddfcae532f84f54b319b403f63591068c5f31daa858` |
| `h3_sparse_attention/landmark_v2_fused_node.py` | 22,224 | `728c0cb7dce46b142b2be6bad4520d9559d292ba2e5c4b0313205acf4eeef6e6` |
| `h3_sparse_attention/landmark_v2_max.py` | 6,216 | `98294f92414ffe94c3a6a4f3312d2e2d371b6dd69c16ef933bc30dcd4005d810` |
| `h3_sparse_attention/landmark_v2_order.py` | 1,542 | `f83f845da11d845b0c6e11398608f5623a2b5f64dd3611b12abf720035f171c8` |
| `h3_sparse_attention/landmark_v2_route.py` | 12,029 | `db6d7a77a5c27d4db121c6d99b5bfcfd265af4a8adbcb37ff4af20e71280d705` |
| `h3_sparse_attention/landmark_v2_terminal.py` | 10,213 | `12c28f8f58475255291cc88a8d897deaee5177d9d874fab3972babd8fbe12ef3` |
| `h3_sparse_attention/mahalanobis_kmeans.py` | 17,928 | `87ebe1f714915ab5892a218445eb935c56ce60ac8380ed67932dc8bac60a76dd` |
| `h3_sparse_attention/reblock_hierarchy.py` | 10,653 | `89f79ed7e8911616779dcc5060aebc031c7cb5b875a41b3aff176c608abe2471` |

## 持ち込んでいないファイル

ZIP の中の次のファイルは、写していません。

- ComfyUI を読み込むもの: `__init__.py`（ComfyUI の入口）、`comfyui_nodes.py`（ノード。製品は、設定の値と呼び出し方を突き合わせるために読んだだけです）、`comfyui_spark_split.py`、`comfyui_dmad.py`。
- 導入や組み立ての道具: `install.py`・`kernel_builder.py`・`package_manager.py`・`scripts/`・`native/`・`patches/`・`pyproject.toml`・`requirements.txt`・`spark_h3_build.json`。
- ComfyUI の画面やワークフロー: `web/`・`workflows/`。
- そのほか: `assets/`・`docs/`・`README.md`・`INSTALL.zh-CN.md`・`convert_larryvrh_lora_comfyui.py`。
- 組み立て済みの部品 `wheelhouse/spark_h3_kitchen-…whl`。部品は、インストーラーが作者の GitHub の Releases から取得して H3 の仮想環境に入れます（再ホストはしません）。

## 作者のファイルの振る舞いで、知っておくこと

- `comfyui_reblock_plan.py` は、パッケージの中からの相対の import（`from .h3_sparse_attention…`）に失敗すると、自分のフォルダーを `sys.path` の先頭に足して、最上位の `h3_sparse_attention` として読み込み直します（作者の単体テストのための道）。製品の糊は、使えるかの検査（`engine_h3.spark_h3.probe`）で、その計画が使うモジュールをパッケージの中の名前で先に読み込み、この道に入らないようにしています。
- `h3_sparse_attention/` のいくつかのモジュールは、読み込んだ時点で `H3_LMV2_…` などの環境変数を読みます。製品は、これらの環境変数を設定しません（作者の既定のまま動きます）。
