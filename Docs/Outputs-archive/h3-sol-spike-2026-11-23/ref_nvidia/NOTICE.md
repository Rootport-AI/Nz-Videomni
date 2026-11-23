# 照合の基準の実装の出どころ

`decoder.py` は、NVIDIA の Sol-Engine の公開リポジトリから、手を加えずに写した 1 ファイルです（照合の基準として実行するためだけに置いています）。

- リポジトリ: GitHub の `NVlabs/Sana`（取得したときの木の識別子 `cf67a3acad7511acb85053e56a05e5dd24c82ba0`）
- パス: `models/Open-H3-Regenerate-2K/open_h3/fast_vae/decoder.py`
- SHA-256: `87f25f35bb1ceebbcae4abcfb0439d522dfbc0c233d3fce84a3e19f12ce242de`
- 著作者: NVIDIA（StreamRender-H3 の高速な動画VAE の作業。ファイルの先頭の注記のとおり）
- ライセンス: Apache License 2.0（リポジトリの README の表示による）。このファイル自体にはライセンスの注記がありません。取得のときにリポジトリの LICENSE の本文を写せなかった（中身が「404: Not Found」だった）ため、本文の確認は製品への実装を相談する前にやり直します（計画の付録の「ライセンス」の項）。

使い方の約束:

- 台本は、このファイルの `NonCausalViTDecoder` だけを、パスを指定して直接読み込みます。同じリポジトリの高速版の系統（`noncausal_fast3.py` など、追加のライブラリを読みに行く部分）は写していませんし、読み込みません。
- このファイルが読み込むのは `torch` と `math` だけです。
