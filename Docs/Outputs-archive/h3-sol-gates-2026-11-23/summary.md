# Sol-H3-VAE 段 1 の実機の確認（記録）

| 項目 | 内容 | 判定 |
|---|---|---|
| S1 | 製品の復号器と実験の構成 B の一致（実験の潜在 5 本） | pass |
| S2 | 製品の経路で Sol（T2V・連結 2 本・V2V・Inpainting。Turbo LoRA 0.8・4 ステップ） | pass |
| S3 | ふだんの使い方（Turbo LoRA 0.8・4 ステップ・1280×768・124 フレーム × 5 本）の Off と Sol | pass |
| S4 | 切り替え（資産のパスを無いファイルに向けた記述子の写し） | pass |
| S5 | Off の経路（F0: ワーカーの自己試験の照合値／F1: R1 の条件の API） | pass |
| S6 | 壊れたファイル（実物の先頭 64 MiB の写し） | pass |
| S7 | 常駐 on で Off → Sol → Off | pass |
| S8 | 製品の経路での VRAM（2560×1472・192 フレーム × 2 本・Sol） | fail |
| S8_off | S8 の対照（同じ条件で sol_vae を指定しない Off） | pass |
| S9 | Sol の重みの読み込みの時間 | pass |
| S10 | 上限の境目（2560×1472。T'=112 は Sol、T'=117 は too_large） | pass |

合格の条件:

- S1: 5 本とも uint8 が 1 ビットまで一致し、製品の映像の MD5 が実験の B.mp4 と一致
- S2: 4 本とも完了し、sol_vae_used が on、vae_source が sol、フレーム数と大きさが合う
- S3: 2 本とも完了（532 フレーム・音声）。Sol は sol_vae_used on。復号の段の秒と VRAM の山を記録
- S4: 2 本とも完了し、Sol のジョブは on->off（理由 file_missing）、映像のストリームが同じシードの Off と一致
- S5: F0 の mp4 の sha256 が 0da6111c…、F1 の映像と音声のストリームが R1 の記録と一致し、記録に sol_vae の欄が無い
- S6: テキストエンコーダーより前に失敗し、文が「Sol-H3-VAE のファイル（…）の照合で、合わない所が 1 か所…」
- S7: 3 本とも完了し、vae_source が file → sol → resident、Sol は on
- S8: 完了し、sol_vae_used on。復号の段の VRAM の山（確保と予約）と、共有 GPU メモリの増分が 0 付近
- S8_off: 完了し、sol_vae_used が無い。共有 GPU メモリの時系列を S8 と比べる（判定は記録だけ）
- S9: 記録だけ（OS のファイルのキャッシュを通さない読み取りの秒と、from_file の 3 回の秒）
- S10: 内側は Sol で完走し共有 GPU メモリの増分が 0 付近、超えたほうは通常の VAE（reason too_large）で完走
