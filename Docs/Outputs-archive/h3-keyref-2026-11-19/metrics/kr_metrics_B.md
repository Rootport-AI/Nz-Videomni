| ラン | フレーム数 | 先頭 PSNR | key との PSNR (1・4・8・17・24) | 隣の PSNR の最小 (1〜24) | 後ろの中央値 (25〜) | 印 | キーフレームの PSNR | キーフレーム付近の印 | 行数 text／cond／ref_video／ref_audio／video | 秒 encode／denoise／decode | VRAM の山 MB | L の自己検査 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_010_f17_harness | 124 | - | - | 20.91 | 24.56 | freeze@3 freeze@7 freeze@11 freeze@15 freeze@19 freeze@23 | 17: 28.75 | freeze@7(kf17) freeze@11(kf17) freeze@15(kf17) freeze@19(kf17) freeze@23(kf17) freeze@27(kf17) | 1086／960／950／0／35520 | 17.34／407.12／37.03 | 6345 | - |
| b1_010_f17_product | 124 | - | - | 20.91 | 24.56 | freeze@3 freeze@7 freeze@11 freeze@15 freeze@19 freeze@23 | 17: 28.75 | freeze@7(kf17) freeze@11(kf17) freeze@15(kf17) freeze@19(kf17) freeze@23(kf17) freeze@27(kf17) | 1086／960／950／0／35520 | 15.82／406.99／36.95※ | 6345※ | - |
| b2_001_f17 | 124 | - | - | 21.74 | 24.54 | freeze@3 freeze@7 freeze@11 freeze@15 freeze@19 freeze@23 | 17: 29.19 | freeze@7(kf17) freeze@11(kf17) freeze@15(kf17) freeze@19(kf17) freeze@23(kf17) freeze@27(kf17) | 1086／0／950／0／35520 | 17.03／392.38／37.17 | 6345 | ok (0 の行 960、一致 [True]) |
| b3_011_f17 | 124 | - | - | 23.87 | 25.14 | freeze@3 freeze@7 freeze@11 freeze@15 freeze@19 freeze@23 | 17: 29.28 | freeze@7(kf17) freeze@11(kf17) freeze@15(kf17) freeze@19(kf17) freeze@23(kf17) freeze@27(kf17) | 1086／960／950／0／35520 | 16.99／407.56／37.27 | 6345 | ok (0 の行 960、一致 [True]) |

※ 門の 2 本目（1 本目が確保した領域を使い回すので、ほかのランと比べられない）。

印: 隣り合うフレームの PSNR ÷ 同じランのフレーム 25 以降の中央値が 1.3 以上で freeze、0.7 以下で jump（「印」は 1〜24、「キーフレーム付近の印」は各キーフレームの前後 12 フレーム。kfN はキーフレームのフレーム番号）。PSNR は mp4 を ffmpeg で rgb24 にした全画素・全チャンネル、ピーク 255（同一フレームは 99.0）。キーフレームは製品と同じ前処理の画布。
