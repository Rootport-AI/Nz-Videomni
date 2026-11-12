# S3 summary (h3 fp8 / w4a8 spike, 2026-11-12)

| condition | load s | total s | s/step (mean) | peak VRAM alloc / reserved MiB | self-check | notes |
|---|---|---|---|---|---|---|
| int8 | 0.18 | 244.68 | 8.785 | 6332 / 7484 | True |  |
| fp8 | 0.14 | 288.98 | 11.753 | 6332 / 7484 | True |  |
| w4a8_kijai | 0.13 | 241.75 | 9.652 | 6332 / 7484 | True |  |
| w4a8_eros | 0.13 | 121.91 | 9.608 | 6332 / 7484 | True |  |
| fp8_dequant | 0.13 | 359.55 | 15.468 | 6332 / 7484 | True |  |
