"""Vendored Video-Depth-Anything (Small / vits) inference core.

Third-party code — see ``README.md`` for provenance and the full list of
modifications, and ``LICENSE`` for the Apache-2.0 text it ships under.

Nothing here is imported at package-import time: ``engine/preprocess/depth.py``
does ``from engine.preprocess.vda.video_depth_anything.video_depth import
VideoDepthAnything`` LAZILY inside ``_ensure_loaded`` so that a canny/pose-only
job (or an environment without torchvision) never pays for — or fails on — this
subtree.
"""
