"""The Ref2VA references' latent cache. The body is ``engine/ref_latent_cache.py``,
shared by H3, LTX 2.3 and LTX 2.5; this module only re-exports it under the old name."""

from engine.ref_latent_cache import RefLatentCache, media_identity  # noqa: F401
