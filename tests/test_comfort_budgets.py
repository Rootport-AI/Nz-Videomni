"""Comfort-budget table (``LimitsConfig.comfort_budgets``), 2026-08-31.

Pins the default table's shape and the one intentional asymmetry in it: "ltx"
(LTX 2.3) carries no empty-``requires`` row because its default (non-pruned
VAE decoder) configuration's comfort boundary is NOT monotone in tokens (see
``config._default_comfort_budgets`` docstring and Docs/COMFORT_LIMIT_TABLE.md
§4.8 for the calibration itself). A client with no matching row is expected
to fall back to the legacy ``spill_free_frames`` table instead, so "ltx"
getting a default row here would be a regression, not a fix.

No validators are added (the file has none; the frontend normalizes), so
these tests only pin the DEFAULT shape plus a plain yaml-override smoke test.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from config import PROJECT_ROOT, AppConfig, load_config

LTX_REQUIRES = {
    "attention_backend": "sage",
    "block_swap_prefetch": True,
    "keep_resident": True,
    "fused_gguf_dequant_kernel": True,
    "vae_mode": "prune_vaed",
}


def _default_budgets():
    return AppConfig().limits.comfort_budgets


def test_family_id_set_matches_engine_registry():
    """The table's keys are exactly the engine family ids
    (services.engines.FAMILY_BY_ID), no more and no less.

    ``services.engines`` is cheap to import here: it only holds id ->
    (module path, class name) string tables and imports each adapter
    lazily (see the module's own docstring), so this doesn't drag in the
    ~2000-line LTX adapters or their PIL/chain_math/api.models chain."""
    from services.engines import FAMILY_BY_ID

    assert set(_default_budgets()) == set(FAMILY_BY_ID)


def test_ltx_has_exactly_one_row_with_the_five_calibrated_requires():
    ltx = _default_budgets()["ltx"]
    assert len(ltx.rows) == 1
    row = ltx.rows[0]
    assert row.requires == LTX_REQUIRES
    assert (row.single_budget, row.chain_budget) == (44880, 40000)


def test_ltx_has_no_empty_requires_row():
    """The deliberate asymmetry: LTX 2.3's default configuration has no safe
    token ceiling (see module docstring), so it must NOT get a catch-all row.
    A future edit adding one here would silently make the marker "smart" for
    the non-monotone default configuration, which is the exact regression
    this table's design forbids."""
    ltx = _default_budgets()["ltx"]
    assert not any(row.requires == {} for row in ltx.rows)


def test_ltx25_has_one_unconditional_row():
    ltx25 = _default_budgets()["ltx25"]
    assert len(ltx25.rows) == 1
    row = ltx25.rows[0]
    assert row.requires == {}
    assert (row.single_budget, row.chain_budget) == (44880, 44880)


def test_both_families_use_the_default_token_factors():
    budgets = _default_budgets()
    for family_id in ("ltx", "ltx25"):
        profile = budgets[family_id]
        assert profile.spatial_factor == 32
        assert profile.temporal_factor == 8


def test_default_profiles_publish_the_2026_09_05_outpaint_budgets():
    """§3-135: the Outpainting (Edit tab) comfort-token lines from the
    2026-09-05 all-on recalibration (Docs/COMFORT_LIMIT_TABLE.md §9) --
    ltx=42,240 (unchanged from the prior calibration), ltx25=46,080 (the one
    line that moved)."""
    budgets = _default_budgets()
    assert budgets["ltx"].outpaint_budget == 42240
    assert budgets["ltx25"].outpaint_budget == 46080


def test_comfort_rows_do_not_carry_an_outpaint_budget_of_their_own():
    """J1 (2026-09-05, Docs/VERIFICATION_LOG.md §98.11): the outpaint line is
    a fixed per-family value, not a per-row one -- it must not ride along on
    ``ComfortRow`` (which is keyed by acceleration ``requires``), only on the
    profile itself."""
    budgets = _default_budgets()
    for family_id in ("ltx", "ltx25"):
        for row in budgets[family_id].rows:
            assert not hasattr(row, "outpaint_budget")


def test_engine_comfort_profile_defaults_outpaint_budget_to_none():
    """An uncalibrated family (no ruling yet) must default to ``None`` --
    "no line": the client draws no outpaint comfort warning at all, and a
    future family must NOT inherit the two calibrated families' numbers (see
    ``EngineComfortProfile.outpaint_budget`` docstring, J1's "do not copy the
    precedent")."""
    from config import EngineComfortProfile

    assert EngineComfortProfile().outpaint_budget is None


def test_yaml_can_override_the_comfort_budgets_table(tmp_path: Path):
    """``config.yaml`` overriding ``limits.comfort_budgets`` replaces the
    whole table (no per-row merge) -- same overwrite discipline as every
    other dict-valued limits field (see spill_free_frames tests).

    ``requires`` is written as real YAML (a quoted string plus a boolean,
    not a Python literal pasted into a dict) so this also pins that
    ``keep_resident``'s ``bool`` survives yaml -> pydantic -> ``model_dump``
    (the JSON wire shape) as an actual bool, not ``1``/``"True"``/etc."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "limits": {
                    "comfort_budgets": {
                        "ltx": {
                            "spatial_factor": 16,
                            "temporal_factor": 4,
                            "rows": [
                                {
                                    "requires": {
                                        "attention_backend": "sage",
                                        "keep_resident": True,
                                    },
                                    "single_budget": 1000,
                                    "chain_budget": 500,
                                }
                            ],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert set(cfg.limits.comfort_budgets) == {"ltx"}
    ltx = cfg.limits.comfort_budgets["ltx"]
    assert (ltx.spatial_factor, ltx.temporal_factor) == (16, 4)
    assert len(ltx.rows) == 1
    assert (ltx.rows[0].single_budget, ltx.rows[0].chain_budget) == (1000, 500)
    req = ltx.rows[0].requires
    assert req["attention_backend"] == "sage"
    assert req["keep_resident"] is True
    dumped = cfg.model_dump()["limits"]["comfort_budgets"]["ltx"]["rows"][0]["requires"]
    assert dumped["keep_resident"] is True


def test_config_yaml_example_spill_free_frames_matches_the_2026_08_31_recalibration():
    """``config.yaml.example`` is the template new installs copy from
    (setup.bat) -- it must carry the same re-measured legacy fallback table
    as the repository's own config.yaml, not the pre-2026-08-31 values."""
    example_path = PROJECT_ROOT / "config.yaml.example"
    raw = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert raw["limits"]["spill_free_frames"] == {
        "512x320": 481,
        "960x576": 481,
        "1280x768": 273,
        "1920x1088": 161,
        "2560x1472": 81,
    }
