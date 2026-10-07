"""Comfort-budget table (``limits.comfort_budgets``), §1-31 (2026-10-06).

The table's source of truth is the base-model manifests' ``comfort`` block
(``scripts/manifests/*.json``), validated and keyed by engine family at
startup (``api/context.py`` ``build_comfort_budgets``). ``config.yaml`` can no
longer override it: ``load_config`` drops the key with a WARNING.

Pins the SHIPPED rows (Docs/COMFORT_LIMIT_TABLE.md §1): every row carries
``requires.weight_class``; LTX 2.3's 4bit and Q6_K rows exist only for the
all-five-toggles-on configuration (the default configuration falls back to
``spill_free_frames``), its 8bit row and all LTX 2.5 rows are unconditional.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.context import build_comfort_budgets
from config import PROJECT_ROOT, LimitsConfig, load_config
from conftest import _make_args, base_model_descriptor, build_model_layout
from services.base_models import load_base_models

ALL_ON = {
    "attention_backend": "sage",
    "block_swap_prefetch": True,
    "keep_resident": True,
    "fused_gguf_dequant_kernel": True,
    "vae_mode": "prune_vaed",
}

#: A minimal valid comfort block for synthetic descriptors.
COMFORT = {
    "spatial_factor": 32,
    "temporal_factor": 8,
    "outpaint_budget": 1000,
    "rows": [
        {
            "requires": {"weight_class": "8bit", "keep_resident": True},
            "single_budget": 300,
            "chain_budget": 200,
        },
    ],
}


def _shipped():
    return build_comfort_budgets(load_base_models(PROJECT_ROOT / "scripts" / "manifests"))


def _rows(profile) -> list[tuple[dict, int, int]]:
    return [(row.requires, row.single_budget, row.chain_budget) for row in profile.rows]


# --- the shipped table --------------------------------------------------------- #


def test_family_id_set_matches_engine_registry():
    """The table's keys are exactly the engine family ids
    (services.engines.FAMILY_BY_ID), no more and no less."""
    from services.engines import FAMILY_BY_ID

    assert set(_shipped()) == set(FAMILY_BY_ID)


def test_ltx_rows():
    """LTX 2.3: 4bit and Q6_K only for all-on, 8bit unconditional
    (Docs/COMFORT_LIMIT_TABLE.md §1)."""
    assert _rows(_shipped()["ltx"]) == [
        ({"weight_class": "4bit", **ALL_ON}, 42840, 42240),
        ({"weight_class": "8bit"}, 32640, 32384),
        ({"weight_class": "q6k", **ALL_ON}, 43200, 40832),
    ]


def test_ltx_has_no_class_less_row_and_no_default_configuration_row():
    """LTX 2.3's default configuration has no token line (its boundary follows
    the VAE decode chunk count, Docs/COMFORT_LIMIT_TABLE.md §4.8): it must fall
    back to ``spill_free_frames``. So no row without a weight class, and the
    4bit/Q6_K rows carry the five all-on keys."""
    for requires, _single, _chain in _rows(_shipped()["ltx"]):
        assert "weight_class" in requires
        if requires["weight_class"] in ("4bit", "q6k"):
            assert {k: requires.get(k) for k in ALL_ON} == ALL_ON


def test_ltx25_rows_carry_only_the_weight_class():
    assert _rows(_shipped()["ltx25"]) == [
        ({"weight_class": "4bit"}, 46920, 46376),
        ({"weight_class": "8bit"}, 38760, 39424),
        ({"weight_class": "q6k"}, 43344, 43648),
    ]


def test_alpha_gen_budgets():
    """AlphaGen (POST /generate/alpha) one-stage line: provisional, exactly half
    of each LTX 2.5 row's single line (gate 0, VERIFICATION_LOG §151; owner
    ruling 2026-10-07). LTX 2.3 has no AlphaGen, so its rows carry None."""
    budgets = _shipped()
    assert [(r.requires, r.alpha_gen_budget) for r in budgets["ltx25"].rows] == [
        ({"weight_class": "4bit"}, 23460),
        ({"weight_class": "8bit"}, 19380),
        ({"weight_class": "q6k"}, 21672),
    ]
    for row in budgets["ltx25"].rows:
        assert row.alpha_gen_budget * 2 == row.single_budget
    assert [r.alpha_gen_budget for r in budgets["ltx"].rows] == [None, None, None]


def test_both_families_use_the_default_token_factors():
    budgets = _shipped()
    for family_id in ("ltx", "ltx25"):
        profile = budgets[family_id]
        assert (profile.spatial_factor, profile.temporal_factor) == (32, 8)


def test_outpaint_budgets_are_per_family():
    """Docs/COMFORT_LIMIT_TABLE.md §9: one line per family, not per weight class."""
    budgets = _shipped()
    assert budgets["ltx"].outpaint_budget == 42240
    assert budgets["ltx25"].outpaint_budget == 46080


def test_comfort_rows_do_not_carry_an_outpaint_budget_of_their_own():
    """J1 (Docs/VERIFICATION_LOG.md §98.11): the outpaint line is a fixed
    per-family value on the profile, never on a row."""
    for profile in _shipped().values():
        for row in profile.rows:
            assert not hasattr(row, "outpaint_budget")


def test_engine_comfort_profile_defaults_outpaint_budget_to_none():
    """An uncalibrated family must default to ``None`` ("no line")."""
    from config import EngineComfortProfile

    assert EngineComfortProfile().outpaint_budget is None


# --- config.yaml can no longer override it ------------------------------------ #


def test_code_default_is_empty():
    assert LimitsConfig().comfort_budgets == {}


def test_yaml_comfort_budgets_is_ignored_with_a_warning(tmp_path: Path, caplog):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "limits": {
                    "max_width": 1280,
                    "comfort_budgets": {
                        "ltx": {"rows": [{"requires": {}, "single_budget": 1, "chain_budget": 1}]}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="ltx.config"):
        cfg = load_config(cfg_path)
    assert cfg.limits.comfort_budgets == {}
    assert cfg.limits.max_width == 1280  # the rest of limits still applies
    warnings = [r for r in caplog.records if "comfort_budgets" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


# --- startup: descriptors -> GET /config -------------------------------------- #


def _boot(tmp_path: Path, descriptors: list[dict]):
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", **build_model_layout(tmp_path, descriptors)},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
        "tracking": {"backend": "mock"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return main.build_app(_make_args(cfg_path.as_posix()))


def _with_comfort(comfort, base_id: str = "LTX23") -> dict:
    descriptor = base_model_descriptor(base_id)
    descriptor["comfort"] = comfort
    return descriptor


def test_hermetic_client_serves_an_empty_table(client):
    """conftest's descriptor carries no ``comfort``: the table is ``{}``."""
    assert client.get("/api/v1/config").json()["limits"]["comfort_budgets"] == {}


def test_descriptor_comfort_reaches_get_config(tmp_path: Path):
    app = _boot(tmp_path, [_with_comfort(COMFORT)])
    with TestClient(app) as c:
        limits = c.get("/api/v1/config").json()["limits"]
    # model_dump fills the optional row key the synthetic block leaves out
    expected = copy.deepcopy(COMFORT)
    expected["rows"][0]["alpha_gen_budget"] = None
    assert limits["comfort_budgets"] == {"ltx": expected}
    # a bool survives JSON -> pydantic -> model_dump as a real bool
    assert limits["comfort_budgets"]["ltx"]["rows"][0]["requires"]["keep_resident"] is True
    # the compatibility values keep flowing unchanged
    assert limits["chain_comfort_token_budget"] == 40000
    assert limits["single_comfort_token_budget"] == 44880


@pytest.mark.parametrize(
    "weight_class",
    ["4-bit", "fp8", None],
    ids=["misspelled", "unknown", "missing"],
)
def test_bad_weight_class_fails_startup(tmp_path: Path, weight_class):
    comfort = copy.deepcopy(COMFORT)
    requires = comfort["rows"][0]["requires"]
    if weight_class is None:
        del requires["weight_class"]
    else:
        requires["weight_class"] = weight_class
    with pytest.raises(RuntimeError, match="weight_class"):
        _boot(tmp_path, [_with_comfort(comfort)])


def test_invalid_comfort_shape_fails_startup(tmp_path: Path):
    comfort = copy.deepcopy(COMFORT)
    comfort["rows"][0]["single_budget"] = "lots"
    with pytest.raises(RuntimeError, match="LTX23: comfort"):
        _boot(tmp_path, [_with_comfort(comfort)])


def test_comfort_that_is_not_an_object_fails_startup(tmp_path: Path):
    with pytest.raises(RuntimeError, match="'comfort' must be an object"):
        _boot(tmp_path, [_with_comfort([1, 2])])


def test_two_descriptors_of_one_family_with_comfort_fail_startup(tmp_path: Path):
    descriptors = [_with_comfort(COMFORT), _with_comfort(COMFORT, base_id="LTX23B")]
    with pytest.raises(RuntimeError, match="'ltx' already has a comfort table"):
        _boot(tmp_path, descriptors)


# --- config.yaml.example ------------------------------------------------------- #


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


def test_config_yaml_example_leaves_the_two_comfort_scalars_to_config_py():
    """``config.yaml.example`` keeps ``chain_comfort_token_budget`` and
    ``single_comfort_token_budget`` as commented-out examples only: a copied
    config.yaml that wrote them would freeze the values and stop future
    default changes from reaching the install (台帳 §1-72)."""
    example_path = PROJECT_ROOT / "config.yaml.example"
    raw = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert "chain_comfort_token_budget" not in raw["limits"]
    assert "single_comfort_token_budget" not in raw["limits"]
    published = load_config(example_path).limits
    defaults = LimitsConfig()
    assert published.chain_comfort_token_budget == defaults.chain_comfort_token_budget
    assert published.single_comfort_token_budget == defaults.single_comfort_token_budget
