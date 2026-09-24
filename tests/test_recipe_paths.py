"""services/recipe_paths.relativize_recipe_paths (台帳 §3-164).

Absolute local paths in the recipe become store-relative (``outputs/...``,
``uploads/...``), the same shape as ``output.path``. Pure function, no I/O
besides ``Path.resolve`` on the two base directories.
"""

from __future__ import annotations

from pathlib import Path

from services.recipe_paths import relativize_recipe_paths


def _rel(obj, tmp_path):
    return relativize_recipe_paths(
        obj, output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )


# ------------------------------------------------------------ the four leaves


def test_relative_value_is_unchanged(tmp_path):
    obj = {"output": {"path": "outputs/job1/output.mp4"}}
    assert _rel(obj, tmp_path) == obj


def test_under_output_dir(tmp_path):
    value = str(tmp_path / "outputs" / "job1" / "inpaint_canvas.mp4")
    assert _rel({"canvas_path": value}, tmp_path) == {
        "canvas_path": "outputs/job1/inpaint_canvas.mp4"
    }


def test_under_upload_dir(tmp_path):
    value = str(tmp_path / "uploads" / "audios" / "abc" / "input.wav")
    assert _rel({"source_audio_path": value}, tmp_path) == {
        "source_audio_path": "uploads/audios/abc/input.wav"
    }


def test_elsewhere_keeps_from_the_last_store_component(tmp_path):
    # A job recorded before a rename/move: the path is outside both stores.
    old = "Z:\\old\\Nz-LTX23-backend\\uploads\\videos\\v1\\input.mp4"
    assert _rel({"source_path": old}, tmp_path) == {"source_path": "uploads/videos/v1/input.mp4"}
    # The LAST store component wins.
    nested = "Z:\\outputs\\proj\\uploads\\masks\\m1\\input.mp4"
    assert _rel({"mask_path": nested}, tmp_path) == {"mask_path": "uploads/masks/m1/input.mp4"}


def test_elsewhere_without_a_store_component_keeps_the_file_name(tmp_path):
    assert _rel({"source_path": "Z:\\somewhere\\clip.mp4"}, tmp_path) == {
        "source_path": "clip.mp4"
    }


# ---------------------------------------------------------- shape and scope


def test_recursion_over_dicts_and_lists(tmp_path):
    up = str(tmp_path / "uploads" / "images" / "i1" / "input.png")
    obj = {
        "a2v": {"source_audio_path": up},
        "items": [{"mask_path": up}, [{"canvas_path": up}], "untouched"],
    }
    assert _rel(obj, tmp_path) == {
        "a2v": {"source_audio_path": "uploads/images/i1/input.png"},
        "items": [
            {"mask_path": "uploads/images/i1/input.png"},
            [{"canvas_path": "uploads/images/i1/input.png"}],
            "untouched",
        ],
    }


def test_none_and_non_path_keys_pass_through(tmp_path):
    absolute = str(tmp_path / "uploads" / "x.wav")
    obj = {"source_path": None, "prompt": absolute, "path_count": 3, "paths": [absolute]}
    assert _rel(obj, tmp_path) == obj


def test_input_is_not_mutated(tmp_path):
    value = str(tmp_path / "outputs" / "j" / "a.mp4")
    obj = {"inpaint": {"canvas_path": value}}
    _rel(obj, tmp_path)
    assert obj == {"inpaint": {"canvas_path": value}}


# ------------------------------------------------------------- edge cases


def test_different_drive_does_not_raise(tmp_path):
    # relative_to across drives raises ValueError; that must fall through to
    # rule 4, not escape.
    drive = "Q:" if not str(tmp_path).upper().startswith("Q:") else "R:"
    value = f"{drive}\\data\\outputs\\job9\\output.mp4"
    assert _rel({"source_path": value}, tmp_path) == {"source_path": "outputs/job9/output.mp4"}


def test_backslash_rooted_path_without_a_drive_goes_to_rule_4(tmp_path):
    # On Windows a leading-backslash path has no drive, so ``Path.is_absolute()``
    # is False; it still has a root, so it is not treated as relative and the
    # user-name part is not kept.
    value = "\\Users\\someone\\uploads\\a.wav"
    assert _rel({"source_path": value}, tmp_path) == {"source_path": "uploads/a.wav"}


def test_nested_stores_use_the_deeper_base(tmp_path):
    # upload_dir inside output_dir: an upload maps to ``uploads/...``, while a
    # file directly under the output store still maps to ``outputs/...``.
    output_dir = tmp_path / "outputs"
    upload_dir = output_dir / "uploads"
    upload_value = str(upload_dir / "audios" / "a1" / "input.wav")
    output_value = str(output_dir / "job1" / "output.mp4")
    result = relativize_recipe_paths(
        {"source_audio_path": upload_value, "canvas_path": output_value},
        output_dir=output_dir,
        upload_dir=upload_dir,
    )
    assert result == {
        "source_audio_path": "uploads/audios/a1/input.wav",
        "canvas_path": "outputs/job1/output.mp4",
    }


def test_base_given_unresolved_still_matches_the_resolved_value(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "sub").mkdir()
    unresolved_output_dir = tmp_path / "sub" / ".." / "outputs"
    value = str((tmp_path / "outputs" / "j1" / "output.mp4").resolve())
    result = relativize_recipe_paths(
        {"canvas_path": value},
        output_dir=unresolved_output_dir,
        upload_dir=tmp_path / "uploads",
    )
    assert result == {"canvas_path": "outputs/j1/output.mp4"}


def test_windows_case_difference_is_absorbed(tmp_path):
    value = str(tmp_path / "uploads" / "a" / "input.wav").upper()
    result = _rel({"source_audio_path": value}, tmp_path)
    assert result["source_audio_path"].lower() == "uploads/a/input.wav"
    assert Path(result["source_audio_path"]).is_absolute() is False
