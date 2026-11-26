"""The installer's Spark-H3 step (``Install-SparkH3Kitchen`` in
scripts/install_ltx.ps1), read as text — nothing is run, nothing is fetched.

What is pinned here:

1. the ZIP table (GPU capability -> build tag, ZIP name, SHA-256) is exactly the
   Windows rows of the release's ``SHA256SUMS.txt`` (copied below as a LITERAL,
   so the script is compared against the release, not against itself), and the
   release tag / wheel version are ``comfyui-v0.2.0`` / ``0.2.37``;
2. the table and the version agree with what the engine checks at load time
   (``engine_h3/spark_h3.py`` ``ARCHITECTURES`` / ``KITCHEN_VERSION``);
3. a SHA-256 mismatch (and every other failure of the step) is a warning, never
   ``throw`` / ``exit``, and the mismatch reaches the summary;
4. the wheel stays out of the venv's recorded state (freeze file, direct pins).
"""

from __future__ import annotations

import ast
import re

from config import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "install_ltx.ps1"
FREEZE = PROJECT_ROOT / "engine_h3" / "venv-engine-h3.freeze.txt"
ENGINE = PROJECT_ROOT / "engine_h3" / "spark_h3.py"

#: The Windows rows of the comfyui-v0.2.0 release's SHA256SUMS.txt, by
#: ``torch.cuda.get_device_capability()`` (as the installer spells it).
RELEASE_ZIPS: dict[str, tuple[str, str, str]] = {
    "8.6": (
        "sm86",
        "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm86-cu130.zip",
        "83e2455162ff7e9550c6fdcbda5534c7768dbecf60593cbb3223deeb7fe6c5d1",
    ),
    "8.9": (
        "sm89",
        "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm89-cu130.zip",
        "d610d19859fffe0eda69f0075749c4b4c1f5cd0ff0bc4967cbfac3466e007391",
    ),
    "12.0": (
        "sm120",
        "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm120-cu130.zip",
        "87034b19523bd02090bb54f7fae1f872ff5edbff9b60ec8ae1de93938ef8ca23",
    ),
}


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8-sig")


def _code_lines(text: str) -> list[str]:
    """Lines with PowerShell ``#`` comments dropped (the step's strings hold no ``#``,
    so cutting at the first one is enough)."""
    return [line.split("#", 1)[0] for line in text.splitlines()]


def _function(text: str, name: str) -> str:
    """The body of a top-level ``function <name>`` (up to its closing ``}`` at column 0)."""
    match = re.search(rf"^function {re.escape(name)}\b.*?^\}}", text, re.S | re.M)
    assert match, f"the installer no longer defines {name}"
    return match.group(0)


def _zip_table(text: str) -> dict[str, tuple[str, str, str]]:
    block = re.search(r"^\$sparkH3Zips = @\{(.*?)^\}", text, re.S | re.M)
    assert block, "the installer no longer declares $sparkH3Zips"
    rows = re.findall(
        r"'([0-9.]+)'\s*=\s*@\{\s*Arch = '(sm\d+)';\s*Zip = '([^']+)';\s*Sha256 = '([0-9a-f]{64})'\s*\}",
        block.group(1),
    )
    return {cap: (arch, zip_name, sha) for cap, arch, zip_name, sha in rows}


def test_zip_table_is_the_release_sha256sums():
    assert _zip_table(_script()) == RELEASE_ZIPS


def test_release_tag_and_wheel_version_are_pinned():
    text = _script()
    assert (
        "$sparkH3ReleaseUrl = 'https://github.com/zechengtang/Spark-H3/releases/download/comfyui-v0.2.0'"
        in text
    )
    assert "$sparkH3KitchenVersion = '0.2.37'" in text
    # The wheel pick and the "already installed" test share one prefix, which
    # carries the GPU tag (a GPU swap re-installs).
    prefix = _function(text, "Get-SparkH3VersionPrefix")
    assert '"$sparkH3KitchenVersion+spark.h3.$Arch.cu130."' in prefix
    body = _function(text, "Install-SparkH3Kitchen")
    assert "$installed.StartsWith($wantPrefix)" in body
    assert "$wheels.Count -ne 1" in body
    assert "uv pip install --python $PythonPath --no-deps $wheel.FullName" in body


def _engine_constants() -> dict[str, object]:
    """``ARCHITECTURES`` / ``KITCHEN_VERSION`` of engine_h3/spark_h3.py, read without
    importing it (it imports torch, which the app venv does not have)."""
    found: dict[str, object] = {}
    for node in ast.parse(ENGINE.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"ARCHITECTURES", "KITCHEN_VERSION"}:
                found[target.id] = ast.literal_eval(node.value)
    return found


def test_table_agrees_with_the_engine_probe():
    engine = _engine_constants()
    assert engine["KITCHEN_VERSION"] == "0.2.37"
    installer = {
        tuple(int(part) for part in cap.split(".")): arch
        for cap, (arch, _zip, _sha) in _zip_table(_script()).items()
    }
    assert installer == engine["ARCHITECTURES"]


def test_no_failure_of_the_step_stops_the_install():
    text = _script()
    for name in ("Install-SparkH3Kitchen", "Get-SparkH3VersionPrefix", "Find-SparkH3Wheels"):
        code = "\n".join(_code_lines(_function(text, name)))
        assert not re.search(r"\bthrow\b", code, re.I), name
        assert not re.search(r"\bexit\b", code, re.I), name


def test_outer_try_catches_every_cmdlet_failure_without_rethrowing():
    """``$ErrorActionPreference = "Stop"``: any cmdlet of the fetch/install part can
    throw, so its outer ``try`` (at the function's own indentation) needs a
    ``catch`` that warns and returns, not just a ``finally``. The two probes
    before it are guarded the same way."""
    body = _function(_script(), "Install-SparkH3Kitchen")
    outer = re.search(
        r"\n    \$stage = ''\n    try \{\n(?:(?!\n    \}).)*?\n    \} catch \{\n(.*?)\n    \} finally \{\n",
        body,
        re.S,
    )
    assert outer, "the outer try of the fetch/install part has no catch before its finally"
    catch = "\n".join(_code_lines(outer.group(1)))
    assert "Write-Warn" in catch
    assert "return @($warnings)" in catch
    assert not re.search(r"\bthrow\b", catch, re.I)
    # Every `try` at the function's own indentation has a catch (the two probes
    # and the outer one): three trys, three catches.
    assert len(re.findall(r"\n    try \{", body)) == 3
    assert len(re.findall(r"\n    \} catch \{", body)) == 3


def test_sha256_mismatch_warns_installs_and_reaches_the_summary():
    text = _script()
    body = _function(text, "Install-SparkH3Kitchen")
    mismatch = re.search(r"if \(\$actual -eq \$expected\) \{(.*?)\n        \}\n", body, re.S)
    assert mismatch, "the SHA-256 comparison changed shape"
    branch = mismatch.group(1)
    assert "Write-Warn $w" in branch
    assert "$warnings += $w" in branch
    assert "return" not in branch.split("} else {", 1)[1]
    # The install still follows the check.
    assert body.index("Get-FileHash -Algorithm SHA256") < body.index("uv pip install")
    # Returned from the H3 block and re-shown with the weights' warnings.
    assert "$SparkH3HashWarnings = @(Install-SparkH3Kitchen -PythonPath $h3Py)" in text
    assert "$HashWarnings = @($SparkH3HashWarnings)" in text
    assert text.index("$SparkH3HashWarnings = @(Install-SparkH3Kitchen") < text.index(
        "$HashWarnings = @($SparkH3HashWarnings)"
    )


def test_step_runs_after_the_h3_venv_in_the_minimaxh3_block():
    text = _script()
    block = re.search(r"^if \(\$foundIds -contains 'MiniMaxH3'\) \{\n(.*?)^\}", text, re.S | re.M)
    assert block
    inner = block.group(1)
    assert inner.index("Ensure-EngineVenv -VenvPath $h3Venv") < inner.index("Install-SparkH3Kitchen")


def test_wheel_stays_out_of_the_venv_state():
    freeze_pins = [
        line.strip()
        for line in FREEZE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [p for p in freeze_pins if re.match(r"spark[-_]h3[-_]kitchen", p, re.I)]
    # The freeze says who installs it (a comment, so Get-EngineStateHash ignores it).
    assert "Install-SparkH3Kitchen" in FREEZE.read_text(encoding="utf-8")
    pins = re.search(r"^\$h3DirectPins = @\((.*?)^\)", _script(), re.S | re.M)
    assert pins
    assert "spark" not in pins.group(1).lower()
