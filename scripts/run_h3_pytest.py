r"""Run the MiniMax H3 engine tests inside .venv-engine-h3 (台帳 §1-93).

The engine venv has no pytest, so the app venv's site-packages is APPENDED to
sys.path (appended, never prepended: torch / transformers / comfy_kitchen must
keep resolving to the engine venv). --noconftest because the app conftest
builds a FastAPI app this venv cannot import. Same shape as the LTX 2.5 runner
(Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py),
kept under version control here so the tests can be run from a clone.
Targets come from argv (default: tests/test_h3_engine.py).

    .venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv" / "Lib" / "site-packages"))
sys.dont_write_bytecode = True

import pytest  # noqa: E402

args = sys.argv[1:] or [str(ROOT / "tests" / "test_h3_engine.py")]
raise SystemExit(pytest.main(args + ["--noconftest", "-p", "no:cacheprovider", "--rootdir", str(ROOT)]))
