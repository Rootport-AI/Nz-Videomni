"""§3-90 implementer A: run tests inside .venv-engine-ltx25.

Same shape as outputs/ltx25-sage-gate/run_ltx25_pytest.py (the documented
runner): that venv has no pytest, so the app venv's site-packages is APPENDED
to sys.path (appended, never prepended: torch / ltx_core must keep resolving to
the engine venv). --noconftest because the app conftest builds a FastAPI app
this venv cannot import. Targets come from argv.
"""
import sys
from pathlib import Path

ROOT = Path("S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni")
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv" / "Lib" / "site-packages"))

import pytest  # noqa: E402

args = sys.argv[1:] or [str(ROOT / "tests")]
raise SystemExit(pytest.main(args + ["--noconftest", "-p", "no:cacheprovider"]))
