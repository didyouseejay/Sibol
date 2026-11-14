"""Repository hygiene checks for notebooks (no Jupyter installation needed).

- Every notebook is valid nbformat-4 JSON.
- Working notebooks (notebooks/*.ipynb) carry no embedded images and stay
  under 1 MB, so heavy outputs are not committed. Figures belong in
  outputs/figures/ instead. Git-ignored run copies (notebooks/*_result.ipynb)
  are exempt.
- Working notebooks never request a download directly; downloading is only
  possible through the DOWNLOAD flag and src.data.flowers102.

Run from the project root with:
    python -m pytest tests/
"""

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
# Executed copies downloaded from Colab (notebooks/*_result.ipynb) are git-ignored
# local records, so they may embed images; the hygiene rules below skip them.
LOCAL_RESULT_SUFFIX = "_result.ipynb"
ALL_NOTEBOOKS = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
WORKING_NOTEBOOKS = sorted(p for p in NOTEBOOK_DIR.glob("*.ipynb")
                           if not p.name.endswith(LOCAL_RESULT_SUFFIX))
MAX_NOTEBOOK_BYTES = 1_000_000


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code(nb):
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebooks_exist():
    assert WORKING_NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_v4(path):
    nb = _load(path)
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert "id" in cell, f"{path.name}: cell without id"


@pytest.mark.parametrize("path", WORKING_NOTEBOOKS, ids=lambda p: p.name)
def test_working_notebook_has_no_heavy_outputs(path):
    assert path.stat().st_size < MAX_NOTEBOOK_BYTES, f"{path.name} is too large; clear its outputs"
    for cell in _load(path)["cells"]:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            assert not any(k.startswith("image/") for k in data), (
                f"{path.name} has embedded images; save figures to outputs/figures and clear outputs"
            )


@pytest.mark.parametrize("path", WORKING_NOTEBOOKS, ids=lambda p: p.name)
def test_working_notebook_never_forces_download(path):
    code = _code(_load(path)).replace(" ", "")
    assert "download=True" not in code, f"{path.name} passes download=True directly"
