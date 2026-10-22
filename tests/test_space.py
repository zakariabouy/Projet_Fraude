import shutil
import subprocess
import sys

import yaml

from src import space
from src.config import ROOT
from src.report import readme_headline


def test_space_src_copy_imports_on_its_own(tmp_path):
    """The modules copied into the Space must cover everything the dashboard imports."""
    (tmp_path / "src").mkdir()
    for module in space.src_dependencies():
        shutil.copy2(ROOT / "src" / module, tmp_path / "src" / module)
    code = f"import sys; sys.path.insert(0, r'{tmp_path}'); import src.predict, src.cost, src.data"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_space_requirements_are_pinned():
    lines = space.space_requirements().split()
    assert {line.split("==")[0] for line in lines} == space.SPACE_PACKAGES
    assert all("==" in line for line in lines)


def test_publishing_records_the_space_and_keeps_comments(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    shutil.copy2(ROOT / "configs" / "config.yaml", config)
    monkeypatch.setattr(space, "CONFIG_PATH", config)
    space.set_space_link("someone/fraud-demo")
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["links"]["hf_space"] == (
        "someone/fraud-demo"
    )
    assert "# Hugging Face Space id" in config.read_text(encoding="utf-8")


def test_headline_badges_follow_the_links():
    res = {
        "split": {"test": {"rows": 1000}},
        "cost_reduction": {
            "xgb_topk_profitable_vs_xgb_raw_0.5": {"value": 0.3, "lo": 0.1, "hi": 0.5}
        },
    }
    offline = readme_headline(res, {"hf_space": None, "github_repo": None})
    assert "huggingface.co/spaces" not in offline and "actions/workflows" not in offline
    online = readme_headline(res, {"hf_space": "u/demo", "github_repo": "u/repo"})
    assert "huggingface.co/spaces/u/demo" in online
    assert "github.com/u/repo/actions/workflows/ci.yml/badge.svg" in online
    assert "30%" in online
