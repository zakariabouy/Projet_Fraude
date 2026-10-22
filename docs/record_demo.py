"""Record the dashboard GIF shown at the top of the README, from the live demo page.

    pip install playwright             # drives the Edge or Chrome already installed: no download
    python docs/record_demo.py         # the published Space (links.hf_space in the config)
    python docs/record_demo.py --url http://127.0.0.1:8765   # or a local copy:
                                       # python -m http.server 8765 --directory dist/hf_space

The browser plays a visitor's path: first screen, the analysts' capacity raised to 20 alerts
per hour, the alerts with their SHAP reason codes, the model tab.
-> docs/dashboard_demo.gif + docs/dashboard.png
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS.parent))

from src.config import load_config  # noqa: E402


def live_url() -> str | None:
    """Address of the published static Space, e.g. https://user-name.static.hf.space/."""
    space = (load_config().get("links") or {}).get("hf_space")
    if not space:
        return None
    host = space.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    return f"https://{host}.static.hf.space/"


def frame(page: Page) -> Image.Image:
    shot = page.screenshot(animations="disabled", caret="hide", timeout=60_000)
    return Image.open(io.BytesIO(shot)).convert("RGB")


def record(url: str, width: int) -> list[tuple[Image.Image, int]]:
    frames = []
    with sync_playwright() as p:
        browser = None
        args = ["--disable-gpu"]  # headless screenshots can hang waiting for a GPU frame
        for channel in ("msedge", "chrome"):
            try:
                browser = p.chromium.launch(channel=channel, args=args)
                break
            except Exception:  # try the next installed browser
                continue
        if browser is None:
            browser = p.chromium.launch(args=args)  # needs `playwright install chromium`
        page = browser.new_page(
            viewport={"width": width, "height": round(width * 0.625)}, color_scheme="light"
        )
        page.goto(url)
        page.locator("#cards .card").first.wait_for(timeout=120_000)
        page.wait_for_timeout(800)
        frames.append((frame(page), 4000))

        page.locator("#capacity_per_hour").fill("20")  # K = 10 -> 20 alerts per hour
        page.wait_for_timeout(800)
        frames.append((frame(page), 3500))
        page.locator("#reset").click()

        page.get_by_role("tab", name="Alerts").click()
        page.wait_for_timeout(600)
        frames.append((frame(page), 4000))
        page.locator("#alerts-table tr.clickable").nth(1).click()  # another alert explained
        page.wait_for_timeout(500)
        frames.append((frame(page), 3000))

        page.get_by_role("tab", name="Model").click()
        page.wait_for_timeout(1500)
        frames.append((frame(page), 3000))
        browser.close()
    return frames


def save_gif(frames: list[tuple[Image.Image, int]], path: Path, width: int) -> None:
    resized = [
        img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
        for img, _ in frames
    ]
    palette = [img.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for img in resized]
    palette[0].save(
        path,
        save_all=True,
        append_images=palette[1:],
        duration=[ms for _, ms in frames],
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", help="default: the published Space")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--gif-width", type=int, default=960)
    args = parser.parse_args()
    url = args.url or live_url()
    if not url:
        parser.error("no Space published yet: pass --url")

    frames = record(url, args.viewport_width)
    frames[0][0].save(DOCS / "dashboard.png", optimize=True)
    save_gif(frames, DOCS / "dashboard_demo.gif", args.gif_width)
    size = (DOCS / "dashboard_demo.gif").stat().st_size / 1e6
    print(f"{len(frames)} frames from {url} -> docs/dashboard_demo.gif ({size:.1f} MB)")


if __name__ == "__main__":
    main()
