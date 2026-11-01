"""予告おじゃまのアイコンの見本(src/vision/templates/pending_*.png)を実機の画像から作る

種類ごとに数枚の実例(画像, 側, 列)を、1枚目を基準に位置合わせして中央値を取る。
実例の画像は debug_frames_history(git 管理外)にあるもの(2026-10-11〜10-12 の実機、Steam 版)。
キャリブレーションは撮影時のもの(config/calibration.json)を使う。

使い方:
  venv\\Scripts\\python tools\\make_pending_icon_templates.py [--images debug_frames_history]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.capture.calibrate import CONFIG_PATH, load_calibration  # noqa: E402
from src.vision.pending_icons import BAND_H, CELL_PX, CENTER_Y, KINDS, TEMPLATE_DIR, icon_mask, locate  # noqa: E402

# 種類 → [(画像名の先頭, 側, 列)]。1つ目を基準にする
EXAMPLES = {
    "small_self": [("036_turn27", "self", 2), ("036_turn27", "self", 3), ("043_turn18", "self", 5),
                   ("spawn_wait_enemy_0218", "self", 4), ("spawn_wait_enemy_0218", "self", 5)],
    "small_enemy": [("spawn_wait_enemy_0091", "enemy", 5), ("mismatch_self_0053.30s", "enemy", 5)],
    "large": [("036_turn27", "self", 1), ("043_turn18", "self", 1), ("043_turn18", "self", 2), ("043_turn18", "self", 3),
              ("spawn_wait_enemy_0218", "self", 1), ("mismatch_self_0053.30s", "enemy", 2), ("mismatch_self_0053.30s", "enemy", 3)],
    "rock": [("036_turn27", "self", 0), ("043_turn18", "self", 0), ("spawn_wait_enemy_0091", "enemy", 1),
             ("spawn_wait_enemy_0091", "enemy", 2), ("mismatch_self_0053.30s", "enemy", 1)],
    "star": [("spawn_wait_enemy_0091", "enemy", 0), ("spawn_wait_enemy_0218", "self", 0)],
    "moon": [("flash_self_0338.28s_0338.11", "enemy", 0), ("030_turn27", "enemy", 0), ("021_turn30", "enemy", 0)],
    "crown": [("mismatch_self_0053.30s", "enemy", 0)],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(ROOT / "debug_frames_history"))
    parser.add_argument("--calibration", default=str(CONFIG_PATH))
    args = parser.parse_args()
    cal = load_calibration(Path(args.calibration))
    region = cal.capture_region()
    origin = (region.left, region.top)
    names = sorted(Path(args.images).glob("*.png"))
    cache: dict[tuple[str, str], np.ndarray] = {}

    def band(stem: str, who: str) -> np.ndarray:
        if (stem, who) not in cache:
            path = next(p for p in names if p.stem.startswith(stem))
            img = cv2.imread(str(path))
            pc = cal.self_player if who == "self" else cal.enemy_player
            cache[(stem, who)] = pc.geometry().warp(img, cell_px=CELL_PX, rows_above=2, origin=origin)[:BAND_H]
        return cache[(stem, who)]

    for kind, examples in EXAMPLES.items():
        r = KINDS[kind][0]
        mask = icon_mask(kind)
        stem, who, col = examples[0]
        cx = col * CELL_PX + CELL_PX // 2
        ref = band(stem, who)[CENTER_Y - r : CENTER_Y + r, cx - r : cx + r].copy()
        crops = []
        for stem, who, col in examples:
            (x, y), score = locate(band(stem, who), col, ref, mask)
            crops.append(band(stem, who)[y : y + 2 * r, x : x + 2 * r].astype(np.float32))
            print(f"{kind}: {stem} {who} 列{col + 1} 一致度 {score:.2f}")
        out = TEMPLATE_DIR / f"pending_{kind}.png"
        cv2.imwrite(str(out), np.median(np.stack(crops), axis=0).astype(np.uint8))
        print(f"  → {out}")


if __name__ == "__main__":
    main()
