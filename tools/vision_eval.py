"""画像認識の評価ツール(正解ラベル付きの実機画像で一致率と混同行列を出す)

データは vision_eval/ に置く(画面キャプチャを含むので git 管理外):
  vision_eval/images/<名前>.png   画面画像(キャプチャ範囲そのもの。debug_frames_history の画像)
  vision_eval/labels/<名前>.json  正解ラベル(形式は src/vision/eval_dataset.py)
  vision_eval/review/<名前>.png   確認用の画像(盤面を正規化してラベルの文字を重ねたもの)

使い方:
  # 画像を追加(今の認識結果を初期ラベルにする。verified=false)
  venv\\Scripts\\python tools\\vision_eval.py add debug_frames_history\\013_turn12.png [--screen result]
  # 確認用の画像を作る(名前を省略すると未確認のもの全部)。ラベルと今の読みが違うマスは赤枠
  venv\\Scripts\\python tools\\vision_eval.py review [名前...]
  # ラベル(JSON)を目視で直したら確認済みにする
  venv\\Scripts\\python tools\\vision_eval.py verify 名前...
  # 採点(確認済みのみ。--all で未確認も含める)
  venv\\Scripts\\python tools\\vision_eval.py run [--all] [--errors]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.capture.calibrate import CONFIG_PATH, CalibrationResult, load_calibration  # noqa: E402
from src.engine.puyo_field import COLS, ROWS  # noqa: E402
from src.vision.eval_dataset import compare, load_label, make_label, read_frame, save_label, summarize  # noqa: E402

DATA = ROOT / "vision_eval"
IMAGES, LABELS, REVIEW = DATA / "images", DATA / "labels", DATA / "review"
REVIEW_CELL_PX = 48


def _load_image(path: Path) -> np.ndarray:
    # 日本語を含むパスでも読めるように imdecode を使う
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _case(name: str) -> tuple[dict, np.ndarray, CalibrationResult]:
    label = load_label(LABELS / f"{name}.json")
    image = _load_image(IMAGES / label["image"])
    return label, image, CalibrationResult.from_dict(label["calibration"])


def cmd_add(args: argparse.Namespace) -> None:
    cal = load_calibration(Path(args.calibration))
    region = cal.capture_region()
    for src in map(Path, args.images):
        image = _load_image(src)
        if image.shape[:2] != (region.height, region.width):
            print(f"{src.name}: 大きさ {image.shape[1]}x{image.shape[0]} がキャプチャ範囲 {region.width}x{region.height} と違うので飛ばします")
            continue
        name = src.stem
        if (LABELS / f"{name}.json").exists() and not args.force:
            print(f"{name}: 既にあります(--force で上書き)")
            continue
        IMAGES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, IMAGES / src.name)
        save_label(LABELS / f"{name}.json", make_label(src.name, cal, read_frame(image, cal), args.screen))
        print(f"{name}: 追加しました(screen={args.screen})")


def _text(img: np.ndarray, s: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.7) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_review(label: dict, image: np.ndarray, cal: CalibrationResult) -> np.ndarray:
    """両者の盤面を正規化して並べ、各マスにラベルの文字を描く。今の読みと違うマスは赤枠で囲み、読みを右下に出す。"""
    reading = read_frame(image, cal)
    region = cal.capture_region()
    origin = (region.left, region.top)
    c = REVIEW_CELL_PX
    panels = []
    for who, pc in (("self", cal.self_player), ("enemy", cal.enemy_player)):
        want, got = label["players"][who], reading[who]
        board = pc.geometry().warp(image, cell_px=c, rows_above=1, origin=origin)  # 行0 = 予告帯(13段目)
        for row in range(ROWS - 1):
            for col in range(COLS):
                x, y = col * c, (row + 1) * c
                cv2.rectangle(board, (x, y), (x + c - 1, y + c - 1), (90, 90, 90), 1)
                w, g = want["board"][row][col], got["board"][row][col]
                _text(board, w, (x + 4, y + 20), (255, 255, 255))
                if w != g and w != "*":
                    cv2.rectangle(board, (x + 1, y + 1), (x + c - 2, y + c - 2), (0, 0, 255), 2)
                    _text(board, g, (x + c - 18, y + c - 6), (0, 0, 255), 0.6)
        _text(board, f"pending {want['pending']} (read {got['pending']})", (4, c - 8), (0, 255, 255), 0.55)
        # ネクスト枠は盤面と同じ高さの列に縦に並べる
        col_w = 2 * c
        side = np.zeros((board.shape[0], col_w, 3), np.uint8)
        y = c
        for key, (x0, y0, w0, h0) in zip(("next1", "next2"), pc.next_rects):
            crop = image[y0 - origin[1] : y0 - origin[1] + h0, x0 - origin[0] : x0 - origin[0] + w0]
            if crop.size == 0:
                continue
            scale = col_w / crop.shape[1]
            crop = cv2.resize(crop, (col_w, max(1, int(crop.shape[0] * scale))))
            crop = crop[: side.shape[0] - y]
            side[y : y + crop.shape[0]] = crop
            for i in range(2):
                wl, gl = want[key][i], got[key][i]
                color = (255, 255, 255) if wl == gl or wl == "*" else (0, 0, 255)
                _text(side, f"{wl}" + ("" if wl == gl else f"({gl})"), (4, y + 22 + i * crop.shape[0] // 2), color)
            y += crop.shape[0] + 10
        _text(side, who, (4, 24), (0, 255, 0))
        panels.append(np.hstack([board, side]))
    out = np.hstack([panels[0], np.full((panels[0].shape[0], 12, 3), 40, np.uint8), panels[1]])
    _text(out, f"screen={label['screen']} verified={label['verified']}", (4, out.shape[0] - 8), (0, 255, 0), 0.6)
    return out


def cmd_review(args: argparse.Namespace) -> None:
    names = args.names or [p.stem for p in sorted(LABELS.glob("*.json")) if not load_label(p)["verified"]]
    REVIEW.mkdir(parents=True, exist_ok=True)
    for name in names:
        label, image, cal = _case(name)
        out = REVIEW / f"{name}.png"
        cv2.imencode(".png", render_review(label, image, cal))[1].tofile(str(out))
        print(out)


def cmd_verify(args: argparse.Namespace) -> None:
    for name in args.names:
        path = LABELS / f"{name}.json"
        label = load_label(path)
        label["verified"] = True
        save_label(path, label)
        print(f"{name}: 確認済みにしました")


def cmd_run(args: argparse.Namespace) -> None:
    results = []
    for path in sorted(LABELS.glob("*.json")):
        label = load_label(path)
        if not label["verified"] and not args.all:
            continue
        _, image, cal = _case(path.stem)
        res = compare(path.stem, label, read_frame(image, cal))
        results.append(res)
        if args.errors and res.errors:
            print(f"--- {path.stem}")
            for e in res.errors:
                print(f"    {e}")
    if not results:
        print("採点できる画像がありません(verify で確認済みにするか --all を付ける)")
        return
    print(summarize(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="画像を追加(今の認識結果を初期ラベルにする)")
    p.add_argument("images", nargs="+")
    p.add_argument("--calibration", default=str(CONFIG_PATH))
    p.add_argument("--screen", default="game", choices=("game", "result", "other"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_add)
    p = sub.add_parser("review", help="確認用の画像を作る")
    p.add_argument("names", nargs="*")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("verify", help="確認済みにする")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("run", help="採点する")
    p.add_argument("--all", action="store_true", help="未確認のラベルも含める")
    p.add_argument("--errors", action="store_true", help="画像ごとの誤りを表示する")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
