"""予告おじゃまのアイコンの種類を見分けて、降ってくるおじゃまの個数に換算する

盤面の上(13段目の位置の帯)に、相手から送られたおじゃまがアイコンで最大6個、大きい順に左から並ぶ。
アイコンは種類ごとに個数が違う(ぷよぷよの規則: 小=1、大=6、岩=30、星=180、月=360、王冠=720、彗星=1440)。
以前は列ごとの白い画素の割合で「アイコンが何個あるか」を数えていて、岩(30個)も1個と数え、大きな攻撃を
大幅に少なく見積もっていた(分析文書 P0-A)。

見分け方はテンプレート一致(カラー、TM_CCOEFF_NORMED)。見本は実機の画像から種類ごとに数枚を位置合わせして中央値を
取ったもの(tools/make_pending_icon_templates.py、1マス 32px の盤面座標)。小さいアイコンは下半分の内側が半透明で
背景(自分側=青、相手側=赤)が透けるので、外側の輪と上半分だけで比べる。読み方は2段(PendingIconReader.read):
  1. 位置: 見本の円の外を平均色で塗り、マスク無しで帯全体の一致を計算する(速い。予告が無ければここで終わる)。
  2. 種類: 見つけた位置だけマスク付きの一致で決める(正確。速い方式は光の演出の下で大と小の順位が逆転した)。
所要時間(1盤面): 予告なし 約1.7ms、予告あり 約4ms。

確かめ方(2026-10-20、debug_frames_history):
  - 目で数えた正解 38本(予告の出ている帯を無作為に選んだもの): 新 38/38 一致、旧(アイコンの数) 1/38。
    ただし月の見本は正解の4本のうち3本から作った。
  - 個数の換算(ぷよぷよの規則どおりか)は、連鎖の点数(ログの再現値)と予告の並びが合うことで確かめた: 相手の5連鎖
    8,140点(116個)で30個降った後に [岩 岩 大 大 大 大](=84、残り86 のうち表示は6個まで)、8,070点(115個)で
    [岩 岩 岩 大 大 大]。
  - 残る誤り: 光の演出の下で大を小と読む・小を1つ見落とす(数フレームで消える)、入ってくる途中で重なった2個を1個と読む。
彗星は実機の画像が無く見本が無い(見つからない列は従来どおり白い割合で1個と数える)。表示は6個までなので、
それを超える分(例: 115個 = 岩3 大4 小1 のうち 岩3 大3 まで)は読めない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).parent / "templates"
CELL_PX = 32
COLS = 6
# 帯の切り出し: geometry.warp(rows_above=2) の上から BAND_H 画素(14段目の上〜13段目)。アイコンの中心は y≈45〜49
BAND_H = 72
CENTER_Y = 47
SEARCH = 5  # 中心からのずれの探索幅(画素)

# 種類 → (見本の半径, 個数)
KINDS: dict[str, tuple[int, int]] = {
    "small_self": (11, 1),
    "small_enemy": (11, 1),
    "large": (15, 6),
    "rock": (15, 30),
    "star": (15, 180),
    "moon": (15, 360),
    "crown": (15, 720),
}
# 1段目(位置): 円の外を平均色で塗った見本の一致度。本物は大 p1 0.46 / 小 p1 0.40(光の下)、空の列は大 p99 0.38 /
# 小 中央 0.40・p99 0.44。取りこぼさないよう低めに取り、2段目で落とす
FAST_BIG_MIN = 0.38
FAST_SMALL_MIN = 0.47
FAST_SMALL_RING_MIN = 0.40  # 0.40〜0.47 の小の候補は外側の輪が灰色の時だけ
MAX_POSITIONS = 8
# 2段目(種類): マスク付きの一致度。本物は大・岩・星・王冠 0.6〜1.0(光の下 0.6〜0.8)、空の列 ≤0.55。
# 小は本物 ≥0.66(光の下 0.58〜0.65 で輪が灰色)、空の列 0.55〜0.65(相手側の赤い背景、輪は灰色でない)
BIG_SCORE_MIN = 0.60
SMALL_SCORE_MIN = 0.66
SMALL_SCORE_WITH_RING_MIN = 0.58
SMALL_RING_GREY_MIN = 0.15
REFINE = 2  # 2段目で1段目の最良位置の周りを探す幅(画素)
CLASSIFY_SEARCH = 8  # 2段目で位置の周りを探す幅(画素)。1段目の位置は小さい見本が大きいアイコンの右側に一致して7〜9px ずれることがある(001_turn10)
PEAK_RADIUS = 3  # 一致度の山を探す幅(画素)
MIN_ICON_DISTANCE = 22  # これより近い候補は同じアイコン(並んだアイコンの間隔は 32px)


def icon_mask(kind: str) -> np.ndarray:
    r = KINDS[kind][0]
    yy, xx = np.mgrid[0 : 2 * r, 0 : 2 * r]
    d2 = (yy - r + 0.5) ** 2 + (xx - r + 0.5) ** 2
    m = d2 <= (r - 0.5) ** 2
    if kind.startswith("small"):
        # 下半分の内側は半透明で背景が透けるので比べない
        m &= ~((d2 <= (r - 4) ** 2) & (yy >= r - 1))
    return m.astype(np.uint8) * 255


def locate(band: np.ndarray, col: int, tpl: np.ndarray, mask: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """列 col の付近で見本が最もよく一致する位置(左上)と一致度。"""
    r = tpl.shape[0] // 2
    cx = col * CELL_PX + CELL_PX // 2
    y0, y1 = max(0, CENTER_Y - r - SEARCH), min(band.shape[0], CENTER_Y + r + SEARCH)
    x0, x1 = max(0, cx - r - SEARCH), min(band.shape[1], cx + r + SEARCH)
    win = band[y0:y1, x0:x1]
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return None, -1.0
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, score, _, (lx, ly) = cv2.minMaxLoc(res)
    return (x0 + lx, y0 + ly), float(score)


def ring_grey_fraction(band: np.ndarray, xy: tuple[int, int], r: int) -> float:
    """小さいアイコンの外側の輪(灰色)に当たる画素のうち、灰色(低彩度・中〜高明度)の割合。"""
    x, y = xy
    patch = cv2.cvtColor(band[y : y + 2 * r, x : x + 2 * r], cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[0 : 2 * r, 0 : 2 * r]
    d = np.sqrt((yy - r + 0.5) ** 2 + (xx - r + 0.5) ** 2)
    ring = (d >= r - 4) & (d <= r - 1)
    grey = (patch[..., 1] < 70) & (patch[..., 2] > 100)
    return float(grey[ring].mean())


@dataclass(frozen=True)
class IconReading:
    kinds: tuple[str, ...]  # 見つけたアイコンの種類を左から順に("small"/"large"/"rock"/"star"/"crown")
    centers: tuple[int, ...]  # 各アイコンの中心の x(盤面座標、1マス 32px)
    scores: tuple[float, ...]

    def value(self) -> int:
        return sum(KINDS["small_self" if k == "small" else k][1] for k in self.kinds)

    def column_has_icon(self, col: int) -> bool:
        return any(col * CELL_PX <= c < (col + 1) * CELL_PX for c in self.centers)


class PendingIconReader:
    def __init__(self, template_dir: Path = TEMPLATE_DIR) -> None:
        self.templates: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.filled: dict[str, np.ndarray] = {}  # 円の外を平均色で塗った見本(マスク無しの速い一致に使う)
        for kind in KINDS:
            path = template_dir / f"pending_{kind}.png"
            tpl = cv2.imread(str(path))
            if tpl is None:
                raise FileNotFoundError(f"予告アイコンの見本が読めません: {path}")
            self.templates[kind] = tpl
            self.masks[kind] = icon_mask(kind)
            inside = self.masks[kind] > 0
            filled = tpl.copy()
            filled[~inside] = tpl[inside].mean(axis=0).astype(np.uint8)
            self.filled[kind] = filled

    def read(self, band_bgr: np.ndarray) -> IconReading:
        """帯の画像(warp(rows_above=2) の上 BAND_H 画素、幅 6 列)からアイコンを読む。

        アイコンは列の中央に並ぶが、増えた時は右から滑って入ってくるので列からずれる(実機 10-12 の 28.64s:
        3個とも約12px右)。そこで列に縛らず2段で読む:
          1. 位置: 円の外を平均色で塗った見本で帯全体の一致を計算し(見本ごとに1回、速い)、x 方向の一致度の山を
             候補にして、高い順に近すぎない(MIN_ICON_DISTANCE)ものを残す。予告が無ければここで終わる。
          2. 種類: 残った位置だけマスク付きの一致(正確だが遅い)で種類を決める。速い方式は光の演出の下で一致度が
             大きく下がり(大 0.75→0.45)、小さい見本の方が高くなって種類を取り違えた(001_turn10)。"""
        cands: list[tuple[float, int]] = []  # (速い一致度, 中心x)
        profiles: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}  # 種類 → (x ごとの速い一致度, その y, 半径, 帯の y0)
        for kind, tpl in self.filled.items():
            r = tpl.shape[0] // 2
            y0 = max(0, CENTER_Y - r - SEARCH)
            strip = band_bgr[y0 : min(band_bgr.shape[0], CENTER_Y + r + SEARCH)]
            if strip.shape[0] < tpl.shape[0]:
                continue
            res = np.nan_to_num(cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
            prof = res.max(axis=0)
            rows = res.argmax(axis=0)
            profiles[kind] = (prof, rows, r, y0)
            small = kind.startswith("small")
            floor = FAST_SMALL_RING_MIN if small else FAST_BIG_MIN
            local_max = cv2.dilate(prof.reshape(1, -1), np.ones((1, 2 * PEAK_RADIUS + 1), np.uint8)).ravel()
            for x in np.flatnonzero((prof >= floor) & (prof >= local_max)):
                score = float(prof[x])
                if small and score < FAST_SMALL_MIN:
                    # 一致度が低い小の候補は、外側の輪が灰色の時だけ(枠の白い線と背景の組み合わせを除く)
                    if ring_grey_fraction(band_bgr, (int(x), y0 + int(rows[x])), r) < SMALL_RING_GREY_MIN:
                        continue
                cands.append((score, int(x) + r))
        positions: list[int] = []
        for score, cx in sorted(cands, reverse=True):
            if len(positions) >= MAX_POSITIONS:
                break
            if all(abs(cx - p) >= MIN_ICON_DISTANCE for p in positions):
                positions.append(cx)
        found: list[tuple[float, int, str]] = []
        for cx in positions:
            # 2段目は、速い一致度がその位置で最も高い「大きい見本1つ」と「小さい見本1つ」だけで比べる(全種類だと
            # 予告のある帯で 7ms かかった)。大きい見本どうし・小さい見本どうしの順位は速い方式でも崩れない
            # 2段目のマスク付きの一致は、その見本が1段目で最もよく一致した位置(左上)の ±REFINE 画素だけで計算する
            # (位置を広く探すと 1盤面 約4ms かかった。見本ごとの最良位置を使うので、小さい見本が大きいアイコンの右側に
            # 一致して列の中心から 7〜9px ずれても(001_turn10)正しい位置で比べられる)
            near: dict[str, tuple[float, int, int]] = {}
            for k, (prof, rows, r, y0) in profiles.items():
                lo, hi = max(0, cx - r - CLASSIFY_SEARCH), min(len(prof), cx - r + CLASSIFY_SEARCH + 1)
                if hi <= lo:
                    continue
                x = lo + int(np.argmax(prof[lo:hi]))
                near[k] = (float(prof[x]), x, y0 + int(rows[x]))
            big_kind = max((k for k in near if not k.startswith("small")), key=lambda k: near[k][0], default=None)
            small_kind = max((k for k in near if k.startswith("small")), key=lambda k: near[k][0], default=None)
            spots = {k: near[k][1:] for k in (big_kind, small_kind) if k is not None}
            kind, score, center = self._classify_at(band_bgr, cx, list(spots), spots)
            if kind is not None:
                found.append((score, center, kind))
        taken: list[tuple[float, int, str]] = []
        for item in sorted(found, reverse=True):
            if len(taken) >= COLS:
                break
            if all(abs(item[1] - t[1]) >= MIN_ICON_DISTANCE for t in taken):
                taken.append(item)
        taken.sort(key=lambda t: t[1])
        return IconReading(tuple(t[2] for t in taken), tuple(t[1] for t in taken), tuple(t[0] for t in taken))

    def _classify_at(
        self, band_bgr: np.ndarray, cx: int, kinds: list[str] | None = None, spots: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[str | None, float, int]:
        """中心 cx 付近のアイコンの種類をマスク付きの一致で決める(kinds の見本だけ。省略時は全種類)。
        spots があれば、その見本は左上 (x, y) の ±REFINE 画素だけ探す(無ければ cx の周りの帯の高さ全体)。
        (種類, 一致度, 中心x) を返す。アイコンでなければ種類 None。"""
        big: tuple[str | None, float, int] = (None, -1.0, cx)
        small: tuple[float, tuple[int, int] | None, int] = (-1.0, None, cx)
        for kind in kinds or list(self.templates):
            tpl = self.templates[kind]
            r = tpl.shape[0] // 2
            if spots and kind in spots:
                sx, sy = spots[kind]
                y0, y1 = max(0, sy - REFINE), min(band_bgr.shape[0], sy + 2 * r + REFINE)
                x0, x1 = max(0, sx - REFINE), min(band_bgr.shape[1], sx + 2 * r + REFINE)
            else:
                y0, y1 = max(0, CENTER_Y - r - SEARCH), min(band_bgr.shape[0], CENTER_Y + r + SEARCH)
                x0, x1 = max(0, cx - r - CLASSIFY_SEARCH), min(band_bgr.shape[1], cx + r + CLASSIFY_SEARCH)
            win = band_bgr[y0:y1, x0:x1]
            if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
                continue
            res = np.nan_to_num(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=self.masks[kind]), nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, s, _, (lx, ly) = cv2.minMaxLoc(res)
            if kind.startswith("small"):
                if s > small[0]:
                    small = (float(s), (x0 + lx, y0 + ly), x0 + lx + r)
            elif s > big[1]:
                big = (kind, float(s), x0 + lx + r)
        is_small = small[0] >= SMALL_SCORE_MIN or (
            small[0] >= SMALL_SCORE_WITH_RING_MIN
            and small[1] is not None
            and ring_grey_fraction(band_bgr, small[1], KINDS["small_self"][0]) >= SMALL_RING_GREY_MIN
        )
        # 小さいアイコンにも大きい見本(特に星)がそこそこ一致するので、大きい見本の方が高い時だけ大きいアイコンとする
        if big[1] >= BIG_SCORE_MIN and (big[1] > small[0] or not is_small):
            return big[0], big[1], big[2]
        if is_small:
            return "small", small[0], small[2]
        return None, max(big[1], small[0]), cx


_default: PendingIconReader | None = None


def default_reader() -> PendingIconReader:
    global _default
    if _default is None:
        _default = PendingIconReader()
    return _default
