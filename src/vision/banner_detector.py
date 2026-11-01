"""「全消し」バナーの検出と、その画素を盤面の読み取りから除外するためのマスク

全消しボーナスを持っている間、盤面の 9〜11段目・1〜5列目あたりに「全消し」の文字が表示され
続ける(ボーナスを消費する連鎖まで。実機 2026-09-20 13:56 では試合のほぼ全体)。文字は白い本体と
オレンジの縁取りで、空マスの上では黄ぷよ・おじゃまとして読まれ、偽の4連結(YYYY)が取り込み待ちを
起こし、確定盤面にも混入した。ぷよは文字より手前に描かれる(文字はぷよの無い所だけに見える)ので、
文字の画素だけを分類から除けば、その下の空マスは空、手前のぷよは通常どおり読める。

検出はテンプレート一致(カラー、TM_CCOEFF_NORMED)。テンプレートは実機の 000_turn1.png(空盤面に
文字だけ)を盤面座標(1マス 32px)に変換して切り出したもの。文字は出現直後に少し動く(y が 12px 違う
フレームがある)ので、9〜11段目付近を探索する。
得点の実測(debug_frames_history 2,819枚): 文字あり ≥0.66(ぷよに大きく隠れた場合)、文字なし ≤0.49。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).parent / "templates" / "all_clear_banner.png"

# 探索範囲(盤面座標の y、1マス 32px。row 1〜5 = 12〜8段目)
SEARCH_Y0 = 32
SEARCH_Y1 = 224
# テンプレート全体の一致がこれ以上なら文字あり
BANNER_SCORE_MIN = 0.58
# テンプレートと画素が一致した(BGR の差の合計がこれ未満)所を文字の画素とみなす
BANNER_PIXEL_DIFF_MAX = 90
# 一度検出した文字は、ぷよに大きく隠れてテンプレート全体の一致度が下がっても(実機 2026-09-20 14:41:
# 積みが11段まで来て一致度 0.47、文字の一部だけ見えて YYY と読まれた)、インクの画素がこの割合以上
# 一致している間は見えているとみなす(実測: 隠れた文字 0.65〜0.78、文字なしの盤面 ≤0.55 は
# カットインの白と橙で、文字を検出していない状態では使わない)
BANNER_KEEP_INK_FRAC_MIN = 0.3
# インクの一致がこの回数続けて下回ったら文字は消えたとみなす
BANNER_LOST_FRAMES = 10


@dataclass(frozen=True)
class BannerMatch:
    score: float
    x: int  # 盤面座標での左上
    y: int


class AllClearBannerDetector:
    """テンプレート一致で文字を探し、見つけた後は隠れても位置を覚えて追い続ける(盤面ごとに1個持つ)。"""

    def __init__(self, template_path: Path = TEMPLATE_PATH) -> None:
        self.active: BannerMatch | None = None  # 追跡中の文字の位置
        self._lost = 0
        tpl = cv2.imread(str(template_path))
        if tpl is None:
            raise FileNotFoundError(f"全消しバナーのテンプレートが読めません: {template_path}")
        self.template = tpl
        hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # 文字のインク(白い本体+オレンジの縁取り)。透けている背景は除く
        self.ink = ((s < 70) & (v > 170)) | ((h >= 5) & (h <= 25) & (s > 120) & (v > 150))

    def _best(self, warped_bgr: np.ndarray) -> BannerMatch | None:
        th, tw = self.template.shape[:2]
        roi = warped_bgr[SEARCH_Y0:SEARCH_Y1, :]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None
        result = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if not np.isfinite(score):
            return None
        return BannerMatch(float(score), int(loc[0]), int(loc[1]) + SEARCH_Y0)

    def find(self, warped_bgr: np.ndarray) -> BannerMatch | None:
        """状態を持たない検出(テンプレート全体の一致度だけで判定)。"""
        best = self._best(warped_bgr)
        return best if best is not None and best.score >= BANNER_SCORE_MIN else None

    def _ink_frac(self, warped_bgr: np.ndarray, match: BannerMatch) -> float:
        th, tw = self.template.shape[:2]
        return float(self.pixel_mask(warped_bgr, match)[match.y : match.y + th, match.x : match.x + tw].sum() / self.ink.sum())

    def observe(self, warped_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        """1フレーム観測し、(文字の画素マスク or None, 一致度) を返す。

        一致度が閾値以上なら文字を検出して位置を覚える。以後は一致度が下がっても、覚えた位置(または
        今回の最良位置)でインクの画素が BANNER_KEEP_INK_FRAC_MIN 以上一致していれば見えているとみなす。
        """
        best = self._best(warped_bgr)
        if best is not None and best.score >= BANNER_SCORE_MIN:
            self.active, self._lost = best, 0
            return self.pixel_mask(warped_bgr, best), best.score
        if self.active is None:
            return None, 0.0
        candidates = [self.active] + ([best] if best is not None else [])
        match = max(candidates, key=lambda m: self._ink_frac(warped_bgr, m))
        frac = self._ink_frac(warped_bgr, match)
        if frac >= BANNER_KEEP_INK_FRAC_MIN:
            self._lost = 0
            self.active = BannerMatch(frac, match.x, match.y)
            return self.pixel_mask(warped_bgr, match), frac
        self._lost += 1
        if self._lost >= BANNER_LOST_FRAMES:
            self.active = None
            return None, 0.0
        return self.pixel_mask(warped_bgr, match), frac  # 消えたと確定するまでは覚えた位置で除く(害はない)

    def pixel_mask(self, warped_bgr: np.ndarray, match: BannerMatch) -> np.ndarray:
        """文字が実際に見えている画素(テンプレートのインクと一致する所)の bool マスク(盤面全体の大きさ)。
        手前にぷよが描かれている所はテンプレートと違う色なので含まれない。"""
        th, tw = self.template.shape[:2]
        mask = np.zeros(warped_bgr.shape[:2], dtype=bool)
        region = warped_bgr[match.y : match.y + th, match.x : match.x + tw]
        if region.shape[:2] != (th, tw):
            return mask
        diff = np.abs(region.astype(np.int32) - self.template.astype(np.int32)).sum(axis=2)
        hit = (self.ink & (diff < BANNER_PIXEL_DIFF_MAX)).astype(np.uint8)
        # 縁取りの点模様やアンチエイリアスの隙間を埋める
        hit = cv2.dilate(hit, np.ones((3, 3), np.uint8))
        mask[match.y : match.y + th, match.x : match.x + tw] = hit.astype(bool)
        return mask


_default: AllClearBannerDetector | None = None


def default_detector() -> AllClearBannerDetector:
    global _default
    if _default is None:
        _default = AllClearBannerDetector()
    return _default
