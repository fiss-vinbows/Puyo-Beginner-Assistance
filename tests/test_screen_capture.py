"""画面取得(src/capture/screen_capture.py)のテスト。実際の画面の小さな領域を取得する。"""

from __future__ import annotations

import unittest

import numpy as np

from src.capture.screen_capture import CaptureRegion, ScreenCapture


class GrabBgrTest(unittest.TestCase):
    def test_grab_bgr_matches_grab_reversed(self) -> None:
        # grab_bgr(2026-10-21) は grab の RGB を BGR に並べ替えたものと同じ画素で、OpenCV がそのまま使える連続配列
        region = CaptureRegion(left=0, top=0, width=64, height=48)
        try:
            cap = ScreenCapture()
        except Exception as exc:  # noqa: BLE001  画面の無い環境(サービス等)では飛ばす
            self.skipTest(f"画面を取得できない: {exc}")
        with cap:
            rgb = cap.grab(region)
            bgr = cap.grab_bgr(region)
        self.assertEqual(bgr.shape, (48, 64, 3))
        self.assertEqual(bgr.dtype, np.uint8)
        self.assertTrue(bgr.flags["C_CONTIGUOUS"])
        # 2回の取得の間に画面が変わることがあるので、多くの画素が一致することだけ確かめる
        same = np.all(bgr == rgb[:, :, ::-1], axis=2).mean()
        self.assertGreater(same, 0.9)



class PrefetchCaptureTest(unittest.TestCase):
    REGION = CaptureRegion(left=0, top=0, width=4, height=3)

    def test_frames_in_order_and_next_is_prefetched(self) -> None:
        # 受け取ったらすぐ次の1枚を取り始め、読み取りの間に用意しておく(2026-10-21)
        import threading
        import time

        from src.capture.screen_capture import PrefetchCapture

        calls: list[float] = []
        lock = threading.Lock()

        def grab(region):
            with lock:
                n = len(calls)
                calls.append(time.perf_counter())
            time.sleep(0.02)  # 取得に 20ms かかる
            return np.full((region.height, region.width, 3), n, np.uint8)

        with PrefetchCapture(grab=grab) as cap:
            first = cap.grab_bgr(self.REGION)
            time.sleep(0.05)  # 読み取りの間に次の1枚が用意される
            t = time.perf_counter()
            second = cap.grab_bgr(self.REGION)
            waited = time.perf_counter() - t
            third = cap.grab_bgr(self.REGION)
        self.assertEqual((int(first[0, 0, 0]), int(second[0, 0, 0]), int(third[0, 0, 0])), (0, 1, 2))
        self.assertLess(waited, 0.01)  # 用意してあるので待たない
        self.assertLessEqual(len(calls), 4)  # 受け取られた分(と次の1枚)しか取らない

    def test_error_in_capture_thread_is_raised(self) -> None:
        from src.capture.screen_capture import PrefetchCapture

        def grab(region):
            raise OSError("取得できない")

        with PrefetchCapture(grab=grab) as cap:
            with self.assertRaises(OSError):
                cap.grab_bgr(self.REGION)

    def test_other_region_is_rejected(self) -> None:
        from src.capture.screen_capture import PrefetchCapture

        with PrefetchCapture(grab=lambda region: np.zeros((region.height, region.width, 3), np.uint8)) as cap:
            cap.grab_bgr(self.REGION)
            with self.assertRaises(ValueError):
                cap.grab_bgr(CaptureRegion(left=1, top=0, width=4, height=3))

    def test_real_screen(self) -> None:
        from src.capture.screen_capture import PrefetchCapture

        try:
            with PrefetchCapture() as cap:
                a = cap.grab_bgr(self.REGION)
                b = cap.grab_bgr(self.REGION)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"画面を取得できない: {exc}")
        self.assertEqual(a.shape, (3, 4, 3))
        self.assertEqual(b.shape, (3, 4, 3))


if __name__ == "__main__":
    unittest.main()
