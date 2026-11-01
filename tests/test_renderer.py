"""オーバーレイ描画のテスト(オフスクリーンの QImage に描画する)"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtGui, QtWidgets  # noqa: E402

from src.capture.calibrate import load_calibration  # noqa: E402
from src.engine.ama_client import AttackInfo, FieldAnalysis  # noqa: E402
from src.overlay.renderer import (  # noqa: E402
    OverlayDrawData,
    OverlayLayout,
    format_panel_lines,
    PANEL_ENEMY_COLOR,
    PANEL_LINES,
    PANEL_SELF_COLOR,
    PANEL_TEXT_COLOR,
    layout_panel_rows,
    panel_lines,
    panel_slots,
    plan_dots,
    render_overlay,
    render_perf,
    wrap_line,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _analysis(count: int, score: int, harass_count: int = 0) -> FieldAnalysis:
    mk = lambda c, s: AttackInfo(count=c, score=s, garbage=s // 70, x=0, rotation="UP", frame=0, all_clear=False)  # noqa: E731
    return FieldAnalysis(immediate=mk(count, score), max=mk(count, score), harass=mk(harass_count, 300), potential=0, garbage_max=score // 70)


class PanelLinesTest(unittest.TestCase):
    def test_lines(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "harass", _analysis(7, 17700), _analysis(6, 10560, 3))
        lines = format_panel_lines(data)
        # 自分・相手の見出しとおじゃまの個数は出さない(文字色で区別する)
        self.assertEqual(lines, ["提案: 潰し", "7連鎖 17,700点", "6連鎖 10,560点", "  潰し 3連鎖 300点"])
        self.assertFalse(any("おじゃま" in line or "自分" in line or "相手" in line for line in lines))

    def test_fire_chain_line(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "main", _analysis(7, 17700), fire_chain=(6, 10420))
        lines = format_panel_lines(data)
        self.assertEqual(lines[1], "この手で 6連鎖 10,420点")
        self.assertEqual(lines[2], "7連鎖 17,700点")

    def test_form_line(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", form="移行: 弥生時代 → フキゲンGTR")
        self.assertEqual(format_panel_lines(data)[1], "移行: 弥生時代 → フキゲンGTR")

    def test_line_colors(self) -> None:
        # 自分の連鎖・点数は青、相手は濃いピンク(潰しの行も同じ色)、他は白
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700, 2), _analysis(6, 10560, 3))
        colors = {line: color for line, color in panel_lines(data)}
        self.assertEqual(colors["提案: 構築"], PANEL_TEXT_COLOR)
        self.assertEqual(colors["7連鎖 17,700点"], PANEL_SELF_COLOR)
        self.assertEqual(colors["  潰し 2連鎖 300点"], PANEL_SELF_COLOR)
        self.assertEqual(colors["6連鎖 10,560点"], PANEL_ENEMY_COLOR)
        self.assertEqual(colors["  潰し 3連鎖 300点"], PANEL_ENEMY_COLOR)

    def test_all_clear_label(self) -> None:
        # 全消しは日本語表記でも「All Green」(利用者の指定 2026-10-25)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "all_clear")
        self.assertEqual(format_panel_lines(data)[0], "提案: All Green")

    def test_built_chain_line(self) -> None:
        # 組んでいる連鎖(2026-10-25): ネクネクまでで撃てる最大より大きい時だけ、別の行に出す(色は相手の行と同じ)
        enemy = _analysis(2, 420)
        enemy.potential_chain, enemy.potential = 5, 4840
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), enemy)
        colors = {line: color for line, color in panel_lines(data)}
        self.assertEqual(colors["  組んでいる 5連鎖 4,840点"], PANEL_ENEMY_COLOR)
        self.assertEqual(panel_lines(data, "en")[-1][0], "  Built: 5-chain 4,840 pts")
        enemy.potential_chain = 2  # 撃てる最大と同じなら出さない
        self.assertFalse(any("組んでいる" in line for line in format_panel_lines(data)))

    def test_later_fire_and_firing_lines(self) -> None:
        # 撃てる最大が先の手で撃つ連鎖なら「@何手目」と、相手は今撃てる連鎖も出す。相手の連鎖中は撃っている連鎖だけを出す
        # (実機 2026-10-25 12:23: 「9連鎖」は3手目に撃つ場合で、相手はすぐ撃てる6連鎖を撃った)
        enemy = _analysis(9, 35100)
        enemy.max.frame = 2
        enemy.immediate.count, enemy.immediate.score = 6, 18530
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), enemy)
        lines = panel_lines(data)
        self.assertEqual(lines[-2:], [("9連鎖 35,100点 @3", PANEL_ENEMY_COLOR), ("  今撃つと 6連鎖 18,530点", PANEL_ENEMY_COLOR)])
        self.assertEqual([line for line, _ in panel_lines(data, "en")[-2:]], ["9-chain 35,100 pts @3", "  Fire now: 6-chain 18,530 pts"])
        data.enemy_firing = (6, 18530, True)
        self.assertEqual(panel_lines(data)[-1], ("発火中 6連鎖 18,530点", PANEL_ENEMY_COLOR))
        self.assertEqual(panel_lines(data, "en")[-1][0], "Firing: 6-chain 18,530 pts")
        data.enemy_firing = (6, 18530, False)
        self.assertEqual(format_panel_lines(data)[-1], "発火 6連鎖 18,530点")
        data.enemy_firing = (0, 0, True)
        self.assertEqual(format_panel_lines(data)[-1], "発火中(段数は再現できず)")

    def test_enemy_sub_lines_fit_slots(self) -> None:
        # 行の位置の固定(クラウド側)と補足の行(今撃つと・組んでいる・潰し)の合わせ: 相手の補足は2行まで、上から順に出す
        enemy = _analysis(9, 35100, 3)
        enemy.max.frame = 2
        enemy.immediate.count, enemy.immediate.score = 6, 18530
        enemy.potential_chain, enemy.potential = 11, 60000
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), enemy)
        rows = layout_panel_rows(panel_slots(data), len, 1000)
        self.assertEqual(len(rows), PANEL_LINES)
        self.assertEqual([r[0] for r in rows[-3:]], ["9連鎖 35,100点 @3", "  今撃つと 6連鎖 18,530点", "  組んでいる 11連鎖 60,000点"])

    def test_chaining_blanks_building_chain_info(self) -> None:
        # 自分の連鎖の消化中は、この手で・構想・自分の連鎖の行を空白に(位置は固定)。相手の行は残す
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700, 2), _analysis(6, 10560, 3),
                               fire_chain=(2, 360), plan_text="構想: 9連鎖 31,140点 @12", note="連鎖中: 次の手を待機", chaining=True)
        self.assertEqual(format_panel_lines(data), ["提案: 構築", "連鎖中: 次の手を待機", "6連鎖 10,560点", "  潰し 3連鎖 300点"])
        rows = layout_panel_rows(panel_slots(data), len, 1000)
        self.assertEqual(len(rows), PANEL_LINES)
        self.assertEqual(rows[9][0], "6連鎖 10,560点")

    def test_self_firing_shown_while_chaining(self) -> None:
        # 自分の連鎖中は組んでいる連鎖の行を隠すが、撃っている連鎖は自分の行に出す(利用者 2026-10-25)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), _analysis(6, 10560),
                               chaining=True, self_firing=(5, 7550, True))
        colors = dict(panel_lines(data))
        self.assertEqual(colors.get("発火中 5連鎖 7,550点"), PANEL_SELF_COLOR)
        self.assertNotIn("7連鎖 17,700点", colors)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", chaining=True, self_firing=(3, 0, True))
        self.assertIn("発火中 3連鎖(点数は再現できず)", format_panel_lines(data))

    def test_missing_analysis(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build")
        self.assertEqual(panel_lines(data)[1:], [("--", PANEL_SELF_COLOR), ("--", PANEL_ENEMY_COLOR)])


class RenderTest(unittest.TestCase):
    def test_render_draws_dots_at_cell_centers(self) -> None:
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(2, 360), _analysis(1, 40))
        render_overlay(painter, layout, data)
        painter.end()
        # 最下段左端(1段目1列目)の中心は赤、その上は黄
        ax, ay = cal.self_player.geometry().cell_center(12, 0)
        cx, cy = cal.self_player.geometry().cell_center(11, 0)
        self.assertEqual(image.pixelColor(round(ax), round(ay)).red(), 235)
        self.assertEqual(image.pixelColor(round(cx), round(cy)).green(), 210)
        # パネルは盤面(x=272〜)の左枠外、下端に揃う(半透明の黒)
        self.assertGreater(image.pixelColor(200, 860).alpha(), 100)
        self.assertEqual(image.pixelColor(640, 860).alpha(), 0)

    def test_waiting_hides_dots_but_keeps_panel(self) -> None:
        # 手番の切り替わり直後(waiting): 前の手番のドットは描かず、パネルだけ残す
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(2, 360), _analysis(1, 40), waiting=True)
        render_overlay(painter, layout, data)
        painter.end()
        ax, ay = cal.self_player.geometry().cell_center(12, 0)
        self.assertEqual(image.pixelColor(round(ax), round(ay)).alpha(), 0)
        self.assertGreater(image.pixelColor(200, 860).alpha(), 100)

    def test_panel_stays_inside_left_margin(self) -> None:
        # 長い行は盤面の左の余白に収まるよう折り返され、パネルが画面外(x<0)に出ない
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", note="相手が発火: 提案を更新 (仕掛けに対応: 相殺後も勝ち)" * 2)
        render_overlay(painter, layout, data)
        painter.end()
        left = cal.self_player.geometry().grid_to_screen(0, 12)[0]
        # パネルの右端は盤面の左、左端は画面内
        self.assertGreater(image.pixelColor(round(left) - 20, 860).alpha(), 100)
        self.assertEqual(image.pixelColor(0, 860).alpha(), 0)

    def test_panel_size_is_fixed(self) -> None:
        # パネルの大きさは文言によらず固定(利用者の指定 2026-10-25)。行が多すぎる時は連鎖の行を残して白の行を省く
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        left = round(cal.self_player.geometry().grid_to_screen(0, 12)[0])

        def panel_box(data: OverlayDrawData) -> tuple[int, int, int, int]:
            image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
            image.fill(QtGui.QColor(0, 0, 0, 0))
            painter = QtGui.QPainter(image)
            render_overlay(painter, layout, data)
            painter.end()
            pts = [(x, y) for x in range(0, left, 2) for y in range(0, 1080, 2) if image.pixelColor(x, y).alpha() > 100]
            return min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts)

        short = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", waiting=True)
        long = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700, 2), _analysis(6, 10560, 3),
                               note="相手が発火: 提案を更新" * 3, form="狙い: GTR", plan_text="構想: 9連鎖 31,140点 @12", fire_chain=(2, 360), waiting=True)
        self.assertEqual(panel_box(short), panel_box(long))
        # 行数を超える種類(長いメモ)は割り当ての行数で「…」にして収め、下の行の位置は動かさない
        rows = layout_panel_rows(panel_slots(long), len, 12)
        self.assertEqual(len(rows), PANEL_LINES)
        self.assertTrue(rows[4][0].endswith("…"))

    def test_rows_keep_positions_when_info_missing(self) -> None:
        # 情報が無い種類は空白の行にして、他の行の位置を固定する(情報が増えて行が間に入ると見づらい 2026-10-25)
        few = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), _analysis(6, 10560))
        many = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700, 2), _analysis(6, 10560, 3),
                               note="相手の連鎖が終了: 提案を更新", form="狙い: GTR", plan_text="構想: 9連鎖 31,140点 @12", fire_chain=(2, 360))
        a = layout_panel_rows(panel_slots(few), len, 1000)
        b = layout_panel_rows(panel_slots(many), len, 1000)
        self.assertEqual(len(a), PANEL_LINES)
        self.assertEqual(len(b), PANEL_LINES)
        for line in ("提案: 構築", "7連鎖 17,700点", "6連鎖 10,560点"):
            self.assertEqual([r[0] for r in a].index(line), [r[0] for r in b].index(line))
        self.assertEqual(a[2][0], "")  # この手で(無し)は空白

    def test_2p_draws_on_right_board_and_panel_outside_right(self) -> None:
        # 2P を操作する時(自分と相手を入れ替えたキャリブレーション): ドットは右の盤面、
        # パネルは右の盤面の外側(右)に置き、画面中央(盤面の左)には出さない
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json").swapped()
        self.assertTrue(cal.self_on_right())
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        geom = cal.self_player.geometry()
        right = geom.grid_to_screen(6, 12)[0]
        layout = OverlayLayout(geom, cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0, panel_right_space=1920 - right)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", note="相手が発火: 提案を更新" * 3)
        render_overlay(painter, layout, data)
        painter.end()
        ax, ay = geom.cell_center(12, 0)
        self.assertEqual(image.pixelColor(round(ax), round(ay)).red(), 235)
        left = geom.grid_to_screen(0, 12)[0]
        self.assertGreater(image.pixelColor(round(right) + 20, 860).alpha(), 100)
        self.assertEqual(image.pixelColor(round(left) - 20, 860).alpha(), 0)
        self.assertEqual(image.pixelColor(1919, 860).alpha(), 0)

    def test_1p_calibration_is_not_right(self) -> None:
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        self.assertFalse(cal.self_on_right())

    def test_wrap_line(self) -> None:
        self.assertEqual(wrap_line("abcdef", len, 4), ["abcd", "ef"])
        self.assertEqual(wrap_line("abc", len, 4), ["abc"])
        self.assertEqual(wrap_line("", len, 4), [""])
        # 英語表記は空白で折り返す。空白の無い長い語は文字単位
        self.assertEqual(wrap_line("Move: Extend chain now", len, 12, True), ["Move: Extend", "chain now"])
        self.assertEqual(wrap_line("abcdefgh ij", len, 4, True), ["abcd", "efgh", "ij"])

    def test_plan_dots(self) -> None:
        # 最終的な構想(2026-10-24): 今の盤面で空のマスだけを描く。おじゃまは描かない。発火の組ぷよのマスに印を付ける
        settled = [["empty"] * 6 for _ in range(13)]
        settled[12][0] = "red"
        plan = [["empty"] * 6 for _ in range(13)]
        plan[12][0] = "red"
        plan[12][1] = "blue"
        plan[11][1] = "garbage"
        plan[11][0] = "green"
        dots = plan_dots(plan, settled, [(11, 0), (10, 0)])
        self.assertEqual(sorted(dots), [(11, 0, "green", True), (12, 1, "blue", False)])
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        data = OverlayDrawData((12, 5), (11, 5), "red", "yellow", "build", plan_dots=dots, plan_text="構想: 9連鎖 31,140点 @12")
        render_overlay(painter, layout, data)
        painter.end()
        bx, by = cal.self_player.geometry().cell_center(12, 1)
        self.assertGreater(image.pixelColor(round(bx), round(by)).alpha(), 0)
        self.assertIn("構想: 9連鎖 31,140点 @12", format_panel_lines(data))
        # 計算中(waiting)は描かない
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        data.waiting = True
        render_overlay(painter, layout, data)
        painter.end()
        self.assertEqual(image.pixelColor(round(bx), round(by)).alpha(), 0)

    def test_perf_outside_board_top(self) -> None:
        # CPU 使用率と観測の FPS(2026-10-24): 1P は盤面の上端の左の枠外に描き、盤面の中には描かない
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        render_perf(painter, layout, "CPU 45% / 観測 33fps")
        painter.end()
        x0, y0 = cal.self_player.geometry().grid_to_screen(0, 0)
        self.assertGreater(image.pixelColor(round(x0) - 20, round(y0) + 8).alpha(), 100)  # 盤面の左上の枠外
        cx, cy = cal.self_player.geometry().cell_center(2, 2)
        self.assertEqual(image.pixelColor(round(cx), round(cy)).alpha(), 0)  # 盤面の中は透明のまま

    def test_next_move_dots(self) -> None:
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", next_axis_cell=(12, 5), next_child_cell=(11, 5), next_axis_color="green", next_child_color="purple")
        render_overlay(painter, layout, data)
        painter.end()
        nx, ny = cal.self_player.geometry().cell_center(12, 5)
        self.assertGreater(image.pixelColor(round(nx), round(ny)).green(), 150)


if __name__ == "__main__":
    unittest.main()
