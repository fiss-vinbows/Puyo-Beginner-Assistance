"""PBA のプラグイン(2026-10-28、利用者「様々な機能が搭載されて複雑になりすぎているので、標準機能をベースにプラグインで拡張する」)

標準機能: 支援モード・シミュレーター・定跡の登録・エンジンの登録(YamaNEKO 搭載)・配ぷよ読み・CPU の使用・パネルの表記(日本語・英語)・ログ。
それ以外はプラグインが足す。プラグインは plugins/<名前>/plugin.py に PLUGIN = Plugin(...) を書いて置く(src/plugin_loader.py が読む)。

プラグインができること(今の段階):
- autoplay_sides: 全自動の操作を許可する側("1p"・"2p")。どれかのプラグインが許可した側だけ、全自動の設定(キー割り当て・仮想パッド・
  設置キャンセル・段差キャンセル・先行入力・切り替えキー・キーボードで仮想パッドを操作)がメイン画面に出て、全自動で動かせる
- nameplate_guard: このプラグインが許可した側の全自動は、その側のネームプレートに「2P」(テトリス2)か「Player 2」(eスポーツ)(ローカル対戦のゲスト)が
  見えている間だけ動く(オンライン対戦では相手の名前が出るので動かない。不正防止)。他のプラグインが同じ側を条件なしで
  許可していれば、その側は条件なし
- manual_calibration: 画面の手動キャリブレーション(対戦用・一人用)のボタンを出す(Switch のキャプチャボード用)
- window_resize: PC 版のゲームのウィンドウを決まった大きさ(1600×900 など)に合わせる項目をメイン画面に出す
- debug_log: 認識結果と提案をログに出力する(debug_logs/)切り替えをメイン画面に出す
- languages: 情報パネルの表記を足す(行ごとの置き換え関数)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    label: str  # メイン画面の「パネルの表記」に出す名前
    transform: Callable[[str], str]  # 日本語のパネルの1行 → この表記の1行
    word_break: bool = False  # 折り返しを空白で行う(英語のように単語を切らない)


@dataclass(frozen=True)
class Plugin:
    id: str
    name: str
    description: str = ""
    # 配布してよいか。False のプラグイン(全自動モード)は配布物(exe)に入れない
    distributable: bool = True
    autoplay_sides: frozenset[str] = frozenset()
    nameplate_guard: bool = False
    manual_calibration: bool = False
    debug_log: bool = False
    window_resize: bool = False
    languages: dict[str, Language] = field(default_factory=dict)
