# ライセンスと第三者のソフトウェア

## PBA 本体

PBA のソースコードは MIT ライセンスです(`LICENSE`)。

Copyright (c) 2026 fiss-vinbows

## 配布する実行ファイル(PBA.exe)について

配布物の PBA.exe には、GPL v3 の PyQt6 が入っています。そのため、**PBA.exe を含む配布物の全体は GPL v3 の条件で配布します**。

- GPL v3 の全文は `licenses/` に入っています。
- 対応するソースコードは https://github.com/fiss-vinbows/Puyo-Beginner-Assistance で公開しています。
- PBA 自身のソースコードは、単独では MIT ライセンスのまま使えます。

## 同梱しているもの

ライセンス文の全文は、配布物の `licenses/` フォルダに入っています。

| ソフトウェア | 用途 | ライセンス |
|---|---|---|
| [ama](https://github.com/citrus610/ama)(ama-bridge.exe) | 思考エンジン。入出力のブリッジ(bridge/)は PBA で追加 | MIT |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | 画面(GUI) | GPL v3 |
| [Qt 6](https://www.qt.io/)(PyQt6-Qt6) | 画面(GUI)の土台 | LGPL v3 |
| [PyQt6-sip](https://pypi.org/project/PyQt6-sip/) | PyQt6 の土台 | BSD-2-Clause |
| [NumPy](https://numpy.org/) | 数値計算 | BSD-3-Clause(同梱の部品のライセンスは numpy の licenses) |
| [OpenCV](https://opencv.org/)(opencv-python) | 画像認識 | Apache-2.0(同梱の部品のライセンスは LICENSE-3RD-PARTY) |
| [python-mss](https://github.com/BoboTiG/python-mss) | 画面の取り込み | MIT |
| [Python](https://www.python.org/)(PyInstaller が同梱) | 実行環境 | PSF License |
| [PyInstaller](https://pyinstaller.org/) のブートローダー | exe の起動部分 | GPL v2(例外条項により配布物には及ばない) |

ama-bridge.exe は WinLibs(MinGW-w64)の g++ で静的リンクしています。GCC の実行時ライブラリは、GCC Runtime Library Exception により配布物に条件が及びません。

## 同梱していないもの

- 思考エンジン YamaNEKO(別のリポジトリ)
- ぷよぷよテトリス2・ぷよぷよeスポーツ(株式会社セガの著作物)。PBA は、画面に表示された内容を読み取るだけです。
