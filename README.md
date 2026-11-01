# PBA (Puyo Beginner Assistance)

ぷよぷよテトリス2のぷよぷよ画面をリアルタイムで認識し、最善手(組ぷよの置き場所)を
オーバーレイ表示する支援AI。[TBA (Tetris Beginner Assistance)](../Tetris) のぷよぷよ版。

## 構成

- `external/ama-main/` — 思考エンジン [ama](https://github.com/citrus610/ama)(MIT)。
  `bridge/main.cpp` は本プロジェクトで追加した標準入出力JSONブリッジ。
  `makefile` は Windows(MinGW-w64)向けに修正済み(ソース一覧の補完・静的リンク・`-march=native`除去)。
- `src/engine/` — `puyo_field.py`(盤面表現)、`ama_client.py`(ブリッジ呼び出し)、`tracker.py`(認識の時系列から手番・連鎖・確定盤面を追跡)
- `src/capture/` — 画面キャプチャ、`geometry.py`(盤面4隅の透視変換)、`calibrate.py`(キャリブレーションUI)
- `src/vision/` — `cell_classifier.py`(1マスの判定)、`board_reader.py`(盤面・予告おじゃま・ネクスト)、`color_mapper.py`(4色→amaスロット)
- `src/overlay/renderer.py` — 着地マスのドットと情報パネルの描画
- `src/app.py` — アプリ本体(メインウィンドウ、支援モードのループ、オーバーレイ)
- `tests/` — 単体テスト・統合テスト。`tests/fixtures/` に動画から抜いた1080pフレームとキャリブレーション
- `tools/build_ama.bat` — ama ブリッジのビルド
- `tools/replay_video.py` — 録画動画で認識→追跡→思考をリプレイ検証
- `tools/preview_overlay.py` — フレーム画像にオーバーレイを合成して見た目を確認

## セットアップ

### 1. Python環境

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 2. ama ブリッジのビルド(初回のみ)

MinGW-w64 の g++ が必要(MSVC では ama の GCC 拡張がコンパイルできない)。

```bash
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --source winget
```

インストール後:

```bash
tools\build_ama.bat
```

`external/ama-main/bin/bridge/ama-bridge.exe` が生成されればOK(静的リンク済み、DLL不要)。

## 起動

```bash
start.bat
```

1. 「キャリブレーションを実行」→ 対戦開始直後(盤面が空に近い状態)で **Aキー** を押すと
   自分・相手の盤面4隅を自動検出する。緑の格子を矢印キーで微調整して Enter。
   続けてネクスト枠(大きい方・小さい方)の左上・右下をクリック(自分→相手の順)。
   一人用(とことん等)モードは画面レイアウトが対戦と異なるため、「一人用モード」に
   チェックを入れてから別途キャリブレーションする(`config/calibration_solo.json` に保存。
   相手側の指定は不要)。
2. 「支援モード開始」→ 組ぷよを設置するたびに、次の組ぷよの置き場所(軸=大きいドット、
   子=小さいドット)と、自分・相手の「発火したら何連鎖・何点」が表示される。

## ブリッジのプロトコル

1行1JSON。起動直後に `{"ready": true}` を返す。詳細は `external/ama-main/bridge/main.cpp` 冒頭。

```
build  : {"field": [13行(上から)], "queue": ["RG", "BY"]} -> {"x", "r", "score", "axis", "child", ...}
analyze: {"mode": "analyze", "field", "queue"} -> 即発火/最大/潰しの連鎖数・点数・おじゃま換算
battle : {"mode": "battle", "self": {...}, "enemy": {...}} -> 相手を凝視した手 + label(構築/潰し/本線発火/受け/全消し)
```

- マスは `R`/`Y`/`G`/`B`/`#`(おじゃま)/`.`(空)。ama は4色固定。
- `queue` の各要素は「軸ぷよ・子ぷよ」の2文字。先頭が現在の組ぷよ。
- `x` は軸ぷよの列(左から0)、`r` は子ぷよの位置(UP/RIGHT/DOWN/LEFT)。
- `axis`/`child` は着地マス (x, y)、y は最下段が0。

## テスト

```bash
venv\Scripts\python -m unittest discover -s tests -v
```

ブリッジ実行ファイルが存在する環境でのみ統合テストが走り、未ビルドなら自動スキップされる。

## 実測(1P 自己プレイ、`puyop` 100手)

`-O3 -flto` 静的リンク版で 1手あたり平均 75ms、38手で 112,640点(約12連鎖)を構築。

## ライセンス

PBA本体は [MIT License](LICENSE)。ama も MIT License(`external/ama-main/LICENSE`)。
