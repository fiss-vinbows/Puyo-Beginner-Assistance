"""ログ出力プラグイン: 認識結果と提案を debug_logs/ に出力する切り替えをメイン画面に出す(不具合の調査用)。

ログには画面の画像も残る(debug_frames_history/ など)。
"""

from src.plugin_api import Plugin

PLUGIN = Plugin(
    id="debug_log",
    name="ログ出力",
    description="認識結果と提案を debug_logs/ に出力します(不具合の調査用。画面の画像も保存します)",
    debug_log=True,
)
