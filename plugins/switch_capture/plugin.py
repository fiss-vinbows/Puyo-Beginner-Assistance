"""Switch 対応プラグイン: キャプチャボードの画面を手動でキャリブレーションする(対戦用・一人用)。"""

from src.plugin_api import Plugin

PLUGIN = Plugin(
    id="switch_capture",
    name="Switch 対応",
    description="キャプチャボードの画面を手動でキャリブレーションします(対戦用・一人用)",
    manual_calibration=True,
)
