"""言語プラグイン「mokou」: 日本語の表記で、11〜19連鎖の発火を呼び名で出す(遊び)。"""

from src.plugin_api import Language, Plugin

from .mokou import to_mokou

PLUGIN = Plugin(
    id="language_mokou",
    name="mokou 表記",
    description="情報パネルの表記に「mokou」を足します(日本語で、11〜19連鎖の発火を「11の刃」のような呼び名で出す)",
    languages={"mokou": Language("mokou", to_mokou)},
)
