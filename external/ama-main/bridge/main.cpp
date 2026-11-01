// ama ブリッジ (PBA: Puyo Beginner Assistance 用)
//
// 標準入力から1行1メッセージのJSONを受け取り、ama の探索結果を標準出力へ
// 1行のJSONで返す。Cold Clear 2 のような「別プロセスに盤面を渡して手をもらう」
// 形で Python から使うための入口。ama 本体(core/, ai/)は上流のまま(無改変)。
// 2026-10-21 の方針(PBA に搭載する思考エンジンは自作AIの YamaNEKO だけ。ama は手を加えず素の状態で使うか
// 対戦相手にする)により、ブリッジは入出力の変換だけを行い、ama の判断を変える処理(初心者モード・開幕の形の
// 選択と重み・本線発火の目安・将来のツモ)は外した。要求にそれらの項目があっても読まない。
//
// 共通の表記:
//   field: 13行(上が13段目=隠し段、下が1段目)×6列の文字列。R/Y/G/B/#/.
//   queue: ["RG", "BY", ...] 現在の組ぷよ, ネクスト, (ネクネク...)。各要素は「軸, 子」
//   座標 (x, y): x は左から0始まり、y は最下段が0
//   r: 子ぷよの位置 UP/RIGHT/DOWN/LEFT
//
// モード (省略時は "build"):
//   build   1P用ビームサーチで連鎖構築の最善手を返す
//           入力 {"field", "queue", "width"?, "depth"?}
//           出力 {"x", "r", "score", "axis", "child", "candidates", "path", "ms"}
//           path: 置き場所までの操作手順(1要素=1フレームの入力。LEFT/RIGHT/CW/CCW/DROP/NONE/…)
//   analyze 盤面と組ぷよ列から「発火したら何連鎖・何点になるか」を調べる(凝視・点数表示用)
//           入力 {"mode": "analyze", "field", "queue"(2組以上), "target_point"?}
//           出力 {"immediate": 1手目で発火できる最大, "max": 手持ちの組ぷよで発火できる最大,
//                 "harass": 3連鎖以下の潰し最大, "potential": 将来の本線の見込み点,
//                 "garbage_max": max を おじゃま個数換算,
//                 "fire_points": 1個置くと連鎖が起きる発火点 [{"x","y"(置く前の高さ),"color","count","score"}]
//                                を点数の高い順, "ms"}
//           immediate/max/harass は {"count", "score", "garbage", "x", "r", "frame"}
//   battle  相手の盤面を凝視した対戦思考(ai::think)。潰し・受け・本線発火も判断する
//           入力 {"mode": "battle", "self": Player, "enemy": Player, "target_point"?}
//           出力に "form": 設置後の盤面に最も合う形の名前(GTR/FRON/SGTR、無ければ null)を含む(build/battle 共通。表示用)
//             Player = {"field", "queue"(2組以上), "all_clear"?, "bonus"?,
//                       "attack"?(そのプレイヤーに降る予告おじゃま個数。ama 内部の意味(送った量)とは
//                       逆なのでブリッジで入れ替える), "attack_frame"?(そのプレイヤーの連鎖が
//                       終わるまでの手数目安), "dropping"?(落下中のおじゃま個数),
//                       "chain_bonus"?(進行中の連鎖に全消しボーナスが乗っている),
//                       "chain_field"?(点滅から復元した発火直前の盤面。消して連鎖を再現する),
//                       "attack_frame_est"?(enemy のみ。PBA が経過時間から見積もった、連鎖が終わるまでに置ける手数)}
//             enemy_unknown: true なら相手の状態は認識失敗時の仮のもの(記録用)
//           出力 {"x", "r", "eval", "axis", "child", "label", "self": analyze結果,
//                 "enemy": analyze結果, "fire_reason": 常に null(互換のため残す),
//                 "incoming"/"incoming_source"/"incoming_chain": 相手の連鎖を再現できた時の着弾量, "ms"}
//             label: "main"(本線発火) / "harass"(潰し) / "all_clear"(全消し) /
//                    "accept"(受け・対応しながら構築) / "build"(土台構築)
// "quit" または EOF で終了。

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <climits>
#include <tuple>
#include <optional>
#include <vector>
#include "../ai/ai.h"

using json = nlohmann::json;

// ぷよぷよ通/ぷよテトのおじゃまぷよ1個あたりの点数
constexpr i32 DEFAULT_TARGET_POINT = 70;

static const char* direction_name(direction::Type r)
{
    switch (r) {
    case direction::Type::UP: return "UP";
    case direction::Type::RIGHT: return "RIGHT";
    case direction::Type::DOWN: return "DOWN";
    case direction::Type::LEFT: return "LEFT";
    }
    return "UP";
}

// 実行ファイルと同じフォルダの config.json から評価重みを読む。
// 無ければ既定の重みで動作する。
static search::Configs load_configs(const std::string& exe_path)
{
    search::Configs configs;
    std::string dir = exe_path;
    auto pos = dir.find_last_of("/\\");
    dir = (pos == std::string::npos) ? "." : dir.substr(0, pos);

    std::ifstream file(dir + "/config.json");
    if (!file.good()) {
        return configs;
    }
    try {
        json js;
        file >> js;
        if (js.contains("build")) {
            // ama の config.json 形式 (build/freestyle/fast/ac の4種)
            from_json(js["build"], configs.build);
            if (js.contains("freestyle")) from_json(js["freestyle"], configs.freestyle);
            if (js.contains("fast")) from_json(js["fast"], configs.fast);
            if (js.contains("ac")) from_json(js["ac"], configs.ac);
        } else {
            // puyop 形式 (build 重みが直下)
            from_json(js, configs.build);
        }
    } catch (...) {
        // 壊れた設定は無視して既定値を使う
    }
    return configs;
}

// ---- 入力の解釈 ----

static std::optional<std::string> parse_field(const json& rows, Field& field)
{
    if (!rows.is_array() || rows.size() != 13) {
        return "field must have 13 rows";
    }
    for (int i = 0; i < 13; ++i) {
        std::string row = rows[i].get<std::string>();
        if (row.size() != 6) {
            return "each row must have 6 cells";
        }
        int y = 12 - i;
        for (int x = 0; x < 6; ++x) {
            cell::Type c = cell::from_char(row[x]);
            if (c != cell::Type::NONE) {
                field.set_cell(x, y, c);
            }
        }
    }
    return {};
}

static std::optional<std::string> parse_queue(const json& items, cell::Queue& queue)
{
    if (!items.is_array()) {
        return "queue must be an array";
    }
    for (const auto& item : items) {
        std::string s = item.get<std::string>();
        if (s.size() != 2) {
            return "each queue item must have 2 chars";
        }
        cell::Type a = cell::from_char(s[0]);
        cell::Type b = cell::from_char(s[1]);
        if (a == cell::Type::NONE || b == cell::Type::NONE || a == cell::Type::GARBAGE || b == cell::Type::GARBAGE) {
            return "queue must consist of R/Y/G/B";
        }
        queue.push_back({a, b});
    }
    if (queue.empty()) {
        return "queue is empty";
    }
    return {};
}

static std::optional<std::string> parse_player(const json& js, gaze::Player& player)
{
    if (auto err = parse_field(js.at("field"), player.field)) return err;
    if (auto err = parse_queue(js.at("queue"), player.queue)) return err;
    if (player.queue.size() < 2) {
        return "battle/analyze requires at least 2 pairs in queue";
    }
    player.all_clear = js.value("all_clear", false);
    player.bonus = js.value("bonus", 0);
    player.attack = js.value("attack", 0);
    player.attack_chain = js.value("attack_chain", 0);
    player.attack_frame = js.value("attack_frame", 0);
    player.dropping = js.value("dropping", 0);
    return {};
}

// ---- 出力の組み立て ----

// 手を置いて連鎖まで進めた後の盤面を13行の文字列(上から)で返す。
// 呼び出し側(PBA)が「提案通りに置いた後の盤面」を予測して次の手を先読みするのに使う
static json field_after_placement(Field field, const move::Placement& p, const cell::Pair& pair, json& out)
{
    field.drop_pair(p.x, p.r, pair);
    auto mask = field.pop();
    auto chain = chain::get_score(mask);
    json rows = json::array();
    for (int y = 12; y >= 0; --y) {
        std::string row;
        for (int x = 0; x < 6; ++x) {
            row.push_back(cell::to_char(field.get_cell(x, y)));
        }
        rows.push_back(row);
    }
    out["field_after"] = rows;
    out["chain_after"] = {{"count", chain.count}, {"score", chain.score}};
    return rows;
}

// 置き場所までの操作手順(1要素=1フレームの入力)を返す。PBA の全自動モード(Steam版でキーボード入力を
// 送って画像認識を試験する)用。ama の path::Finder が回し・床蹴り・180回転の待ちフレームまで含めて生成する
static void input_path(Field& field, const move::Placement& p, const cell::Pair& pair, json& out)
{
    auto queue = path::Finder::find(field, p, pair);
    if (queue.empty()) {
        return;
    }
    auto expanded = path::Finder::get_queue_convert_m180(queue);
    json frames = json::array();
    for (auto mv : expanded) {
        switch (mv) {
        case path::Input::LEFT: frames.push_back("LEFT"); break;
        case path::Input::RIGHT: frames.push_back("RIGHT"); break;
        case path::Input::CW: frames.push_back("CW"); break;
        case path::Input::CCW: frames.push_back("CCW"); break;
        case path::Input::CW_LEFT: frames.push_back("CW_LEFT"); break;
        case path::Input::CW_RIGHT: frames.push_back("CW_RIGHT"); break;
        case path::Input::CCW_LEFT: frames.push_back("CCW_LEFT"); break;
        case path::Input::CCW_RIGHT: frames.push_back("CCW_RIGHT"); break;
        case path::Input::DROP: frames.push_back("DROP"); break;
        case path::Input::WAIT: frames.push_back("WAIT"); break;
        case path::Input::TOUCH: frames.push_back("TOUCH"); break;
        case path::Input::NONE: frames.push_back("NONE"); break;
        default: return;  // M180 が残っている等、送れない手順は返さない
        }
    }
    out["path"] = frames;
}

// 着地マス (軸, 子) を計算する
static void landing_cells(Field& field, const move::Placement& p, json& out)
{
    u8 heights[6];
    field.get_heights(heights);
    int x = p.x;
    int cx = x + direction::get_offset_x(p.r);
    int axis_y, child_y;
    switch (p.r) {
    case direction::Type::UP:
        axis_y = heights[x];
        child_y = heights[x] + 1;
        break;
    case direction::Type::DOWN:
        child_y = heights[x];
        axis_y = heights[x] + 1;
        break;
    default:
        axis_y = heights[x];
        child_y = heights[cx];
        break;
    }
    out["x"] = x;
    out["r"] = direction_name(p.r);
    out["axis"] = {x, axis_y};
    out["child"] = {cx, child_y};
}

static json attack_json(const dfs::attack::Data& a, const move::Placement& p, i32 target_point)
{
    return json{
        {"count", a.count},
        {"score", a.score},
        {"garbage", a.score / target_point},
        {"x", p.x},
        {"r", direction_name(p.r)},
        {"frame", a.frame},
        {"all_clear", a.all_clear},
    };
}

// 盤面と組ぷよ列から「発火したら何連鎖・何点になるか」を集計する
static json analyze_field(Field field, const cell::Queue& queue, i32 target_point)
{
    auto asearch = dfs::attack::search(field, queue);

    dfs::attack::Data immediate, max, harass;
    move::Placement p_immediate, p_max, p_harass;

    for (auto& c : asearch.candidates) {
        auto consider = [&](const dfs::attack::Data& a) {
            if (a.score > max.score) { max = a; p_max = c.placement; }
            if (a.frame == 0 && a.score > immediate.score) { immediate = a; p_immediate = c.placement; }
            if (a.count < 4 && a.score > harass.score) { harass = a; p_harass = c.placement; }
        };
        for (auto& a : c.attacks) consider(a);
        // attacks_detect は「あと1個足せば発火」の見込み。実際に置ける手ではないので集計から除く
    }

    // 将来の本線の見込み(静止探索)
    i32 potential = 0;
    dfs::quiet::search(field, 8, 3, [&](dfs::quiet::Result q) {
        potential = std::max(potential, q.chain.score);
    });

    // 発火点: 列ごとに1個のぷよを落として連鎖が起きる (列, 色) を探す。
    // 組ぷよに依らない「どこに何色を置けば本線が発火するか」の案内用(相手の仕掛けへの対応)
    json fire_points = json::array();
    {
        std::vector<std::tuple<i32, i32, i32, char, i32>> found;  // score, count, x, color, y
        const cell::Type colors[] = { cell::Type::RED, cell::Type::YELLOW, cell::Type::GREEN, cell::Type::BLUE };
        for (i32 x = 0; x < 6; ++x) {
            i32 y = field.get_height(x);
            if (y >= 12) continue;  // 13段目には置けない(見えないので案内しない)
            for (auto color : colors) {
                Field f = field;
                f.drop_puyo(x, color);
                auto mask = f.pop();
                auto chain = chain::get_score(mask);
                if (chain.count <= 0) continue;
                found.emplace_back(chain.score, chain.count, x, cell::to_char(color), y);
            }
        }
        std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        for (auto& [score, count, x, color, y] : found) {
            fire_points.push_back({{"x", x}, {"y", y}, {"color", std::string(1, color)}, {"count", count}, {"score", score}});
        }
    }

    return json{
        {"fire_points", fire_points},
        {"immediate", attack_json(immediate, p_immediate, target_point)},
        {"max", attack_json(max, p_max, target_point)},
        {"harass", attack_json(harass, p_harass, target_point)},
        {"potential", potential},
        {"garbage_max", max.score / target_point},
    };
}

// ---- 各モード ----

// 上流の形パターン(beam/form.h の list)の名前。list と同じ順(表示用。ama 本体には足さない)
static const char* const FORM_NAMES[] = { "GTR", "FRON", "SGTR" };
static_assert(sizeof(FORM_NAMES) / sizeof(FORM_NAMES[0]) == beam::form::COUNT, "形の名前は beam::form::list と同じ数");

// 設置後の盤面に最も合う形の名前を out["form"] に入れる(合う形が無ければ null)。表示用で、探索には影響しない
static void best_form_after(Field field, const move::Placement& p, const cell::Pair& pair, json& out)
{
    field.drop_pair(p.x, p.r, pair);
    field.pop();
    u8 heights[6];
    field.get_heights(heights);
    i32 best = -1;
    i32 best_score = 0;
    for (usize i = 0; i < beam::form::COUNT; ++i) {
        const i32 score = beam::form::evaluate(field, heights, beam::form::list[i]);
        if (score > best_score) {
            best_score = score;
            best = static_cast<i32>(i);
        }
    }
    if (best < 0) out["form"] = nullptr;
    else out["form"] = FORM_NAMES[best];
}

static json handle_build(const json& req, const search::Configs& configs)
{
    Field field;
    if (auto err = parse_field(req.at("field"), field)) return json{{"error", *err}};
    cell::Queue queue;
    if (auto err = parse_queue(req.at("queue"), queue)) return json{{"error", *err}};

    beam::Configs bc;
    if (req.contains("width")) bc.width = req["width"].get<size_t>();
    if (req.contains("depth")) bc.depth = req["depth"].get<size_t>();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = beam::search_multi(field, queue, configs.build, bc);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (result.candidates.empty()) {
        return json{{"error", "no valid placement"}, {"ms", ms}};
    }

    auto best = result.candidates[0];
    json out;
    landing_cells(field, best.placement, out);
    out["score"] = best.score;
    input_path(field, best.placement, queue[0], out);
    field_after_placement(field, best.placement, queue[0], out);
    best_form_after(field, best.placement, queue[0], out);
    out["ms"] = ms;

    json candidates = json::array();
    for (const auto& c : result.candidates) {
        candidates.push_back({{"x", c.placement.x}, {"r", direction_name(c.placement.r)}, {"score", c.score}});
    }
    out["candidates"] = candidates;
    return out;
}

static json handle_analyze(const json& req)
{
    gaze::Player player;
    if (auto err = parse_player(req, player)) return json{{"error", *err}};
    i32 target_point = req.value("target_point", DEFAULT_TARGET_POINT);

    auto t0 = std::chrono::high_resolution_clock::now();
    json out = analyze_field(player.field, player.queue, target_point);
    auto t1 = std::chrono::high_resolution_clock::now();
    out["ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return out;
}

// ---- 相手の連鎖の再現 ----
constexpr i32 ALL_CLEAR_BONUS_GARBAGE = 30;  // 全消しボーナス(個)

// 相手が連鎖中の時、その連鎖の着弾量と連鎖後の盤面。発火直前の盤面(引き金込み)を消して再現できる時だけ
// 値を持つ(2026-09-20 の方針: 見積もりで代用しない。点滅フレームから復元した盤面(flash)、または追跡している
// 盤面に消える組がある(replay)時が「再現できる」)。再現できない時は ama に生の予告アイコン数を渡す
struct EnemyChainEstimate
{
    i32 garbage = 0;      // 着弾するおじゃま(個)
    i32 count = 0;        // 連鎖数
    Field after;          // 連鎖後の相手の盤面
    std::string source;   // "flash_replay"(点滅から復元した盤面) / "replay"(追跡盤面から再現)
};

static std::optional<EnemyChainEstimate> replay_enemy_chain(const gaze::Player& enemy, i32 target_point, i32 chain_bonus, const std::optional<Field>& chain_field)
{
    EnemyChainEstimate e;
    if (chain_field) {
        e.after = *chain_field;
        auto mask = e.after.pop();
        auto chain = chain::get_score(mask);
        if (chain.count > 0) {
            e.garbage = (chain.score + chain_bonus) / target_point;
            e.count = chain.count;
            e.source = "flash_replay";
            return e;
        }
    }
    e.after = enemy.field;
    auto mask = e.after.pop();
    auto chain = chain::get_score(mask);
    if (chain.count > 0) {
        e.garbage = (chain.score + chain_bonus) / target_point;
        e.count = chain.count;
        e.source = "replay";
        return e;
    }
    return std::nullopt;
}

static json handle_battle(const json& req, const search::Configs& configs)
{
    gaze::Player self, enemy;
    if (auto err = parse_player(req.at("self"), self)) return json{{"error", std::string("self: ") + *err}};
    if (auto err = parse_player(req.at("enemy"), enemy)) return json{{"error", std::string("enemy: ") + *err}};
    i32 target_point = req.value("target_point", DEFAULT_TARGET_POINT);
    // JSON の "attack" は「そのプレイヤーに降る予告おじゃま」(画面に見える値)。ama の gaze::Player.attack は
    // 「そのプレイヤーが相手に送ったおじゃま」(ai::think の balance = self.attack - enemy.attack が正なら
    // 相手の盤面に落ちる)なので入れ替える。入れ替えないと ama は常に「自分が攻撃中」と解釈し、
    // 相手の攻撃を受けている分岐(本線での返し・受け)に入らない(実機 2026-09-17: 相手の7連鎖に無反応で負け)
    std::swap(self.attack, enemy.attack);
    // 全消しボーナス: ぷよぷよテトリス2 では次の連鎖に +30個(target_point 換算)が乗る。ama は
    // これを bonus(点数)で受け取る設計(attack_send = (score + bonus) / target_point、とどめ判定の
    // enemy.all_clear * 30 は相手側だけ)なので、フラグだけで渡されたら bonus を補う
    if (self.all_clear && self.bonus == 0) self.bonus = ALL_CLEAR_BONUS_GARBAGE * target_point;
    if (enemy.all_clear && enemy.bonus == 0) enemy.bonus = ALL_CLEAR_BONUS_GARBAGE * target_point;
    // 相手が連鎖中で、その連鎖を盤面から再現できる時だけ、合計の着弾量を ama に渡す
    // (予告アイコン数は途中の値。再現できない時は生の値のまま)
    const i32 enemy_chain_bonus = req.at("enemy").value("chain_bonus", false) ? ALL_CLEAR_BONUS_GARBAGE * target_point : 0;
    std::optional<EnemyChainEstimate> replayed;
    if (enemy.attack_frame > 0) {
        std::optional<Field> chain_field;
        const json& ej = req.at("enemy");
        if (ej.contains("chain_field")) {
            Field f;
            if (auto err = parse_field(ej.at("chain_field"), f)) return json{{"error", std::string("enemy.chain_field: ") + *err}};
            chain_field = f;
        }
        replayed = replay_enemy_chain(enemy, target_point, enemy_chain_bonus, chain_field);
        if (replayed) {
            enemy.attack = std::max(enemy.attack, replayed->garbage);
            enemy.attack_frame = std::max(1, std::min(4, replayed->count / 2));
            // PBA が経過時間から見積もった「連鎖が終わるまでに置ける手数」があれば使う。固定値では経過時間が
            // 反映されず、6連鎖の間ずっと「3手後」の攻撃を狙い続けて間に合わなかった(実機 2026-09-27 15:24)
            if (ej.contains("attack_frame_est")) {
                enemy.attack_frame = std::max(0, ej.at("attack_frame_est").get<i32>());
            }
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // ama の本来の流れ: 操作中に構築探索(search::Thread)を回し、その結果を
    // ai::think に渡す。ただし search::Thread は組ぷよが3組以上あると「与えられた
    // 組ぷよの数だけ」の深さで beam::search するため(ama のクライアントは乱数で
    // 延長した長いキューを渡す前提)、ネクスト2組しか見えない画面認識では探索が
    // 浅すぎて構築の手が弱くなる。ここでは build モードと同じ search_multi
    // (乱数で depth まで延長した複数キューを並列探索)で構築候補を作る。
    search::Result bsearch;
    bsearch.build = beam::search_multi(self.field, self.queue, configs.build);
    cell::Queue short_queue = { self.queue[0], self.queue[1] };
    bsearch.freestyle = dfs::build::search(self.field, short_queue, configs.freestyle);
    bsearch.fast = dfs::build::search(self.field, short_queue, configs.fast);
    bsearch.ac = dfs::build::search(self.field, short_queue, configs.ac);

    // 判断の診断用(ai::think が比較に使う値。ai::think は値渡しなので呼ぶ前の値と同じ)
    const i32 diag_enemy_attack = std::max(0, enemy.attack - self.attack);
    const i32 diag_attack_frame = enemy.attack_frame;
    const i32 diag_bonus = self.bonus;

    auto result = ai::think(self, enemy, bsearch, configs, target_point);

    // 手の意味(ラベル)を判定する。ai::think は理由を返さないので、
    // 選ばれた手で発火する攻撃があり、その点数が評価値と一致すれば「発火」とみなす。
    std::string label = "build";
    auto self_attacks = dfs::attack::search(self.field, self.queue);
    for (auto& c : self_attacks.candidates) {
        if (!(c.placement == result.placement)) continue;
        for (auto& a : c.attacks) {
            if (a.frame == 0 && a.score == result.eval) {
                if (a.all_clear) label = "all_clear";
                else if (a.count >= 4) label = "main";
                else label = "harass";
            }
        }
    }
    if (label == "build" && enemy.attack > self.attack) {
        label = "accept";
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    json out;
    landing_cells(self.field, result.placement, out);
    out["eval"] = result.eval;
    input_path(self.field, result.placement, self.queue[0], out);
    field_after_placement(self.field, result.placement, self.queue[0], out);
    best_form_after(self.field, result.placement, self.queue[0], out);
    out["label"] = label;
    out["fire_reason"] = nullptr;  // ama の判断を差し替えない(本線発火の目安は 2026-10-21 に外した)
    // 相手の攻撃を受けている時、本線で返せたか(ai.cpp Step 2 の条件)を診断として返す
    // (実機 2026-09-27 12:44: 206個に対し「最大8連鎖」があるのに返さなかった理由を切り分ける)
    if (diag_enemy_attack > 0) {
        const dfs::attack::Data* in_time = nullptr;
        const dfs::attack::Data* any = nullptr;
        for (auto& c : self_attacks.candidates) {
            for (const auto* list : { &c.attacks, &c.attacks_detect }) {
                for (auto& a : *list) {
                    if (any == nullptr || a.score > any->score) any = &a;
                    if (a.frame <= diag_attack_frame && (in_time == nullptr || a.score > in_time->score)) in_time = &a;
                }
            }
        }
        auto brief = [&](const dfs::attack::Data* a) -> json {
            if (a == nullptr) return nullptr;
            return json{{"score", a->score}, {"count", a->count}, {"frame", a->frame},
                        {"send", (a->score + diag_bonus) / target_point}};
        };
        Field accept_field = self.field;
        out["diag"] = {
            {"enemy_attack", diag_enemy_attack},
            {"attack_frame", diag_attack_frame},
            {"accept_limit", gaze::get_accept_limit(accept_field)},
            {"best_in_time", brief(in_time)},
            {"best_any", brief(any)},
        };
    }
    // 自分に降ると再現したおじゃま(個)とその根拠(相手の連鎖を再現できた時だけ。表示・記録用)
    if (replayed) {
        out["incoming"] = replayed->garbage;
        out["incoming_source"] = replayed->source;
        out["incoming_chain"] = replayed->count;
    }
    out["self"] = analyze_field(self.field, self.queue, target_point);
    out["enemy"] = analyze_field(enemy.field, enemy.queue, target_point);
    out["ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    out["queue_used"] = self.queue.size();  // 構築のビームサーチに使った既知のツモの組数
    return out;
}

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);

    search::Configs configs = load_configs(argc > 0 ? argv[0] : "");

    std::cout << json{{"ready", true}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "quit") break;
        json out;
        try {
            json req = json::parse(line);
            std::string mode = req.value("mode", "build");
            if (mode == "build") {
                out = handle_build(req, configs);
            } else if (mode == "analyze") {
                out = handle_analyze(req);
            } else if (mode == "battle") {
                out = handle_battle(req, configs);
            } else {
                out = json{{"error", "unknown mode: " + mode}};
            }
        } catch (const std::exception& e) {
            out = json{{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}
