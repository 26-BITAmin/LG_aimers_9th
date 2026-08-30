"""`src/ftt.py` 의 **피처 그룹 정의** — 헤드가 무엇을 읽는지를 정하는 유일한 파일.

아키텍처는 이 딕셔너리들의 껍데기다. `shuf` 위약이 정확히 "구성 내용"만 지운
팔이라, **scheme − shuf 격차가 곧 이 파일의 값어치**다.

규칙
  · 이름은 `enc["num_cols"]` 의 수치 컬럼명 또는 `mlp_infer.EMB_COLS` 의 범주 컬럼명.
    오타는 `_check_cover` 가 KeyError 로 즉시 죽인다.
  · **겹쳐도 된다.** 한 피처가 여러 그룹에 들어가면 여러 헤드가 그걸 읽는다.
  · 빠지면 그 피처는 쿼리만 던지고 **아무 헤드도 안 읽는다** = 사실상 삭제다.
    `_check_cover` 가 미포함 목록을 출력한다. 확인하고 넘어갈 것.
  · **총 헤드 수는 scheme 마다 같아야 한다** (`N_HEAD`). 안 그러면 재는 것이
    "분할 효과"가 아니라 "헤드 수 효과"가 된다. `full` 대조군도 같은 수를 쓴다.

전역 헤드는 **0개**다 (사용자 결정, 2026-08-30). 완전 분할이라 그룹 경계를 넘는
상호작용은 2번째 블록에서 토큰이 이미 그룹 요약을 흡수한 뒤에야 간접적으로 생긴다.
`full` 과의 대비가 가장 선명한 설정이다.
"""

N_HEAD = 16          # 전 scheme 공통. 바꾸면 full 대조군도 같이 바뀐다.

# 컬럼을 이름으로 묶어두면 scheme 사이에서 재사용할 수 있고 오타가 한 곳에 모인다.
_CNT = ["balls_before", "strikes_before", "count_diff", "is_two_strikes",
        "is_three_balls", "is_full_count", "is_first_pitch", "count_state"]
_SIT = ["inning", "outs_before", "run_top_before", "run_bot_before",
        "run_total_before", "score_diff_home", "score_diff_pitcher_team",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
        "home_win_expectancy", "li", "we_margin", "is_pitcher_home",
        "abs_score_diff", "is_late_close", "top_bottom", "base_state"]
_HND = ["pitcher_hand", "batter_hand", "hand_same", "hand_matchup",
        "matchup_success_edge", "matchup_middle_edge"]
_PSK = ["asof_pitcher_n", "miss_pitcher_asof",
        "asof_pitcher_success_rate", "asof_pitcher_success_rate_sm",
        "asof_pitcher_middle_rate", "asof_pitcher_middle_rate_sm",
        "asof_pitcher_reverse_rate", "asof_pitcher_reverse_rate_sm",
        "asof_pitcher_ball_rate", "asof_pitcher_ball_rate_sm",
        "asof_pitcher_strike_rate", "asof_pitcher_strike_rate_sm",
        "fail_middle_share", "pitcher_ball_minus_strike"]
_PFM = ["miss_prev_game",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "form_p1_minus_career", "form_p3_minus_career", "form_p1_minus_p5",
        "form_p3_minus_p5", "form_mid_p1_minus_career"]
_PMX = ["asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_fastball_rate_sm",
        "asof_pitcher_breaking_rate", "asof_pitcher_breaking_rate_sm",
        "asof_pitcher_offspeed_rate", "asof_pitcher_offspeed_rate_sm",
        "pitchmix_entropy", "pitchmix_nonfastball"]
_BAT = ["asof_batter_n", "miss_batter_asof",
        "asof_batter_success_rate", "asof_batter_success_rate_sm",
        "asof_batter_middle_rate", "asof_batter_middle_rate_sm",
        "batter_id", "batter_team_id"]
_IDT = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


# ══════════════════════════════════════════════════════════════════════
# ① sem8 — 의미 축 8그룹. 세분화가 최대, 그룹당 토큰 5~20개.
# ══════════════════════════════════════════════════════════════════════
SEM8 = {
    "count":   _CNT + ["outs_before", "inning"],
    "state":   _SIT,
    "hand":    _HND,
    "p_skill": _PSK,
    "p_form":  _PFM,
    "p_mix":   _PMX,
    "batter":  _BAT,
    "ident":   _IDT,
}
SEM8_PLAN = {g: 2 for g in SEM8}                      # 8 × 2 = 16


# ══════════════════════════════════════════════════════════════════════
# ② sem8ov — 같은 8그룹, 축에 걸치는 피처를 **중복 배치**한다.
#
# 어떤 기준으로 겹치나: "그 그룹의 값을 읽을 때 조건이 되는 것"을 넣는다.
#   · 폼(prev1/3/5)은 커리어 기준선 없이는 뜻이 없다 → p_form 에 커리어 성공률
#   · 구종 구사는 좌우에 따라 달라진다 → p_mix 에 pitcher_hand
#   · 투수 제구력은 카운트에 따라 다르게 발현된다 → p_skill 에 카운트 3개
#   · 좌우 우위는 결국 누구냐에 걸린다 → hand 에 id 2개
# 대가: 헤드당 key 가 11~20 → 15~26 으로 늘어 분할의 순도가 떨어진다.
# ══════════════════════════════════════════════════════════════════════
SEM8OV = {
    "count":   SEM8["count"] + ["hand_matchup", "asof_pitcher_success_rate_sm"],
    "state":   SEM8["state"] + ["count_state", "is_two_strikes"],
    "hand":    SEM8["hand"] + ["pitcher_id", "batter_id",
                               "asof_pitcher_reverse_rate_sm"],
    "p_skill": SEM8["p_skill"] + ["count_state", "is_two_strikes",
                                  "is_three_balls", "pitcher_hand"],
    "p_form":  SEM8["p_form"] + ["asof_pitcher_success_rate",
                                 "asof_pitcher_success_rate_sm",
                                 "asof_pitcher_middle_rate_sm", "asof_pitcher_n"],
    "p_mix":   SEM8["p_mix"] + ["pitcher_hand", "count_state", "batter_hand"],
    "batter":  SEM8["batter"] + ["hand_matchup", "matchup_success_edge",
                                 "matchup_middle_edge", "count_state"],
    "ident":   SEM8["ident"] + ["pitcher_hand", "batter_hand", "asof_pitcher_n",
                                "asof_batter_n"],
}
SEM8OV_PLAN = {g: 2 for g in SEM8OV}                  # 8 × 2 = 16


# ══════════════════════════════════════════════════════════════════════
# ③ subj3 — 주체 축 3그룹. 그룹이 커서 어텐션이 그룹 **안에서** 실제로 고를
#    여지가 크다. 위약과의 격차가 가장 크게 날 수 있는 설정이다.
#
# 좌우·매치업(`_HND`)은 투수와 타자 **양쪽에** 넣는다. 그건 어느 한쪽의 속성이
# 아니라 둘의 관계라서, 한쪽에만 넣으면 다른 쪽 헤드가 그 조건을 못 본다.
# ══════════════════════════════════════════════════════════════════════
SUBJ3 = {
    "pitcher": _PSK + _PFM + _PMX + ["pitcher_hand", "hand_same", "hand_matchup",
                                     "pitcher_id", "pitcher_team_id"],
    "batter":  _BAT + ["batter_hand", "hand_same", "hand_matchup",
                       "matchup_success_edge", "matchup_middle_edge"],
    "context": _CNT + _SIT,
}
# 16 을 3 으로 못 나눈다. 투수 피처가 압도적으로 많으니(38개) 그쪽에 하나 더 준다.
SUBJ3_PLAN = {"pitcher": 6, "batter": 5, "context": 5}    # = 16

CORE = {
    "sem8":   (SEM8, SEM8_PLAN),
    "sem8ov": (SEM8OV, SEM8OV_PLAN),
    "subj3":  (SUBJ3, SUBJ3_PLAN),
}

# ══════════════════════════════════════════════════════════════════════
#  헤드 수 변형 — **문헌이 예측을 못 하는 지점** (2026-08-30)
# ══════════════════════════════════════════════════════════════════════
# 문헌: BERT 헤드의 70~90% 가 제거 가능 · 기계번역은 17% 만 남겨도 92% ·
#       "16 헤드가 8 헤드의 2배가 아니다" · 헤드 중요도가 멱법칙
#       (Michel et al. 2019 계열 · `Finding the Pillars of Strength` arXiv:2305.14380)
#
# ⚠️ 그 증거는 전부 **헤드들이 같은 입력을 볼 때**의 것이다. 논문들이 드는 중복의
#    원인이 정확히 그것이다 — 병렬 설계라 자연히 비슷한 걸 본다. **헤드마다 입력을
#    다르게 강제하는 것은 그 원인을 구조적으로 제거하는 개입**이라, "헤드는
#    중복된다" 가 이 설계로 전이된다는 보장이 없다.
#    [[literature-transfers-by-regime-not-by-method]]
#
# → 그래서 헤드 수 스윕은 **마스크 있는 팔에서** 한다. `full` 에서 재면 그냥 문헌
#   재현이고, 이 설계가 다른지를 못 묻는다.
#
# 그룹은 SEM8 그대로, **그룹당 헤드 수만** 바꾼다:
#   sem8x1  8그룹 × 1 =  8 헤드   (문헌 sweet spot 4~8 쪽)
#   sem8    8그룹 × 2 = 16        (기준)
#   sem8x4  8그룹 × 4 = 32        (사용자 제안: 헤드 ↑ + 용량 ↑)
# ⚠️ `dh` 고정이면 H 가 늘 때 **용량도 는다** (H16 122k → H32 188k).
#    용량을 고정하려면 `--inner 128` 을 같이 준다. 둘 다 재야 갈린다.
HEADVAR = {
    "sem8x1": (SEM8, {g: 1 for g in SEM8}),      # H = 8
    "sem8x4": (SEM8, {g: 4 for g in SEM8}),      # H = 32
}

SCHEMES = dict(CORE, **HEADVAR)

for _n, (_g, _p) in SCHEMES.items():
    assert set(_g) == set(_p), f"{_n}: GROUPS 와 HEAD_PLAN 의 그룹이 다르다"
for _n, (_g, _p) in CORE.items():
    # 핵심 비교(그룹 축)는 헤드 수가 **같아야** 한다 — 안 그러면 재는 것이
    # "분할 효과" 가 아니라 "헤드 수 효과" 가 된다. HEADVAR 는 의도적 예외다.
    assert sum(_p.values()) == N_HEAD, f"{_n}: 핵심 scheme 은 헤드 {N_HEAD}개여야 한다"
