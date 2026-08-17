"""asof 차분 — 커리어 누적에서 **시즌 진행분**을 분리한다.

기전
    `asof_*`는 커리어 누적이다 (§7.44 A: 시즌 경계 연속성 100%).
    §7.44는 이 사실에 **나눗셈**을 걸었다 (era 상대화 → 기각).
    여기서는 **뺄셈**을 건다. 나오는 것이 다르다.

        n_szn = asof_n − n_off(pitcher, season)          이번 시즌 던진 공 수
        k_szn = asof_rate × asof_n − k_off(...)          이번 시즌 성공 횟수

    `k = rate × n`이 정수로 복원된다는 것이 이 모듈의 전제다. 측정했다:
    성공·몰림·타자·구종믹스 전부 `max|r×n − round| < 0.008` (저장 6자리).
    **비율은 뺄 수 없고 횟수는 뺄 수 있다.** 이 한 줄이 채널의 존폐를 가른다.

왜 새 채널인가
    `asof_*`는 커리어, `prev1/3/5_game`은 5경기 창. **시즌 단위 창이 통째로 비어 있다.**
    그리고 트리는 투수별 상수를 뺄 수 없으므로 이 값을 스스로 만들 수 없다.
    §7.34 G가 잰 투수 축 천장(838.8)이 겨냥하는 양이 바로 "이 투수의 현재 레벨"이다.

    §7.52 P4가 그 간격을 숫자로 줬다 — 커리어 성공률은 다음 시즌 레벨을
    **이항 노이즈의 3.4배**(E[z²]=11.84)만큼, 그리고 **체계적으로 한 방향으로**
    (E[z] = −1.2~−1.7) 빗나간다.

리키지 (CLAUDE.md §3.2)
    ⚠️ 오프셋 테이블은 **train에서만** 만들어지고 추론에서는 조회만 된다.
    각 행은 자기 (pitcher_id, season) 키로 searchsorted 할 뿐이다 —
    `test.csv`의 다른 행을 보지 않는다 (Trackman `tm`/`tmb` 조회와 같은 형태).

    시즌 s의 오프셋은 **s 이전 시즌만으로** 만든다. 그래서 학습 행도 자기 시즌의
    미래를 보지 않고, 배포(2025 오프셋 = 2024 종료 시점)와 **같은 구조**가 된다.
    → fold 검증 설정이 배포 설정과 일치한다. §7.22가 "fold 간 외삽은 식별되지
      않는다"고 한 문제를 이 축은 구조적으로 피한다.

검증됨 (2024를 가짜 test로, ≤2023 오프셋)
    복원한 n_szn 과 직접계산 시즌순번의 일치율 **1.000000** (신규투수 19.9% 포함).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# (rate 컬럼, 표본수 컬럼). k = rate × n 이 정수 복원되는 것만 넣는다.
DELTA_SPEC = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
]
ID_OF = {"asof_pitcher_n": "pitcher_id", "asof_pitcher_pitchmix_n": "pitcher_id",
         "asof_batter_n": "batter_id"}


def fit_offsets(train_df, target_seasons=None):
    """train에서만 호출. (엔티티, 시즌) → **그 시즌 시작 시점의 누적 상태**.

    시즌 s의 값은 s 이전 시즌 행만으로 만들어진다. 배포 시 test 시즌(2025)의
    오프셋은 곧 train 마지막 시즌(2024) 종료 시점이다 — 학습 행과 같은 경로.

    반환: {n_col: {"keys": int64[], "n_off": f8[], "k_off": {rate_col: f8[]}}}
    키 = 엔티티 id * 10000 + season  (features.py의 tm 조회와 같은 규약).
    """
    seasons = sorted(train_df["season"].unique().tolist())
    if target_seasons is None:                    # 학습 시즌 + 그 다음 한 해(=test)
        target_seasons = seasons + [max(seasons) + 1]

    art = {}
    for ncol in dict.fromkeys(n for _, n in DELTA_SPEC):
        idcol = ID_OF[ncol]
        rates = [r for r, n in DELTA_SPEC if n == ncol]
        cols = [idcol, "season", ncol] + rates
        d = train_df[cols].copy()

        # 각 (엔티티, 시즌)의 **종료 시점** 상태.
        # asof는 '직전까지'이므로 그 시즌 마지막 행의 값에 그 행 한 건을 더해야
        # 종료 시점이 된다. rate는 그 한 건의 결과를 모르므로 n만 +1 하고
        # k는 그대로 둔다 — 1건 오차는 n_szn ≥ 100 구간에서 무시 가능하고,
        # 2024 가짜 test 검증에서 일치율 1.000000 을 확인했다 (control_success 사용).
        d = d.sort_values([idcol, "season", ncol])
        end = d.groupby([idcol, "season"]).tail(1)
        e = pd.DataFrame({idcol: end[idcol].to_numpy(),
                          "season": end["season"].to_numpy(),
                          "n": end[ncol].to_numpy(dtype=np.float64) + 1.0})
        for r in rates:
            e[r] = (end[r].to_numpy(dtype=np.float64)
                    * end[ncol].to_numpy(dtype=np.float64))
        e = e.fillna(0.0).sort_values([idcol, "season"])

        # 시즌 s의 시작 = 존재하는 시즌 중 s 미만의 **마지막** 종료 상태.
        # (중간에 결장한 시즌이 있어도 마지막 관측으로 이어진다)
        rows = []
        for ent, g in e.groupby(idcol, sort=False):
            sv = g["season"].to_numpy()
            for s in target_seasons:
                j = np.searchsorted(sv, s) - 1          # s 미만의 마지막
                if j < 0:
                    continue                            # 데뷔 전 → 오프셋 0 (조회 실패 = 0)
                rows.append((ent, s, g["n"].to_numpy()[j],
                             *[g[r].to_numpy()[j] for r in rates]))
        t = pd.DataFrame(rows, columns=[idcol, "season", "n"] + rates)

        key = (t[idcol].to_numpy(dtype=np.int64) * 10000
               + t["season"].to_numpy(dtype=np.int64))
        o = np.argsort(key)
        art[ncol] = {"keys": key[o], "n_off": t["n"].to_numpy(dtype=np.float64)[o],
                     "k_off": {r: t[r].to_numpy(dtype=np.float64)[o] for r in rates}}
    return art


def season_to_date(df, art):
    """행 단위 복원. **df 전체 집계 없음** — 각 행이 자기 키로 조회할 뿐이다.

    반환: {n_col: n_szn, rate_col: k_szn}. 미등록(신인·데뷔전)은 오프셋 0 →
    n_szn = 커리어 = 시즌분이 되어 **정의상 옳다** (2024 검증에서 확인).
    """
    out = {}
    for ncol, a in art.items():
        idcol = ID_OF[ncol]
        kv = (df[idcol].to_numpy(dtype=np.int64) * 10000
              + df["season"].to_numpy(dtype=np.int64))
        j = np.clip(np.searchsorted(a["keys"], kv), 0, max(len(a["keys"]) - 1, 0))
        hit = (len(a["keys"]) > 0) & (a["keys"][j] == kv)

        n_now = df[ncol].to_numpy(dtype=np.float64)
        out[ncol] = np.maximum(n_now - np.where(hit, a["n_off"][j], 0.0), 0.0)
        for r, k_off in a["k_off"].items():
            k_now = df[r].to_numpy(dtype=np.float64) * n_now
            out[r] = np.clip(np.nan_to_num(k_now) - np.where(hit, k_off[j], 0.0), 0.0, None)
    return out


# ── ★ 설계 결정 — 여기를 채워야 한다 ─────────────────────────────
def shrink(hits_szn, n_szn, r_career, league, n_career=None):
    """시즌 진행 비율을 저표본에서 어디로, 얼마나 수축시킬 것인가.

    ⚠️ 이름 규약: 이 모듈에서 **`hits`는 성공 횟수**, **`strength`는 수축 강도**다.
       저장소의 `SMOOTH_K`가 후자를 `k`라 부르므로 여기서는 `k`를 쓰지 않는다.
       (초안에서 `k_szn`(횟수)과 `SMOOTH_K`(강도)를 같은 글자로 써서 혼동을 만들었다)

    왜 필요한가
        시즌 초 30구짜리 `k_szn/n_szn`은 거의 순수 노이즈다. n_szn 분위수는
        p5=33 · p25=208 · p50=505 (§ 측정) — test 행의 상당수가 저표본 구간이다.

    §7.52가 좁혀준 것
        ❌ **레퍼토리 변동성으로 k를 투수별화하는 경로는 닫혔다** (P3 +0.035,
           P4 t=−0.46). 이 축은 후보에서 지운다.
        ⚠️ 커리어로 수축하면 **E[z] = −1.2~−1.7의 편의를 그대로 물려받는다**
           (커리어는 다음 시즌을 체계적으로 과대평가한다).

    남은 선택지 — 서로 다른 것을 가정한다
        (a) 커리어로 수축   개인 사전을 지키지만 위 편의를 들여온다.
        (b) 리그 평균으로   편의는 없지만 개인차를 버리고, **2025 리그 평균을 모른다**.
        (c) 손잡이×보직 그룹 평균으로   중간. 보직은 `n_szn / 경과월`로 만들 수 있다.

        그리고 수축 강도(strength)를 상수로 둘지, **n_career 대비 비율**로 둘지가
        같이 걸린다. 후자면 커리어 5000구 베테랑은 시즌 500구로 안 움직이고
        신인은 확 움직인다.

    ⚠️ 편의 보정을 여기에 굽지 않는 선택지도 있다 — `r_szn`과 `r_career`를
       **둘 다** 피처로 내보내면 트리가 그 차이의 사용법을 학습에서 배운다.
       그러면 이 함수의 역할은 **분산 통제 하나로 한정**된다. (기본 권고)

    Args:
        hits_szn:  시즌 진행분 **성공 횟수** (행 단위 배열)
        n_szn:     시즌 진행분 **표본 수** (던진 공 수)
        r_career:  그 행의 커리어 비율 (결측 가능)
        league:    train에서 얼린 리그 평균 스칼라
        n_career:  커리어 표본 수 (강도를 비율형으로 둘 때만 필요)
    Returns:
        r_szn: 수축된 시즌 진행 비율 (행 단위 배열, 결측 없음)

    현재 구현 = **(a) 커리어로 수축, 강도 상수 200** (잠정 기본값).
    측정을 막지 않으려고 권고안을 넣어뒀다. 편의 보정은 여기서 굽지 않는다 —
    `r_szn`과 `r_career`를 둘 다 내보내면 트리가 학습에서 배운다.
    (b)/(c)로 바꾸려면 이 함수만 갈아끼우면 된다.
    """
    strength = 200.0
    prior = np.where(np.isfinite(r_career), r_career, league)
    return (hits_szn + strength * prior) / (np.maximum(n_szn, 0.0) + strength)


# ── 사후 가산 축 레지스트리 ──────────────────────────────────────────
# ⚠️ **새 축을 추가할 때 고칠 곳은 여기 하나다** (`features.CTX_CODERS`와 같은 규약).
#    `features.asof_lookup`은 이 명세를 아티팩트에서 읽으므로 투수/타자를 구분하지
#    않는다. train / verify / predict / build_submit이 자동으로 같은 것을 본다.
#
#   gap_success   투수의 올해 성공률 − 커리어   (§7.55, LB +26.14, 배율 0.93)
#   gap_bsuccess  타자가 올해 상대해온 제구 수준 − 커리어  (§7.57)
#     └ `gap_success`와 **거의 직교**하다: 증분 잔차 t가 6.70 → 6.53으로만 준다.
#       다른 엔티티이기 때문이다. §7.34 G의 `batter_id` 천장 −25.7은 **레벨**을
#       잰 것이고 이건 **변화**다 — 같은 관문이 아니다.
ASOF_SPEC = {
    "gap_success": dict(id="pitcher_id", n_col="asof_pitcher_n",
                        r_col="asof_pitcher_success_rate"),
    "gap_bsuccess": dict(id="batter_id", n_col="asof_batter_n",
                         r_col="asof_batter_success_rate"),
}


def fit_asof_axes(train_df, axes, strength=200.0):
    """배포용 축 리스트를 만든다. **train에서만 호출.**

    axes: [(축이름, β), ...].  β=0인 축은 건너뛴다 (δ 축과 같은 규약).
    """
    off_all = fit_offsets(train_df)
    league = float(train_df["control_success"].mean())
    import features as _f              # 소비 측과 **같은 코드**로 통계를 뽑는다

    out = []
    for name, beta in axes:
        if not beta:
            continue
        spec = ASOF_SPEC[name]
        off = off_all[spec["n_col"]]
        a = dict(spec, name=name, keys=off["keys"], n_off=off["n_off"],
                 k_off=off["k_off"][spec["r_col"]],
                 mu=0.0, sd=1.0, beta=0.0,   # mu/sd를 먼저 0/1로 두고 원단위를 잰다
                 league=league, strength=float(strength))
        raw = _f.asof_lookup(train_df, a)
        # ⚠️ 표준화 통계는 **학습 행에서만** 뽑는다. 검증/추론 행의 평균을 쓰면
        #    그 자체가 리크다 (tmbfit의 교훈).
        a["mu"] = float(np.mean(raw))
        a["sd"] = float(np.std(raw)) or 1.0
        a["beta"] = float(beta)
        out.append(a)
    return out


def fit_asof(train_df, beta, strength=200.0):
    """(구) 단일 축 API — **측정 스크립트 호환용**이다.

    `gap_success` 축 하나를 새 형식으로 돌려준다. 소비 측(`features.logit_adjust`)은
    리스트를 받으므로 호출부에서 `[a]`로 감싼다.

    배포 경로는 `fit_asof_axes`를 쓴다 — 축이 둘 이상이기 때문이다 (§7.57).
    """
    a = fit_asof_axes(train_df, [("gap_success", beta or 1.0)], strength)[0]
    a["beta"] = float(beta)
    return a


__all__ = ["DELTA_SPEC", "ASOF_SPEC", "fit_offsets", "season_to_date",
           "shrink", "fit_asof", "fit_asof_axes"]
