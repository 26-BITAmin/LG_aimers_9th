"""제출 zip 의 `artifacts['asof']` 에 **시즌 진행분 축 하나를 추가**한다 (재학습 없음).

왜 이 도구인가
    `features.asof_lookup` 은 축의 명세(`id`/`n_col`/`r_col`/오프셋표/mu/sd/β)를
    **아티팩트에서 읽는다** — 코드에 축 이름이 하드코딩돼 있지 않다. 그래서 축 추가는
    모델 재학습이 아니라 `artifacts.joblib` 안 리스트에 원소 하나를 붙이는 일이다.
    zip 의 `features.py`·`contam.py` 는 손대지 않는다.

    ⚠️ β 단위. 배포 축은 **표준화**돼 있다 (`v = (gap - mu)/sd`). 원단위 gap 위에서
    잰 β 를 그대로 넣으면 sd 배(~0.02) 만큼 과다하다. β 는 `src/grevaxis.py` 처럼
    **배포 함수로 만든 축** 위에서 골라 와야 한다.

재현 관문 (G0)
    이 도구가 만드는 축은 `src/asofdelta.fit_asof_axes` 산출물이다. 같은 함수로
    챔피언의 기존 3축(`gap_success`·`gap_bsuccess`·`gap_ball`)을 다시 만들면
    mu/sd 가 1e-9 안에서 일치한다 (`src/grevaxis.py --repro` 실측). 그래서 새 축도
    같은 규약으로 만들어졌다고 믿을 수 있다. 이 도구는 실행 때마다 그 대조를 **다시** 한다.

PROB_SHIFT 재보정
    축을 더하면 예측 평균이 움직인다. 확률공간 아핀 뒤의 `PROB_SHIFT` 는 목표 prior 에
    맞춰 고정된 값이라 그 이동을 되돌려야 한다. train.csv 2024행을 프록시로 Δmean 을
    재고 그만큼 뺀다 (`rebase_routing.py`·`rebase_coldstart_batter.py` 와 같은 절차).

사용
    python tools_hj/rebase_asof_axis.py --src hj05_r047f330.zip --out hj08_grev.zip \
        --axis gap_reverse --beta -0.03 --contam
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "src"))
try:                      # Windows 기본 콘솔은 cp949 라 em-dash 하나에 죽는다
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np

from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha


def build_axis(train_df, name, beta, strength, ct):
    import asofdelta
    from gapstrike import EXTRA_SPEC
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    return asofdelta.fit_asof_axes(train_df, [(name, beta)], strength, ct=ct)[0]


def repro_check(train_df, art, strength):
    """기존 축을 우리 코드로 다시 만들어 mu/sd 가 맞는지 — 매 실행마다."""
    import asofdelta
    ok = True
    for a in art["asof"]:
        asofdelta.ASOF_SPEC.setdefault(
            a["name"], {k: a[k] for k in ("id", "n_col", "r_col")})
        m = build_axis(train_df, a["name"], float(a["beta"]), strength,
                       art.get("contam") if "cadj" in a else None)
        good = (abs(m["mu"] - a["mu"]) < 1e-9 and abs(m["sd"] - a["sd"]) < 1e-9
                and len(m["keys"]) == len(a["keys"])
                and np.array_equal(m["keys"], a["keys"])
                and np.allclose(m["k_off"], a["k_off"], atol=0, rtol=0))
        ok &= good
        print(f"  재현 {a['name']:<14} mu {a['mu']:+.9f}/{m['mu']:+.9f}  "
              f"sd {a['sd']:.9f}/{m['sd']:.9f}  {'OK' if good else 'NG'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--contam", action="store_true",
                    help="pre-2023 F 오염 보정(cadj)을 축에 단다 (§7.60)")
    a = ap.parse_args()

    import pandas as pd
    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="asofax_", dir="D:/t1eval")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    pp = os.path.join(work, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    m = SHIFT_RE.findall(src0)
    assert len(m) == 1, f"PROB_SHIFT 패턴 {len(m)}개"
    shift0 = float(m[0][1])

    ajp = os.path.join(work, "model", "artifacts.joblib")
    bundle = joblib.load(ajp)
    art = bundle["artifacts"]
    have = [x["name"] for x in art.get("asof", [])]
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}")
    print(f"  기존 asof 축 {[(x['name'], x['beta']) for x in art['asof']]}")
    assert a.axis not in have, f"{a.axis} 는 이미 있다"
    strength = float(art["asof"][0]["strength"])
    print(f"  축 강도 S = {strength}  (아티팩트 실측 — config 값을 쓰지 않는다)")

    # ── 전 시즌 train 으로 축을 만든다 (챔피언 축과 같은 학습 범위) ──────
    from data import load_csv
    train_df = load_csv(TRAIN_CSV)
    print(f"  train {len(train_df):,}행 (2019~2024)")
    assert repro_check(train_df, art, strength), "재현 관문 실패 — 축을 못 믿는다"

    ct = art.get("contam") if a.contam else None
    ax = build_axis(train_df, a.axis, a.beta, strength, ct)
    print(f"  새 축 {a.axis}  β {a.beta:+.5f}  mu {ax['mu']:+.9f}  sd {ax['sd']:.9f}  "
          f"cadj {ax.get('cadj', '-')}  키 {len(ax['keys']):,}")
    assert float(ax["league"]) == float(art["asof"][0]["league"]), "league 불일치"

    print("추론 ① 축 추가 전 …")
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pq = os.path.join(tmp, "proxy.pkl")
    df.to_pickle(pq)
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    art["asof"] = list(art["asof"]) + [ax]
    joblib.dump(bundle, ajp)

    print("추론 ② 축 추가 후 (레벨 미보정) …")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
    d_ = mid - before
    dm = float(d_.mean())
    print(f"  Δp 평균 {dm:+.6g} · 평균|Δp| {np.abs(d_).mean():.6g} · "
          f"max|Δp| {np.abs(d_).max():.6g}")
    assert np.abs(d_).max() > 0.0, "축이 예측에 반영되지 않았다"
    shift = shift0 - dm
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")
    open(pp, "w", encoding="utf-8").write(
        SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", src0))

    print("추론 ③ 재보정 검증 …")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    r = float((after - before).mean())
    print(f"  재보정 후 Δmean {r:+.3g}  {'OK' if abs(r) < 1e-9 else '경고'}")
    assert abs(r) < 1e-7

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                  and open(os.path.join(work, n), "rb").read()
                  != open(os.path.join(orig, n), "rb").read())
    exp = sorted(["predict.py", "model/artifacts.joblib"])
    print(f"\n변경 격리  {len(diff)}/{len(names)}: {diff}")
    assert diff == exp, f"기대 {exp}"
    assert len(os.path.basename(a.out)) <= 30
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
