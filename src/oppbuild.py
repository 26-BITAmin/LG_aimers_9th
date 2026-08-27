"""§7.185 — s46 빌드: 챔피언(s45_cap2) + 상대 질량 보정 (§7.184 β_p=0.0125, β_b=0.015)

전달 형태
    train 전체(2019~2024)로 만든 엔티티 표 2개 (p1_opp 투수, b1_opp 타자 — LOO,
    recency hl=0.5 ref=2024, k=50)를 model/opp_tables.npz 로 동봉.
    predict.py 가 행별 dict 조회로 v = −(x−μ)/σ (미등장 → 0) 를 만들고
    **최종 확률의 로짓**에 dz = β_p·v_p + β_b·v_b 를 가산한다 — §7.184 측정과
    같은 지점(모든 블렌드·δ·스케일 이후). 상수(μ·σ)는 배포 표 자신에서 얼린다.

레벨 재보정 (§7.178 ② 규약)
    2024 전 행 프록시로 챔피언 평균 vs 신규 평균을 재고 OPP_SHIFT = −Δmean 을
    패치해 평균을 챔피언과 정렬한다. §7.184 의 Δ 는 shift_to_prior 로 레벨을
    소거한 값이므로 배포도 레벨 불변이어야 측정과 같은 물건이다.
    챔피언의 PROB_SHIFT 는 건드리지 않는다.

검증
    ⓐ 표 커버리지 (2024 프록시 행 기준) — §7.184 fold2024 와 비슷해야 한다 (~80%/91%)
    ⓑ max|Δp| > 0 (보정 반영) · 평균|Δp| ~ 0.003 스케일
    ⓒ 패치 후 Δmean < 1e-9 (레벨 불변)
    ⓓ zip = 챔피언 파일 전부 + opp_tables.npz 1개 추가, 교체는 predict.py 만
    ⚠️ 2024 프록시 BSS 는 보지 않는다 — 챔피언 코어가 2024 를 학습에 썼으므로
       in-sample 이다. 채택 근거는 §7.184 홀드아웃 측정으로 끝났다.

실행:  .venv/Scripts/python.exe -u src/oppbuild.py 2>&1 | tee experiments/oppbuild.log
"""
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc
import gnnprobe as G
from config import TRAIN_CSV
from t1core import WORKER

BASE_ZIP = ROOT / "s45_cap2.zip"             # 현 챔피언 1119.4894859009
OUT = ROOT / "s46_opp.zip"
REF_DIR = ASCII / "S45REF"
NEW_DIR = ASCII / "OPPF"
BETA_P, BETA_B = 0.0125, 0.015               # §7.184 밴드 규칙(좌편향) 선정값
CRLF, LF = chr(13) + chr(10), chr(10)

INIT_ANCHOR = "        self.model_dir = model_dir"
INIT_SNIPPET = INIT_ANCHOR + """
        # opp_tables (§7.184) — train 에서 얼린 상대 질량 표. 행 독립 dict 조회.
        self.opp = None
        _op = os.path.join(model_dir, "opp_tables.npz")
        if os.path.exists(_op):
            with np.load(_op) as _t:
                self.opp = dict(
                    p=dict(zip(_t["pid"].tolist(), _t["pv"].tolist())),
                    b=dict(zip(_t["bid"].tolist(), _t["bv"].tolist())),
                    mu_p=float(_t["mu_p"]), sd_p=float(_t["sd_p"]),
                    mu_b=float(_t["mu_b"]), sd_b=float(_t["sd_b"]),
                    beta_p=float(_t["beta_p"]), beta_b=float(_t["beta_b"]))"""

PRED_ANCHOR = ("        p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER)"
               + " + PROB_SHIFT")
PRED_SNIPPET = PRED_ANCHOR + """
        # §7.184 상대 질량 보정 — 최종확률 로짓 가산. 레벨은 OPP_SHIFT 로 재정렬.
        OPP_SHIFT = 0.000000000
        if self.opp is not None:
            _o = self.opp
            _vp = np.array([_o["p"].get(int(k), np.nan)
                            for k in test_df["pitcher_id"].to_numpy()])
            _vb = np.array([_o["b"].get(int(k), np.nan)
                            for k in test_df["batter_id"].to_numpy()])
            _vp = np.where(np.isfinite(_vp), -(_vp - _o["mu_p"]) / _o["sd_p"], 0.0)
            _vb = np.where(np.isfinite(_vb), -(_vb - _o["mu_b"]) / _o["sd_b"], 0.0)
            _dz = _o["beta_p"] * _vp + _o["beta_b"] * _vb
            _pc = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
            p = 1.0 / (1.0 + np.exp(-(np.log(_pc / (1 - _pc)) + _dz))) + OPP_SHIFT"""


def extract(zp, dst):
    if (dst / "predict.py").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dst)


def proxy(dname, out_npz, fresh=False):
    f = ROOT / "experiments" / "cache" / out_npz
    if fresh:
        f.unlink(missing_ok=True)
    if not f.exists():
        wk = ROOT / "experiments" / "cache" / "_opp_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  ❌ {dname}: {r.stderr.strip()[-600:]}")
            sys.exit(1)
    return np.load(f)["p"]


def main():
    t0 = time.time()

    # ── 1. 배포 표: train 전체 (§7.183 과 같은 코드 경로 — skew 방지) ──
    from data import load_csv
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    _fc.SPLITS["ALL"] = lambda sv, mv: (np.ones(len(sv), bool),
                                        np.zeros(len(sv), bool))
    tabs = G.graph_tables(df, s, "ALL", np.random.default_rng(0))
    p_tab = tabs["p1_opp"][1]
    b_tab = tabs["b1_opp"][1]
    mu_p, sd_p = float(p_tab.mean()), float(p_tab.std())
    mu_b, sd_b = float(b_tab.mean()), float(b_tab.std())
    print(f"  표: 투수 {len(p_tab):,} (μ {mu_p:.4f} σ {sd_p:.4f})  "
          f"타자 {len(b_tab):,} (μ {mu_b:.4f} σ {sd_b:.4f})")
    # 커버리지 (2024 행 기준 — 배포 표는 2024 를 포함하므로 사실상 상한)
    d24 = df[s == 2024]
    covp = d24.pitcher_id.isin(p_tab.index).mean()
    covb = d24.batter_id.isin(b_tab.index).mean()
    print(f"  2024 행 커버리지: 투수 {covp:.1%}  타자 {covb:.1%}")
    del df

    # ── 2. 신규 팔 디렉터리 + 표 저장 + predict.py 패치 ──
    extract(BASE_ZIP, REF_DIR)
    shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
    np.savez_compressed(
        NEW_DIR / "model" / "opp_tables.npz",
        pid=p_tab.index.to_numpy(dtype=np.int64),
        pv=p_tab.to_numpy(dtype=np.float64),
        bid=b_tab.index.to_numpy(dtype=np.int64),
        bv=b_tab.to_numpy(dtype=np.float64),
        mu_p=mu_p, sd_p=sd_p, mu_b=mu_b, sd_b=sd_b,
        beta_p=BETA_P, beta_b=BETA_B)

    pred = NEW_DIR / "predict.py"
    src = pred.read_text(encoding="utf-8").replace(CRLF, LF)
    assert "opp_tables" not in src, "이미 패치된 predict.py"
    assert src.count(INIT_ANCHOR) == 1 and src.count(PRED_ANCHOR) == 1
    src = src.replace(INIT_ANCHOR, INIT_SNIPPET)
    src = src.replace(PRED_ANCHOR, PRED_SNIPPET)
    pred.write_text(src, encoding="utf-8", newline="\n")
    print(f"[{time.time()-t0:.0f}s] predict.py 패치 (β_p={BETA_P}, β_b={BETA_B})")

    # ── 3. 프록시 정렬: OPP_SHIFT = −Δmean ──
    p_ref = proxy("S45REF", "opp_ref.npz")
    p_new = proxy("OPPF", "opp_new.npz", fresh=True)
    dmax = float(np.abs(p_new - p_ref).max())
    dmean = float(p_new.mean() - p_ref.mean())
    assert dmax > 1e-4, "보정이 반영되지 않았다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행  "
          f"max|Δp|={dmax:.5f}  평균|Δp|={np.abs(p_new-p_ref).mean():.5f}  "
          f"Δmean={dmean:+.9f}  (레벨 비용 {1e5*dmean**2/0.2497:.3f}점)")

    src = pred.read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"OPP_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    assert m and float(m.group(1)) == 0.0
    src2 = src[:m.start(1)] + f"{-dmean:.9f}" + src[m.end(1):]
    pred.write_text(src2, encoding="utf-8", newline="\n")
    p_new2 = proxy("OPPF", "opp_new2.npz", fresh=True)
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  OPP_SHIFT 0 → {-dmean:.9f}   패치 후 Δmean = {d2:+.12f}  "
          f"{'✅' if abs(d2) < 1e-9 else '⚠️'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    # ── 4. 패키징: 챔피언 전부 + opp_tables.npz, 교체는 predict.py 만 ──
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(BASE_ZIP) as zin, \
         zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for n in names:
            if n == "predict.py":
                zout.writestr(n, src2)
            else:
                zout.writestr(n, zin.read(n))
        zout.write(NEW_DIR / "model" / "opp_tables.npz", "model/opp_tables.npz")
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"(챔피언 {len(names)}개 + opp_tables.npz, predict.py 교체)")
    print(f"기대 LB Δ (§7.184): +7.22 × [0.5, 0.93] = [+3.61, +6.72]  "
          f"— 음수면 되돌리고 기록")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
