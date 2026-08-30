"""챔피언 **실물 파이프라인** 위에서 콜드 패치의 ΔBSS (§7.211).

사용자 지적: *"지금 챔피언이 이거(general_lgb/compact/user_fspec) 쓰고 있는거 아니야?"*
맞다. 그래서 재학습하지 말고 **챔피언 zip 을 그대로 돌린다.**
바꾸는 것은 `model/mlp_emb4_*.npz` **5개뿐**이고, 나머지 77 파일(라우팅 계층 29 포함)은
챔피언 실물 그대로다. 파이프라인 정합은 **100%** 다.

⚠️ 대신 감수하는 것 — 반드시 같이 읽어야 한다
   ① 챔피언은 2019~2024 로 학습했다. 2024 에는 **처음 보는 투수/타자가 0명**이라
      임베딩 index 0 을 아무도 안 탄다 → 패치가 무연산이 된다.
      그래서 `id = -1` 마스킹으로 미지 경로를 태운다 (§7.203·§7.208 과 같은 절차).
   ② 2024 는 in-sample 이라 **절대 BSS 는 의미 없다.** 읽을 것은 두 팔의 **차이**뿐이다.
   ③ 코어가 과도하게 정확하므로 부품이 기여할 여지가 줄어 **Δ 는 과소추정**된다
      (`src/t1gen.py` 가 같은 편향을 기록했다). 보수적이므로 "그래도 이기면 진짜".
   ④ 마스킹은 임베딩만이 아니라 `asof_*` 조회까지 흔들지만 **두 팔이 같은 마스크**라
      상쇄된다. 관문: 알려진 행 Δp == 0.

팔: none(대조) / pitch(emb0[0]) / bat(emb1[0]) / both

실행: .venv/Scripts/python.exe -u src/hjswap.py 2>&1 | tee experiments/hjswap.log
"""
import shutil, subprocess, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).parent.parent
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import numpy as np
from config import TRAIN_CSV
from validation import bss

BASE_ZIP = ROOT / "hj03_dim8_cold.zip"
CACHE = ROOT / "experiments" / "cache"
ARMS = ("none", "pitch", "bat", "both")
UNK_P, UNK_B = 0.199, 0.093          # 2024 실측 신규 비율 (투수 19.9% / 타자 9.3%)
B_BOOT = 4000

WORKER = r"""
import sys, numpy as np, pandas as pd
arm, out = sys.argv[1], sys.argv[2]
d = r"D:/t1eval/" + arm
sys.path.insert(0, d)
va = pd.read_csv(r"{csv}")
va = va[va.season == 2024].reset_index(drop=True)
y = va["control_success"].to_numpy(dtype=np.float64)
rng = np.random.RandomState(0)
mp = np.zeros(len(va), bool); mb = np.zeros(len(va), bool)
for col, frac, holder in (("pitcher_id", {up}, "p"), ("batter_id", {ub}, "b")):
    ids = rng.permutation(np.sort(va[col].unique())); cnt = va[col].value_counts()
    take, tot, goal = [], 0, frac * len(va)
    for i in ids:
        if tot >= goal: break
        take.append(i); tot += int(cnt[i])
    m = va[col].isin(take).to_numpy()
    if holder == "p": mp = m
    else: mb = m
va.loc[mp, "pitcher_id"] = -1
va.loc[mb, "batter_id"] = -1
from predict import Predictor
import predict as _pm
assert _pm.__file__.replace("\\", "/").startswith(d), "그 zip 의 predict.py 가 아니다"
p = Predictor(model_dir=d + "/model")
np.savez_compressed(out, p=np.asarray(p.predict(va), dtype=np.float64),
                    mp=mp, mb=mb, y=y,
                    pid=va["pitcher_id"].to_numpy())
"""

def build_arm(arm):
    d = ASCII / f"HJSW_{arm}"
    if (d / "predict.py").exists(): return d
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    with zipfile.ZipFile(BASE_ZIP) as z: z.extractall(d)
    if arm == "none": return d
    n = 0
    for f in sorted((d / "model").glob("mlp_emb4_*.npz")):
        w = dict(np.load(f, allow_pickle=True))
        ec = [str(c) for c in w["ecols"]]
        for col, tag in (("pitcher_id", "pitch"), ("batter_id", "bat")):
            if arm in (tag, "both"):
                k = f"emb{ec.index(col)}"; e = w[k]
                w[k] = np.vstack([e[1:].mean(0)[None, :], e[1:]]).astype(e.dtype)
                n += 1
        np.savez_compressed(f, **w)
    print(f"  {arm}: 임베딩 {n}개 행 교체 (파일 5개, 나머지 77개는 챔피언 실물)")
    return d

def run(arm):
    f = CACHE / f"hjswap_{arm}.npz"
    if not f.exists():
        wk = CACHE / "_hjswap_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/"))
                            .replace("{up}", repr(UNK_P)).replace("{ub}", repr(UNK_B)),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), f"HJSW_{arm}", str(f)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if not f.exists():
            print(f"  [실패] {arm}\n{(r.stdout or '')[-500:]}\n{(r.stderr or '')[-900:]}")
            sys.exit(1)
    z = np.load(f); return z["p"], z["mp"], z["mb"], z["y"], z["pid"]

def main():
    t0 = time.time()
    print("§7.211 — 챔피언 **실물** 파이프라인 위 콜드 패치 ΔBSS")
    print(f"  변경: mlp_emb4_*.npz 5개만. 라우팅 계층 29파일 포함 나머지 77개는 그대로")
    print(f"  마스킹: 투수 {UNK_P:.1%} · 타자 {UNK_B:.1%} (2024 실측 신규 비율)")
    print("  ⚠️ 2024 는 in-sample — **절대 BSS 는 의미 없다.** 두 팔의 차이만 읽는다\n")
    for a in ARMS: build_arm(a)
    P = {}
    for a in ARMS:
        P[a], mp, mb, y, pid = run(a)
        print(f"  {a:<6} 추론 완료 ({time.time()-t0:.0f}s)")
    ybar = float(y.mean())
    ref = ybar * (1 - ybar)
    _, inv = np.unique(np.where(pid < 0, -1, pid), return_inverse=True)
    n_id = inv.max() + 1
    rng = np.random.default_rng(42)
    n = np.bincount(inv, minlength=n_id)
    seen = np.where(n > 0)[0]
    base = P["none"]
    print(f"\n  마스킹 후 미지 투수 행 {mp.mean():.2%} · 미지 타자 행 {mb.mean():.2%}"
          f"   ȳ {ybar:.6f}   기준 BSS {bss(y, base):.1f} (in-sample, 참고용)")
    print(f"\n  {'팔':<8}{'ΔBSS':>9}{'95% CI':>21}{'P(Δ>0)':>9}"
          f"{'미지행 Δp̄':>13}{'알려진행 Δp̄':>14}")
    s0 = np.bincount(inv, weights=(base - y) ** 2, minlength=n_id)
    for a in ARMS[1:]:
        p = P[a]; d = p - base
        s1 = np.bincount(inv, weights=(p - y) ** 2, minlength=n_id)
        draws = np.empty(B_BOOT)
        for b in range(B_BOOT):
            pick = rng.choice(seen, size=len(seen), replace=True)
            N = n[pick].sum()
            draws[b] = 1e5 * (s0[pick].sum() - s1[pick].sum()) / (N * ref)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        tgt = mp if a == "pitch" else mb if a == "bat" else (mp | mb)
        print(f"  {a:<8}{bss(y,p)-bss(y,base):>+9.2f}   [{lo:+7.2f},{hi:+7.2f}]"
              f"{float(np.mean(draws>0)):>9.3f}{d[tgt].mean():>+13.2e}"
              f"{d[~tgt].mean():>+14.2e}")
    print("\n  · 알려진 행 Δp̄ 가 0 이면 패치가 미지 행만 건드린 것 (기전 격리)")
    print("  · Δ 는 **과소추정** — 코어가 2024 를 학습해 부품 여지가 줄어든다 (t1gen 과 같은 편향)")
    print(f"총 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
