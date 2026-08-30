"""챔피언 `hj03` 의 구조를 **직접 열어** 내 fold 하네스와 항목별로 대조한다 (§7.210).

왜
    §7.209 가 `pitcher_id` 임베딩을 dim 4 로 재서 결론이 틀렸다. 그때 나는 "B1 은
    배포 챔피언 등가" 라는 **물려받은 문장**을 믿었고 확인하지 않았다.
    같은 실수를 막으려면 말이 아니라 **파일에서 뽑은 값**으로 대조해야 한다.

무엇을 대조하나 (전부 hj03 zip 에서 직접 읽는다)
    ① 피처 프레임   `artifacts.joblib["feature_cols"]` vs 내 `build_features` 산출 컬럼
    ② MLP 인코더    `bundle["mlp_enc"]` 의 num_cols · sizes vs 내 `fit_encoder`
    ③ MLP 가중치    hj03 `mlp_emb4_*.npz` 의 키·shape vs 내 fold 학습 산출
    ④ 블렌드 상수   mlp_alpha · mlp_weights · cat_alpha · fspec_alpha · 라우팅 · PROB_*
    ⑤ δ 축          `art["ctx"]` · `art["asof"]` 의 축과 β vs 내 `deploy29` dep
    ⑥ 미대응 목록   hj03 에 있는데 내 하네스에 대응물이 **없는** 모델 파일

실행:  .venv/Scripts/python.exe -u src/hjstruct.py 2>&1 | tee experiments/hjstruct.log
"""
import io
import json
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np

from config import (CONTAM_AXES, DROP_FEATURES, MLP_SEEDS, MLP_VARIANTS,
                    RECENCY_HALF_LIFE, ROOT, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts

ZIP = ROOT / "hj03_dim8_cold.zip"
FOLD = 2024
CACHE = ROOT / "experiments" / "cache"
P_DIM = 8


def ok(b):
    return "✅" if b else "❌"


def main():
    z = zipfile.ZipFile(ZIP)
    print(f"대상 {ZIP.name}   ({ZIP.stat().st_size:,}B)\n")

    # hj03 의 번들을 그대로 읽는다 (joblib 은 파일 경로가 필요하므로 임시로 푼다)
    tmp = Path("D:/t1eval/_hjstruct")
    tmp.mkdir(parents=True, exist_ok=True)
    for n in ("model/artifacts.joblib",):
        (tmp / "artifacts.joblib").write_bytes(z.read(n))
    bundle = joblib.load(tmp / "artifacts.joblib")
    art_hj = bundle["artifacts"]
    cols_hj = list(bundle["feature_cols"])

    # ── 내 하네스가 만드는 프레임 (시즌 < FOLD 로 fit) ──────────────
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tr = df[s < FOLD]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art_me = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                           k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                           center_asof=False)
    Xme = build_features(tr.head(200), art_me)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            Xme[c] = tr.head(200)[c].to_numpy(dtype=np.float32)
    Xme = Xme.drop(columns=[c for c in DROP_FEATURES if c in Xme.columns])
    cols_me = list(Xme.columns)

    print("=" * 90)
    print("① 피처 프레임")
    print("=" * 90)
    print(f"  hj03 feature_cols {len(cols_hj)}개 · 내 build_features {len(cols_me)}개")
    print(f"  {ok(cols_hj == cols_me)} 이름·순서 완전 일치"
          if cols_hj == cols_me else
          f"  ❌ 불일치 — hj03에만 {sorted(set(cols_hj)-set(cols_me))[:6]} / "
          f"내게만 {sorted(set(cols_me)-set(cols_hj))[:6]}"
          + (f"  (집합은 같고 순서만 다름)" if set(cols_hj) == set(cols_me) else ""))

    print()
    print("=" * 90)
    print("② MLP 인코더")
    print("=" * 90)
    enc_hj = bundle.get("mlp_enc")
    if enc_hj is None:
        print("  (번들에 mlp_enc 없음)")
    else:
        import mlp
        import modelabel
        import soften
        y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        aux, _ = modelabel.aux_labels(tr)
        Xtr_full = build_features(tr, art_me)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                Xtr_full[c] = tr[c].to_numpy(dtype=np.float32)
        Xtr_full = Xtr_full.drop(columns=[c for c in DROP_FEATURES
                                          if c in Xtr_full.columns])
        Xs, _ys, _ws, _aux, _h = soften.soften(tr, Xtr_full, y_tr, w, aux)
        enc_me = mlp.fit_encoder(Xs, art_me)
        for k in ("num_cols", "sizes"):
            a, b = enc_hj.get(k), enc_me.get(k)
            same = (list(a) == list(b)) if k == "num_cols" else (list(a) == list(b))
            print(f"  {k:<10} hj03 {len(a)} / 나 {len(b)}   {ok(same)}"
                  + ("" if same else f"   hj03={list(a)[:8]}... 나={list(b)[:8]}..."))
        print(f"  cat_luts 컬럼  hj03 {list(enc_hj['cat_luts'].keys())}")
        print(f"                 나   {list(enc_me['cat_luts'].keys())}")

    print()
    print("=" * 90)
    print("③ MLP emb4 가중치 — 키·shape")
    print("=" * 90)
    w_hj = dict(np.load(io.BytesIO(z.read("model/mlp_emb4_42.npz")), allow_pickle=True))
    p_me = CACHE / f"bcoldw_d{P_DIM}_f{FOLD}_emb4_42.npz"
    if not p_me.exists():
        print(f"  (내 산출물 없음: {p_me.name})")
    else:
        w_me = dict(np.load(p_me, allow_pickle=True))
        w_me.pop("_z", None)
        kh, km = sorted(w_hj), sorted(w_me)
        print(f"  키 목록 {ok(kh == km)}   hj03 {kh}")
        if kh != km:
            print(f"                        나   {km}")
        print(f"  {'배열':<8}{'hj03':>14}{'나':>14}   일치")
        for k in kh:
            if k not in w_me:
                continue
            a, b = w_hj[k], w_me[k]
            sa = str(getattr(a, "shape", "-"))
            sb = str(getattr(b, "shape", "-"))
            # 행 수는 학습 시즌이 달라 당연히 다르다 — **열(차원)** 만 비교한다
            same = (a.ndim == b.ndim and
                    (a.shape[1:] == b.shape[1:] if a.ndim > 1 else True))
            note = "" if k in ("ecols",) else ("  (행수는 학습 개체 수라 달라도 정상)"
                                               if sa != sb and same else "")
            print(f"  {k:<8}{sa:>14}{sb:>14}   {ok(same)}{note}")
        print(f"  ecols  hj03 {[str(c) for c in w_hj['ecols']]}")
        print(f"         나   {[str(c) for c in w_me['ecols']]}")

    print()
    print("=" * 90)
    print("④ 블렌드 상수 — hj03 실제값 vs 내 하네스")
    print("=" * 90)
    pred = z.read("predict.py").decode("utf-8")
    cm = json.loads(z.read("model/cat_members.json"))
    rout = re.findall(r"z\[is_[rf]\] = ([0-9.]+) \* z\[is_[rf]\] \+ ([0-9.]+)", pred)
    lit = {k: re.search(rf"{k}\s*=\s*([0-9.e-]+)", pred).group(1)
           for k in ("PROB_CENTER", "PROB_SCALE", "PROB_SHIFT")}
    paired = re.search(r"z \+= ([0-9.]+) \* ([0-9.]+) \*", pred)
    rows = [
        ("mlp_alpha", bundle.get("mlp_alpha"), 0.75, True),
        ("mlp_weights(emb4합)", round(sum(bundle["mlp_weights"][:5]), 6), 0.875, True),
        ("mlp_weights(noid합)", round(sum(bundle["mlp_weights"][5:]), 6), 0.125, True),
        ("cat_alpha", cm["alpha"], 0.10, True),
        ("cat 멤버수", len(cm["members"]), 12, True),
        ("LGBM 멤버수", len(bundle.get("members") or []), 12, False),
        ("fspec_alpha", bundle.get("fspec_alpha"), 0.0, False),
        ("R 라우팅", tuple(rout[0]) if rout else None, None, False),
        ("F 라우팅", tuple(rout[1]) if len(rout) > 1 else None, None, False),
        ("compact paired 가중", f"{paired.group(1)}*{paired.group(2)}" if paired else None,
         None, False),
        ("PROB_CENTER", lit["PROB_CENTER"], None, False),
        ("PROB_SCALE", lit["PROB_SCALE"], None, False),
        ("logit_delta", round(float(bundle["logit_delta"]), 8), None, False),
    ]
    print(f"  {'항목':<22}{'hj03':>22}{'내 하네스':>16}   판정")
    for name, hv, mv, must in rows:
        if mv is None:
            print(f"  {name:<22}{str(hv):>22}{'대응 없음':>16}   ⚠️ 미반영")
        else:
            print(f"  {name:<22}{str(hv):>22}{str(mv):>16}   {ok(str(hv) == str(mv))}")

    print()
    print("=" * 90)
    print("⑤ δ 축 (사후 로짓 보정) — hj03 아티팩트 안의 축과 β")
    print("=" * 90)
    import deploy29 as D
    for slot in ("ctx", "asof", "mix"):
        v = art_hj.get(slot) or []
        print(f"  hj03 art['{slot}'] {len(v)}개: "
              + ", ".join(f"{a.get('name', a.get('axis', '?'))}={a.get('beta')}"
                          if isinstance(a, dict) else str(a)[:60] for a in v))
    print(f"  내 deploy29 dep = {np.round(D.CTX_AXES and [b for _, b in D.CTX_AXES] + [b for _, b in D.ASOF_AXES], 5).tolist()}")
    print(f"     (config CTX_AXES {D.CTX_AXES} · ASOF_AXES {D.ASOF_AXES})")

    print()
    print("=" * 90)
    print("⑥ hj03 에 있는데 내 하네스에 **대응물이 없는** 모델 가족")
    print("=" * 90)
    names = sorted(n.split("/")[-1] for n in z.namelist() if n.startswith("model/"))
    fam = {}
    for b in names:
        k = re.sub(r"[_.]?(42|7|2024|1234|31337)", "", b)
        k = re.sub(r"\.(txt|npz|cbm|pkl|joblib|json)$", "", k)
        fam[k] = fam.get(k, 0) + 1
    mine = {"lgbm_base", "lgbm_wide", "lgbm_extra", "lgbm_lowff",
            "mlp_emb4", "mlp_noid",
            "cat_base", "cat_wide", "cat_sym", "cat_depthwise",
            "artifacts", "meta", "cat_members"}
    miss, have = [], []
    for k, v in sorted(fam.items()):
        (have if k in mine else miss).append(f"{k} x{v}")
    print("  대응 있음: " + ", ".join(have))
    print()
    print("  ❌ 대응 없음: " + ", ".join(miss))
    print(f"\n  → 총 {sum(fam[k.split(' x')[0]] for k in miss)}개 파일이 내 하네스에 없다.")


if __name__ == "__main__":
    main()
