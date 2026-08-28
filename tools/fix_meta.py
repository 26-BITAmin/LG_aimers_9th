"""제출 zip 의 `model/meta.json` 을 **진실원에서 다시 만든다.**

왜 필요한가
    T1 계보 빌드는 meta 를 이전 zip 에서 **복사**한다 (`t1cap.py`·`rt45build.py`·
    `catf0build.py` 어디에도 갱신이 없다). 그래서 T1 시점 값이 s43 → s45 → s47 →
    sy02_c4 → hj01_dim12 까지 그대로 흘렀다. 원우 §7.193 이 T1 계보 8건을 고쳤으나
    그 뒤 빌드가 다시 옛 meta 를 물려받았다.

    `Predictor` 는 `artifacts.joblib` 을 읽으므로 **채점에는 영향이 없다.** 그러나
    meta 는 코드 검증·인수인계에서 읽히는 문서다. 틀린 문서는 틀린 결론을 만든다
    (§7.171 이 `meta.json` 을 믿고 `fspec_alpha 0.458` 로 읽을 뻔했다).

진실원
    emb_dim / 멤버 구성      model/mlp_*.npz            (ecols · emb 모양)
    가중 · alpha · 멤버 목록  model/artifacts.joblib     (bundle 최상위)
    ctx · asof β             artifacts["ctx"], ["asof"]  (리스트 안의 beta)
    CatBoost                 model/cat_members.json
    확률공간 아핀 · 라우팅     predict.py                  (하드코딩 리터럴)

    ⚠️ 학습 레시피(recency · soft_label · init_score · aux)는 이 파일들에 없다.
       기존 meta 값을 그대로 이어받되 **추정하지 않는다** — 모르는 것은 안 적는다.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile

import joblib
import numpy as np

DERIVED = "_derived_from"


def truth(d):
    """zip 을 푼 디렉터리에서 진실원 값을 모은다."""
    b = joblib.load(os.path.join(d, "model", "artifacts.joblib"))
    art = b["artifacts"]
    src = open(os.path.join(d, "predict.py"), encoding="utf-8").read()

    def rx(p, cast=float):
        m = re.search(p, src)
        return cast(m.group(1)) if m else None

    t = {}
    # MLP 변형 — npz 에서 실제 임베딩 차원을 읽는다
    variants = {}
    for tag in b["mlp_members"]:
        v = tag.rsplit("_", 1)[0]
        if v in variants:
            continue
        z = np.load(os.path.join(d, "model", f"mlp_{tag}.npz"))
        ec = [str(c) for c in z["ecols"]]
        variants[v] = {c: int(z[f"emb{i}"].shape[1]) for i, c in enumerate(ec)}
    t["mlp_emb_dim"] = variants
    t["mlp_members"] = list(b["mlp_members"])
    t["mlp_weights"] = [float(x) for x in b["mlp_weights"]]
    vw, tags = {}, [m.rsplit("_", 1)[0] for m in b["mlp_members"]]
    for v in variants:
        vw[v] = round(sum(w for w, g in zip(b["mlp_weights"], tags) if g == v), 12)
    t["mlp_variant_weight"] = vw
    t["mlp_alpha"] = float(b["mlp_alpha"])
    t["fspec_alpha"] = float(b.get("fspec_alpha") or 0.0)
    t["fspec_members"] = list(b.get("fspec_members") or [])
    t["members"] = list(b["members"])
    t["logit_delta"] = float(b["logit_delta"])
    t["soft_label_f"] = bool(b.get("soft_label_f"))
    t["soft_h"] = {int(k): float(v) for k, v in (b.get("soft_h") or {}).items()}

    # 사후 보정 β — 리스트 안에 들어 있다
    t["ctx_axes"] = {a["name"]: float(a["beta"]) for a in (art.get("ctx") or [])}
    t["asof_axes"] = {a["name"]: float(a["beta"]) for a in (art.get("asof") or [])}
    t["smooth_k"] = art.get("smooth_k")   # 스칼라가 아니라 축별 dict 일 수 있다

    cmj = os.path.join(d, "model", "cat_members.json")
    if os.path.exists(cmj):
        c = json.load(open(cmj, encoding="utf-8"))
        t["cat_alpha"], t["cat_members"] = float(c["alpha"]), list(c["members"])

    t["prob_affine"] = {"center": rx(r"PROB_CENTER = ([0-9.]+)"),
                        "scale": rx(r"PROB_SCALE = ([0-9.]+)"),
                        "shift": rx(r"PROB_SHIFT = ([-0-9.eE+]+)")}
    t["routing"] = {"R_general": rx(r"z\[is_r\] = [0-9.]+ \* z\[is_r\] \+ ([0-9.]+)"),
                    "F_specialist": rx(r"z\[is_f\] = [0-9.]+ \* z\[is_f\] \+ ([0-9.]+)"),
                    "compact_delta": rx(r"z \+= [0-9.]+ \* ([0-9.]+) \*")}
    t["n_lgbm_members"] = len(b["members"])
    t["n_mlp_members"] = len(b["mlp_members"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", help="생략하면 감사만 하고 쓰지 않는다")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="meta_")
    d = os.path.join(tmp, "z")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(d)

    mp = os.path.join(d, "model", "meta.json")
    old = json.load(open(mp, encoding="utf-8"))
    t = truth(d)

    print(f"소스 {os.path.basename(a.src)}\n")
    print("%-24s %-26s %-26s" % ("항목", "meta.json", "진실원"))
    print("-" * 80)
    bad = 0
    for k in sorted(t):
        o = old.get(k, "(없음)")
        if k == "mlp_emb_dim" and "mlp_emb_dim" not in old:
            # 옛 스키마 폴백 — 재작성본은 mlp_emb_dim 을 직접 갖는다
            o = {v: c[0]["emb_dim"] if isinstance(c, list) else c
                 for v, c in (old.get("mlp_variants") or {}).items()}
        same = json.dumps(o, sort_keys=True, default=str) == json.dumps(t[k], sort_keys=True, default=str)
        bad += not same
        if not same:
            print("%-24s %-26s %-26s" % (k, str(o)[:25], str(t[k])[:25]))
    print(f"\n어긋난 항목 {bad}/{len(t)}")

    if not a.out:
        print("\n(--out 이 없어 감사만 했다)")
        return

    new = dict(old)
    new.update(t)
    # 옛 스키마 중 진실원으로 대체된 것을 지운다 — 두 곳에 다른 값이 남으면 안 된다
    new.pop("mlp_variants", None)
    new[DERIVED] = {
        "tool": "tools/fix_meta.py",
        "sources": ["model/artifacts.joblib", "model/mlp_*.npz",
                    "model/cat_members.json", "predict.py"],
        "note": ("학습 레시피(recency_half_life · use_init_score · mlp_aux_* 등)는 "
                 "아티팩트에 없어 이전 meta 값을 그대로 이어받았다. 추정하지 않았다."),
    }
    json.dump(new, open(mp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, sort_keys=True)

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(d, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"\n출력 {a.out}\n  SHA-256 {h}")
    print("  ⚠️ meta.json 은 Predictor 가 읽지 않는다 — **예측은 바이트 동일**이다.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
