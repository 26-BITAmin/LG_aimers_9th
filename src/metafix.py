"""`meta.json` 을 `artifacts.joblib` 에서 **파생**시킨다 — 감사 + 수정.

왜 필요한가
    `meta.json` 은 사람 확인용 스냅숏인데, T1 계보 빌드는 **이전 zip 에서 그대로
    복사**한다 (`t1cap.py`·`rt45build.py`·`catf0build.py` 어디에도 meta 갱신이 없다).
    그래서 T1 시점의 값이 s43 → s45 → s47 → s48 까지 그대로 흘러왔고,
    실제 동작과 어긋난 채로 굳었다. 확인된 것 둘:

        fspec_alpha         meta 0.458  vs  번들 0.0
            → 우리 F전문가 3개는 로드조차 안 된다. T1 이 팀원 `user_fspec` 으로
              그 자리를 대체했고 파일만 남았다 (사용자 확인).
        mlp_variant_weight  meta {emb4 .625, noid .375}  vs  번들 {.875, .125}

    `predict.py` 가 읽는 것은 **번들**이다. 그래서 성능에는 영향이 없다.
    문제는 **코드 검증(9/11)** 이다 — 검증자는 meta.json 을 읽는다.

원칙
    번들에서 **유도할 수 있는 필드만** 고친다. `soft_h`·`recency_half_life`·
    `train_rows`·`lgb_params`·`drop_features` 등은 학습 시점 설정이라 번들에
    없다 → 손대지 않는다. 모르는 것을 지어내지 않는 것이 이 스크립트의 계약이다.

⚠️ **제출된 zip 은 절대 제자리 수정하지 않는다.** 제출물은 기록이다.
   zip 을 고치려면 `--zip IN --out OUT` 으로 **새 파일**을 만든다.

사용
    감사만          .venv/Scripts/python.exe src/metafix.py --dir t1work/model
    디렉터리 수정    .venv/Scripts/python.exe src/metafix.py --dir t1work/model --write
    zip 감사        .venv/Scripts/python.exe src/metafix.py --zip s45_cap2.zip
    zip → 새 zip    .venv/Scripts/python.exe src/metafix.py --zip s47r066f355.zip --out s47_metafix.zip
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib

ROOT = Path(__file__).resolve().parents[1]


def _variant_of(tag):
    """`emb4_31337` → `emb4`.  시드는 마지막 `_` 뒤다."""
    return tag.rsplit("_", 1)[0]


def derive(model_dir):
    """번들에서 유도 가능한 meta 필드만 만든다. 유도 불가 필드는 넣지 않는다."""
    model_dir = Path(model_dir)
    b = joblib.load(model_dir / "artifacts.joblib")
    art = b["artifacts"]
    out = {}

    out["n_features"] = len(b["feature_cols"])
    out["logit_delta"] = float(b["logit_delta"])

    members = list(b.get("members") or [])
    if members:
        out["members"] = members
        out["n_members"] = len(members)

    # ── MLP ────────────────────────────────────────────
    out["mlp_alpha"] = float(b.get("mlp_alpha") or 0.0)
    out["use_mlp"] = out["mlp_alpha"] > 0
    mm = list(b.get("mlp_members") or [])
    mw = list(b.get("mlp_weights") or [])
    if mm:
        out["mlp_members"] = mm
    if mm and mw and len(mm) == len(mw):
        vw = {}
        for tag, w in zip(mm, mw):
            vw[_variant_of(tag)] = vw.get(_variant_of(tag), 0.0) + float(w)
        # 부동소수 잡음 정리 — 합이 1 이어야 한다
        s = sum(vw.values())
        assert abs(s - 1.0) < 1e-9, f"mlp 가중 합이 1이 아니다: {s}"
        out["mlp_variant_weight"] = {k: round(v, 12) for k, v in vw.items()}
        out["mlp_weights"] = [float(v) for v in mw]

    # ── fspec ──────────────────────────────────────────
    #    ⚠️ α=0 이면 `predict.py` 가 로드조차 안 한다. 멤버 목록은 남기되
    #       α 를 진실대로 적어 "파일은 있으나 죽어 있다"가 읽히게 한다.
    out["fspec_alpha"] = float(b.get("fspec_alpha") or 0.0)
    fm = list(b.get("fspec_members") or [])
    if fm:
        out["fspec_members"] = fm

    # ── CatBoost — 진실은 cat_members.json 이다 (predict.py 가 이걸 읽는다) ──
    cm = model_dir / "cat_members.json"
    if cm.exists():
        d = json.loads(cm.read_text(encoding="utf-8"))
        out["cat_alpha"] = float(d["alpha"])
        out["cat_members"] = list(d["members"])
    elif b.get("cat_members"):
        out["cat_alpha"] = float(b.get("cat_alpha") or 0.0)
        out["cat_members"] = list(b["cat_members"])

    # ── 사후 보정 축 ───────────────────────────────────
    ctx = art.get("ctx") or []
    if ctx:
        out["ctx_axes"] = {a["name"]: float(a["beta"]) for a in ctx}
        out["ctx_cells"] = {a["name"]: len(a.get("keys", [])) for a in ctx}
    asof = art.get("asof") or []
    if asof:
        out["asof_axes"] = {a["name"]: float(a["beta"]) for a in asof}
        out["asof_cells"] = {a["name"]: len(a["keys"]) for a in asof}
    out["mix_axes"] = {a["name"]: float(a["beta"]) for a in (art.get("mix") or [])}

    # ⚠️ `smooth_k` 는 유도하지 않는다. 번들 쪽은 **컬럼별 dict** 이고 meta 는
    #    스칼라라 표현이 다르다 — 둘을 같은 값으로 취급하면 없던 불일치를 만든다.
    if art.get("use_player_ids") is not None:
        out["use_player_ids"] = bool(art["use_player_ids"])
    ct = art.get("contam")
    if ct and ct.get("axes"):
        out["contam_axes"] = list(ct["axes"])
    return out


def audit(model_dir):
    """(불일치 목록, 유도값, 현재 meta) — 불일치는 (키, meta값, 진실값)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    truth = derive(model_dir)
    bad = []
    for k, v in truth.items():
        if k not in meta:
            bad.append((k, "(없음)", v))
        elif meta[k] != v:
            # float 는 표시 자릿수 차이를 무시한다
            try:
                if abs(float(meta[k]) - float(v)) < 1e-12:
                    continue
            except (TypeError, ValueError):
                pass
            bad.append((k, meta[k], v))
    return bad, truth, meta


def report(tag, bad):
    if not bad:
        print(f"  {tag}: 일치 ✅")
        return
    print(f"  {tag}: 불일치 {len(bad)}건")
    for k, got, want in bad:
        print(f"    {k}")
        print(f"      meta   {got}")
        print(f"      번들   {want}")


def write_meta(model_dir, truth):
    """meta.json 을 갱신한다. **유도 가능한 키만** 덮어쓰고 나머지는 보존한다."""
    p = Path(model_dir) / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta.update(truth)
    # ⚠️ CRLF 로 쓰면 안 된다 (LightGBM 모델 파일 사고와 같은 계열).
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="model 디렉터리 (artifacts.joblib 이 있는 곳)")
    ap.add_argument("--zip", dest="zf", help="제출 zip (읽기 전용으로 연다)")
    ap.add_argument("--out", help="--zip 과 함께: 고친 **새** zip 경로")
    ap.add_argument("--write", action="store_true", help="--dir 과 함께: 제자리 수정")
    a = ap.parse_args()

    if a.zf:
        z = zipfile.ZipFile(a.zf)
        names = z.namelist()
        mdir = os.path.dirname([n for n in names if n.endswith("artifacts.joblib")][0])
        tmp = Path(tempfile.mkdtemp())
        for n in names:
            if n.startswith(mdir) and not n.endswith("/"):
                (tmp / os.path.basename(n)).write_bytes(z.read(n))
        bad, truth, _ = audit(tmp)
        print(f"[{a.zf}]")
        report(os.path.basename(a.zf), bad)
        if not a.out:
            if bad:
                print("\n  → 고치려면 --out 으로 **새 zip** 경로를 준다."
                      " 제출된 zip 은 제자리 수정하지 않는다.")
            return 1 if bad else 0
        if Path(a.out).exists():
            raise SystemExit(f"이미 있다: {a.out} — 덮어쓰지 않는다")
        write_meta(tmp, truth)
        fixed = (tmp / "meta.json").read_bytes()
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as w:
            for i in z.infolist():
                data = z.read(i.filename)
                if i.filename == f"{mdir}/meta.json":
                    data = fixed
                w.writestr(i, data)
        print(f"\n  → {a.out} 작성. meta.json 만 다르고 나머지 {len(names)-1}개는 바이트 동일.")
        return 0

    if not a.dir:
        raise SystemExit("--dir 또는 --zip 중 하나가 필요하다")
    bad, truth, _ = audit(a.dir)
    print(f"[{a.dir}]")
    report(str(a.dir), bad)
    if bad and a.write:
        p = write_meta(a.dir, truth)
        print(f"\n  → {p} 갱신 ({len(bad)}건)")
    elif bad:
        print("\n  → 고치려면 --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
