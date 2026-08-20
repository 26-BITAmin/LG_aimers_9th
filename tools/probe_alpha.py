#!/usr/bin/env python3
"""MLP_ALPHA 프로브 러너 — DIRECTION.md 후보 #1.

챔피언 스택(`wonwoo` 브랜치)의 코드를 **읽기·실행만** 하고, 튜닝 상수는
이 스크립트가 주입한다. 원우 브랜치 파일은 한 글자도 고치지 않는다.

왜 이렇게 되는가
----------------
`wonwoo:src/train.py`는 상수를 `from config import (..., MLP_ALPHA, ...)`로
가져온다 = **import 시점의 값 복사**다. 따라서 `train`을 import 하기 **전에**
`config.MLP_ALPHA`를 바꿔두면 train이 그 값을 집어간다.

왜 artifacts.joblib을 직접 고치면 안 되는가
-------------------------------------------
`train._finish()`가 블렌드 **이후** 로짓으로 prior shift를 다시 푼다:

    raw_ref = (1-α)·z_lgbm + α·z_mlp
    raw_ref = raw_ref + logit_adjust(...)          # δ 축
    delta   = solve_delta(base + raw_ref, TARGET_PRIOR_2025)

α가 바뀌면 raw_ref가 바뀌고 → delta도 달라져야 한다. `mlp_alpha` 값만
갈아끼우면 prior가 어긋나고, `BSS ≈ −400,000 × (예측평균 − 실제평균)²`이라
레벨에서 크게 잃는다. **반드시 `_finish()`를 다시 태워야 한다.**

왜 `--artifacts-only`로 충분한가
--------------------------------
`train._TRAIN_SHAPING`(저장된 모델을 결정하는 설정 목록)에 `MLP_ALPHA`가
**없다.** α는 LGBM 20개도 MLP 8개도 바꾸지 않고 블렌드 비중일 뿐이다.
그래서 모델을 재사용해도 되고, 피처 해시 가드에도 걸리지 않는다.
(§7.66 — 26분 → 38초)

사용법
------
    # 1) 전체 학습 1회 (모델 20+8개 생성, 약 30분). α는 배포값 그대로.
    python tools/probe_alpha.py --alpha 0.40 --full --out submit/model_a40

    # 2) 이후 α만 바꿔 38초씩 (모델 재사용)
    python tools/probe_alpha.py --alpha 0.30 --from submit/model_a40 --out submit/model_a30
    python tools/probe_alpha.py --alpha 0.25 --from submit/model_a40 --out submit/model_a25

근거: DIRECTION.md §6-1 · wonwoo:MODEL.md §7.65 · §7.66
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_CHAMPION = Path.home() / "Aimers_9th_champion"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha", type=float, required=True,
                   help="주입할 MLP_ALPHA (배포값 0.40)")
    p.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION,
                   help=f"챔피언 워크트리 경로 (기본 {DEFAULT_CHAMPION})")
    p.add_argument("--out", type=Path, required=True,
                   help="산출 모델 디렉터리 (챔피언 워크트리 기준 상대경로도 가능)")
    p.add_argument("--full", action="store_true",
                   help="전체 학습. 생략하면 --artifacts-only (모델 재사용)")
    p.add_argument("--from", dest="reuse", type=Path, default=None,
                   help="--artifacts-only 시 재사용할 모델 디렉터리")
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def main() -> int:
    args = parse_args()
    root = args.champion.resolve()
    src = root / "src"
    if not (src / "train.py").is_file():
        raise SystemExit(f"챔피언 워크트리를 찾을 수 없다: {src}/train.py")

    out_dir = resolve(root, args.out)
    if not args.full:
        if args.reuse is None:
            raise SystemExit("--artifacts-only 경로에는 --from 이 필요하다")
        reuse_dir = resolve(root, args.reuse)
        if not (reuse_dir / "artifacts.joblib").is_file():
            raise SystemExit(f"재사용할 모델이 없다: {reuse_dir}")
        # 모델 파일을 out_dir로 복사한 뒤 그 자리에서 아티팩트만 다시 만든다.
        # (원본 디렉터리를 덮어쓰면 baseline을 잃는다)
        if out_dir.exists():
            raise SystemExit(f"이미 존재한다. 지우거나 다른 이름을 쓸 것: {out_dir}")
        shutil.copytree(reuse_dir, out_dir)
        print(f"모델 재사용: {reuse_dir} → {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # config.MODEL_DIR 은 이 환경변수를 읽는다 (wonwoo:src/config.py).
    os.environ["LGA_MODEL_DIR"] = str(out_dir)

    sys.path.insert(0, str(src))
    os.chdir(root)          # data/ 를 상대경로로 읽는 코드가 있다

    # ⚠️ 순서가 전부다 — train 을 import 하기 **전에** config 를 고친다.
    import config
    before = config.MLP_ALPHA
    config.MLP_ALPHA = float(args.alpha)
    print(f"MLP_ALPHA 주입: {before} → {config.MLP_ALPHA}")
    print(f"MODEL_DIR      : {config.MODEL_DIR}")

    sys.argv = ["train.py"] + ([] if args.full else ["--artifacts-only"])
    print(f"모드           : {'전체 학습' if args.full else '--artifacts-only'}\n")

    import train
    # 값 복사가 실제로 됐는지 확인한다. 침묵 실패를 막는 유일한 방어선이다.
    if train.MLP_ALPHA != float(args.alpha):
        raise SystemExit(
            f"주입 실패: train.MLP_ALPHA={train.MLP_ALPHA} != {args.alpha}. "
            "config import 순서가 바뀌었는지 확인할 것."
        )
    train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
