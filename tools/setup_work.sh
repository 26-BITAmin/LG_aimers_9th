#!/usr/bin/env bash
# 학습 작업 디렉터리 조립 — tools/train_mlp_dim.py 가 쓰는 --work 를 만든다.
#
# 왜 조립이 필요한가
#   학습 루프(mlp.py) · 보조 라벨 복원(modelabel.py) · soft label(soften.py) 은
#   wonwoo 브랜치에 있고, **피처·인코딩·오염보정은 제출 zip 것을 써야 한다.**
#   배포와 같은 피처 경로가 아니면 비교가 성립하지 않는다
#   (원우 t1core.py 방침: "T1 자신의 features.py + artifacts.joblib").
#
# 사용:  bash tools/setup_work.sh <work_dir> <풀어둔_제출zip_dir> [wonwoo_src_경로|git_ref]
#
#   3번째 인자가 **디렉터리면** 그걸 쓴다 (git 불필요 — 배포 번들용).
#   생략하거나 git ref 면 `git archive` 로 가져온다.
set -euo pipefail
W="${1:?work_dir}"; Z="${2:?unzipped_submission_dir}"; SRC="${3:-origin/wonwoo}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

for f in features.py contam.py mlp_infer.py model/artifacts.joblib; do
  [ -e "$Z/$f" ] || { echo "[중단] 제출 zip 에 $f 가 없다: $Z"; exit 1; }
done

rm -rf "$W"; mkdir -p "$W"

# 1) 학습 모듈 (mlp 가 numemb 등 여러 모듈을 import 하므로 src 전체가 필요하다)
if [ -d "$SRC" ]; then
  echo "학습 모듈: 로컬 디렉터리 $SRC"
  cp "$SRC"/*.py "$W/"
else
  git -C "$REPO" cat-file -e "$SRC:src/mlp.py" 2>/dev/null || {
    echo "[중단] '$SRC' 가 디렉터리도 git ref 도 아니다."; exit 1; }
  echo "학습 모듈: git $SRC"
  git -C "$REPO" archive "$SRC" src | tar -x -C "$W" --strip-components=1
fi
# 2) 배포 경로로 덮어쓴다 — 이 세 개가 핵심이다
cp "$Z/features.py" "$Z/contam.py" "$Z/mlp_infer.py" "$W/"
[ -d "$Z/team_reference" ] && cp -r "$Z/team_reference" "$W/"

python3 - "$W" <<'PY'
import sys, os
w = sys.argv[1]
need = ["mlp.py","modelabel.py","soften.py","data.py","config.py",
        "features.py","contam.py","mlp_infer.py"]
miss = [n for n in need if not os.path.exists(os.path.join(w, n))]
print("[중단] 없는 모듈:", miss) if miss else print(f"조립 완료: {w}  ({len(os.listdir(w))}개 파일)")
sys.exit(1 if miss else 0)
PY
