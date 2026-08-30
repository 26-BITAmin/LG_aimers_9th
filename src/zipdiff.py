"""제출 zip 사이의 **실질 차이**를 찾는다 (§7.197 layerdiff 의 재사용 가능 판).

왜 다시 쓰나
    §7.197 이 `sy02_c4` 를 감사할 때 쓴 `scratchpad/layerdiff.py` 는 세션 스크래치라
    남아 있지 않다. 팀 챔피언 `hj03_dim8_cold.zip` 이 이제 디스크에 있으므로
    (§7.203 이 *"zip 이 없어 직접 적층 불가"* 라고 적어둔 제약이 풀렸다)
    같은 감사를 다시 할 수 있어야 한다. 이번엔 `src/` 에 남긴다.

무엇을 하나
    ① 파일 단위 — namelist 합집합, 크기, sha256. 있음/없음/동일/다름으로 분류
    ② `.npz` — 배열별 shape·dtype·max|Δ|·corr. **바이트가 달라도 값이 같을 수 있다**
       (직렬화 차이). §7.197 이 LGBM 4변형에서 corr=1.00000000 인데 바이트가 달랐던
       바로 그 함정이다 — 크기 diff 만 보면 "바뀌었다" 고 오독한다.
    ③ `.py` / `.json` / `.txt` — 텍스트 diff (줄 단위)
    ④ 요약 — "실질 변경" 만 추린다 (값이 실제로 다른 것)

⚠️ 이 도구는 **정적 비교**다. 두 zip 이 같은 입력에 같은 확률을 내는지는 별도 문제다
   (가중치가 같아도 `predict.py` 한 줄이 다르면 출력이 갈린다). 실질 변경 목록을
   받은 뒤 그 파일들을 읽어 판단해야 한다.

실행:
    .venv/Scripts/python.exe -u src/zipdiff.py hj03_dim8_cold.zip sy04_cold.zip
    .venv/Scripts/python.exe -u src/zipdiff.py A.zip B.zip C.zip     (첫 zip 이 기준)
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).parent.parent
TEXT_EXT = {".py", ".json", ".txt", ".md", ".cfg", ".csv"}


def digest(b):
    return hashlib.sha256(b).hexdigest()[:16]


def read_all(path):
    """zip → {이름: (크기, sha256, bytes)}. 디렉터리 엔트리는 뺀다."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            b = z.read(i.filename)
            out[i.filename] = (len(b), digest(b), b)
    return out


def cmp_npz(b0, b1):
    """두 npz 바이트열의 배열별 비교. 반환: (동일한가, 설명 줄들)."""
    lines, same = [], True
    try:
        with np.load(io.BytesIO(b0), allow_pickle=True) as d0, \
             np.load(io.BytesIO(b1), allow_pickle=True) as d1:
            k0, k1 = set(d0.files), set(d1.files)
            if k0 != k1:
                same = False
                lines.append(f"      키 차이  기준만 {sorted(k0-k1)}  비교만 {sorted(k1-k0)}")
            for k in sorted(k0 & k1):
                a, b = d0[k], d1[k]
                if a.shape != b.shape:
                    same = False
                    lines.append(f"      {k:<24} shape {a.shape} vs {b.shape}")
                    continue
                if a.dtype.kind not in "fiub":
                    if not np.array_equal(a, b):
                        same = False
                        lines.append(f"      {k:<24} 비수치 배열 불일치")
                    continue
                af = a.astype(np.float64).ravel()
                bf = b.astype(np.float64).ravel()
                md = float(np.max(np.abs(af - bf))) if af.size else 0.0
                if md == 0.0:
                    lines.append(f"      {k:<24} shape {str(a.shape):<14} **비트 동일**")
                    continue
                same = False
                sd = float(np.std(af)) if af.size > 1 else 0.0
                c = (float(np.corrcoef(af, bf)[0, 1])
                     if af.size > 1 and np.std(af) > 0 and np.std(bf) > 0 else float("nan"))
                lines.append(f"      {k:<24} shape {str(a.shape):<14} "
                             f"max|Δ| {md:.3e}  sd {sd:.4f}  corr {c:.8f}")
    except Exception as e:                                   # noqa: BLE001
        same = False
        lines.append(f"      npz 비교 실패: {e}")
    return same, lines


def cmp_text(b0, b1, name, ctx=0):
    """줄 단위 diff. 반환: (동일한가, 설명 줄들)."""
    import difflib
    t0 = b0.decode("utf-8", "replace").splitlines()
    t1 = b1.decode("utf-8", "replace").splitlines()
    if t0 == t1:
        return True, []
    d = list(difflib.unified_diff(t0, t1, "기준", "비교", n=ctx, lineterm=""))
    return False, ["      " + x for x in d[:80]]


def compare(base_path, other_path):
    A = read_all(base_path)
    B = read_all(other_path)
    print("\n" + "=" * 96)
    print(f"■ 기준 {base_path.name} ({base_path.stat().st_size:,}B)  vs  "
          f"{other_path.name} ({other_path.stat().st_size:,}B)")
    print("=" * 96)

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    both = sorted(set(A) & set(B))
    identical = [n for n in both if A[n][1] == B[n][1]]
    differ = [n for n in both if A[n][1] != B[n][1]]

    print(f"  파일 {len(A)} vs {len(B)}   공통 {len(both)}   "
          f"sha 동일 {len(identical)}   sha 다름 {len(differ)}   "
          f"기준만 {len(only_a)}   비교만 {len(only_b)}")
    for n in only_a:
        print(f"    − 기준에만: {n}  ({A[n][0]:,}B)")
    for n in only_b:
        print(f"    + 비교에만: {n}  ({B[n][0]:,}B)")

    if not differ:
        print("  ★ sha 가 다른 파일이 없다 — 두 zip 은 내용상 동일하다.")
        return []

    print(f"\n  ── sha 가 다른 {len(differ)}개 파일 파고들기 ──")
    real = []
    for n in differ:
        sa, sb = A[n][0], B[n][0]
        ext = Path(n).suffix.lower()
        print(f"\n    {n}   {sa:,}B → {sb:,}B  ({sb-sa:+,}B)")
        if ext == ".npz":
            same, lines = cmp_npz(A[n][2], B[n][2])
            for ln in lines:
                print(ln)
            if same:
                print("      → 값은 같다. **직렬화 차이일 뿐** (§7.197 과 같은 함정)")
            else:
                real.append(n)
        elif ext in TEXT_EXT:
            same, lines = cmp_text(A[n][2], B[n][2], n)
            for ln in lines:
                print(ln)
            if same:
                print("      → 텍스트 동일 (개행·인코딩 차이)")
            else:
                real.append(n)
        else:
            print("      (이진 파일 — 바이트 비교만. 값 비교 불가)")
            real.append(n)

    print(f"\n  ★ 실질 변경 {len(real)}/{len(differ)}: {real}")
    return real


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        print("zip 을 2개 이상 주어야 한다.")
        return
    paths = [ROOT / a if not Path(a).is_absolute() else Path(a) for a in args]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"없는 파일: {p}")
    base = paths[0]
    print(f"기준 zip: {base.name}")
    for other in paths[1:]:
        compare(base, other)


if __name__ == "__main__":
    main()
