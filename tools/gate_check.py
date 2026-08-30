#!/usr/bin/env python3
"""슬롯 사용 전 관문 — 2층 체크리스트를 한 번에 돌린다.

왜 이 도구인가
--------------
검증이 흔들린 이유는 **성격이 다른 두 층을 한 덩어리로 봤기 때문**이다.

  1층  빌드 검증    결정론적 **확인**. "zip 이 의도한 대로 동작하는가"
                    -> 확실해질 수 있다. 전부 통과해야 태운다.
  2층  점수 예측    통계적 **추론**. "LB 가 오르는가"
                    -> 원리적으로 확실해지지 않는다. **크기만** 잰다.

2층을 확실하게 만들려는 시도가 전부 실패했다 (홀드아웃 0/4, 부호 2회 오류).
그래서 이 도구는 2층을 판정하지 않는다. 대신 **답을 적도록 강제**한다:

    "이 후보가 지려면, 무엇이, 얼마나 틀려야 하는가?"

`--risk` 와 `--predict` 없이는 판정을 내지 않는다. `hj07`(−5.93) 과
`hj11`(−1.25) 은 이 문장이 없는 채로 태웠고 둘 다 졌다. `count4` 축은
그 문장을 쓸 수 있었고(라운드 8 `C=4.79`), 썼으면 안 태웠을 것이다.

사용
----
    python3 tools/gate_check.py \\
        --cand  outputs/hj12_r000f267.zip \\
        --champ outputs/hj09_grbc.zip \\
        --risk  "라벨없이 잰 C=0.5836 이 실제 test 에서 2.49배로 커야 진다 (실측 전이오차 1.41배의 1.8배)" \\
        --predict 0.712 --lo 0.548 --hi 0.828

    # 빠른 확인 (격리 실행/행 독립성 생략)
    python3 tools/gate_check.py --cand ... --champ ... --quick --risk ... --predict ...

근거: reports/00_STATUS.md §0 · reports/12_axis_reopening_report.md §23
"""
from __future__ import annotations

import argparse
import hashlib
import unicodedata
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, NG, WARN, SKIP = "✅", "❌", "⚠️ ", "—"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def members(z: Path) -> dict[str, bytes]:
    """zip 구성원의 이름 -> CRC. 디렉터리 항목은 뺀다."""
    with zipfile.ZipFile(z) as f:
        return {i.filename: i.CRC for i in f.infolist() if not i.is_dir()}


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.hard_fail = False
        self.warned: list[str] = []

    def add(self, name: str, mark: str, detail: str) -> None:
        self.rows.append((name, mark, detail))
        if mark == NG:
            self.hard_fail = True
        elif mark == WARN:
            self.warned.append(name)

    @staticmethod
    def _w(t: str) -> int:
        """한글은 터미널에서 두 칸을 차지한다 — len() 으로 맞추면 표가 어긋난다."""
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)

    def show(self, title: str) -> None:
        print(f"\n=== {title} ===")
        w = max(self._w(n) for n, _, _ in self.rows)
        for n, m, d in self.rows:
            print(f"  {m} {n}{' ' * (w - self._w(n))}  {d}")


# ── 1층 ────────────────────────────────────────────────────────
def layer1(a) -> Report:
    R = Report()
    cand, champ = Path(a.cand), Path(a.champ)

    # 1. 변경 격리
    mc, mh = members(cand), members(champ)
    only_c, only_h = sorted(set(mc) - set(mh)), sorted(set(mh) - set(mc))
    diff = sorted(k for k in set(mc) & set(mh) if mc[k] != mh[k])
    changed = diff + only_c + only_h
    # 파일 **추가**는 모델 변경에서 정상이다 — 그 자체로는 실패가 아니다.
    # 이 검사가 답하는 것은 "의도하지 않은 것이 바뀌었나" 하나뿐이므로,
    # 의도 목록(--expect-changed)이 없으면 판정하지 못한다 -> 경고로 내린다.
    allow = set(a.expect_changed.split(",")) if a.expect_changed else None
    if only_h:
        R.add("1 변경 격리", NG, f"챔피언에 있던 파일이 사라졌다 {only_h}")
    elif allow is None:
        R.add("1 변경 격리", WARN,
              f"{len(changed)}/{len(mc)}개 상이  {changed}  — --expect-changed 미지정이라 판정 불가")
    elif set(changed) - allow:
        R.add("1 변경 격리", NG, f"의도 밖 변경 {sorted(set(changed) - allow)}")
    else:
        R.add("1 변경 격리", OK, f"{len(changed)}/{len(mc)}개 상이  {changed}")

    # 2. 레벨 재보정 — 빌더가 이미 재고 출력한다. 값을 인자로 받아 기록만 한다.
    if a.relevel_dmean is None:
        R.add("2 레벨 재보정", SKIP, "--relevel-dmean 미지정 (빌더 출력값을 넣을 것)")
    elif abs(a.relevel_dmean) < 1e-8:
        R.add("2 레벨 재보정", OK, f"프록시 Δmean {a.relevel_dmean:+.3e}  (<1e-8)")
    else:
        R.add("2 레벨 재보정", NG, f"프록시 Δmean {a.relevel_dmean:+.3e}  재보정이 안 됐다")

    # 3. meta.json 감사
    rc, out = run([a.python, "tools/fix_meta.py", "--src", str(cand)])
    m = re.search(r"어긋난 항목\s+(\d+)/(\d+)", out)
    if rc != 0 or not m:
        R.add("3 meta.json", NG, f"감사 실패 (rc={rc})")
    elif int(m.group(1)) == 0:
        R.add("3 meta.json", OK, f"0/{m.group(2)}")
    else:
        R.add("3 meta.json", NG,
              f"{m.group(1)}/{m.group(2)} 어긋남 — **--out 으로 재생성할 것 (반드시 마지막에)**")
    return R


def layer1_verify(a, R: Report, tag: str, zp: Path) -> dict:
    """verify_zip.py 를 돌려 4~8번을 채운다. tag 는 '후보' / '챔피언'."""
    cmd = [a.python, "tools/verify_zip.py", str(zp), "--python", a.python]
    if a.rows:
        cmd += ["--rows", str(a.rows)]
    rc, out = run(cmd)
    got = {"out": out}
    got["crc"] = "CRC 정상" in out
    got["struct"] = "숨은 macOS" in out and "✅ 숨은" in out
    mt = re.search(r"약\s+(\d+)s\s*/\s*(\d+)s\s*\((\d+)%\)", out)
    got["time"] = (int(mt.group(1)), int(mt.group(2)), int(mt.group(3))) if mt else None
    got["fmt"] = "row_id 가 sample_submission" in out and "✅ 확률 범위 정상" in out
    mi = re.search(r"최대\s+([0-9.]+e[-+]?\d+)", out)
    got["indep"] = float(mi.group(1)) if mi else (0.0 if "행 독립성" in out and "❌" not in out.split("행 독립성")[-1] else None)
    got["ran"] = "정상 종료" in out
    return got


def layer2(a) -> bool:
    print("\n=== 2층 — 점수 예측 (확실해지지 않는다. 크기만 잰다) ===")
    print("\n  질문: \"이 후보가 지려면, 무엇이, 얼마나 틀려야 하는가?\"")
    if not a.risk or a.predict is None:
        print(f"\n  {NG} 답이 없다 — --risk 와 --predict 를 지정해야 판정한다.")
        print("     기전을 한 문장으로 못 쓰면 그 후보는 태우지 않는다.")
        print("     (hj07 −5.93 · hj11 −1.25 가 이 문장 없이 태운 결과다)")
        return False
    print(f"\n  답:   {a.risk}")
    lo = a.lo if a.lo is not None else a.predict
    hi = a.hi if a.hi is not None else a.predict
    print(f"\n  예측  {a.predict:+.4f}   봉투 [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0:
        print(f"  {NG} 봉투 하한이 0 이하다 — 오차 범위 안에서 진다.")
        return False
    print(f"  {OK} 봉투 전 구간 양수")
    if a.champ_lb is not None:
        print(f"\n  {a.champ_lb:.4f}  ->  {a.champ_lb + lo:.4f} ~ {a.champ_lb + hi:.4f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand", required=True, help="검증할 제출 zip")
    ap.add_argument("--champ", required=True, help="현 챔피언 zip (기준선)")
    ap.add_argument("--python", default=sys.executable, help="평가 환경 파이썬 (numpy 2.x)")
    ap.add_argument("--rows", type=int, help="verify_zip 행 수 (기본 20000)")
    ap.add_argument("--expect-changed", help="쉼표 구분. 지정하면 그 밖의 변경을 실패로 본다")
    ap.add_argument("--relevel-dmean", type=float, help="빌더가 출력한 재보정 후 프록시 Δmean")
    ap.add_argument("--quick", action="store_true", help="4~8번(격리 실행·행 독립성) 생략")
    ap.add_argument("--risk", help="2층 답: 무엇이 얼마나 틀려야 지는가 (한 문장)")
    ap.add_argument("--predict", type=float, help="예측 ΔScore")
    ap.add_argument("--lo", type=float, help="예측 봉투 하한")
    ap.add_argument("--hi", type=float, help="예측 봉투 상한")
    ap.add_argument("--champ-lb", type=float, help="챔피언 LB 점수 (환산 표시용)")
    a = ap.parse_args()

    cand, champ = Path(a.cand), Path(a.champ)
    for p in (cand, champ):
        if not p.exists():
            sys.exit(f"[중단] 없는 파일: {p}")
    print(f"후보    {cand.name}\n  SHA-256 {sha(cand)}")
    print(f"챔피언  {champ.name}")

    R = layer1(a)
    if a.quick:
        for n in ("4 zip 무결성·구조", "5 격리 실행", "6 추론 시간", "7 출력 형식", "8 행 독립성"):
            R.add(n, SKIP, "--quick")
    else:
        c = layer1_verify(a, R, "후보", cand)
        R.add("4 zip 무결성·구조", OK if c["crc"] and c["struct"] else NG,
              "CRC·구조·파일명" if c["crc"] and c["struct"] else "verify_zip 출력 확인 요")
        R.add("5 격리 실행", OK if c["ran"] else NG,
              "빈 임시 디렉터리 end-to-end" if c["ran"] else "실행 실패")
        if c["time"]:
            s, lim, pct = c["time"]
            R.add("6 추론 시간", OK if pct < 80 else NG, f"{s}s / {lim}s ({pct}%)")
        else:
            R.add("6 추론 시간", NG, "시간을 파싱하지 못했다")
        R.add("7 출력 형식", OK if c["fmt"] else NG,
              "컬럼·row_id 순서·범위·NaN" if c["fmt"] else "형식 불일치")

        # 8. 행 독립성 — **절대값이 아니라 챔피언 대비**로 읽는다
        h = layer1_verify(a, R, "챔피언", champ)
        ci, hi_ = c["indep"], h["indep"]
        if ci is None or hi_ is None:
            R.add("8 행 독립성", NG, "값을 파싱하지 못했다")
        elif ci <= max(hi_ * 3.0, 1e-8):
            R.add("8 행 독립성", OK,
                  f"후보 {ci:.3e}  vs 챔피언 {hi_:.3e}  (같은 자릿수 = BLAS 배치 누적)")
        else:
            R.add("8 행 독립성", NG,
                  f"후보 {ci:.3e} 가 챔피언 {hi_:.3e} 보다 크다 — 우리 변경의 결함")

    R.show("1층 — 빌드 검증 (결정론. 전부 통과해야 태운다)")
    l2 = layer2(a)

    print("\n" + "=" * 62)
    if R.hard_fail:
        print("판정  ❌ 1층 실패 — 태우지 않는다. 위 실패 항목을 고칠 것.")
        return 1
    if not l2:
        print("판정  ❌ 2층 미충족 — 태우지 않는다.")
        print("      (다른 후보가 전혀 없을 때만, 클래스 전적을 근거로 예외)")
        return 1
    skipped = [n for n, m, _ in R.rows if m == SKIP]
    print("판정  ✅ 1층 통과 · 2층 답 있음 — 슬롯 배분 대상")
    if R.warned:
        print(f"      ⚠️ 판정 불가 {R.warned} — 인자를 채워 다시 돌릴 것")
    if skipped:
        print(f"      ⚠️ 미실행 {skipped} — 제출 전에 채울 것")
    print("      max 채점이라 하방은 슬롯 한 장이다. 남은 후보 중 예측이 가장 큰 것부터.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
