"""실험 팔(arm) 정의를 전수로 읽어 **한 팔에서 두 가지 이상이 바뀐 것**을 찾는다.

왜 이 도구인가
    축이 죽은 게 아니라 측정이 죽어 있던 사례가 두 번 나왔다.
      · 부록 D  `ASOF_STRENGTH` — 무효한 근거로 닫힘
      · `ablate17` emb8 — `pitcher_id` 4→8 에 `hid` 128→256(§7.25 에서 −75~−88 로
        확인된 변경)·`dropout`·`wd`·`batter_id` 가 함께 묶여 있었다.
        따로 재니 **LB +7.67** (신영 C-4, 팀 챔피언 1127.1558).

    결론 문장이 아니라 **코드의 팔 정의**를 본다. 대조군과 두 개 이상 다른 팔은
    그 축을 판정할 자격이 없다 — 이긴 성분과 진 성분이 상쇄되기 때문이다.

한계 (읽는 사람이 알아야 한다)
    · 정적 분석이라 팔이 런타임에 조립되면 못 잡는다.
    · "대조군" 은 dict 의 첫 항목으로 가정한다. 아니면 --control 로 지정한다.
    · 교란이 곧 오판은 아니다. **다시 잴 후보 목록**이지 결론이 아니다.
"""
import argparse
import ast
import os
import sys


def unparse(n):
    try:
        return ast.unparse(n)
    except Exception:
        return "?"


def arm_kwargs(node):
    """`dict(a=1, b=2)` · `dict(BASE, a=1)` · `{...}` 에서 {키: 값소스} 를 뽑는다.

    반환 (kwargs, base) — base 는 `dict(BASE, ...)` 의 BASE 이름(없으면 None).
    """
    if isinstance(node, ast.Tuple) and node.elts:          # (설정, 시드수) 형태
        node = node.elts[0]
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict":
        base = unparse(node.args[0]) if node.args else None
        return {k.arg: unparse(k.value) for k in node.keywords if k.arg}, base
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out[k.value] = unparse(v)
        return out, None
    return None, None


def scan_file(path, min_diff, control_name):
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    hits = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        group = node.targets[0].id if isinstance(node.targets[0], ast.Name) else "?"
        arms = {}
        for k, v in zip(node.value.keys, node.value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            kw, base = arm_kwargs(v)
            if kw is not None:
                arms[k.value] = (kw, base)
        if len(arms) < 2:
            continue

        names = list(arms)
        ctrl = control_name if control_name in arms else names[0]
        ckw, cbase = arms[ctrl]
        for name in names:
            if name == ctrl:
                continue
            kw, base = arms[name]
            diff = []
            for key in sorted(set(ckw) | set(kw)):
                a, b = ckw.get(key), kw.get(key)
                if a != b:
                    diff.append((key, a, b))
            if base != cbase:
                diff.append(("<base>", cbase, base))
            if len(diff) >= min_diff:
                hits.append(dict(file=path, line=node.lineno, group=group,
                                 control=ctrl, arm=name, diff=diff,
                                 ctx=lines[node.lineno - 1][:90]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--min-diff", type=int, default=2,
                    help="대조군과 이만큼 이상 다르면 교란으로 본다 (기본 2)")
    ap.add_argument("--control", default=None, help="대조군 팔 이름 (기본: dict 의 첫 항목)")
    a = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(a.root):
        files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]

    hits = []
    for f in sorted(files):
        hits += scan_file(f, a.min_diff, a.control)

    hits.sort(key=lambda h: -len(h["diff"]))
    print(f"파일 {len(files)}개 · 교란 의심 팔 {len(hits)}개 (대조군과 {a.min_diff}개 이상 상이)\n")
    for h in hits:
        rel = os.path.relpath(h["file"], a.root)
        print(f"── {rel}:{h['line']}  {h['group']}   [{h['control']}] vs [{h['arm']}]   {len(h['diff'])}개 상이")
        for key, x, y in h["diff"]:
            print(f"     {key:<14} {str(x)[:34]:<34} -> {str(y)[:34]}")
        print()
    return hits


if __name__ == "__main__":
    main()
