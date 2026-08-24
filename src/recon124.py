# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# §7.124 의 수치를 predict.py / MODEL.md 주석에서 그대로 가져온다
r24   = -0.03043      # 2024 실측 F-R 격차
m     = -0.02202      # 모델이 예측하는 격차 (재기준화 더미 실측)
D_fit = r24 - m       # 적합된 보정량 = +0.00841
r25   = -0.02343      # LB 역산된 2025 실제 격차
D_tru = r25 - m       # 2025 에 필요했던 보정량
K, wF = 400700.0, 0.1142
Kp = K * wF * (1 - wF)

print(f"D_fit = r24 - m = {D_fit:+.6f}     (r24 의 {D_fit/r24:.1%} — 72% 가 상쇄된다)")
print(f"D_tru = r25 - m = {D_tru:+.6f}")
print(f"c(잔차 기준) = D_tru / D_fit = {D_tru/D_fit:.4f}   <-- 이것이 내 c 와 같은 단위")
print(f"c(원격차 기준) = r25 / r24  = {r25/r24:.4f}   <-- §7.124 가 쓴 '77%'")
print()
kap = 1.0
gain = Kp * D_fit**2 * (kap*(2*(D_tru/D_fit) - kap))
print(f"kappa=1 예측 이득 = K'·D_fit²·k(2c-k) = {gain:+.3f}     실측 LB Δ = -1.90")
print(f"   -> 식이 맞다. 소수 둘째 자리까지 재현.")
print(f"이 축 최대 이득 = K'·D_tru² = {Kp*D_tru**2:+.3f}      MODEL.md 기록 '+0.08'")
print()
# 손익분기를 두 단위로
print("손익분기 (kappa=1):")
print(f"  잔차 기준  c >= 0.500  ->  D >= {0.5*D_fit:+.6f}")
print(f"  원격차 기준 r25 >= m + {0.5*D_fit:+.6f} = {m+0.5*D_fit:+.6f}"
      f"  = r24 의 {(m+0.5*D_fit)/r24:.1%}   <-- §7.124 의 '86%'")
print("  => 같은 식이다. 분모만 다르다.")
print()
print("=== 단위 맞춘 비교 (둘 다 잔차 D 기준) ===")
for nm, kap, c in (("§7.124", 1.0, D_tru/D_fit), ("이번 (kappa=0.5)", 0.5, 0.558)):
    be = kap/2
    print(f"  {nm:<18s} kappa={kap:<4} 손익분기 c>={be:.3f}  관측 c={c:.3f}"
          f"  여유 {c/be:.2f}x  {'패' if c<be else '통과'}")
