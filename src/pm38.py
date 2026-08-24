# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

LB37, LB38 = 1110.3461746799, 1109.8068231037
D = LB38 - LB37
print(f"s38 − s37 = {D:+.10f}\n")

KAP = 0.5
S24 = 6.11        # calsub in-sample 오라클 (fold2024, s35+cat10) = K'·||a24||²
S23 = 18.57       # fold2024 전이 +5.72 = S23·κ(2c−κ), c=0.558 에서 역산
c_fold = 0.558

print("① 내 예측이 어디서 틀렸나 — **단위를 또 섞었다**")
print(f"  내가 인용한 fold2024 전이 +5.72 는 **a₂₃ 스케일**이다 (2023 계수를 2024 에 적용).")
print(f"  배포는 **a₂₄** 를 쓴다. ||a₂₄||² / ||a₂₃||² = c² = {c_fold**2:.3f}")
pred_correct = S24 * KAP * (2*c_fold - KAP)
print(f"  배포 등가 예측 = S₂₄·κ(2c−κ) = {S24:.2f}·{KAP}·({2*c_fold:.3f}−{KAP}) = {pred_correct:+.2f}")
print(f"  검산: S₂₃·c² = {S23*c_fold**2:.2f} ≈ S₂₄ {S24:.2f}  ✅ 두 경로가 맞는다")
print(f"  → 올바른 예측은 **+1.9**(배율 1)~**+0.8**(배율 2.27) 이었다. 내가 말한 +2~+6 은 3배 부풀려져 있었다.\n")

print("② 그래도 부호가 틀렸다 — c 를 역산한다")
for m, nm in ((1.0, "배율 1 (스칼라 행)"), (2.27, "배율 2.27 (s37 실측)")):
    c = 0.25 + D*m/S24
    print(f"  {nm:<22s} c = {c:+.3f}")
print(f"\n  손익분기 c = κ/2 = {KAP/2}")
print(f"  fold 간 실측 감쇠   c = {c_fold}")
print(f"  §7.124 의 실측       c = 0.168   ← 완전히 다른 레벨 보정인데 같은 자리\n")

print("③ 레벨 오차의 상한 — 점유율만으로 얼마나 잃을 수 있나")
K = 1e5/(0.4824*(1-0.4824))
for dz, nm in ((0.0, "2024 점유율(중심화 기준)"), (0.003, "더미 점유율(회전 없음, 극단)")):
    print(f"  Δlogit {dz:.4f} → Δp {dz/4:+.6f} → 손실 K·Δp² = {K*(dz/4)**2:.3f} BSS   {nm}")
print(f"  → 점유율 드리프트가 설명할 수 있는 최대치는 −0.23. 관측 −0.54 의 43%.")
print(f"     나머지 −0.31 이상은 **축 자체의 붕괴**다.\n")

print("④ 이 축에 2025 에서 남은 최대 이득")
for c in (0.162, 0.186):
    print(f"  c={c:.3f} 이면  κ*=c,  최대 = S₂₄·c² = {S24*c*c:+.3f} BSS")
print("  → 슬롯 값어치 없음. §7.124 와 같은 결말.")
