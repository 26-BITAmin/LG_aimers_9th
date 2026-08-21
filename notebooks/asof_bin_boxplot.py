#!/usr/bin/env python
"""asof_* 10분위 구간별 제구 성공률 분포 box plot + HTML 리포트.

usage: python asof_bin_boxplot.py [--data-dir ../data] [--out-dir .]

단위 3종: 투수 / 시즌x월 / 등판(경기x투수).
train.csv 에 game_id 가 없어 등판 키는 asof_pitcher_prev{1,3,5}_game_{success,middle}_rate
6개 컬럼의 변화 지점 합집합으로 파생한다.
2019-2022 game_type='F' 라벨오염 행은 제외한다.
무작위 요소 없음 (seed 불필요).
"""
import argparse, os, sys, json, base64
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='../data')
ap.add_argument('--out-dir',  default='.')
A = ap.parse_args()
OUT = A.out_dir; os.makedirs(OUT, exist_ok=True)

ASOF = ['asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_reverse_rate',
 'asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate',
 'asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate',
 'asof_pitcher_prev5_game_success_rate','asof_pitcher_prev1_game_middle_rate',
 'asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate',
 'asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate',
 'asof_pitcher_pitchmix_n','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate',
 'asof_pitcher_offspeed_rate']
PREV = ASOF[6:12]

KEYS=['row_id','season','game_month','game_type','pitcher_id','control_success']
df = pd.read_csv(os.path.join(A.data_dir,'train.csv'), usecols=KEYS+ASOF, encoding='utf-8-sig',
                 dtype={'season':'int16','game_month':'int8','game_type':'category',
                        'pitcher_id':'int32','control_success':'int8'})
for _c in ASOF: df[_c]=df[_c].astype('float32')
assert np.array_equal(df['asof_pitcher_n'].to_numpy(), df['asof_pitcher_pitchmix_n'].to_numpy()), \
    'asof_pitcher_pitchmix_n != asof_pitcher_n (이 데이터에서는 동일해야 함)'
df['ord']=np.arange(len(df),dtype=np.int32)
df = df.sort_values(['pitcher_id','ord'],kind='stable').reset_index(drop=True)
pid=df['pitcher_id'].to_numpy(); same=np.zeros(len(df),bool); same[1:]=pid[1:]==pid[:-1]
bnd=~same
for c in PREV:
    v=df[c].to_numpy(); ch=np.zeros(len(df),bool)
    ch[1:]=same[1:]&~((v[1:]==v[:-1])|(np.isnan(v[1:])&np.isnan(v[:-1]))); bnd|=ch
bnd[0]=True
df['outing_id']=np.cumsum(bnd).astype(np.int32)
df['season_month']=(df['season'].astype(np.int32)*100+df['game_month'].astype(np.int32))
mask_bad=(df['season']<=2022)&(df['game_type']=='F'); n_bad=int(mask_bad.sum())
df=df.loc[~mask_bad].reset_index(drop=True)
BASE=float(df['control_success'].mean()); y=df['control_success'].to_numpy(np.int8)
season=df['season'].to_numpy()
seas_base={s: df.loc[df['season']==s,'control_success'].mean() for s in np.unique(season)}
resid = y - df['season'].map(seas_base).to_numpy()     # 시즌 기저율 제거

UNITS=[('pitcher','pitcher_id',50,'Per-pitcher success rate',(0.26,0.76)),
       ('seasmon','season_month',200,'Per season-month success rate',(0.38,0.66)),
       ('outing','outing_id',10,'Per-outing (game x pitcher) success rate',(0.0,1.0))]

NQ=10; bininfo={}; codes_map={}
for c in ASOF:
    v=df[c].to_numpy(np.float64); ok=~np.isnan(v)
    cat,edges=pd.qcut(v[ok],NQ,retbins=True,duplicates='drop',labels=False)
    nb=int(cat.max())+1
    codes=np.full(len(df),-1,np.int16); codes[ok]=cat.astype(np.int16); codes_map[c]=codes
    isn = c.endswith('_n')
    lbl=[(f"{edges[i]:.0f}~{edges[i+1]:.0f}" if isn else f"{edges[i]:.3f}~{edges[i+1]:.3f}") for i in range(nb)]
    bininfo[c]=dict(n_bins=nb,edges=[float(x) for x in edges],labels=lbl,
                    na_count=int((~ok).sum()),has_na=bool((~ok).sum()),
                    n_unique=int(pd.Series(v[ok]).nunique()))

def collect(c,unitcol,minn):
    codes=codes_map[c]; u=df[unitcol].to_numpy()
    t=pd.DataFrame({'b':codes,'u':u,'y':y})
    gr=t.groupby(['b','u'],sort=True)['y'].agg(['sum','size'])
    gr=gr[gr['size']>=minn]; gr['rate']=gr['sum']/gr['size']
    pooled=t.groupby('b')['y'].agg(['sum','size']); pooled['rate']=pooled['sum']/pooled['size']
    data={};stats={}
    for b,sub in gr.groupby(level=0):
        r=sub['rate'].to_numpy(); n=sub['size'].to_numpy(); pbar=sub['sum'].sum()/n.sum()
        exp_sd=float(np.sqrt(np.mean(pbar*(1-pbar)/n)))
        sd=float(r.std(ddof=1)) if len(r)>1 else np.nan
        rho=float(spearmanr(r,n).statistic) if len(r)>3 else np.nan
        data[int(b)]=r
        stats[int(b)]=dict(n_groups=int(len(r)),rows_kept=int(n.sum()),
            median=float(np.median(r)),q1=float(np.percentile(r,25)),q3=float(np.percentile(r,75)),
            unw_mean=float(r.mean()),obs_sd=sd,exp_sd=exp_sd,
            excess=float(sd/exp_sd) if exp_sd>0 and sd==sd else np.nan,
            rho_rate_size=rho)
    return data,stats,pooled

def panel(ax,c,data,pooled,stats,ylim):
    nb=bininfo[c]['n_bins']
    order=([-1] if bininfo[c]['has_na'] else [])+list(range(nb))
    order=[b for b in order if b in data and len(data[b])>0]
    pos=np.arange(len(order))
    bp=ax.boxplot([data[b] for b in order],positions=pos,widths=.62,showfliers=False,
        patch_artist=True,medianprops=dict(color='#c0392b',lw=1.7),
        whiskerprops=dict(color='#777',lw=.9),capprops=dict(color='#777',lw=.9),
        boxprops=dict(facecolor='#cfe3f5',edgecolor='#4a749a',lw=.9))
    for i,b in enumerate(order):
        if b==-1: bp['boxes'][i].set_facecolor('#f3dcb2'); bp['boxes'][i].set_edgecolor('#a9803c')
    pr=[pooled.loc[b,'rate'] if b in pooled.index else np.nan for b in order]
    um=[stats[b]['unw_mean'] for b in order]
    ax.plot(pos,pr,'o-',color='#111',ms=3.6,lw=1.4,zorder=6)
    ax.plot(pos,um,'s',color='#e08a1e',ms=3.2,zorder=6)
    ax.axhline(BASE,color='#999',ls='--',lw=.9,zorder=1)
    ax.set_xticks(pos); ax.set_xticklabels(['NA' if b==-1 else str(b+1) for b in order],fontsize=7)
    d=pr[-1]-pr[1 if order[0]==-1 else 0]
    ax.set_title(f"{c.replace('asof_','')}   $\\Delta$pooled={d:+.3f}",fontsize=8.5)
    ax.tick_params(axis='y',labelsize=7); ax.grid(axis='y',alpha=.25,lw=.5); ax.set_ylim(*ylim)

results={}
for key,unitcol,minn,ulab,ylim in UNITS:
    fig,axes=plt.subplots(5,4,figsize=(19,15.5)); axes=axes.ravel(); res={}
    for i,c in enumerate(ASOF):
        data,stats,pooled=collect(c,unitcol,minn)
        res[c]=dict(stats=stats,pooled={int(b):dict(rate=float(pooled.loc[b,'rate']),n=int(pooled.loc[b,'size'])) for b in pooled.index})
        panel(axes[i],c,data,pooled,stats,ylim)
    for j in range(len(ASOF),len(axes)): axes[j].axis('off')
    axes[19].text(.02,.95,f"unit: {ulab}\nmin {minn} rows per (bin, unit) cell\n\n"
        "black line = pooled rate (volume-weighted)\n"
        "orange square = unweighted mean over units\n"
        "red line = median, box = IQR, whiskers = 1.5*IQR\n"
        "orange box = NA bin (cold start)\n"
        f"dashed = overall base rate {BASE:.4f}\n"
        f"x = decile 1..10 (low->high)\n\n"
        f"train 2019-2024, contaminated F excluded\nn = {len(df):,} rows",
        fontsize=9,va='top',family='monospace')
    fig.suptitle(f"asof_* decile bins  x  {ulab}",fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.975])
    p=os.path.join(OUT,f"box_{key}.png"); fig.savefig(p,dpi=100); plt.close(fig)
    print("saved",p, os.path.getsize(p)//1024,"KB"); results[key]=res

# ---- 시즌 보정 기울기: bin별 (y - 시즌기저율) 평균 ----
seasadj={}
for c in ASOF:
    b=codes_map[c]
    s=pd.Series(resid).groupby(b).mean()
    seasadj[c]={int(k):float(v) for k,v in s.items()}
json.dump(dict(base_rate=BASE,n_rows=int(len(df)),n_dropped=n_bad,season_base={int(k):float(v) for k,v in seas_base.items()},
               bininfo=bininfo,results=results,season_adjusted=seasadj),
          open(os.path.join(OUT,'stats.json'),'w'),indent=1)
print("done")


# ================= 요약 테이블 =================
S=json.load(open(os.path.join(OUT,'stats.json')))
ASOF=list(S['bininfo'].keys()); BASE=S['base_rate']
rows=[]
for c in ASOF:
    nb=S['bininfo'][c]['n_bins']
    pooled=S['results']['pitcher'][c]['pooled']
    pr=[pooled[str(b)]['rate'] for b in range(nb)]
    sa=S['season_adjusted'][c]
    sar=[sa[str(b)] for b in range(nb)]
    na = pooled.get('-1')
    d_raw = pr[-1]-pr[0]
    d_adj = sar[-1]-sar[0]
    mono = spearmanr(range(nb), pr).statistic
    # 시즌×월 단위 bin1 vs bin10 박스 겹침
    sm=S['results']['seasmon'][c]['stats']
    b1,b10=sm['0'],sm[str(nb-1)]
    overlap = not (b1['q3']<b10['q1'] or b10['q3']<b1['q1'])
    # 과분산 (관측SD / 이항SD)
    exc_p=np.nanmean([S['results']['pitcher'][c]['stats'][str(b)]['excess'] for b in range(nb)])
    exc_o=np.nanmean([S['results']['outing'][c]['stats'][str(b)]['excess'] for b in range(nb)])
    exc_s=np.nanmean([S['results']['seasmon'][c]['stats'][str(b)]['excess'] for b in range(nb)])
    # 볼륨가중 효과: pooled 폭 vs 투수 unweighted 폭
    up=[S['results']['pitcher'][c]['stats'][str(b)]['unw_mean'] for b in range(nb)]
    d_unw = up[-1]-up[0]
    rows.append(dict(feature=c, bins=nb, d_pooled=d_raw, d_seasonadj=d_adj, d_unw_pitcher=d_unw,
        wt_share=(1-d_unw/d_raw) if abs(d_raw)>1e-9 else np.nan,
        mono_spearman=mono, na_n=(na['n'] if na else 0), na_rate=(na['rate'] if na else np.nan),
        exc_pitcher=exc_p, exc_seasmon=exc_s, exc_outing=exc_o))
t=pd.DataFrame(rows).sort_values('d_pooled',key=abs,ascending=False)
pd.set_option('display.width',250)
print(t.round(4).to_string(index=False))
t.to_csv(os.path.join(OUT,'summary.csv'),index=False)


# ================= HTML 리포트 =================
S=json.load(open(os.path.join(OUT,'stats.json'))); BASE=S['base_rate']; N=S['n_rows']
ASOF=list(S['bininfo'].keys())
summ=pd.read_csv(os.path.join(OUT,'summary.csv'))

def b64(p):
    return base64.b64encode(open(p,'rb').read()).decode()

UNITS=[('pitcher','투수별','min 50행/셀'),('seasmon','시즌×월별','min 200행/셀'),('outing','등판별 (경기×투수)','min 10행/셀')]

def var_share(e):
    if not np.isfinite(e) or e<=0: return float('nan')
    return max(0.0, 1-1/e**2)

# 요약 테이블
srows=[]
for _,r in summ.iterrows():
    srows.append(f"""<tr><td class=f>{r.feature}</td><td class=n>{r.d_pooled:+.4f}</td><td class=n>{r.d_seasonadj:+.4f}</td>
<td class=n>{r.d_unw_pitcher:+.4f}</td><td class=n>{'' if not np.isfinite(r.wt_share) else f'{r.wt_share*100:.0f}%'}</td>
<td class=n>{r.mono_spearman:+.3f}</td><td class=n>{int(r.na_n):,}</td><td class=n>{'' if not np.isfinite(r.na_rate) else f'{r.na_rate:.4f}'}</td>
<td class=n>{r.exc_pitcher:.2f}</td><td class=n>{r.exc_seasmon:.2f}</td><td class=n>{r.exc_outing:.2f}</td></tr>""")

# 피처별 상세
det=[]
for c in ASOF:
    info=S['bininfo'][c]; nb=info['n_bins']
    pooled=S['results']['pitcher'][c]['pooled']
    order=(['-1'] if info['has_na'] else [])+[str(b) for b in range(nb)]
    hdr="".join(f"<th>{'NA' if b=='-1' else int(b)+1}</th>" for b in order)
    def row(vals,fmt="{:.4f}"):
        return "".join(f"<td class=n>{'' if v is None or (isinstance(v,float) and not np.isfinite(v)) else fmt.format(v)}</td>" for v in vals)
    rng=[('' if b=='-1' else info['labels'][int(b)]) for b in order]
    lines=[f"<tr><th class=rh>구간 범위</th>"+"".join(f"<td class=rg>{'결측' if b=='-1' else info['labels'][int(b)]}</td>" for b in order)+"</tr>",
           f"<tr><th class=rh>행 수</th>"+row([pooled[b]['n'] for b in order],"{:,}")+"</tr>",
           f"<tr><th class=rh>pooled 성공률</th>"+row([pooled[b]['rate'] for b in order])+"</tr>",
           f"<tr><th class=rh>시즌보정 편차</th>"+row([S['season_adjusted'][c].get(b) for b in order],"{:+.4f}")+"</tr>"]
    for k,ko,_ in UNITS:
        st=S['results'][k][c]['stats']
        lines.append(f"<tr class=sep><th class=rh>{ko} · 셀 수</th>"+row([st[b]['n_groups'] if b in st else None for b in order],"{:,}")+"</tr>")
        lines.append(f"<tr><th class=rh>{ko} · 중앙값</th>"+row([st[b]['median'] if b in st else None for b in order])+"</tr>")
        lines.append(f"<tr><th class=rh>{ko} · IQR</th>"+row([(st[b]['q3']-st[b]['q1']) if b in st else None for b in order])+"</tr>")
        lines.append(f"<tr><th class=rh>{ko} · 과분산비</th>"+row([st[b]['excess'] if b in st else None for b in order],"{:.2f}")+"</tr>")
    det.append(f"""<details><summary><code>{c}</code> &nbsp;<span class=meta>고유값 {info['n_unique']:,} · 구간 {nb}개 · 결측 {info['na_count']:,}</span></summary>
<div class=tw><table class=dt><tr><th class=rh></th>{hdr}</tr>{''.join(lines)}</table></div></details>""")

figs=""
for k,ko,mm in UNITS:
    figs+=f"""<h3>{ko} 단위 <span class=meta>({mm})</span></h3>
<div class=tw><img src="data:image/png;base64,{b64(os.path.join(OUT,f'box_{k}.png'))}" alt="{k}"></div>"""

HTML=f"""<title>asof 구간별 성공률 분포</title>
<style>
:root{{--bg:#fff;--fg:#1a1d21;--mut:#6b7280;--line:#e2e5e9;--acc:#1d4e89;--warn:#8a4b00;--card:#f7f8fa}}
:root:not([data-theme=light]) {{}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#14171a;--fg:#e6e8ea;--mut:#9aa3ad;--line:#2b3037;--acc:#7db3e8;--warn:#e0a45c;--card:#1c2025}}}}
:root[data-theme=dark]{{--bg:#14171a;--fg:#e6e8ea;--mut:#9aa3ad;--line:#2b3037;--acc:#7db3e8;--warn:#e0a45c;--card:#1c2025}}
body{{background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;
max-width:1180px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.3px}}
h2{{font-size:19px;margin:38px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}}
h3{{font-size:15px;margin:26px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);font-size:13.5px;margin-bottom:22px}}
code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.tw{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
img{{max-width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}}
table{{border-collapse:collapse;font-size:12.5px;min-width:100%}}
th,td{{border:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}}
th{{background:var(--card);font-weight:600}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}}
td.f{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px}}
td.rg{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--mut)}}
th.rh{{position:sticky;left:0;background:var(--card);z-index:1;font-weight:500;font-size:12px}}
tr.sep th,tr.sep td{{border-top:2px solid var(--line)}}
details{{border:1px solid var(--line);border-radius:6px;margin:8px 0;padding:0}}
summary{{cursor:pointer;padding:9px 12px;font-family:ui-monospace,Menlo,monospace;font-size:13px}}
details[open] summary{{border-bottom:1px solid var(--line);background:var(--card)}}
details .tw{{padding:10px 12px}}
.meta{{color:var(--mut);font-size:12px;font-family:-apple-system,sans-serif}}
.box{{background:var(--card);border-left:3px solid var(--acc);padding:12px 16px;margin:14px 0;border-radius:0 6px 6px 0}}
.box.w{{border-left-color:var(--warn)}}
ul{{padding-left:22px}} li{{margin:5px 0}}
.k{{color:var(--acc);font-weight:600}}
</style>

<h1>asof_* 10분위 구간별 제구 성공률 분포</h1>
<div class=sub>DACON 236743 · train.csv 2019–2024 · 2019–2022 <code>game_type='F'</code> 라벨오염 {S['n_dropped']:,}행 제외 후 <b>{N:,}행</b> · 전체 성공률 {BASE:.4f}</div>

<h2>그림 읽는 법</h2>
<div class=box>
<b>박스 = 분포, 선 = 실제 값.</b> 한 구간의 성공률 자체는 스칼라 하나라 박스가 안 그려진다.
그래서 구간 안을 단위(투수 / 시즌×월 / 등판)로 쪼개 각 단위의 성공률을 모아 박스로 그렸다.
<ul>
<li><b>검은 선</b> = pooled 성공률(행 수 가중). 이게 실제 그 구간의 성공률이다.</li>
<li><b>주황 사각</b> = 단위별 성공률의 <i>비가중</i> 평균. 검은 선과 벌어지면 볼륨 가중 효과가 있다는 뜻.</li>
<li><b>빨간 선/박스</b> = 중앙값 / IQR. <b>주황 박스</b> = 결측(cold-start) 구간.</li>
<li><b>과분산비</b> = 관측 SD ÷ 이항 노이즈 SD. 1.0이면 박스 폭이 전부 표본 노이즈, 클수록 진짜 이질성.</li>
</ul>
</div>

<h2>핵심 발견</h2>
<div class=box>
<ol>
<li><span class=k><code>asof_pitcher_pitchmix_n</code>은 <code>asof_pitcher_n</code>과 완전히 동일한 컬럼이다.</span>
1,475,092행 전부에서 비트 단위로 일치(<code>np.array_equal → True</code>). 설명서에는 "구종 사용 이력 표본 수"로 다르게 적혀 있지만 값이 같다. 피처 하나 버려도 정보 손실 0.</li>

<li><span class=k>등판(경기×투수) 단위 박스 폭의 약 70%는 순수 이항 노이즈다.</span>
과분산비가 전 피처 1.17–1.25 범위 → 분산 기준 실제 이질성 지분은 <code>1−1/1.2²≈31%</code>.
"오늘 컨디션"에 해당하는 등판 단위 변동은 데이터에서 거의 안 보인다. 등판 단위 피처를 만들 유인이 약하다.</li>

<li><span class=k>pooled 기울기의 상당 부분이 볼륨 가중에서 온다.</span>
<code>asof_pitcher_success_rate</code>: pooled Δ(1→10구간) = +0.131이지만 투수 비가중 Δ = +0.085.
차이 35%는 "상위 구간에 오래 머무는 고볼륨 투수"가 만든 것이다. 구간 내 상관 spearman(성공률, 셀 크기)이
1구간 −0.28에서 10구간 +0.48까지 오르는 것과 일치한다(엄밀한 단조는 아니고 3→4, 6→7구간에서 소폭 역전).</li>

<li><span class=k>시즌 드리프트가 <code>*_n</code> 계열의 기울기를 통째로 뒤집는다.</span>
<code>asof_pitcher_n</code>: 원 Δ = +0.006(무신호처럼 보임) → 시즌 기저율 보정 후 Δ = <b>+0.033</b>.
<code>asof_batter_n</code>: 원 Δ = −0.027 → 보정 후 <b>+0.006</b>(신호 소멸). 누적 카운트는 후반 시즌에 몰려 있어
시즌 기저율 하락과 교란된다. 이 두 피처는 시즌 보정 없이 해석하면 안 된다.</li>

<li><span class=k>rate 계열의 구간–성공률 관계는 시간에 걸쳐 안정적이다.</span>
시즌×월 단위 박스에서 <code>asof_pitcher_success_rate</code>의 1구간(0.45)과 10구간(0.58) IQR이 전혀 겹치지 않는다.
양쪽 구간이 각각 200행 이상인 시즌–월 셀 <b>42개 전부</b>에서 10구간 &gt; 1구간이고, 격차는 최소 +0.027 · 중앙값 +0.112다
→ 2025 외삽 시 이 관계의 <b>부호와 순서</b>는 믿을 만하다.
단 <b>수준(level)</b>은 시즌마다 내려가므로 절대 확률은 보정이 필요하다.</li>

<li><span class=k>투수 단위 과분산비가 피처의 설명력 순위를 준다.</span>
구간으로 자른 뒤에도 남는 투수 간 이질성이 작을수록 그 피처가 투수 차이를 많이 설명한 것이다.
<code>success_rate</code> 1.32 &lt; <code>reverse_rate</code> 1.43 &lt; <code>prev3/5_game_success</code> 1.57 ≪ <code>pitcher_n</code> 2.42.
이전 EDA의 단변량 BSS 순위(899 / 823 / 702·832)와 같은 순서다.</li>
</ol>
</div>

<div class="box w">
<b>이 그림으로 말할 수 없는 것.</b> 전부 단변량 주변(marginal) 관계다. 구간 간 성공률 차이는 다른 변수(시즌·투수 정체성·볼륨)와
교란돼 있고, 시즌 보정 열만 그중 하나를 뺐을 뿐이다. 조건부 효과나 피처 채택 근거로 쓰려면 모델 안에서 다시 재야 한다.
또한 분위 경계는 이 train 분포에서 계산된 것이라 2025 test 분포가 이동하면 같은 경계가 같은 분위를 뜻하지 않는다.
<br><br>
<b>등판 단위의 타자 계열 3개는 특히 조심.</b> <code>asof_batter_*</code>는 타자마다 값이 바뀌어 한 등판이 10개 구간에 흩어진다.
min 10행 필터를 통과하는 셀은 "한 등판에서 같은 구간의 타자를 10명 이상 상대한" 경우뿐이라 행 커버리지가 <b>61.5–65%</b>로 떨어지고
긴 등판(선발)에 편향된다. 투수 계열은 92.9–97.2%, 투수 단위는 96.8–99.7%, 시즌×월 단위는 99.8% 이상으로 문제없다.
</div>

<h2>박스플롯</h2>
{figs}

<h2>피처 요약</h2>
<div class=tw><table>
<tr><th>피처</th><th class=n>Δpooled<br>(10−1구간)</th><th class=n>Δ시즌보정</th><th class=n>Δ투수<br>비가중</th><th class=n>볼륨가중<br>기여</th>
<th class=n>단조성<br>spearman</th><th class=n>결측 행</th><th class=n>결측<br>성공률</th><th class=n>과분산<br>투수</th><th class=n>과분산<br>시즌월</th><th class=n>과분산<br>등판</th></tr>
{''.join(srows)}
</table></div>
<div class=meta style="margin-top:6px">Δpooled 절댓값 내림차순. 볼륨가중 기여 = 1 − Δ비가중/Δpooled. 음수면 비가중 쪽 기울기가 더 크다는 뜻.</div>

<h2>피처별 구간 상세</h2>
{''.join(det)}

<h2>재현</h2>
<div class=box>
<code>python notebooks/asof_bin_boxplot.py</code> (data/train.csv 필요, 약 3분, 무작위 요소 없음)<br>
구간화 <code>pd.qcut(q=10, duplicates='drop')</code>, 결측은 별도 NA 구간. 19개 컬럼 모두 10구간이 온전히 생성됨.<br>
등판 키는 원본에 game_id가 없어 파생: <code>asof_pitcher_prev{{1,3,5}}_game_{{success,middle}}_rate</code> 6개 컬럼의 변화 지점 합집합.
검증 — 투수 내 <code>asof_pitcher_n</code>이 row_id 순서로 정확히 +1씩 증가(1,474,300쌍 전부), 파생 등판 45,861개,
등판당 중앙값 19투구, 투수–시즌당 중앙값 14등판.<br>
환경: Python 3.11, pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.9, scipy.
</div>
"""
open(os.path.join(OUT,'asof_bin_boxplot_report.html'),'w',encoding='utf-8').write(HTML)
