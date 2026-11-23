# ============================================================
# v4 — 다자산 × 다전략 포트폴리오 (국내 상장 ETF 약 300개 → 저상관 자산 선별)
#
#   1층  유니버스: 국내 상장 ETF 중 시가총액 상위 300개
#        (레버리지·인버스·현금성(CD금리·KOFR 등) 제외)
#   2층  자산 선별: 학습구간 수익률 상관관계로 계층 군집화
#        → 군집마다 대표 1개 = 서로 상관 낮은 자산 N_SELECT개
#   3층  전략 3개 (선별 자산에 각각 적용)
#        ① 리스크 패리티 : 변동성 역수 비중 (기본 바닥)
#        ② 추세 + AI 필터 : 120일 수익률 > 0 인 자산만 보유
#                          × 오토인코더 이상도에 따라 노출 100/50/0%
#        ③ 급락 반등 : 최근 변동성 급등 + 20일 하락 자산 매수 (방향 사전 고정)
#   4층  전략 합성: 동일 비중(1/3) → 포트폴리오 변동성 타깃 10%
#   5층  리스크 리포트: 자산별 위험 기여도, VaR, 최악 20일, 2020년 3월 시나리오
#
#   검증: 워크포워드 (3년 학습 → 1년 검증). 자산 선별·AI 학습 모두 학습구간만 사용
# 코랩용 / API 키 불필요
# ============================================================

# %% [셀 1] 설치 (코랩에서 한 번만)
# !pip install -q finance-datareader koreanize-matplotlib

# %% [셀 2] 임포트 & 설정
import os
import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

START = "2014-01-01"
UNIVERSE_SIZE = 300          # 유니버스 크기
N_SELECT = 12                # 최종 선별 자산 수 (= 군집 수)
MIN_VOL = 0.03               # 연변동성 3% 미만 = 현금성 → 선별 제외
CACHE_FILE = "etf_close_cache.pkl"   # 가격 캐시 (두 번째 실행부터 빠름)

# 이름에 포함되면 제외 (레버리지/인버스/현금성)
EXCLUDE_WORDS = ["레버리지", "인버스", "2X", "3X", "곱버스", "Leverage", "Inverse",
                 "CD금리", "KOFR", "SOFR", "머니마켓", "단기자금", "초단기", "단기채", "단기통안", "금리액티브"]

# 유니버스 조회 실패 시 사용할 기본 목록
FALLBACK = {"069500": "KODEX 200", "229200": "KODEX 코스닥150",
            "143850": "TIGER 미국S&P500선물(H)", "133690": "TIGER 미국나스닥100",
            "148070": "KOSEF 국고채10년", "114260": "KODEX 국고채3년",
            "132030": "KODEX 골드선물(H)", "261240": "KODEX 미국달러선물",
            "130680": "TIGER 원유선물Enhanced(H)", "192090": "TIGER 차이나CSI300",
            "195930": "TIGER 유로스탁스50(합성 H)", "091160": "KODEX 반도체",
            "091170": "KODEX 은행", "117460": "KODEX 에너지화학", "266420": "KODEX 헬스케어"}

FEE = 0.00015
SLIPPAGE = 0.001
COST = FEE + SLIPPAGE

TRAIN_DAYS = 756
TEST_DAYS = 252
MIN_TEST_DAYS = 60
REBAL = 20                   # 20거래일(약 1달)마다 리밸런싱

# 전략 파라미터 (사전 고정 — 결과 보고 바꾸지 않기)
VOL_LOOKBACK = 60
TREND_LOOKBACK = 120
SPIKE_RATIO = 1.5            # 20일 변동성 / 120일 변동성 > 1.5 → 변동성 급등
PORT_TARGET_VOL = 0.10       # 포트폴리오 목표 연변동성 10%
AI_WIN = 20
AI_EPOCHS = 300
AI_Q_HALF = 0.80
AI_Q_ZERO = 0.95
SEED = 42

print("[1/10] 설정 완료")


# %% [셀 3] 유니버스 구성 (ETF 약 300개)
def build_universe():
    try:
        import FinanceDataReader as fdr
        lst = fdr.StockListing("ETF/KR")
        print(f"[유니버스] 국내 상장 ETF {len(lst)}개 조회 | 컬럼: {list(lst.columns)}")
        code_col = "Symbol" if "Symbol" in lst.columns else ("Code" if "Code" in lst.columns else lst.columns[0])
        name_col = "Name" if "Name" in lst.columns else lst.columns[1]
        lst = lst.copy()
        keep = pd.Series(True, index=lst.index)
        for w in EXCLUDE_WORDS:
            keep = keep & ~lst[name_col].astype(str).str.contains(w, case=False, regex=False)
        print(f"[유니버스] 레버리지/인버스/현금성 제외 → {int(keep.sum())}개")
        lst = lst[keep]
        for c in ["MarCap", "Amount", "Volume"]:
            if c in lst.columns:
                lst = lst.sort_values(c, ascending=False)
                print(f"[유니버스] '{c}' 기준 상위 {UNIVERSE_SIZE}개 선택")
                break
        lst = lst.head(UNIVERSE_SIZE)
        codes = lst[code_col].astype(str).str.zfill(6)
        return dict(zip(codes, lst[name_col].astype(str)))
    except Exception as e:
        print("[유니버스] 조회 실패 → 기본 목록 사용:", e)
        return dict(FALLBACK)


UNIVERSE = build_universe()
print(f"[2/10] 유니버스 {len(UNIVERSE)}개")


# %% [셀 4] 가격 수집 (캐시 사용)
def download_prices(universe):
    import FinanceDataReader as fdr
    close = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        try:
            close = pd.read_pickle(CACHE_FILE)
            print(f"[가격] 캐시에서 {close.shape[1]}개 종목 불러옴")
        except Exception as e:
            print("[가격] 캐시 읽기 실패:", e)
            close = pd.DataFrame()
    need = [c for c in universe if c not in close.columns]
    print(f"[가격] 새로 받을 종목 {len(need)}개 (개당 약 0.5초)")
    series = {}
    fails = []
    t0 = time.time()
    for i, code in enumerate(need):
        try:
            d = fdr.DataReader(code, START)
            if d is not None and len(d) > 0 and "Close" in d.columns:
                series[code] = d["Close"]
            else:
                fails.append(code)
        except Exception:
            fails.append(code)
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{len(need)} 완료 ({time.time() - t0:.0f}초)")
    if len(series) > 0:
        close = pd.concat([close, pd.DataFrame(series)], axis=1)
    close = close.sort_index()
    try:
        close.to_pickle(CACHE_FILE)
    except Exception as e:
        print("[가격] 캐시 저장 실패:", e)
    if fails:
        print(f"[가격] 실패 {len(fails)}개: {fails[:10]}{' ...' if len(fails) > 10 else ''}")
    cols = [c for c in universe if c in close.columns]
    return close[cols]


close = download_prices(UNIVERSE)
close = close[close.index >= pd.Timestamp(START)]
close = close.replace(0, np.nan)
close = close[close.notna().sum(axis=1) >= 3]        # 데이터가 거의 없는 날 제거
close = close.ffill(limit=5)                          # 하루이틀 빈 값만 채움

R_df = close / close.shift(1) - 1
bad = [c for c in R_df.columns if (R_df[c].abs() > 0.3).any()]
if bad:
    print(f"  ⚠️ 하루 ±30% 넘는 이상치 종목 {len(bad)}개 제외(분할/오류 의심): "
          f"{[UNIVERSE.get(c, c) for c in bad[:8]]}")
    R_df = R_df.drop(columns=bad)
    close = close.drop(columns=bad)

CODES = list(R_df.columns)
NAMES = {c: UNIVERSE.get(c, c) for c in CODES}
DATES = R_df.index
N = len(DATES)
R = R_df.values                    # (N, 자산수), 상장 전은 nan
CL = close.values
print(f"[3/10] 가격 정리 완료 | 자산 {len(CODES)}개 × {N}일 | {DATES[0].date()} ~ {DATES[-1].date()}")


# %% [셀 5] 공통 함수 (지표 / 저상관 군집 선별)
def ann_vol(cols_idx, t, n):
    """t일까지 n일 연변동성 (cols_idx 자산들)"""
    x = R[t - n + 1:t + 1, cols_idx]
    return np.nanstd(x, axis=0) * np.sqrt(252)


def metrics_from_returns(r):
    r = np.nan_to_num(np.asarray(r, dtype=float))
    eq = np.cumprod(1 + r)
    years = len(r) / 252
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Vol": r.std() * np.sqrt(252)}


def cluster_assets(corr, n_clusters):
    """
    평균연결 계층 군집화 (직접 구현, scipy 불필요)
    거리 = sqrt(0.5 × (1 - 상관계수)) : 상관 1 → 거리 0, 상관 -1 → 거리 1
    """
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    n = len(dist)
    D = dist.copy()
    np.fill_diagonal(D, np.inf)
    members = {i: [i] for i in range(n)}
    active = np.ones(n, dtype=bool)
    while active.sum() > n_clusters:
        Dm = np.where(active[:, None] & active[None, :], D, np.inf)
        k = int(np.argmin(Dm))
        i, j = divmod(k, n)
        ni, nj = len(members[i]), len(members[j])
        new_row = (ni * D[i] + nj * D[j]) / (ni + nj)
        D[i, :] = new_row
        D[:, i] = new_row
        D[i, i] = np.inf
        active[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        members[i] = members[i] + members[j]
        del members[j]
    return [members[i] for i in range(n) if active[i]]


def select_assets(a, b):
    """
    학습구간 [a, b) 에 데이터가 모두 있는 자산 → 군집화 → 군집 대표(군집 내 평균상관이 가장 높은 '중심' 자산)
    반환: 선별 자산 인덱스 리스트, 선별 평균상관, 전체 평균상관
    """
    Rt = R[a:b]
    full = np.where(~np.isnan(Rt).any(axis=0))[0]
    vol = np.nanstd(Rt[:, full], axis=0) * np.sqrt(252)
    elig = full[vol >= MIN_VOL]
    if len(elig) < N_SELECT * 2:
        return None, len(elig), np.nan, np.nan
    corr = np.corrcoef(Rt[:, elig].T)
    corr = np.nan_to_num(corr)
    clusters = cluster_assets(corr, N_SELECT)
    picks = []
    for mem in clusters:
        if len(mem) == 1:
            picks.append(mem[0])
        else:
            sub = corr[np.ix_(mem, mem)]
            avg = (sub.sum(axis=1) - 1) / (len(mem) - 1)
            picks.append(mem[int(np.argmax(avg))])
    sel = [int(elig[p]) for p in picks]
    off = ~np.eye(len(elig), dtype=bool)
    all_avg = corr[off].mean()
    sc = np.corrcoef(Rt[:, sel].T)
    sel_avg = sc[~np.eye(len(sel), dtype=bool)].mean()
    return sel, len(elig), sel_avg, all_avg


print("[4/10] 공통 함수 준비 완료")


# %% [셀 6] AI 필터 (오토인코더 이상도 → 노출 배율)
AI_OK = True
try:
    import torch
    import torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[5/10] PyTorch 사용 가능 | device = {DEVICE}")
except Exception as e:
    AI_OK = False
    print("[5/10] PyTorch 없음 → AI 필터 끔(배율 1.0):", e)

if AI_OK:
    class MarketAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=3):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                         nn.Linear(32, 8), nn.ReLU(),
                                         nn.Linear(8, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(),
                                         nn.Linear(8, 32), nn.ReLU(),
                                         nn.Linear(32, input_dim))

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z


def ai_overlay(sel, a, b):
    """
    선별 자산들의 '시장 상태'(평균수익, 평균 절대수익, 자산 간 수익 분산) 20일 패턴을
    학습구간으로만 오토인코더 학습 → 이상도 높으면 노출 축소 (1.0 / 0.5 / 0.0)
    """
    if not AI_OK:
        return np.ones(N), np.zeros(N), (0, 0)
    sub = np.nan_to_num(R[:, sel])
    feats = np.column_stack([sub.mean(axis=1), np.abs(sub).mean(axis=1), sub.std(axis=1)])
    mu = feats[a:b].mean(axis=0)
    sd = feats[a:b].std(axis=0) + 1e-8
    fz = np.clip((feats - mu) / sd, -5, 5)
    idx = np.clip(np.arange(N)[:, None] + np.arange(-AI_WIN + 1, 1)[None, :], 0, N - 1)
    X = fz[idx].reshape(N, -1).astype(np.float32)

    torch.manual_seed(SEED)
    model = MarketAutoencoder(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_tr = torch.tensor(X[a + AI_WIN:b]).to(DEVICE)
    for ep in range(AI_EPOCHS):
        recon, _ = model(x_tr)
        loss = loss_fn(recon, x_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        xa = torch.tensor(X).to(DEVICE)
        recon, _ = model(xa)
        err = ((recon - xa) ** 2).mean(dim=1).cpu().numpy()
    th_h = np.quantile(err[a + AI_WIN:b], AI_Q_HALF)
    th_z = np.quantile(err[a + AI_WIN:b], AI_Q_ZERO)
    scale = np.ones(N)
    scale[err > th_h] = 0.5
    scale[err > th_z] = 0.0
    return scale, err, (th_h, th_z)


# %% [셀 7] 전략 3개 + 합성 + 시뮬레이터
def tgt_risk_parity(sel, t, ai):
    v = ann_vol(sel, t, VOL_LOOKBACK)
    inv = np.where(v > 0, 1 / np.where(v > 0, v, 1), 0)
    return inv / inv.sum() if inv.sum() > 0 else np.zeros(len(sel))


def tgt_trend_ai(sel, t, ai):
    base = tgt_risk_parity(sel, t, ai)
    past = CL[t - TREND_LOOKBACK, sel]
    now = CL[t, sel]
    mom = np.where(past > 0, now / np.where(past > 0, past, 1) - 1, 0)
    w = np.where(mom > 0, base, 0)           # 추세 없는 자산 몫은 현금
    return w * ai[t]


def tgt_spike(sel, t, ai):
    v20 = ann_vol(sel, t, 20)
    v120 = ann_vol(sel, t, 120)
    r20 = CL[t, sel] / CL[t - 20, sel] - 1
    trig = (v120 > 0) & (v20 / np.where(v120 > 0, v120, 1) > SPIKE_RATIO) & (r20 < 0)
    return trig.astype(float) / len(sel)


SLEEVES = {"① 리스크패리티": tgt_risk_parity,
           "② 추세+AI": tgt_trend_ai,
           "③ 급락반등": tgt_spike}


def tgt_combined(sel, t, ai):
    """
    전략 3개 동일비중 합성 → 포트폴리오 변동성 10%에 맞게 확대/축소
    - 변동성이 낮으면 놀고 있는 현금까지 투자 (단, 총 투자비중 100% 이하 = 빚투 없음)
    - 변동성이 높으면 비중 축소
    """
    w = np.zeros(len(sel))
    for fn in SLEEVES.values():
        w = w + fn(sel, t, ai) / len(SLEEVES)
    if w.sum() <= 0:
        return w
    x = np.nan_to_num(R[t - VOL_LOOKBACK + 1:t + 1][:, sel])
    cov = np.cov(x.T) * 252
    pv = math.sqrt(max(float(w @ cov @ w), 0))
    if pv > 0:
        scale = min(PORT_TARGET_VOL / pv, 1.0 / w.sum())
        w = w * scale
    return w


def tgt_equal(sel, t, ai):
    return np.ones(len(sel)) / len(sel)


def simulate(s_idx, e_idx, sel, target_fn, ai):
    """
    일별 시뮬레이션: t일 종가에 목표 결정 → t+1일 종가 체결 → t+2일 수익부터 반영
    비중은 매일 가격 변동에 따라 자연스럽게 변함(drift), 리밸런싱 때 회전율 × 비용 차감
    """
    k = len(sel)
    w = np.zeros(k)
    pending = None
    rets = np.zeros(e_idx - s_idx)
    expo = np.zeros(e_idx - s_idx)
    for i, t in enumerate(range(s_idx, e_idx)):
        pr = 0.0
        if i > 0:
            r = np.nan_to_num(R[t, sel])
            pr = float(w @ r)
            if 1 + pr > 0:
                w = w * (1 + r) / (1 + pr)
        cost = 0.0
        if pending is not None:
            cost = np.abs(pending - w).sum() * COST
            w = pending.copy()
            pending = None
        rets[i] = pr - cost
        expo[i] = w.sum()
        if i % REBAL == 0:
            pending = np.nan_to_num(target_fn(sel, t, ai))
    return rets, expo


print("[6/10] 전략/시뮬레이터 준비 완료")


# %% [셀 8] 워크포워드 실행
ALL = list(SLEEVES.keys()) + ["★ 합성(변동성10%)", "동일비중(선별자산)"]
FN = dict(SLEEVES)
FN["★ 합성(변동성10%)"] = tgt_combined
FN["동일비중(선별자산)"] = tgt_equal
oos = {k: [] for k in ALL}
oos_dates = []
fold_rows = []
sel_history = []
t0 = time.time()
start = 1
fold = 0
while start + TRAIN_DAYS < N:
    a = start
    b = start + TRAIN_DAYS
    e = min(b + TEST_DAYS, N)
    if e - b < MIN_TEST_DAYS:
        break
    sel, n_elig, sel_avg, all_avg = select_assets(a, b)
    if sel is None:
        print(f"[건너뜀] {DATES[a].date()}~{DATES[b - 1].date()} 학습 가능 자산 {n_elig}개 (최소 {N_SELECT * 2}개 필요)")
        start += TEST_DAYS
        continue
    fold += 1
    print(f"\n[폴드 {fold}] 학습 {DATES[a].date()}~{DATES[b - 1].date()} | 검증 {DATES[b].date()}~{DATES[e - 1].date()}")
    print(f"    후보 {n_elig}개 → 선별 {len(sel)}개 | 평균상관 전체 {all_avg:.2f} → 선별 {sel_avg:.2f}")
    print("    선별: " + ", ".join(NAMES[CODES[i]] for i in sel))
    try:
        ai, _, _ = ai_overlay(sel, a, b)
        row = {"fold": fold, "검증": f"{DATES[b].date()}~{DATES[e - 1].date()}",
               "선별평균상관": round(sel_avg, 2)}
        for name in ALL:
            r, ex = simulate(b - 1, e, sel, FN[name], ai)
            oos[name].extend(list(r[1:]))
            row[name] = round(float(np.prod(1 + r[1:]) - 1), 3)
        oos_dates.extend(list(DATES[b:e]))
        fold_rows.append(row)
        sel_history.append({"fold": fold, "sel": sel})
        print("    검증 수익: " + " | ".join(f"{k} {row[k]:+.1%}" for k in ALL))
    except Exception as ex:
        print("    [폴드 오류]", ex)
    start += TEST_DAYS
print(f"\n[7/10] 워크포워드 완료 ({time.time() - t0:.0f}초)")


# %% [셀 9] 결과: 성과 비교 / 전략 간 상관 / 차트
pd.set_option("display.width", 220)
if len(oos_dates) == 0:
    raise SystemExit("검증 가능한 폴드가 없습니다 (데이터 기간 부족).")
print("\n===== 폴드별 검증 수익 =====")
print(pd.DataFrame(fold_rows).to_string(index=False))

oos_df = pd.DataFrame(oos, index=pd.DatetimeIndex(oos_dates))
bench_code = "069500" if "069500" in CODES else None
if bench_code:
    oos_df["KODEX 200 보유"] = np.nan_to_num(R_df[bench_code].reindex(oos_df.index).values)

print("\n===== 표본외 성과 (검증구간만 이어붙임) =====")
for col in oos_df.columns:
    m = metrics_from_returns(oos_df[col].values)
    ok = "✅" if (m["Sharpe"] >= 1.5 and m["MDD"] >= -0.20) else ("△ MDD만 통과" if m["MDD"] >= -0.20 else "❌")
    print(f"  {col:20s} CAGR {m['CAGR']:7.1%} Sharpe {m['Sharpe']:5.2f} MDD {m['MDD']:7.1%} 변동성 {m['Vol']:5.1%} | {ok}")

print("\n===== 전략 간 상관 (낮을수록 섞는 효과 큼) =====")
print(oos_df[list(SLEEVES.keys())].corr().round(2).to_string())

try:
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 2]})
    for col in oos_df.columns:
        lw = 2.5 if col.startswith("★") else 1.2
        ax[0].plot(oos_df.index, np.cumprod(1 + oos_df[col].values), label=col, lw=lw)
    ax[0].set_title("표본외 자산곡선")
    ax[0].legend(fontsize=8)
    last_sel = sel_history[-1]["sel"]
    cm =np.corrcoef(np.nan_to_num(R[-TRAIN_DAYS:, last_sel]).T)
    im = ax[1].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [NAMES[CODES[i]][:12] for i in last_sel]
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(labels)))
    ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].set_title("마지막 폴드 선별자산 상관 (최근 3년)")
    plt.colorbar(im, ax=ax[1])
    plt.tight_layout()
    plt.show()
except Exception as ex:
    print("[차트 오류]", ex)
print("[8/10] 결과 출력 완료")


# %% [셀 10] 오늘 기준 포트폴리오 + 알라딘식 리스크 리포트
try:
    a = N - TRAIN_DAYS
    b = N
    sel, n_elig, sel_avg, all_avg = select_assets(a, b)
    if sel is None:
        raise ValueError(f"최근 3년 데이터가 있는 자산이 {n_elig}개뿐")
    ai, err, (th_h, th_z) = ai_overlay(sel, a, b)
    t = N - 1
    w = tgt_combined(sel, t, ai)
    parts = {name: fn(sel, t, ai) / len(SLEEVES) for name, fn in SLEEVES.items()}

    print(f"\n===== {DATES[-1].date()} 기준 목표 포트폴리오 (★ 합성) =====")
    print(f"  후보 {n_elig}개 → 선별 {len(sel)}개 | 평균상관 {all_avg:.2f} → {sel_avg:.2f}")
    print(f"  AI 이상도 {err[-1]:.3f} (50%컷 {th_h:.3f}, 0%컷 {th_z:.3f}) → 추세전략 노출 {ai[-1]:.0%}")
    table = pd.DataFrame({"코드": [CODES[i] for i in sel],
                          "이름": [NAMES[CODES[i]] for i in sel],
                          "목표비중": w})
    for name in parts:
        table[name] = parts[name]
    table = table.sort_values("목표비중", ascending=False)
    fmt = table.copy()
    for c in ["목표비중"] + list(parts.keys()):
        fmt[c] = fmt[c].map(lambda v: f"{v:6.1%}")
    print(fmt.to_string(index=False))
    print(f"  투자 합계 {w.sum():.1%} | 현금 {max(0.0, 1 - w.sum()):.1%}")

    # --- 리스크 리포트 ---
    x = np.nan_to_num(R[-252:, sel])
    cov = np.cov(x.T) * 252
    pv = math.sqrt(max(float(w @ cov @ w), 0))
    print("\n===== 리스크 리포트 =====")
    print(f"  예상 연변동성 {pv:.1%}")
    if pv > 0:
        rc = w * (cov @ w) / (pv ** 2)
        rc_tab = pd.DataFrame({"이름": [NAMES[CODES[i]] for i in sel], "비중": w, "위험기여": rc})
        rc_tab = rc_tab.sort_values("위험기여", ascending=False)
        print("  [자산별 위험 기여도] (비중은 작아도 위험을 많이 차지하는 자산 확인)")
        for _, r_ in rc_tab.iterrows():
            print(f"    {r_['이름'][:22]:22s} 비중 {r_['비중']:6.1%} → 위험기여 {r_['위험기여']:6.1%}")
    port_hist = np.nan_to_num(R[:, sel]) @ w
    avail = ~np.isnan(R[:, sel]).any(axis=1)
    ph = port_hist[avail]
    var95 = np.quantile(port_hist[-252:], 0.05)
    print(f"  1일 VaR(95%, 최근 1년): {var95:.2%} → 100일 중 5일은 이보다 크게 손실 가능")
    if len(ph) > 20:
        roll20 = pd.Series(ph).rolling(20).apply(lambda s: np.prod(1 + s) - 1, raw=True)
        print(f"  최악 20일 손실(현재 비중을 과거에 적용, {int(avail.sum())}일): {roll20.min():.1%}")
    s2020 = (DATES >= pd.Timestamp("2020-02-19")) & (DATES <= pd.Timestamp("2020-03-19"))
    if s2020.sum() > 0:
        ok_assets = ~np.isnan(R[s2020][:, sel]).any(axis=0)
        loss = np.prod(1 + np.nan_to_num(R[s2020][:, sel]) @ w) - 1
        print(f"  시나리오 2020.2.19~3.19 코로나 폭락: {loss:.1%} "
              f"(당시 데이터 있는 자산 {int(ok_assets.sum())}/{len(sel)}개 기준, 없는 자산은 0% 가정)")
    print("\n  ※ 실거래 전 필수: 일일 최대 손실 한도, 긴급 전체 매도(panic_sell), KIS 모의투자 연동")
except Exception as ex:
    print("[오늘 포트폴리오 오류]", ex)

print("\n[10/10] 완료")
print("※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
