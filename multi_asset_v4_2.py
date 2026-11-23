# ============================================================
# v4.2 — 다자산 포트폴리오: 수익↑ 위험↓ 시도 (v4.1 + 달러 복원 + 다중추세 + 현금활용 배율)
#
# v4.1 대비 변경점
#   ① 카테고리 3(파생) 다시 사용 → 달러선물·국채선물 등 '비레버리지' 선물 ETF 복원
#      (레버리지/인버스는 이름 키워드로 계속 제외)
#   ② ②+ 다중추세+AI: 60·120·250일 추세 3개 투표로 비중 결정 (한 기간 의존 위험 감소)
#   ③ ×1.5 배율: 레버리지 상품 없이 놀고 있는 현금을 투자 (총 100% 이하)
#   ④ 사전 고정 판정: 다중추세 채택 여부 / 배율 확대 여부 자동 판정
#
# 그대로 유지: 자산군 ETF 유니버스, 저상관 군집 선별, 현금 몫 예금 이자,
#             예금 대비 Sharpe, ★ 합성(축소만), AI 기여도 비교, 리스크 리포트
#
# 검증: 워크포워드 (3년 학습 → 1년 검증). 자산 선별·AI 학습 모두 학습구간만 사용
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
UNIVERSE_SIZE = 300
N_SELECT = 12
MIN_VOL = 0.03
CACHE_FILE = "etf_close_cache.pkl"      # v4와 같은 캐시 재사용 (새 종목만 추가 다운로드)

# 카테고리 제외 (FinanceDataReader ETF 목록의 Category 번호)
#   v4.1 실행 결과로 확인: 1 국내지수, 2 국내 업종/테마, 3 파생(레버리지·인버스·선물), 4 해외주식, 5 원자재, 6 채권, 7 기타
#   카테고리 3은 다시 사용 → 달러선물·국채선물 같은 비레버리지 선물 ETF 복원 (레버리지/인버스는 아래 키워드로 제외)
EXCLUDE_CATEGORIES = [2]

EXCLUDE_WORDS = ["레버리지", "인버스", "2X", "3X", "곱버스", "Leverage", "Inverse", "선물인버스",
                 "CD금리", "KOFR", "SOFR", "머니마켓", "단기자금", "초단기", "단기채", "단기통안", "금리액티브"]

CASH_CODE = "153130"                    # KODEX 단기채권 = 예금 대용

FALLBACK = {"069500": "KODEX 200", "229200": "KODEX 코스닥150",
            "143850": "TIGER 미국S&P500선물(H)", "133690": "TIGER 미국나스닥100",
            "148070": "KOSEF 국고채10년", "114260": "KODEX 국고채3년",
            "132030": "KODEX 골드선물(H)", "261240": "KODEX 미국달러선물",
            "130680": "TIGER 원유선물Enhanced(H)", "192090": "TIGER 차이나CSI300",
            "195930": "TIGER 유로스탁스50(합성 H)", "329200": "TIGER 리츠부동산인프라",
            "245710": "ACE 베트남VN30(합성)", "305080": "TIGER 미국채10년선물"}

FEE = 0.00015
SLIPPAGE = 0.001
COST = FEE + SLIPPAGE

TRAIN_DAYS = 756
TEST_DAYS = 252
MIN_TEST_DAYS = 60
REBAL = 20

# 전략 파라미터 (사전 고정)
VOL_LOOKBACK = 60
TREND_LOOKBACK = 120
TREND_LOOKBACKS = [60, 120, 250]   # 다중추세: 3개월·6개월·1년 (사전 고정)
SCALE_K = 1.5                      # 현금 활용 배율 (사전 고정)
PORT_TARGET_VOL = 0.10       # 이 변동성을 넘을 때만 축소
AI_WIN = 20
AI_EPOCHS = 300
AI_Q_HALF = 0.80
AI_Q_ZERO = 0.95
SEED = 42

# 예금 대비 판정 기준 (사전 고정)
STRONG_EXCESS = 0.04         # 연 4%p 이상 초과 + 예금대비 Sharpe 1.0 이상 → 의미 있는 우위
STRONG_SHARPE = 1.0
PARTIAL_EXCESS = 0.02        # 연 2%p 이상 초과 + 예금대비 Sharpe 0.5 이상 → 일부만 소액
PARTIAL_SHARPE = 0.5

print("[1/10] 설정 완료")


# %% [셀 3] 유니버스 구성 (자산군 ETF)
def build_universe():
    try:
        import FinanceDataReader as fdr
        lst = fdr.StockListing("ETF/KR")
        print(f"[유니버스] 국내 상장 ETF {len(lst)}개 조회")
        code_col = "Symbol" if "Symbol" in lst.columns else ("Code" if "Code" in lst.columns else lst.columns[0])
        name_col = "Name" if "Name" in lst.columns else lst.columns[1]
        lst = lst.copy()

        if "Category" in lst.columns:
            print("[유니버스] 카테고리별 개수와 예시 ← 제외 대상이 맞는지 확인하세요")
            excl = [str(c) for c in EXCLUDE_CATEGORIES]
            for cat, g in lst.groupby("Category"):
                flag = "제외" if str(cat) in excl else "사용"
                ex = ", ".join(g[name_col].astype(str).head(4))
                print(f"   카테고리 {cat}: {len(g):4d}개 [{flag}] 예) {ex}")
            lst = lst[~lst["Category"].astype(str).isin(excl)]
            print(f"[유니버스] 카테고리 제외 후 {len(lst)}개")
        else:
            print("[유니버스] ⚠️ Category 컬럼 없음 → 업종/테마 제외 못 함")

        keep = pd.Series(True, index=lst.index)
        for w in EXCLUDE_WORDS:
            keep = keep & ~lst[name_col].astype(str).str.contains(w, case=False, regex=False)
        lst = lst[keep]
        print(f"[유니버스] 레버리지/인버스/현금성 제외 후 {len(lst)}개")
        for c in ["MarCap", "Amount", "Volume"]:
            if c in lst.columns:
                lst = lst.sort_values(c, ascending=False)
                break
        lst = lst.head(UNIVERSE_SIZE)
        codes = lst[code_col].astype(str).str.zfill(6)
        return dict(zip(codes, lst[name_col].astype(str)))
    except Exception as e:
        print("[유니버스] 조회 실패 → 기본 목록 사용:", e)
        return dict(FALLBACK)


UNIVERSE = build_universe()
print(f"[2/10] 유니버스 {len(UNIVERSE)}개")


# %% [셀 4] 가격 수집 (캐시) + 예금 대용 수익률
def download_prices(codes):
    import FinanceDataReader as fdr
    close = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        try:
            close = pd.read_pickle(CACHE_FILE)
            print(f"[가격] 캐시에서 {close.shape[1]}개 종목 불러옴")
        except Exception as e:
            print("[가격] 캐시 읽기 실패:", e)
            close = pd.DataFrame()
    need = [c for c in codes if c not in close.columns]
    print(f"[가격] 새로 받을 종목 {len(need)}개 (개당 약 1초)")
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
    return close


raw = download_prices(list(UNIVERSE) + [CASH_CODE])
raw = raw[raw.index >= pd.Timestamp(START)]
cash_px = raw[CASH_CODE] if CASH_CODE in raw.columns else None

close = raw[[c for c in UNIVERSE if c in raw.columns]].copy()
close = close.replace(0, np.nan)
close = close[close.notna().sum(axis=1) >= 3]
close = close.ffill(limit=5)

R_df = close / close.shift(1) - 1
bad = [c for c in R_df.columns if (R_df[c].abs() > 0.3).any()]
if bad:
    print(f"  ⚠️ 하루 ±30% 넘는 이상치 종목 {len(bad)}개 제외: {[UNIVERSE.get(c, c) for c in bad[:8]]}")
    R_df = R_df.drop(columns=bad)
    close = close.drop(columns=bad)

CODES = list(R_df.columns)
NAMES = {c: UNIVERSE.get(c, c) for c in CODES}
DATES = R_df.index
N = len(DATES)
R = R_df.values
CL = close.values

# 예금 대용 일수익률 (현금 몫 이자 + 예금 재예치 기준선)
if cash_px is not None and cash_px.notna().sum() > 100:
    cp = cash_px.reindex(DATES).ffill()
    cr = (cp / cp.shift(1) - 1).values
    CASH_R = np.where(np.isnan(cr), 0.025 / 252, cr)     # 상장 전 등 빈 날은 연 2.5% 가정
    print(f"[예금 대용] KODEX 단기채권 연환산 {(1 + CASH_R).prod() ** (252 / N) - 1:.2%}")
else:
    CASH_R = np.full(N, 0.025 / 252)
    print("[예금 대용] 단기채권 데이터 없음 → 연 2.5% 고정 가정")
print(f"[3/10] 가격 정리 완료 | 자산 {len(CODES)}개 × {N}일 | {DATES[0].date()} ~ {DATES[-1].date()}")


# %% [셀 5] 공통 함수 (지표 / 저상관 군집 선별)
def ann_vol(cols_idx, t, n):
    x = R[t - n + 1:t + 1, cols_idx]
    return np.nanstd(x, axis=0) * np.sqrt(252)


def metrics_vs_cash(r, cash):
    """일반 성과 + 예금 대비 초과수익/Sharpe"""
    r = np.nan_to_num(np.asarray(r, dtype=float))
    eq = np.cumprod(1 + r)
    years = len(r) / 252
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    ex = r - cash
    ex_ann = ex.mean() * 252
    ex_sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    return {"CAGR": cagr, "MDD": mdd, "Vol": r.std() * np.sqrt(252),
            "Excess": ex_ann, "ExSharpe": ex_sharpe}


def verdict(m):
    if m["Excess"] >= STRONG_EXCESS and m["ExSharpe"] >= STRONG_SHARPE:
        return "의미 있는 우위 (모의투자 후 단계적 확대)"
    if m["Excess"] >= PARTIAL_EXCESS and m["ExSharpe"] >= PARTIAL_SHARPE:
        return "약한 우위 (대부분 예금 + 일부만 소액)"
    return "예금 재예치가 합리적"


def cluster_assets(corr, n_clusters):
    """평균연결 계층 군집화 (거리 = sqrt(0.5×(1-상관)))"""
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
    """학습구간 데이터만으로 군집화 → 군집마다 중심 자산 1개 (수익률로 고르지 않음)"""
    Rt = R[a:b]
    full = np.where(~np.isnan(Rt).any(axis=0))[0]
    vol = np.nanstd(Rt[:, full], axis=0) * np.sqrt(252)
    elig = full[vol >= MIN_VOL]
    if len(elig) < N_SELECT * 2:
        return None, len(elig), np.nan, np.nan
    corr = np.nan_to_num(np.corrcoef(Rt[:, elig].T))
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
    all_avg = corr[~np.eye(len(elig), dtype=bool)].mean()
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
    """선별 자산의 시장 상태 20일 패턴 → 학습구간으로만 AE 학습 → 이상도 높으면 노출 축소"""
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


# %% [셀 7] 전략 + 합성 + 시뮬레이터
def tgt_risk_parity(sel, t, ai):
    v = ann_vol(sel, t, VOL_LOOKBACK)
    inv = np.where(v > 0, 1 / np.where(v > 0, v, 1), 0)
    return inv / inv.sum() if inv.sum() > 0 else np.zeros(len(sel))


def trend_weights(sel, t):
    base = tgt_risk_parity(sel, t, None)
    past = CL[t - TREND_LOOKBACK, sel]
    now = CL[t, sel]
    mom = np.where(past > 0, now / np.where(past > 0, past, 1) - 1, 0)
    return np.where(mom > 0, base, 0)        # 추세 없는 자산 몫은 현금(예금 이자)


def tgt_trend_ai(sel, t, ai):
    return trend_weights(sel, t) * ai[t]


def tgt_trend_only(sel, t, ai):
    return trend_weights(sel, t)


def tgt_combined(sel, t, ai):
    """① 리스크패리티 + ② 추세+AI 반반 → 변동성 10% 넘으면 축소만 (늘리지 않음)"""
    w = (tgt_risk_parity(sel, t, ai) + tgt_trend_ai(sel, t, ai)) / 2
    if w.sum() <= 0:
        return w
    x = np.nan_to_num(R[t - VOL_LOOKBACK + 1:t + 1][:, sel])
    cov = np.cov(x.T) * 252
    pv = math.sqrt(max(float(w @ cov @ w), 0))
    if pv > PORT_TARGET_VOL:
        w = w * (PORT_TARGET_VOL / pv)
    return w


def tgt_equal(sel, t, ai):
    return np.ones(len(sel)) / len(sel)


def multi_trend_weights(sel, t):
    """
    추세 기간 3개 투표 → 3개 모두 상승이면 리스크패리티 비중 100%, 2개 67%, 1개 33%, 0개 현금
    """
    base = tgt_risk_parity(sel, t, None)
    votes = np.zeros(len(sel))
    for lb in TREND_LOOKBACKS:
        past = CL[t - lb, sel]
        now = CL[t, sel]
        mom = np.where(past > 0, now / np.where(past > 0, past, 1) - 1, 0)
        votes = votes + (mom > 0).astype(float)
    return base * votes / len(TREND_LOOKBACKS)


def tgt_multi_trend_ai(sel, t, ai):
    return multi_trend_weights(sel, t) * ai[t]


def make_scaled(fn, k):
    """비중 × k배 (레버리지 상품 없이 현금 활용, 합계 100% 이하)"""
    def scaled(sel, t, ai):
        w = fn(sel, t, ai) * k
        s = w.sum()
        if s > 1.0:
            w = w / s
        return w
    return scaled


STRATS = {"① 리스크패리티": tgt_risk_parity,
          "② 추세+AI": tgt_trend_ai,
          "②' 추세(AI없음)": tgt_trend_only,
          "②+ 다중추세+AI": tgt_multi_trend_ai,
          f"② 추세+AI ×{SCALE_K}": make_scaled(tgt_trend_ai, SCALE_K),
          f"②+ 다중추세+AI ×{SCALE_K}": make_scaled(tgt_multi_trend_ai, SCALE_K),
          "★ 합성(①+②)": tgt_combined,
          "동일비중(선별자산)": tgt_equal}


def simulate(s_idx, e_idx, sel, target_fn, ai):
    """
    t일 종가 결정 → t+1 종가 체결 → t+2 수익부터 반영
    투자하지 않은 현금 몫은 예금 대용 수익(CASH_R)을 받음
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
            cash_w = max(0.0, 1 - w.sum())
            pr = float(w @ r) + cash_w * CASH_R[t]
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
oos = {k: [] for k in STRATS}
oos_expo = {k: [] for k in STRATS}
oos_idx = []
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
        print(f"[건너뜀] {DATES[a].date()}~{DATES[b - 1].date()} 학습 가능 자산 {n_elig}개 (최소 {N_SELECT * 2}개)")
        start += TEST_DAYS
        continue
    fold += 1
    print(f"\n[폴드 {fold}] 학습 {DATES[a].date()}~{DATES[b - 1].date()} | 검증 {DATES[b].date()}~{DATES[e - 1].date()}")
    print(f"    후보 {n_elig}개 → 선별 {len(sel)}개 | 평균상관 {all_avg:.2f} → {sel_avg:.2f}")
    print("    선별: " + ", ".join(NAMES[CODES[i]] for i in sel))
    try:
        ai, _, _ = ai_overlay(sel, a, b)
        row = {"fold": fold, "검증": f"{DATES[b].date()}~{DATES[e - 1].date()}"}
        for name, fn in STRATS.items():
            r, ex = simulate(b - 1, e, sel, fn, ai)
            oos[name].extend(list(r[1:]))
            oos_expo[name].extend(list(ex[1:]))
            row[name] = round(float(np.prod(1 + r[1:]) - 1), 3)
        row["예금"] = round(float(np.prod(1 + CASH_R[b:e]) - 1), 3)
        oos_idx.extend(list(range(b, e)))
        fold_rows.append(row)
        sel_history.append(sel)
        print("    검증 수익: " + " | ".join(f"{k} {row[k]:+.1%}" for k in list(STRATS) + ["예금"]))
    except Exception as ex:
        print("    [폴드 오류]", ex)
    start += TEST_DAYS
print(f"\n[7/10] 워크포워드 완료 ({time.time() - t0:.0f}초)")


# %% [셀 9] 결과: 예금 대비 판정 / 차트
pd.set_option("display.width", 220)
if len(oos_idx) == 0:
    raise SystemExit("검증 가능한 폴드가 없습니다.")
print("\n===== 폴드별 검증 수익 =====")
print(pd.DataFrame(fold_rows).to_string(index=False))

oos_dates = DATES[oos_idx]
cash_oos = CASH_R[oos_idx]
res = pd.DataFrame(oos, index=oos_dates)
res["예금 재예치(단기채)"] = cash_oos
if "069500" in CODES:
    res["KODEX 200 보유"] = np.nan_to_num(R_df["069500"].values[oos_idx])

print("\n===== 예금 대비 비교 (표본외, 검증구간만 이어붙임) =====")
print("  예금대비 Sharpe = (전략수익 - 예금수익) 평균 ÷ 그 변동성  ← 진짜 비교 기준")
summary = []
for col in res.columns:
    m = metrics_vs_cash(res[col].values, cash_oos)
    expo = np.mean(oos_expo[col]) if col in oos_expo else np.nan
    v = "-" if col.startswith("예금") else verdict(m)
    summary.append({"전략": col, "CAGR": f"{m['CAGR']:.1%}", "MDD": f"{m['MDD']:.1%}",
                    "변동성": f"{m['Vol']:.1%}", "예금대비 초과/년": f"{m['Excess']:+.1%}",
                    "예금대비 Sharpe": round(m["ExSharpe"], 2),
                    "평균투자비중": "-" if np.isnan(expo) else f"{expo:.0%}", "판정": v})
print(pd.DataFrame(summary).to_string(index=False))

m_ai = metrics_vs_cash(res["② 추세+AI"].values, cash_oos)
m_no = metrics_vs_cash(res["②' 추세(AI없음)"].values, cash_oos)
print(f"\n  [AI 기여도] 예금대비 Sharpe: 추세+AI {m_ai['ExSharpe']:.2f} vs 추세만 {m_no['ExSharpe']:.2f} | "
      f"MDD {m_ai['MDD']:.1%} vs {m_no['MDD']:.1%} → "
      f"{'AI 필터가 도움 됨' if m_ai['ExSharpe'] > m_no['ExSharpe'] + 0.05 else 'AI 필터의 뚜렷한 기여 없음'}")

# ----- v4.2 사전 고정 판정 -----
m_multi = metrics_vs_cash(res["②+ 다중추세+AI"].values, cash_oos)
print("\n===== v4.2 사전 고정 판정 =====")
adopt_multi = (m_multi["ExSharpe"] > m_ai["ExSharpe"]) and (m_multi["MDD"] >= -0.12)
print(f"  [다중추세] 예금대비 Sharpe {m_multi['ExSharpe']:.2f} vs 기존 {m_ai['ExSharpe']:.2f} | MDD {m_multi['MDD']:.1%} "
      f"→ {'✅ 다중추세 채택' if adopt_multi else '❌ 기존 120일 추세 유지'}")
core = "②+ 다중추세+AI" if adopt_multi else "② 추세+AI"
m_core = m_multi if adopt_multi else m_ai
scaled_name = ("②+ 다중추세+AI" if adopt_multi else "② 추세+AI") + f" ×{SCALE_K}"
m_sc = metrics_vs_cash(res[scaled_name].values, cash_oos)
adopt_scale = (m_sc["ExSharpe"] >= m_core["ExSharpe"] - 0.1) and (m_sc["MDD"] >= -0.15)
print(f"  [×{SCALE_K} 배율] 예금대비 Sharpe {m_sc['ExSharpe']:.2f} vs 1배 {m_core['ExSharpe']:.2f} | MDD {m_sc['MDD']:.1%} "
      f"| 초과 {m_sc['Excess']:+.1%} vs {m_core['Excess']:+.1%} → {'✅ 배율 확대 유효' if adopt_scale else '❌ 1배 유지'}")
FINAL = scaled_name if adopt_scale else core
print(f"  ▶ 최종 운용 후보: '{FINAL}'")

best = max([c for c in STRATS], key=lambda c: metrics_vs_cash(res[c].values, cash_oos)["ExSharpe"])
mb = metrics_vs_cash(res[best].values, cash_oos)
print(f"\n  ▶ 최종 판정: 가장 나은 전략 '{best}' (예금대비 초과 {mb['Excess']:+.1%}/년, Sharpe {mb['ExSharpe']:.2f}) → {verdict(mb)}")
print("    ※ 백테스트 성과는 실전에서 보통 30~50% 줄어듦. 또 '가장 나은 것 고르기' 자체가 사후 선택임")

try:
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 2]})
    for col in res.columns:
        lw = 2.5 if col.startswith("★") or col.startswith("예금") else 1.2
        ls = "--" if col.startswith("예금") else "-"
        ax[0].plot(res.index, np.cumprod(1 + res[col].values), label=col, lw=lw, ls=ls)
    ax[0].set_title("표본외 자산곡선 (예금 재예치 = 점선)")
    ax[0].legend(fontsize=8)
    last_sel = sel_history[-1]
    cm = np.corrcoef(np.nan_to_num(R[-TRAIN_DAYS:, last_sel]).T)
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


# %% [셀 10] 오늘 기준 포트폴리오 + 리스크 리포트
try:
    a = N - TRAIN_DAYS
    b = N
    sel, n_elig, sel_avg, all_avg = select_assets(a, b)
    if sel is None:
        raise ValueError(f"최근 3년 데이터가 있는 자산이 {n_elig}개뿐")
    ai, err, (th_h, th_z) = ai_overlay(sel, a, b)
    t = N - 1
    w = STRATS[FINAL](sel, t, ai)

    print(f"\n===== {DATES[-1].date()} 기준 목표 포트폴리오 ('{FINAL}') =====")
    print(f"  후보 {n_elig}개 → 선별 {len(sel)}개 | 평균상관 {all_avg:.2f} → {sel_avg:.2f}")
    print(f"  AI 이상도 {err[-1]:.3f} (50%컷 {th_h:.3f}, 0%컷 {th_z:.3f}) → 추세 노출 {ai[-1]:.0%}")
    table = pd.DataFrame({"코드": [CODES[i] for i in sel],
                          "이름": [NAMES[CODES[i]] for i in sel],
                          "목표비중": w}).sort_values("목표비중", ascending=False)
    table["목표비중"] = table["목표비중"].map(lambda v: f"{v:6.1%}")
    print(table.to_string(index=False))
    print(f"  투자 합계 {w.sum():.1%} | 나머지 {max(0.0, 1 - w.sum()):.1%} 는 예금·단기채·CMA")

    x = np.nan_to_num(R[-252:, sel])
    cov = np.cov(x.T) * 252
    pv = math.sqrt(max(float(w @ cov @ w), 0))
    print("\n===== 리스크 리포트 =====")
    print(f"  예상 연변동성 {pv:.1%}")
    if pv > 0:
        rc = w * (cov @ w) / (pv ** 2)
        rc_tab = pd.DataFrame({"이름": [NAMES[CODES[i]] for i in sel], "비중": w, "위험기여": rc})
        for _, r_ in rc_tab.sort_values("위험기여", ascending=False).iterrows():
            print(f"    {r_['이름'][:22]:22s} 비중 {r_['비중']:6.1%} → 위험기여 {r_['위험기여']:6.1%}")
    port_hist = np.nan_to_num(R[:, sel]) @ w
    print(f"  1일 VaR(95%, 최근 1년): {np.quantile(port_hist[-252:], 0.05):.2%}")
    avail = ~np.isnan(R[:, sel]).any(axis=1)
    ph = port_hist[avail]
    if len(ph) > 20:
        roll20 = pd.Series(ph).rolling(20).apply(lambda s: np.prod(1 + s) - 1, raw=True)
        print(f"  최악 20일 손실(현재 비중, {int(avail.sum())}일 기준): {roll20.min():.1%}")
    s2020 = (DATES >= pd.Timestamp("2020-02-19")) & (DATES <= pd.Timestamp("2020-03-19"))
    if s2020.sum() > 0:
        ok_assets = ~np.isnan(R[s2020][:, sel]).any(axis=0)
        loss = np.prod(1 + np.nan_to_num(R[s2020][:, sel]) @ w) - 1
        print(f"  시나리오 2020.2.19~3.19 코로나 폭락: {loss:.1%} (데이터 있는 자산 {int(ok_assets.sum())}/{len(sel)}개)")
    print("\n  ※ 실거래 전 필수: 일일 최대 손실 한도, 긴급 전체 매도(panic_sell), KIS 모의투자 연동")
except Exception as ex:
    print("[오늘 포트폴리오 오류]", ex)

print("\n[10/10] 완료")
print("※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
