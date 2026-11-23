# ============================================================
# KODEX 코스닥150레버리지 v2 — 이평선 + ATR손절 + 변동성타깃 + AI비중조절 + 워크포워드
#   ① 신호: 1배 ETF(229200 KODEX 코스닥150) 이평선  → 노이즈 감소
#   ② 매매: 레버리지 ETF(233740)
#   ③ 손절: ATR(변동성) 배수 기반  → 레버리지 일상 변동에 덜 털림
#   ④ 비중: 변동성 타깃 (변동성 클수록 비중 축소)  → MDD 직접 억제
#   ⑤ AI  : 오토인코더 이상도에 따라 비중 100% / 50% / 0%
#   ⑥ 검증: 워크포워드 (3년 학습 → 1년 검증, 1년씩 이동)
# 코랩용 / API 키 불필요
# ============================================================

# %% [셀 1] 설치 (코랩에서 한 번만)
# !pip install -q finance-datareader yfinance

# %% [셀 2] 임포트 & 설정
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIGNAL_TICKER = "229200"   # 신호용: KODEX 코스닥150 (1배)
TRADE_TICKER = "233740"    # 매매용: KODEX 코스닥150레버리지 (2배)
START = "2016-01-01"

FEE = 0.00015              # 수수료(편도)
SLIPPAGE = 0.001           # 슬리피지(편도)
COST = FEE + SLIPPAGE

# 이평선 탐색 범위
SHORT_LIST = [3, 5, 7, 10, 12, 15, 20, 25, 30]
LONG_LIST = [10, 15, 20, 30, 40, 50, 60, 80, 100, 120]
MA_TYPES = ["SMA", "EMA"]

# ATR 손절 (매수가 - K × ATR 아래로 떨어지면 청산, 0 = 손절 없음)
ATR_N = 14
ATR_K_LIST = [0, 2, 3, 4]

# 변동성 타깃 비중
TARGET_VOL = 0.25          # 목표 연변동성 25% (레버리지 ETF 자체는 보통 50~60%)
VOL_N = 20                 # 최근 20일 변동성으로 계산
MAX_W = 1.0                # 최대 비중 100%
MIN_W = 0.05               # 5% 미만이면 그냥 0 (불필요한 소액 매매 방지)
REBAL_BAND = 0.2           # 목표비중과 20%p 이상 차이날 때만 비중 조절 (수수료 절약)

# 워크포워드
TRAIN_DAYS = 756           # 약 3년 학습
TEST_DAYS = 252            # 약 1년 검증
MIN_ENTRIES = 3            # 학습구간 최소 진입 횟수

# AI 필터
AI_WIN = 20                # 최근 20일 수익률 패턴
AI_EPOCHS = 200
AI_Q_HALF = 0.80           # 이상도 상위 20% → 비중 50%
AI_Q_ZERO = 0.95           # 이상도 상위 5%  → 비중 0%

RUN_ABLATION = True        # True: 구성요소별 효과 비교(시간 3배)

print("[1/8] 설정 완료")


# %% [셀 3] 데이터 수집
def load_data(ticker, start):
    """FinanceDataReader 우선, 실패 시 yfinance"""
    df = None
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(ticker, start)
        print(f"[데이터] {ticker} FinanceDataReader 수집 성공")
    except Exception as e:
        print(f"[데이터] {ticker} FDR 실패: {e} → yfinance 시도")
        try:
            import yfinance as yf
            df = yf.download(ticker + ".KS", start=start, auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"[데이터] {ticker} yfinance 수집 성공")
        except Exception as e2:
            print(f"[데이터] {ticker} yfinance 도 실패: {e2}")
            return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[df["Volume"] > 0].dropna()
    return df


sig = load_data(SIGNAL_TICKER, START)
trd = load_data(TRADE_TICKER, START)
if sig is None or trd is None:
    raise SystemExit("데이터 수집 실패 — 인터넷/종목코드 확인")

# 두 종목 날짜 맞추기
common = sig.index.intersection(trd.index)
sig = sig.loc[common]
trd = trd.loc[common]
N = len(trd)
print(f"[2/8] 공통 데이터 {N}일 | {common[0].date()} ~ {common[-1].date()}")
if N < TRAIN_DAYS + TEST_DAYS:
    raise SystemExit("데이터가 워크포워드에 부족합니다.")


# %% [셀 4] 지표 미리 계산 (이평선 / ATR / 변동성 비중)
def calc_ma(close, window, ma_type):
    if ma_type == "SMA":
        return close.rolling(window).mean()
    return close.ewm(span=window, adjust=False).mean()


# 이평선은 '신호용 1배 ETF' 종가로 계산 (리스트로 바꿔 두면 반복문이 빨라짐)
ma_cache = {}
for mt in MA_TYPES:
    for w in sorted(set(SHORT_LIST + LONG_LIST)):
        ma_cache[(mt, w)] = calc_ma(sig["Close"], w, mt).tolist()

# ATR은 '매매용 레버리지 ETF' 기준
prev_c = trd["Close"].shift(1)
tr_range = pd.concat([trd["High"] - trd["Low"],
                      (trd["High"] - prev_c).abs(),
                      (trd["Low"] - prev_c).abs()], axis=1).max(axis=1)
ATR = tr_range.rolling(ATR_N).mean().tolist()

# 변동성 타깃 비중 = 목표변동성 / 최근변동성 (최대 MAX_W)
realized_vol = trd["Close"].pct_change().rolling(VOL_N).std() * np.sqrt(252)
VOLW = np.clip(TARGET_VOL / realized_vol, 0, MAX_W).fillna(0).values

O = trd["Open"].tolist()
H = trd["High"].tolist()
L = trd["Low"].tolist()
C = trd["Close"].tolist()
print(f"[3/8] 지표 계산 완료 | 현재 연변동성 {realized_vol.iloc[-1]:.0%} → 변동성 비중 {VOLW[-1]:.0%}")


# %% [셀 5] 백테스트 엔진 (비중 매매 + ATR 손절)
def backtest(a, b, ma_s, ma_l, atr_k, scale):
    """
    인덱스 a ~ b-1 구간 백테스트
    - 종가에 신호 판단 → 다음날 시가 체결
    - 매수 구간(단기MA > 장기MA)이면 목표비중 = scale[t] (변동성×AI)
    - 매도 구간이면 전량 매도
    - ATR 손절: 매수가 - K×ATR 터치 시 청산, 데드크로스 전까지 재진입 금지
    반환: 자산곡선, 진입횟수, 일별 비중
    """
    n = b - a
    equity = np.ones(n)
    weights = np.zeros(n)
    stock = 0.0         # 주식 평가금액
    cash = 1.0          # 현금
    target = 0.0        # 내일 시가 목표비중
    has_order = False
    stop_px = 0.0
    stopped = False
    entries = 0

    for i in range(1, n):
        t = a + i

        # 1) 보유분 갭 반영 (전일종가 → 시가)
        if stock > 0:
            stock = stock * O[t] / C[t - 1]

        # 2) 시가 주문 체결
        if has_order:
            total = stock + cash
            want = target * total
            trade = want - stock
            if stock == 0 and want > 0:          # 신규 진입
                entries += 1
                atr_prev = ATR[t - 1]
                if atr_k > 0 and not math.isnan(atr_prev):
                    stop_px = O[t] - atr_k * atr_prev
                else:
                    stop_px = 0.0
            cash = cash - trade - abs(trade) * COST
            stock = want
            has_order = False

        # 3) 장중: ATR 손절 or 시가→종가
        if stock > 0:
            if stop_px > 0 and L[t] <= stop_px:
                exit_px = min(O[t], stop_px)
                stock = stock * exit_px / O[t]
                cash = cash + stock * (1 - COST)
                stock = 0.0
                stopped = True
            else:
                stock = stock * C[t] / O[t]

        equity[i] = stock + cash
        cur_w = stock / equity[i] if equity[i] > 0 else 0.0
        weights[i] = cur_w

        # 4) 종가 신호 → 내일 주문 예약
        ms = ma_s[t]
        ml = ma_l[t]
        if math.isnan(ms) or math.isnan(ml):
            continue
        if ms <= ml:
            stopped = False
            new_t = 0.0
        elif stopped:
            new_t = 0.0
        else:
            new_t = scale[t]
            if new_t < MIN_W:
                new_t = 0.0

        if new_t == 0.0 and cur_w > 0:
            target = 0.0
            has_order = True
        elif new_t > 0 and (cur_w == 0 or abs(new_t - cur_w) > REBAL_BAND):
            target = new_t
            has_order = True

    return equity, entries, weights


def metrics(equity, entries=0, weights=None):
    ret = pd.Series(equity).pct_change().fillna(0)
    years = len(equity) / 252
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    expo = float(np.mean(weights)) if weights is not None else 1.0
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Entries": entries, "Exposure": expo}


print("[4/8] 백테스트 엔진 준비 완료")


# %% [셀 6] AI 이상도 → 비중 배율 (폴드마다 학습구간 데이터로만 학습)
AI_OK = True
try:
    import torch
    import torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[5/8] PyTorch 사용 가능 | device = {DEVICE}")
except Exception as e:
    AI_OK = False
    print("[5/8] PyTorch 없음 → AI 필터 끔:", e)

# 신호용 ETF의 최근 20일 수익률 창 (t일 종가까지만 사용)
sig_ret = sig["Close"].pct_change().fillna(0).values
X_RAW = np.zeros((N, AI_WIN), dtype=np.float32)
for t in range(AI_WIN, N):
    X_RAW[t] = sig_ret[t - AI_WIN + 1:t + 1]

if AI_OK:
    class MarketAutoencoder(nn.Module):
        def __init__(self, input_dim=AI_WIN, latent_dim=3):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                         nn.Linear(64, 16), nn.ReLU(),
                                         nn.Linear(16, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(),
                                         nn.Linear(16, 64), nn.ReLU(),
                                         nn.Linear(64, input_dim))

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z


def ai_scale_for_fold(a, b):
    """
    학습구간 [a, b) 로만 오토인코더 학습 → 전체 날짜의 이상도 계산
    이상도 ≤ 80% 분위 → 1.0 / 80~95% → 0.5 / 95% 초과 → 0.0
    반환: (비중배율 배열, 이상도 배열, 임계값2개)
    """
    s0 = a + AI_WIN
    mu = X_RAW[s0:b].mean()
    sd = X_RAW[s0:b].std() + 1e-8
    X = (X_RAW - mu) / sd

    torch.manual_seed(42)
    model = MarketAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_tr = torch.tensor(X[s0:b]).to(DEVICE)
    for ep in range(AI_EPOCHS):
        recon, _ = model(x_tr)
        loss = loss_fn(recon, x_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x_all = torch.tensor(X).to(DEVICE)
        recon, _ = model(x_all)
        err = ((recon - x_all) ** 2).mean(dim=1).cpu().numpy()

    th_half = np.quantile(err[s0:b], AI_Q_HALF)
    th_zero = np.quantile(err[s0:b], AI_Q_ZERO)
    scale = np.ones(N)
    scale[err > th_half] = 0.5
    scale[err > th_zero] = 0.0
    return scale, err, (th_half, th_zero)


# %% [셀 7] 워크포워드 최적화
def add_robust(res):
    """이웃 조합 평균 샤프 (과최적화 방지)"""
    res = res.copy()
    res["robust"] = np.nan
    for i, r in res.iterrows():
        si = SHORT_LIST.index(r["short"])
        li = LONG_LIST.index(r["long"])
        near = res[(res["type"] == r["type"]) & (res["atr_k"] == r["atr_k"]) &
                   (res["short"].isin(SHORT_LIST[max(0, si - 1):si + 2])) &
                   (res["long"].isin(LONG_LIST[max(0, li - 1):li + 2]))]
        res.loc[i, "robust"] = near["Sharpe"].mean()
    return res


def optimize(a, b, scale):
    """[a, b) 구간 그리드 탐색 → robust 1등 조합 반환"""
    rows = []
    for mt in MA_TYPES:
        for s in SHORT_LIST:
            for lw in LONG_LIST:
                if s >= lw:
                    continue
                for k in ATR_K_LIST:
                    try:
                        eq, ent, w = backtest(a, b, ma_cache[(mt, s)], ma_cache[(mt, lw)], k, scale)
                        m = metrics(eq, ent, w)
                        m.update({"type": mt, "short": s, "long": lw, "atr_k": k})
                        rows.append(m)
                    except Exception as e:
                        print("[오류]", mt, s, lw, k, e)
    res = add_robust(pd.DataFrame(rows))
    cand = res[res["Entries"] >= MIN_ENTRIES]
    if len(cand) == 0:
        cand = res
    return cand.sort_values("robust", ascending=False).iloc[0]


def walk_forward(use_vol=True, use_ai=True, label=""):
    print(f"\n----- 워크포워드 시작: {label} -----")
    volw = VOLW if use_vol else np.full(N, MAX_W)
    oos_eq = [1.0]
    oos_w = [0.0]
    folds = []
    start = 0
    t0 = time.time()
    while start + TRAIN_DAYS < N:
        a = start
        b = start + TRAIN_DAYS
        e = min(b + TEST_DAYS, N)
        try:
            if use_ai and AI_OK:
                ai, _, _ = ai_scale_for_fold(a, b)
            else:
                ai = np.ones(N)
            scale = (volw * ai).tolist()

            best = optimize(a, b, scale)
            mt, s, lw, k = best["type"], int(best["short"]), int(best["long"]), int(best["atr_k"])

            # 검증구간 실행 (b-1 일을 기준점으로 시작)
            eq, ent, w = backtest(b - 1, e, ma_cache[(mt, s)], ma_cache[(mt, lw)], k, scale)
            mt_test = metrics(eq, ent, w)
            last = oos_eq[-1]
            for v in eq[1:]:
                oos_eq.append(last * v)
            for v in w[1:]:
                oos_w.append(v)

            folds.append({"학습": f"{trd.index[a].date()}~{trd.index[b - 1].date()}",
                          "검증": f"{trd.index[b].date()}~{trd.index[e - 1].date()}",
                          "조합": f"{mt} {s}/{lw} ATRx{k}",
                          "학습Sharpe": round(best["Sharpe"], 2),
                          "검증Sharpe": round(mt_test["Sharpe"], 2),
                          "검증수익": round(eq[-1] - 1, 3),
                          "검증MDD": round(mt_test["MDD"], 3),
                          "평균비중": round(mt_test["Exposure"], 2)})
            print(f"  폴드 {len(folds)} | {folds[-1]['검증']} | {folds[-1]['조합']} | "
                  f"검증 수익 {eq[-1] - 1:+.1%} MDD {mt_test['MDD']:.1%}")
        except Exception as ex:
            print("  [폴드 오류]", ex)
        start += TEST_DAYS

    first_test = TRAIN_DAYS - 1
    idx = trd.index[first_test:first_test + len(oos_eq)]
    eq_s = pd.Series(oos_eq, index=idx)
    w_s = pd.Series(oos_w, index=idx)
    print(f"  완료 ({time.time() - t0:.0f}초)")
    return eq_s, w_s, pd.DataFrame(folds)


configs = [("전체(변동성+AI)", True, True)]
if RUN_ABLATION:
    configs.append(("AI 없음", True, False))
    configs.append(("변동성타깃 없음(v1과 유사)", False, False))

results = {}
for label, uv, ua in configs:
    results[label] = walk_forward(uv, ua, label)

print("\n[6/8] 워크포워드 완료")


# %% [셀 8] 결과 요약 & 차트
pd.set_option("display.width", 220)
main_label = configs[0][0]
eq_main, w_main, folds_main = results[main_label]
print(f"\n===== 폴드별 결과 ({main_label}) =====")
print(folds_main.to_string(index=False))

print("\n===== 표본외(검증구간만 이어붙인) 성과 비교 =====")
bh = trd["Close"].loc[eq_main.index]
bh = (bh / bh.iloc[0]).values
mb = metrics(bh)
print(f"  {'단순보유(233740)':22s} CAGR {mb['CAGR']:7.1%} Sharpe {mb['Sharpe']:5.2f} MDD {mb['MDD']:7.1%}")
for label in results:
    eq_s, w_s, _ = results[label]
    m = metrics(eq_s.values, 0, w_s.values)
    print(f"  {label:22s} CAGR {m['CAGR']:7.1%} Sharpe {m['Sharpe']:5.2f} MDD {m['MDD']:7.1%} 평균비중 {m['Exposure']:.0%}")

m_main = metrics(eq_main.values, 0, w_main.values)
print("\n  목표 체크(표본외) → Sharpe≥1.5:", "✅" if m_main["Sharpe"] >= 1.5 else "❌",
      "| MDD≤20%:", "✅" if m_main["MDD"] >= -0.20 else "❌")

fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for label in results:
    ax[0].plot(results[label][0].index, results[label][0].values, label=label)
ax[0].plot(eq_main.index, bh, label="Buy & Hold 233740", alpha=0.5)
ax[0].set_yscale("log")
ax[0].set_title("Walk-forward out-of-sample equity (log)")
ax[0].legend()
ax[1].fill_between(w_main.index, w_main.values, step="post", alpha=0.6)
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Position weight (full model)")
plt.tight_layout()
plt.show()
print("[7/8] 차트 완료")


# %% [셀 9] 오늘 기준 신호 (최근 3년으로 다시 최적화)
def next_cross_price(close, s, lw, ma_type):
    """내일 종가가 얼마면 단기MA = 장기MA 가 되는지 역산"""
    if ma_type == "SMA":
        A = close.iloc[-(s - 1):].sum() if s > 1 else 0.0
        B = close.iloc[-(lw - 1):].sum()
        return (s * B - lw * A) / (lw - s)
    a_s, a_l = 2 / (s + 1), 2 / (lw + 1)
    es = calc_ma(close, s, "EMA").iloc[-1]
    el = calc_ma(close, lw, "EMA").iloc[-1]
    return ((1 - a_l) * el - (1 - a_s) * es) / (a_s - a_l)


try:
    a = N - TRAIN_DAYS
    b = N
    if AI_OK:
        ai, err, (th_h, th_z) = ai_scale_for_fold(a, b)
    else:
        ai, err, th_h, th_z = np.ones(N), np.zeros(N), 0, 0
    scale = (VOLW * ai).tolist()
    best = optimize(a, b, scale)
    mt, s, lw, k = best["type"], int(best["short"]), int(best["long"]), int(best["atr_k"])
    eq, ent, w = backtest(a, b, ma_cache[(mt, s)], ma_cache[(mt, lw)], k, scale)

    ms, ml = ma_cache[(mt, s)][-1], ma_cache[(mt, lw)][-1]
    bull = ms > ml
    px = next_cross_price(sig["Close"], s, lw, mt)
    target_today = scale[-1] if bull else 0.0
    if target_today < MIN_W:
        target_today = 0.0

    print(f"\n===== {trd.index[-1].date()} 기준 신호 =====")
    print(f"  최근 3년 최적 조합: {mt} {s}/{lw} (신호: {SIGNAL_TICKER}) | ATR 손절 x{k}")
    print(f"  {SIGNAL_TICKER} 종가 {sig['Close'].iloc[-1]:,.0f} | 단기MA {ms:,.0f} | 장기MA {ml:,.0f} → "
          f"{'매수 구간' if bull else '관망 구간'}")
    print(f"  변동성 비중 {VOLW[-1]:.0%} × AI 배율 {ai[-1]:.1f} "
          f"(이상도 {err[-1]:.2f}, 50%컷 {th_h:.2f}, 0%컷 {th_z:.2f})")
    print(f"  → 모델상 현재 비중 {w[-1]:.0%} / 목표 비중 {target_today:.0%}")
    if bull:
        print(f"  매도 타점: {SIGNAL_TICKER} 종가가 {px:,.0f}원 아래 마감 → 다음날 시가 전량 매도")
    else:
        print(f"  매수 타점: {SIGNAL_TICKER} 종가가 {px:,.0f}원 위 마감 → 다음날 시가 매수(목표비중만큼)")
    if px <= 0 or abs(px / sig["Close"].iloc[-1] - 1) > 0.3:
        print("  (참고) 타점이 현재가와 30% 이상 차이 → 내일 하루 만에 크로스될 가능성은 사실상 없음")
    if k > 0:
        print(f"  손절가(지금 진입 시): {C[-1] - k * ATR[-1]:,.0f}원 ({TRADE_TICKER} 기준, ATR {ATR[-1]:,.0f})")
except Exception as e:
    print("[오늘 신호 오류]", e)

print("\n[8/8] 완료")
print("※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
