# ============================================================
# KODEX 코스닥150레버리지(233740) 이동평균 조합 최적화 + AI 필터
# - 구글 코랩용: 셀 단위(# %%)로 나눠 붙여넣거나 통째로 실행
# - API 키 불필요 (공개 시세만 사용) → .env 없이 실행 가능
# ============================================================

# %% [셀 1] 라이브러리 설치 (코랩에서 한 번만 실행)
# !pip install -q finance-datareader yfinance

# %% [셀 2] 임포트 & 설정값
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKER = "233740"          # KODEX 코스닥150레버리지
START = "2016-01-01"       # 상장(2015-12) 직후부터
TRAIN_RATIO = 0.7          # 앞 70% = 최적화(학습), 뒤 30% = 검증(테스트)

FEE = 0.00015              # 증권사 수수료(편도) 0.015%
SLIPPAGE = 0.001           # 슬리피지(편도) 0.1%
COST = FEE + SLIPPAGE      # ETF는 매도 거래세 없음

# 탐색할 이동평균 조합
SHORT_LIST = [3, 5, 7, 10, 12, 15, 20, 25, 30]
LONG_LIST = [10, 15, 20, 30, 40, 50, 60, 80, 100, 120]
MA_TYPES = ["SMA", "EMA"]
STOP_LIST = [0.0, 0.05, 0.08, 0.10]   # 손절폭 (0 = 손절 없음)
MIN_TRADES = 5                        # 학습구간 최소 거래 횟수

print("[1/7] 설정 완료 | 1회 거래비용(편도):", COST)


# %% [셀 3] 데이터 수집
def load_data(ticker, start):
    """FinanceDataReader 우선, 실패 시 yfinance 사용"""
    df = None
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(ticker, start)
        print("[데이터] FinanceDataReader 로 수집 성공")
    except Exception as e:
        print("[데이터] FDR 실패:", e, "→ yfinance 시도")
        try:
            import yfinance as yf
            df = yf.download(ticker + ".KS", start=start, auto_adjust=False, progress=False)
            # 최신 yfinance는 컬럼이 2단(MultiIndex)이라 1단으로 펴기
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print("[데이터] yfinance 로 수집 성공")
        except Exception as e2:
            print("[데이터] yfinance 도 실패:", e2)
            return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[df["Volume"] > 0]          # 거래정지일 제거
    df = df.dropna()
    return df


df = load_data(TICKER, START)
if df is None or len(df) < 300:
    raise SystemExit("데이터가 부족합니다. 인터넷 연결/종목코드를 확인하세요.")
print(f"[2/7] 데이터 {len(df)}일 | {df.index[0].date()} ~ {df.index[-1].date()}")


# %% [셀 4] 이동평균 계산 & 백테스트 함수
def calc_ma(close, window, ma_type):
    """SMA 또는 EMA 계산 (EMA는 adjust=False → 다음 타점 가격 역산 가능)"""
    if ma_type == "SMA":
        return close.rolling(window).mean()
    return close.ewm(span=window, adjust=False).mean()


def backtest(o, h, l, c, ma_s, ma_l, stop_loss=0.0, entry_ok=None, cost=COST):
    """
    규칙
    - 종가 기준 단기MA > 장기MA 이면 '매수 상태', 아니면 '매도 상태'
    - 신호는 그날 종가에 확인 → 다음날 시가에 주문 체결 (미래 데이터 사용 방지)
    - 손절: 장중 저가가 (매수가 × (1-손절폭)) 이하면 그 가격에 청산
            (갭하락이면 시가에 청산). 손절 후엔 데드크로스가 나올 때까지 재진입 금지
    - entry_ok: AI 필터 (False 인 날은 신규 매수 금지)
    반환: 자산곡선(np.array), 거래별 수익률 리스트
    """
    n = len(c)
    equity = np.ones(n)
    hold = 0            # 보유 여부
    entry = 0.0         # 매수가
    pending = 0         # 다음날 시가 주문: 1=매수, -1=매도
    stopped = False     # 손절 후 재진입 잠금
    trades = []

    for t in range(1, n):
        f = 1.0  # 오늘 하루 자산 변화 배율

        # 1) 보유 중이면 전일 종가 → 오늘 시가 갭 반영
        if hold == 1:
            f = f * (o[t] / c[t - 1])

        # 2) 시가에 대기 주문 체결
        if pending == 1 and hold == 0:
            hold = 1
            entry = o[t]
            f = f * (1 - cost)
        elif pending == -1 and hold == 1:
            hold = 0
            f = f * (1 - cost)
            trades.append(o[t] / entry * (1 - cost) ** 2 - 1)
        pending = 0

        # 3) 장중: 손절 체크 또는 시가 → 종가 반영
        if hold == 1:
            stop_px = entry * (1 - stop_loss)
            if stop_loss > 0 and l[t] <= stop_px:
                exit_px = min(o[t], stop_px)
                f = f * (exit_px / o[t]) * (1 - cost)
                trades.append(exit_px / entry * (1 - cost) ** 2 - 1)
                hold = 0
                stopped = True
            else:
                f = f * (c[t] / o[t])

        equity[t] = equity[t - 1] * f

        # 4) 종가에 신호 판단 → 내일 시가 주문 예약
        if np.isnan(ma_s[t]) or np.isnan(ma_l[t]):
            continue
        bull = ma_s[t] > ma_l[t]
        if not bull:
            stopped = False          # 데드크로스 → 손절 잠금 해제
        ai_ok = True if entry_ok is None else bool(entry_ok[t])

        if hold == 0 and bull and (not stopped) and ai_ok:
            pending = 1
        elif hold == 1 and not bull:
            pending = -1

    return equity, trades


def metrics(equity, trades):
    """CAGR, 샤프, MDD, 거래수, 승률 계산"""
    ret = pd.Series(equity).pct_change().fillna(0)
    years = len(equity) / 252
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    win = np.mean([1 if x > 0 else 0 for x in trades]) if len(trades) > 0 else 0
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": mdd,
            "Trades": len(trades), "WinRate": win, "Final": equity[-1]}


print("[3/7] 백테스트 함수 준비 완료")


# %% [셀 5] 학습구간 그리드 탐색
split = int(len(df) * TRAIN_RATIO)
train = df.iloc[:split]
test = df.iloc[split:]
print(f"[4/7] 학습: {train.index[0].date()}~{train.index[-1].date()} ({len(train)}일) | "
      f"검증: {test.index[0].date()}~{test.index[-1].date()} ({len(test)}일)")


def run_period(data, ma_type, s, lw, stop, entry_ok=None):
    """특정 구간에서 한 조합 백테스트 (MA는 전체 데이터로 계산 후 잘라서 사용 → 검증구간 초반 공백 방지)"""
    ma_s_all = calc_ma(df["Close"], s, ma_type)
    ma_l_all = calc_ma(df["Close"], lw, ma_type)
    idx = df.index.get_indexer(data.index)
    eq, tr = backtest(data["Open"].values, data["High"].values, data["Low"].values,
                      data["Close"].values, ma_s_all.values[idx], ma_l_all.values[idx],
                      stop, None if entry_ok is None else entry_ok[idx])
    return eq, tr


# MA는 미리 한 번씩만 계산해 두기 (속도)
ma_cache = {}
for mt in MA_TYPES:
    for w in sorted(set(SHORT_LIST + LONG_LIST)):
        ma_cache[(mt, w)] = calc_ma(df["Close"], w, mt).values

o_tr, h_tr, l_tr, c_tr = (train[k].values for k in ["Open", "High", "Low", "Close"])
rows = []
t0 = time.time()
for mt in MA_TYPES:
    for s in SHORT_LIST:
        for lw in LONG_LIST:
            if s >= lw:
                continue
            for stop in STOP_LIST:
                try:
                    eq, tr = backtest(o_tr, h_tr, l_tr, c_tr,
                                      ma_cache[(mt, s)][:split], ma_cache[(mt, lw)][:split], stop)
                    m = metrics(eq, tr)
                    m.update({"type": mt, "short": s, "long": lw, "stop": stop})
                    rows.append(m)
                except Exception as e:
                    print("[오류]", mt, s, lw, stop, e)
res = pd.DataFrame(rows)
print(f"[5/7] {len(res)}개 조합 탐색 완료 ({time.time() - t0:.1f}초)")

# --- 과최적화 방지: 이웃 조합 평균 샤프(robust score) ---
# 한 점만 튀는 조합보다 '주변 조합도 다 좋은' 영역을 고른다
res["robust"] = np.nan
for i, r in res.iterrows():
    si = SHORT_LIST.index(r["short"])
    li = LONG_LIST.index(r["long"])
    near = res[(res["type"] == r["type"]) & (res["stop"] == r["stop"]) &
               (res["short"].isin(SHORT_LIST[max(0, si - 1):si + 2])) &
               (res["long"].isin(LONG_LIST[max(0, li - 1):li + 2]))]
    res.loc[i, "robust"] = near["Sharpe"].mean()

cand = res[res["Trades"] >= MIN_TRADES].sort_values("robust", ascending=False)
top = cand.head(10).copy()

# 상위 10개를 검증구간에서 재평가
test_sharpe, test_mdd, test_cagr = [], [], []
for _, r in top.iterrows():
    eq, tr = run_period(test, r["type"], int(r["short"]), int(r["long"]), r["stop"])
    m = metrics(eq, tr)
    test_sharpe.append(m["Sharpe"])
    test_mdd.append(m["MDD"])
    test_cagr.append(m["CAGR"])
top["test_Sharpe"] = test_sharpe
top["test_MDD"] = test_mdd
top["test_CAGR"] = test_cagr

pd.set_option("display.width", 200)
print("\n===== 학습구간 상위 10개 조합 (robust 기준) + 검증 성과 =====")
print(top[["type", "short", "long", "stop", "Sharpe", "robust", "CAGR", "MDD",
           "Trades", "WinRate", "test_Sharpe", "test_CAGR", "test_MDD"]].round(3).to_string(index=False))


# %% [셀 6] 최종 조합 선택 & 성과 비교 (vs 단순 보유)
best = top.iloc[0]
B_TYPE, B_S, B_L, B_STOP = best["type"], int(best["short"]), int(best["long"]), float(best["stop"])
print(f"\n[6/7] 선택 조합: {B_TYPE} {B_S}/{B_L} | 손절 {B_STOP * 100:.0f}%")

for name, data in [("학습", train), ("검증", test), ("전체", df)]:
    eq, tr = run_period(data, B_TYPE, B_S, B_L, B_STOP)
    m = metrics(eq, tr)
    bh = data["Close"].values / data["Close"].values[0]
    mb = metrics(bh, [])
    print(f"  [{name}] 전략 CAGR {m['CAGR']:6.1%} Sharpe {m['Sharpe']:5.2f} MDD {m['MDD']:6.1%} "
          f"거래 {m['Trades']:3d} 승률 {m['WinRate']:.0%} | 보유 CAGR {mb['CAGR']:6.1%} MDD {mb['MDD']:6.1%}")

eq_full, _ = run_period(df, B_TYPE, B_S, B_L, B_STOP)
m_test = metrics(*run_period(test, B_TYPE, B_S, B_L, B_STOP))
print("  목표 체크(검증구간) → Sharpe≥1.5:", "✅" if m_test["Sharpe"] >= 1.5 else "❌",
      "| MDD≤20%:", "✅" if m_test["MDD"] >= -0.20 else "❌")

# 차트: 자산곡선 + 샤프 히트맵
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].plot(df.index, eq_full, label=f"{B_TYPE} {B_S}/{B_L} SL{B_STOP:.0%}")
ax[0].plot(df.index, df["Close"] / df["Close"].iloc[0], label="Buy & Hold", alpha=0.6)
ax[0].axvline(test.index[0], color="red", ls="--", label="Train | Test")
ax[0].set_yscale("log")
ax[0].legend()
ax[0].set_title("Equity (log)")

hm = res[(res["type"] == B_TYPE) & (res["stop"] == B_STOP)].pivot(index="short", columns="long", values="Sharpe")
im = ax[1].imshow(hm.values, cmap="RdYlGn", aspect="auto")
ax[1].set_xticks(range(len(hm.columns)))
ax[1].set_xticklabels(hm.columns)
ax[1].set_yticks(range(len(hm.index)))
ax[1].set_yticklabels(hm.index)
ax[1].set_xlabel("Long MA")
ax[1].set_ylabel("Short MA")
ax[1].set_title(f"Train Sharpe heatmap ({B_TYPE}, SL {B_STOP:.0%})")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
plt.show()


# %% [셀 7] AI 독창 지표 필터 (Autoencoder 이상도) — MA 신호와 결합
# 최근 20일 수익률 패턴을 오토인코더로 학습 → 복원오차가 크면 '평소와 다른 시장'
# → 이상도가 높은 날엔 신규 매수를 막는다 (학습구간 데이터로만 학습 = 미래 누수 없음)
try:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[7/7] AI 필터 학습 시작 | device = {device}")

    WIN = 20
    r = df["Close"].pct_change().fillna(0).values
    mu, sd = r[:split].mean(), r[:split].std()
    rz = (r - mu) / sd                          # 학습구간 기준 표준화

    X = np.zeros((len(df), WIN), dtype=np.float32)
    for t in range(WIN, len(df)):
        X[t] = rz[t - WIN + 1:t + 1]            # t일 종가까지만 사용

    class MarketAutoencoder(nn.Module):
        def __init__(self, input_dim=WIN, latent_dim=3):
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

    torch.manual_seed(42)
    model = MarketAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_train = torch.tensor(X[WIN:split]).to(device)
    for ep in range(300):
        recon, _ = model(x_train)
        loss = loss_fn(recon, x_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0:
            print(f"  [Autoencoder] epoch {ep} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_all = torch.tensor(X).to(device)
        recon, _ = model(x_all)
        anomaly = ((recon - x_all) ** 2).mean(dim=1).cpu().numpy()   # ← AI 이상도 지표

    # 이상도 상위 몇 %를 '매수 금지'로 할지 학습구간에서 고르기
    print("\n  상위 5개 MA 조합 × AI 필터 임계값 비교 (학습 → 검증)")
    best_cfg = None
    for q in [1.0, 0.95, 0.90, 0.80]:
        th = np.quantile(anomaly[WIN:split], q)
        entry_ok = anomaly <= th if q < 1.0 else np.ones(len(df), dtype=bool)
        for _, rr in top.head(5).iterrows():
            a = metrics(*run_period(train, rr["type"], int(rr["short"]), int(rr["long"]), rr["stop"], entry_ok))
            b = metrics(*run_period(test, rr["type"], int(rr["short"]), int(rr["long"]), rr["stop"], entry_ok))
            label = "필터없음" if q == 1.0 else f"상위{(1 - q) * 100:.0f}%차단"
            print(f"  {label:8s} {rr['type']} {int(rr['short']):2d}/{int(rr['long']):3d} SL{rr['stop']:.0%} | "
                  f"학습 Sharpe {a['Sharpe']:5.2f} MDD {a['MDD']:6.1%} | 검증 Sharpe {b['Sharpe']:5.2f} MDD {b['MDD']:6.1%}")
            if best_cfg is None or a["Sharpe"] > best_cfg[0]:
                best_cfg = (a["Sharpe"], q, rr, th)
    print(f"\n  → 학습 Sharpe 기준 최적: {best_cfg[2]['type']} {int(best_cfg[2]['short'])}/"
          f"{int(best_cfg[2]['long'])} SL{best_cfg[2]['stop']:.0%} + AI필터 q={best_cfg[1]}")
    print(f"  → 오늘 AI 이상도: {anomaly[-1]:.3f} (임계값 {best_cfg[3]:.3f}) → "
          f"{'매수 허용' if anomaly[-1] <= best_cfg[3] else '매수 금지(이상 구간)'}")
except Exception as e:
    print("[AI 필터] 건너뜀:", e)


# %% [셀 8] 현재 상태 & 다음 매수/매도 타점 가격
def next_cross_price(close, s, lw, ma_type):
    """내일 종가가 얼마면 단기MA = 장기MA 가 되는지 역산 (= 크로스 타점 가격)"""
    if ma_type == "SMA":
        A = close.iloc[-(s - 1):].sum() if s > 1 else 0.0
        B = close.iloc[-(lw - 1):].sum()
        return (s * B - lw * A) / (lw - s)
    a_s, a_l = 2 / (s + 1), 2 / (lw + 1)
    es = calc_ma(close, s, "EMA").iloc[-1]
    el = calc_ma(close, lw, "EMA").iloc[-1]
    return ((1 - a_l) * el - (1 - a_s) * es) / (a_s - a_l)


try:
    close = df["Close"]
    ms = calc_ma(close, B_S, B_TYPE).iloc[-1]
    ml = calc_ma(close, B_L, B_TYPE).iloc[-1]
    px = next_cross_price(close, B_S, B_L, B_TYPE)
    print(f"\n===== {df.index[-1].date()} 기준 신호 ({B_TYPE} {B_S}/{B_L}) =====")
    print(f"  종가 {close.iloc[-1]:,.0f} | 단기MA {ms:,.0f} | 장기MA {ml:,.0f}")
    if ms > ml:
        print(f"  상태: 매수(보유) 구간 → 다음 종가가 {px:,.0f}원 아래면 데드크로스 = 다음날 시가 매도")
        if B_STOP > 0:
            print(f"  + 손절: 매수가 × {1 - B_STOP:.2f} 이하 터치 시 즉시 청산")
    else:
        print(f"  상태: 관망(현금) 구간 → 다음 종가가 {px:,.0f}원 위면 골든크로스 = 다음날 시가 매수")
    if px <= 0 or abs(px / close.iloc[-1] - 1) > 0.3:
        print("  (참고) 타점 가격이 현재가와 30% 이상 차이 → 내일 하루 만에 크로스될 가능성은 사실상 없음")
except Exception as e:
    print("[타점 계산 오류]", e)

print("\n※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
