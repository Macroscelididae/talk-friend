# ============================================================
# v3.1 — 사전 고정(pre-registered) 재검증: 오토인코더 지표의 20일 예측력
#
# v3 결과에서 나온 '단 하나의 가설'만 검증한다. 조건은 실행 전에 모두 고정.
#   가설: 오토인코더(AE) 지표는 20일 뒤 수익률을 예측한다
#
#   [고정 조건]
#   - 지표: AE 잠재축 중 학습 IC가 가장 큰 축(ae_best) + AE 이상도(ae_anomaly)
#           ※ 잠재축 번호(z0, z1, z2)는 학습할 때마다 바뀌므로 '번호'가 아니라
#             '학습구간에서 가장 예측력 높은 축'을 고른다 (학습 데이터만 사용)
#   - 합성: 두 지표 표준화 평균 → 학습 시작값(seed) 5개 평균 (앙상블)
#   - 예측 기간: 20일 / 리밸런싱: 20일마다
#   - 매매: 229200(1배) 롱전용, 합성점수 > 학습구간 중앙값 → 매수, 아니면 현금
#   - 리스크: 변동성 타깃 비중 + ATR×3 손절
#   - 합격 기준(실행 전 고정):
#       앙상블 검증 IC 평균 ≥ 0.05, 방향 적중률 ≥ 80%, seed 5개 중 4개 이상 IC 평균 > 0.03
#
#   [비교 기준선] 단순 지표 2개와 비교 → AE가 '단순 계산 이상'을 잡는지 확인
#   - ret20 : 최근 20일 수익률
#   - range20 : 최근 20일 평균 장중 변동폭
#
# 코랩용 / API 키 불필요
# ============================================================

# %% [셀 1] 설치 (코랩에서 한 번만)
# !pip install -q finance-datareader yfinance koreanize-matplotlib

# %% [셀 2] 임포트 & 설정
import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # 그래프 한글 (없어도 실행됨)
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

TICKER = "229200"           # KODEX 코스닥150 (1배) — 신호 + 매매
START = "2016-01-01"

FEE = 0.00015
SLIPPAGE = 0.001
COST = FEE + SLIPPAGE

TRAIN_DAYS = 756            # 약 3년 학습
TEST_DAYS = 252             # 약 1년 검증
MIN_TEST_DAYS = 60          # 검증일이 이보다 적은 마지막 폴드는 생략

WIN = 20                    # 최근 20일 패턴 입력
HZ = 20                     # 예측 기간 20일 (고정)
REBAL = 20                  # 20일마다 리밸런싱 (고정)
AE_EPOCHS = 300
SEEDS = [42, 7, 123, 2024, 99]

# 합격 기준 (고정)
PASS_IC = 0.05
PASS_HIT = 0.80
PASS_SEED_IC = 0.03
PASS_SEED_COUNT = 4

# 리스크 관리
TARGET_VOL = 0.25
VOL_N = 20
MAX_W = 1.0
MIN_W = 0.05
ATR_N = 14
STOP_ATR_K = 3

print("[1/9] 설정 완료 (모든 조건 사전 고정)")


# %% [셀 3] 데이터 수집 & 점검
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


df = load_data(TICKER, START)
if df is None:
    raise SystemExit("데이터 수집 실패")
N = len(df)
DATES = df.index
print(f"[2/9] 데이터 {N}일 | {DATES[0].date()} ~ {DATES[-1].date()}")
if N < TRAIN_DAYS + TEST_DAYS:
    raise SystemExit("데이터가 워크포워드에 부족합니다.")

chg = df["Close"].pct_change()
big = chg[chg.abs() > 0.25]
if len(big) > 0:
    print(f"  ⚠️ 하루 ±25% 넘는 날 {len(big)}일 → 데이터 확인 필요")
    for day, c in big.items():
        print(f"     {day.date()} {c:+.1%}")
else:
    print("  ✅ 하루 ±25% 넘는 이상치 없음")


# %% [셀 4] 특징 / 정답 / 기준선 / 리스크 지표
O = df["Open"].values
Hh = df["High"].values
L = df["Low"].values
C = df["Close"].values

feat = pd.DataFrame(index=DATES)
feat["ret1"] = np.log(df["Close"] / df["Close"].shift(1))
feat["gap"] = np.log(df["Open"] / df["Close"].shift(1))
feat["range"] = (df["High"] - df["Low"]) / df["Close"]
feat["clv"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-8)
feat["vol"] = np.log(df["Volume"] / df["Volume"].rolling(20).mean())
feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
F = feat.values.astype(np.float32)

# 정답: t일 신호 → t+1 시가 매수 → t+1+20 시가 매도 수익률
FWD = np.full(N, np.nan)
for t in range(N - 1 - HZ):
    FWD[t] = O[t + 1 + HZ] / O[t + 1] - 1

# 비교 기준선 (AI 아님, 단순 계산)
BASELINES = {
    "ret20": np.log(df["Close"] / df["Close"].shift(20)).values,
    "range20": feat["range"].rolling(20).mean().values,
}

# 리스크 지표
prev_c = df["Close"].shift(1)
tr_rng = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
ATR = tr_rng.rolling(ATR_N).mean().values
rv = df["Close"].pct_change().rolling(VOL_N).std() * np.sqrt(252)
VOLW = np.clip(TARGET_VOL / rv, 0, MAX_W).fillna(0).values

WARMUP = WIN + 20
print("[3/9] 특징/정답/기준선 준비 완료")


def calc_ic(x, y):
    """순위상관(Spearman IC)"""
    try:
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30:
            return np.nan
        xs = pd.Series(x[m])
        if xs.std() == 0:
            return np.nan
        return xs.rank().corr(pd.Series(y[m]).rank())
    except Exception:
        return np.nan


def sign_test_p(hits, n):
    """방향이 우연히 hits번 이상 맞을 확률 (동전 던지기 기준, 단측)"""
    if n == 0:
        return np.nan
    total = 0
    for k in range(hits, n + 1):
        total += math.comb(n, k)
    return total / (2 ** n)


def zscore_by_train(x, tr_idx):
    mu = np.nanmean(x[tr_idx])
    sd = np.nanstd(x[tr_idx]) + 1e-8
    return np.clip((x - mu) / sd, -5, 5)


# %% [셀 5] 오토인코더
try:
    import torch
    import torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/9] PyTorch 사용 가능 | device = {DEVICE}")
except Exception as e:
    raise SystemExit(f"v3.1 은 PyTorch 가 꼭 필요합니다: {e}")


class MarketAutoencoder(nn.Module):
    """20일×5특징(100차원) → 3차원 압축"""
    def __init__(self, input_dim=WIN * 5, latent_dim=3):
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


def make_windows(Fz):
    idx = np.arange(N)[:, None] + np.arange(-WIN + 1, 1)[None, :]
    idx = np.clip(idx, 0, N - 1)
    return Fz[idx]


def ae_one_seed(a, b, tr_idx, seed):
    """
    학습구간으로 AE 학습 (seed 고정) → 지표 계산
    반환: ae_best, ae_anomaly, 합성점수 (모두 학습 IC 방향으로 맞춘 표준화 값) + 설명 정보
    """
    mu = F[a + WARMUP:b].mean(axis=0)
    sd = F[a + WARMUP:b].std(axis=0) + 1e-8
    Fz = ((F - mu) / sd).clip(-5, 5).astype(np.float32)
    Xw = make_windows(Fz).reshape(N, WIN * 5)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MarketAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_tr = torch.tensor(Xw[a + WARMUP:b]).to(DEVICE)
    for ep in range(AE_EPOCHS):
        recon, _ = model(x_tr)
        loss = loss_fn(recon, x_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x_all = torch.tensor(Xw).to(DEVICE)
        recon, z = model(x_all)
        err = ((recon - x_all) ** 2).mean(dim=1).cpu().numpy()
        z = z.cpu().numpy()

    # 잠재축 3개 중 학습 IC 절댓값이 가장 큰 축 선택 (학습 데이터만 사용)
    ic_z = [calc_ic(z[tr_idx, i], FWD[tr_idx]) for i in range(3)]
    ic_abs = [abs(v) if not np.isnan(v) else 0 for v in ic_z]
    best_i = int(np.argmax(ic_abs))
    sgn_best = np.sign(ic_z[best_i]) if ic_abs[best_i] > 0 else 1.0
    ae_best = sgn_best * zscore_by_train(z[:, best_i], tr_idx)

    ic_an = calc_ic(err[tr_idx], FWD[tr_idx])
    sgn_an = np.sign(ic_an) if not np.isnan(ic_an) and ic_an != 0 else 1.0
    ae_anom = sgn_an * zscore_by_train(err, tr_idx)

    comp = (ae_best + ae_anom) / 2
    info = {"best_axis": best_i, "best_train_IC": ic_z[best_i],
            "anom_train_IC": ic_an, "loss": loss.item()}
    return {"ae_best": ae_best, "ae_anomaly": ae_anom, "comp": comp}, info


# %% [셀 6] 매매 엔진 (20일 리밸런싱, 롱전용)
def backtest(a, b, comp, threshold, always_long=False):
    """
    a~b-1 구간. 리밸런싱일(a에서 20일 간격) 종가에 판단 → 다음날 시가 체결
    - 합성점수 > 기준선 → 변동성 타깃 비중으로 매수 / 아니면 현금
    - ATR 손절 시 다음 리밸런싱까지 현금
    - always_long=True: 신호 무시하고 항상 매수 (타이밍 효과 비교용)
    """
    n = b - a
    equity = np.ones(n)
    weights = np.zeros(n)
    stock = 0.0
    cash = 1.0
    order = None
    stop_px = 0.0
    for i in range(1, n):
        t = a + i
        if stock > 0:
            stock = stock * O[t] / C[t - 1]
        if order is not None:
            total = stock + cash
            want = order * total
            trade = want - stock
            if stock == 0 and want > 0:
                if STOP_ATR_K > 0 and not math.isnan(ATR[t - 1]):
                    stop_px = O[t] - STOP_ATR_K * ATR[t - 1]
                else:
                    stop_px = 0.0
            cash = cash - trade - abs(trade) * COST
            stock = want
            order = None
        if stock > 0:
            if stop_px > 0 and L[t] <= stop_px:
                exit_px = min(O[t], stop_px)
                stock = stock * exit_px / O[t]
                cash = cash + stock * (1 - COST)
                stock = 0.0
            else:
                stock = stock * C[t] / O[t]
        equity[i] = stock + cash
        weights[i] = stock / equity[i] if equity[i] > 0 else 0.0

        # 리밸런싱일: 0, 20, 40 ... 번째 날 종가
        if i % REBAL == 0 or i == 1:
            go = always_long or (not np.isnan(comp[t]) and comp[t] > threshold)
            w = VOLW[t] if go else 0.0
            if w < MIN_W:
                w = 0.0
            order = w
    return equity, weights


def metrics(equity, weights=None):
    ret = pd.Series(equity).pct_change().fillna(0)
    years = len(equity) / 252
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    expo = float(np.mean(weights)) if weights is not None else 1.0
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Exposure": expo}


print("[5/9] 엔진 준비 완료")


# %% [셀 7] 워크포워드 (폴드 × seed 5개)
rows = []          # 지표별 IC 기록
dir_rows = []      # 방향 해석 기록
oos = {"AE 타이밍": {"eq": [1.0], "w": [0.0]},
       "항상 보유(변동성 비중)": {"eq": [1.0], "w": [0.0]}}
t0 = time.time()
start = 0
fold = 0
while start + TRAIN_DAYS < N:
    a = start
    b = start + TRAIN_DAYS
    e = min(b + TEST_DAYS, N)
    if e - b < MIN_TEST_DAYS:
        break
    fold += 1
    tr_idx = np.arange(a + WARMUP, b - HZ - 1)   # 정답이 검증구간에 걸치는 날 제외
    te_idx = np.arange(b, e)
    print(f"\n[폴드 {fold}] 학습 {DATES[a].date()}~{DATES[b - 1].date()} | 검증 {DATES[b].date()}~{DATES[e - 1].date()}")
    try:
        comps = []
        for seed in SEEDS:
            inds, info = ae_one_seed(a, b, tr_idx, seed)
            comps.append(inds["comp"])
            for name in inds:
                rows.append({"fold": fold, "seed": seed, "indicator": name,
                             "test_IC": calc_ic(inds[name][te_idx], FWD[te_idx])})
            dir_rows.append({"fold": fold, "seed": seed, "선택축": f"z{info['best_axis']}",
                             "축_학습IC": round(info["best_train_IC"], 3),
                             "이상도_학습IC": round(info["anom_train_IC"], 3)})
            print(f"    seed {seed:4d} | loss {info['loss']:.3f} | 선택축 z{info['best_axis']} "
                  f"(학습IC {info['best_train_IC']:+.3f}) | 이상도 학습IC {info['anom_train_IC']:+.3f}")

        # 앙상블 = seed 5개 합성점수 평균
        ens = np.mean(np.vstack(comps), axis=0)
        ens_ic = calc_ic(ens[te_idx], FWD[te_idx])
        rows.append({"fold": fold, "seed": "앙상블", "indicator": "comp", "test_IC": ens_ic})

        # 기준선 (학습 IC로 방향만 맞춤)
        for bname, bx in BASELINES.items():
            ic_tr = calc_ic(bx[tr_idx], FWD[tr_idx])
            sgn = np.sign(ic_tr) if not np.isnan(ic_tr) and ic_tr != 0 else 1.0
            rows.append({"fold": fold, "seed": "기준선", "indicator": bname,
                         "test_IC": sgn * calc_ic(bx[te_idx], FWD[te_idx])})

        # 매매
        threshold = np.nanmedian(ens[tr_idx])
        eq, w = backtest(b - 1, e, ens, threshold)
        eq2, w2 = backtest(b - 1, e, ens, threshold, always_long=True)
        for name, (q, ww) in {"AE 타이밍": (eq, w), "항상 보유(변동성 비중)": (eq2, w2)}.items():
            last = oos[name]["eq"][-1]
            oos[name]["eq"].extend([last * v for v in q[1:]])
            oos[name]["w"].extend(list(ww[1:]))
        print(f"    → 앙상블 검증 IC {ens_ic:+.3f} | 검증 수익: AE 타이밍 {eq[-1] - 1:+.1%} / 항상보유 {eq2[-1] - 1:+.1%}")
    except Exception as ex:
        print("    [폴드 오류]", ex)
    start += TEST_DAYS

print(f"\n[6/9] 워크포워드 완료 ({time.time() - t0:.0f}초)")


# %% [셀 8] 결과 ① 예측력 판정
ic_df = pd.DataFrame(rows)
pd.set_option("display.width", 220)


def summarize(sub):
    col = sub["test_IC"].dropna()
    n = len(col)
    hits = int((col > 0).sum())
    return {"폴드수": n, "IC평균": col.mean(), "적중": f"{hits}/{n}",
            "적중률": hits / n if n else np.nan,
            "t값": col.mean() / (col.std() / np.sqrt(n)) if n > 1 and col.std() > 0 else np.nan,
            "우연확률(p)": sign_test_p(hits, n)}


print("\n===== 폴드별 검증 IC (20일) =====")
pivot = ic_df[ic_df["seed"].isin(["앙상블", "기준선"])].pivot_table(
    index="fold", columns="indicator", values="test_IC")
print(pivot.round(3).to_string())

print("\n===== seed별 요약 (AE 합성점수) =====")
seed_rows = []
for seed in SEEDS:
    s = summarize(ic_df[(ic_df["seed"] == seed) & (ic_df["indicator"] == "comp")])
    s["seed"] = seed
    seed_rows.append(s)
seed_df = pd.DataFrame(seed_rows)[["seed", "폴드수", "IC평균", "적중", "t값", "우연확률(p)"]]
print(seed_df.round(3).to_string(index=False))

print("\n===== 지표별 요약 (seed 5개 전체) =====")
sum_rows = []
for name in ["ae_best", "ae_anomaly", "comp"]:
    sub = ic_df[(ic_df["indicator"] == name) & (ic_df["seed"].isin(SEEDS))]
    g = sub.groupby("fold")["test_IC"].mean().to_frame("test_IC")   # 폴드별 seed 평균
    s = summarize(g)
    s["지표"] = name + " (seed평균)"
    sum_rows.append(s)
ens_sum = summarize(ic_df[(ic_df["seed"] == "앙상블")])
ens_sum["지표"] = "앙상블 합성점수"
sum_rows.append(ens_sum)
for bname in BASELINES:
    s = summarize(ic_df[(ic_df["seed"] == "기준선") & (ic_df["indicator"] == bname)])
    s["지표"] = f"기준선 {bname}"
    sum_rows.append(s)
print(pd.DataFrame(sum_rows)[["지표", "폴드수", "IC평균", "적중", "t값", "우연확률(p)"]].round(3).to_string(index=False))

# --- 사전 고정 기준으로 판정 ---
good_seeds = int((seed_df["IC평균"] > PASS_SEED_IC).sum())
c1 = ens_sum["IC평균"] >= PASS_IC
c2 = ens_sum["적중률"] >= PASS_HIT
c3 = good_seeds >= PASS_SEED_COUNT
print("\n===== 사전 고정 기준 판정 =====")
print(f"  ① 앙상블 IC 평균 {ens_sum['IC평균']:.3f} ≥ {PASS_IC}  → {'✅' if c1 else '❌'}")
print(f"  ② 방향 적중률 {ens_sum['적중률']:.0%} ≥ {PASS_HIT:.0%}  → {'✅' if c2 else '❌'}")
print(f"  ③ IC 평균 > {PASS_SEED_IC} 인 seed {good_seeds}/{len(SEEDS)} ≥ {PASS_SEED_COUNT}  → {'✅' if c3 else '❌'}")
best_base = max(BASELINES, key=lambda k: summarize(ic_df[(ic_df['seed'] == '기준선') & (ic_df['indicator'] == k)])["IC평균"])
base_ic = summarize(ic_df[(ic_df["seed"] == "기준선") & (ic_df["indicator"] == best_base)])["IC평균"]
print(f"  (참고) 가장 나은 단순 기준선 {best_base} IC 평균 {base_ic:.3f} → "
      f"{'AE가 더 나음' if ens_sum['IC평균'] > base_ic else '⚠️ 단순 기준선보다 못함 → AE의 추가 가치 의문'}")
PASSED = c1 and c2 and c3
print(f"\n  ▶ 최종: {'✅ 가설 통과 — 모의투자 검증 단계로 진행 가능' if PASSED else '❌ 가설 기각 — v3의 20일 결과는 우연일 가능성이 큼'}")

print("\n===== 방향 해석 (학습구간 IC 부호) =====")
print("  이상도_학습IC > 0 : 이상도가 높을수록(시장이 평소와 다를수록) 20일 뒤 수익률이 높았음 → 급변 후 반등형")
print("  이상도_학습IC < 0 : 이상도가 높을수록 20일 뒤 수익률이 낮았음 → 급변 후 추가 하락형")
dir_df = pd.DataFrame(dir_rows)
print(dir_df.groupby("fold")[["축_학습IC", "이상도_학습IC"]].mean().round(3).to_string())
print(f"  이상도 방향이 양(+)인 폴드: {(dir_df.groupby('fold')['이상도_학습IC'].mean() > 0).sum()}/{dir_df['fold'].nunique()}")


# %% [셀 9] 결과 ② 매매 성과 & 차트
first = TRAIN_DAYS - 1
L_oos = len(oos["AE 타이밍"]["eq"])
idx = DATES[first:first + L_oos]
print("\n===== 표본외 매매 성과 (검증구간만 이어붙임) =====")
bh = C[first:first + L_oos] / C[first]
mb = metrics(bh)
print(f"  {'단순 보유 229200':22s} CAGR {mb['CAGR']:7.1%} Sharpe {mb['Sharpe']:5.2f} MDD {mb['MDD']:7.1%}")
for name in oos:
    m = metrics(np.array(oos[name]["eq"]), np.array(oos[name]["w"]))
    ok = "✅" if (m["Sharpe"] >= 1.5 and m["MDD"] >= -0.20) else "❌"
    print(f"  {name:22s} CAGR {m['CAGR']:7.1%} Sharpe {m['Sharpe']:5.2f} MDD {m['MDD']:7.1%} "
          f"평균비중 {m['Exposure']:.0%} | 목표 {ok}")
print("  ※ 'AE 타이밍'이 '항상 보유'보다 나아야 AE 신호에 실제 가치가 있다는 뜻")

try:
    fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [2, 1]})
    for name in oos:
        ax[0].plot(idx, oos[name]["eq"], label=name)
    ax[0].plot(idx, bh, label="단순 보유 229200", alpha=0.5)
    ax[0].set_yscale("log")
    ax[0].set_title("표본외 자산곡선 (log)")
    ax[0].legend()
    piv = pivot.copy()
    xs = np.arange(len(piv.index))
    width = 0.8 / len(piv.columns)
    for j, col in enumerate(piv.columns):
        ax[1].bar(xs + j * width, piv[col].values, width, label=col)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].axhline(PASS_IC, color="tab:green", ls="--", lw=0.8)
    ax[1].set_xticks(xs + 0.4 - width / 2)
    ax[1].set_xticklabels([f"F{i}" for i in piv.index])
    ax[1].set_title("폴드별 검증 IC (20일)")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
except Exception as ex:
    print("[차트 오류]", ex)
print("[7/9] 결과 출력 완료")


# %% [셀 10] 오늘 기준 신호 (최근 3년, seed 5개 앙상블)
try:
    a = N - TRAIN_DAYS
    b = N
    tr_idx = np.arange(a + WARMUP, b - HZ - 1)
    print(f"\n최근 3년({DATES[a].date()}~{DATES[-1].date()})으로 재학습 중...")
    comps = []
    for seed in SEEDS:
        inds, info = ae_one_seed(a, b, tr_idx, seed)
        comps.append(inds["comp"])
    ens = np.mean(np.vstack(comps), axis=0)
    threshold = np.nanmedian(ens[tr_idx])

    # 합성점수가 무엇과 닮았는지 (해석용)
    print("\n  [해석] 합성점수와 단순 특징(20일 평균)의 상관 — 학습구간")
    for col in ["ret1", "range", "vol", "clv", "gap"]:
        s20 = feat[col].rolling(20).mean().values
        print(f"    {col:6s}(20일평균): {calc_ic(ens[tr_idx], s20[tr_idx]):+.2f}")

    go = ens[-1] > threshold
    w = VOLW[-1] if go else 0.0
    print(f"\n===== {DATES[-1].date()} 기준 신호 =====")
    print(f"  합성점수 {ens[-1]:+.3f} / 기준선(학습 중앙값) {threshold:+.3f} → {'매수' if go else '현금'}")
    if not PASSED:
        print("  ⚠️ 위 판정에서 가설이 기각됨 → 이 신호는 참고용일 뿐, 매매 근거로 쓰지 마세요")
    elif go and w >= MIN_W:
        print(f"  다음 리밸런싱(20거래일 주기) 시가에 229200 비중 {w:.0%} 매수 | "
              f"손절가 약 {C[-1] - STOP_ATR_K * ATR[-1]:,.0f}원")
    else:
        print("  현금 유지, 다음 리밸런싱일에 재판단")
except Exception as ex:
    print("[오늘 신호 오류]", ex)

print("\n[9/9] 완료")
print("※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
