# ============================================================
# v3 — AI 독창 지표 예측력(IC) 검증 → 검증 통과 지표로만 매매
#
#   1단계: AI 독창 지표 생성 (폴드마다 '학습구간 데이터로만' 학습)
#          ① Autoencoder 잠재벡터 3개 + 이상도
#          ② Attention 예측값 / 어텐션 엔트로피 / 최근 집중도
#          ③ Genetic 진화 수식 2개
#   2단계: 지표가 미래 수익률을 예측하는지 측정 (IC = 순위상관)
#          → 학습구간에서 정한 방향(부호)이 검증구간에서도 맞는지 확인
#   3단계: 학습 IC가 충분한 지표만 합성 → 상승/하락/중립 판단
#          → 4가지 매매 방식 비교 (1배/2배 × 롱전용/인버스 전환)
#
# 신호 계산: 229200 (KODEX 코스닥150, 1배)
# 매매 대상: 229200(1배) / 233740(2배 레버리지) / 251340(코스닥150선물인버스)
# 기존 기술적 지표(RSI·MACD 등) 사용 안 함
# 코랩용 / API 키 불필요
# ============================================================

# %% [셀 1] 설치 (코랩에서 한 번만)
# !pip install -q finance-datareader yfinance koreanize-matplotlib

# %% [셀 2] 임포트 & 설정
import time
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # 그래프 한글 깨짐 방지 (없어도 실행됨)
except Exception:
    pass

TICKERS = {"x1": "229200",    # KODEX 코스닥150 (1배) — 신호 계산 + 매매
           "x2": "233740",    # KODEX 코스닥150레버리지 (2배)
           "inv": "251340"}   # KODEX 코스닥150선물인버스 (-1배)
START = "2016-01-01"

FEE = 0.00015
SLIPPAGE = 0.001
COST = FEE + SLIPPAGE

# 워크포워드
TRAIN_DAYS = 756            # 약 3년 학습
TEST_DAYS = 252             # 약 1년 검증

# AI 지표
WIN = 20                    # 최근 20일 패턴을 입력으로 사용
H_MAIN = 5                  # 매매에 쓰는 예측 기간: 5일 뒤 수익률
IC_HORIZONS = [5, 20]       # IC 측정 기간 (5일, 20일)
AE_EPOCHS = 300
ATTN_EPOCHS = 200
GA_POP = 60                 # 유전 알고리즘 개체 수
GA_GEN = 15                 # 세대 수
GA_DEPTH = 3                # 수식 최대 깊이
SEED = 42

# 지표 합성 & 매매 규칙
IC_MIN = 0.03               # 학습구간 |IC| 가 이 값 이상인 지표만 사용
SMOOTH = 3                  # 합성 점수 3일 평균 (잦은 매매 방지)
Q_LONG = 0.65               # 합성 점수가 학습구간 상위 35% → 상승 판단
Q_SHORT = 0.35              # 하위 35% → 하락 판단

# 리스크 관리
TARGET_VOL = 0.25           # 변동성 타깃 (종목별)
VOL_N = 20
MAX_W = 1.0
MIN_W = 0.05
REBAL_BAND = 0.2
ATR_N = 14
STOP_ATR_K = 3              # 진입가 - 3×ATR 손절 (0 = 손절 없음)

# 비교할 매매 방식: (상승 시 매수 종목, 하락 시 매수 종목 or None)
CONFIGS = {"1배 롱전용": ("x1", None),
           "1배 롱+인버스": ("x1", "inv"),
           "2배 롱전용": ("x2", None),
           "2배 롱+인버스": ("x2", "inv")}

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan 평균 경고 숨김
random.seed(SEED)
np.random.seed(SEED)
print("[1/9] 설정 완료")


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


dfs = {}
for key in TICKERS:
    dfs[key] = load_data(TICKERS[key], START)
    if dfs[key] is None:
        raise SystemExit(f"{TICKERS[key]} 데이터 수집 실패")

common = dfs["x1"].index
for key in dfs:
    common = common.intersection(dfs[key].index)
for key in dfs:
    dfs[key] = dfs[key].loc[common]
N = len(common)
DATES = common
print(f"[2/9] 공통 데이터 {N}일 | {common[0].date()} ~ {common[-1].date()}")
if N < TRAIN_DAYS + TEST_DAYS:
    raise SystemExit("데이터가 워크포워드에 부족합니다.")

# 데이터 이상치 점검 (하루 ±25% 넘는 날)
for key in dfs:
    r = dfs[key]["Close"].pct_change().abs()
    n_big = int((r > 0.25).sum())
    if n_big > 0:
        print(f"  ⚠️ {TICKERS[key]}: 하루 ±25% 넘는 날 {n_big}일 → 액면분할/데이터 오류 확인 필요")


# %% [셀 4] 기본 특징 & 미래수익률 & 리스크 지표
# 종목별 가격 배열
O, H, L, C, ATR, VOLW = {}, {}, {}, {}, {}, {}
for key in dfs:
    d = dfs[key]
    O[key] = d["Open"].values
    H[key] = d["High"].values
    L[key] = d["Low"].values
    C[key] = d["Close"].values
    prev_c = d["Close"].shift(1)
    tr_rng = pd.concat([d["High"] - d["Low"], (d["High"] - prev_c).abs(),
                        (d["Low"] - prev_c).abs()], axis=1).max(axis=1)
    ATR[key] = tr_rng.rolling(ATR_N).mean().values
    rv = d["Close"].pct_change().rolling(VOL_N).std() * np.sqrt(252)
    VOLW[key] = np.clip(TARGET_VOL / rv, 0, MAX_W).fillna(0).values

# AI 입력용 기본 특징 5개 (229200 기준, 모두 t일 종가까지 정보만 사용)
d = dfs["x1"]
feat = pd.DataFrame(index=d.index)
feat["ret1"] = np.log(d["Close"] / d["Close"].shift(1))                       # 일간 수익률
feat["gap"] = np.log(d["Open"] / d["Close"].shift(1))                         # 시가 갭
feat["range"] = (d["High"] - d["Low"]) / d["Close"]                           # 장중 변동폭
feat["clv"] = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / (d["High"] - d["Low"] + 1e-8)  # 종가 위치
feat["vol"] = np.log(d["Volume"] / d["Volume"].rolling(20).mean())            # 거래량 강도
feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
F = feat.values.astype(np.float32)          # (N, 5)
FEAT_NAMES = list(feat.columns)

# 유전 알고리즘 재료(터미널): 기본 특징 + 5일/20일 수익률
GA_TERMS = {name: feat[name].values for name in FEAT_NAMES}
GA_TERMS["ret5"] = np.log(d["Close"] / d["Close"].shift(5)).fillna(0).values
GA_TERMS["ret20"] = np.log(d["Close"] / d["Close"].shift(20)).fillna(0).values

# 미래 수익률 (정답): t일 신호 → t+1 시가 매수 → t+1+H 시가 매도 기준
FWD = {}
for hz in IC_HORIZONS:
    f = np.full(N, np.nan)
    for t in range(N - 1 - hz):
        f[t] = O["x1"][t + 1 + hz] / O["x1"][t + 1] - 1
    FWD[hz] = f

WARMUP = WIN + 20   # 20일 거래량 평균 + 20일 창
print("[3/9] 특징/정답/리스크 지표 계산 완료")


def calc_ic(x, y):
    """순위상관(Spearman IC). 유효 데이터 30개 미만이면 nan"""
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


# %% [셀 5] AI 모델 정의 (Autoencoder / Attention)
AI_OK = True
try:
    import torch
    import torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/9] PyTorch 사용 가능 | device = {DEVICE}")
except Exception as e:
    AI_OK = False
    print("[4/9] PyTorch 없음 → Autoencoder/Attention 지표 생략, 유전 지표만 사용:", e)

if AI_OK:
    class MarketAutoencoder(nn.Module):
        """20일×5특징(100차원) → 3차원 압축. 잠재벡터 3개 = 독창 지표"""
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

    class AttentionIndicator(nn.Module):
        """현재 시점이 과거 20일 중 어디를 보는지(어텐션) + 5일 뒤 수익률 예측"""
        def __init__(self, n_feat=5, d_model=32):
            super().__init__()
            self.embed = nn.Linear(n_feat, d_model)
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, 1)
            self.d_model = d_model

        def forward(self, x):
            h = torch.relu(self.embed(x))                           # (B, T, d)
            Q = self.query(h[:, -1:, :])                            # 현재 시점 쿼리
            K = self.key(h)
            V = self.value(h)
            attn = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)  # (B, 1, T)
            ctx = (attn @ V).squeeze(1)
            return self.out(ctx).squeeze(-1), attn.squeeze(1)


def make_windows(Fz):
    """(N, 5) → (N, WIN, 5) 슬라이딩 창 (t일까지의 20일)"""
    idx = np.arange(N)[:, None] + np.arange(-WIN + 1, 1)[None, :]
    idx = np.clip(idx, 0, N - 1)
    return Fz[idx]


def ai_indicators(a, b, tr_idx):
    """Autoencoder + Attention 지표 (학습구간 tr_idx 로만 학습)"""
    out = {}
    mu = F[a + WARMUP:b].mean(axis=0)
    sd = F[a + WARMUP:b].std(axis=0) + 1e-8
    Fz = ((F - mu) / sd).clip(-5, 5).astype(np.float32)
    Xw = make_windows(Fz)                                   # (N, WIN, 5)

    # ---- ① Autoencoder ----
    torch.manual_seed(SEED)
    ae = MarketAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x_tr = torch.tensor(Xw[a + WARMUP:b].reshape(-1, WIN * 5)).to(DEVICE)
    for ep in range(AE_EPOCHS):
        recon, _ = ae(x_tr)
        loss = loss_fn(recon, x_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ae.eval()
    with torch.no_grad():
        x_all = torch.tensor(Xw.reshape(-1, WIN * 5)).to(DEVICE)
        recon, z = ae(x_all)
        err = ((recon - x_all) ** 2).mean(dim=1).cpu().numpy()
        z = z.cpu().numpy()
    for i in range(3):
        out[f"ae_z{i}"] = z[:, i]
    out["ae_anomaly"] = err
    print(f"    [AE] 최종 loss {loss.item():.4f}")

    # ---- ② Attention (5일 뒤 수익률 예측 학습, 정답은 학습구간 것만) ----
    y = FWD[H_MAIN][tr_idx]
    y_mu, y_sd = y.mean(), y.std() + 1e-8
    y_z = np.clip((y - y_mu) / y_sd, -3, 3).astype(np.float32)
    torch.manual_seed(SEED)
    at = AttentionIndicator().to(DEVICE)
    opt = torch.optim.Adam(at.parameters(), lr=1e-3, weight_decay=1e-4)
    xs = torch.tensor(Xw[tr_idx]).to(DEVICE)
    ys = torch.tensor(y_z).to(DEVICE)
    for ep in range(ATTN_EPOCHS):
        pred, _ = at(xs)
        loss = loss_fn(pred, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    at.eval()
    with torch.no_grad():
        pred, attn = at(torch.tensor(Xw).to(DEVICE))
        pred = pred.cpu().numpy()
        attn = attn.cpu().numpy()                           # (N, WIN)
    out["attn_pred"] = pred                                 # 예측값 자체
    out["attn_entropy"] = -(attn * np.log(attn + 1e-8)).sum(axis=1)   # 집중도(낮을수록 특정 시점 집중)
    out["attn_recent"] = attn[:, -5:].sum(axis=1)           # 최근 5일에 둔 비중
    print(f"    [Attention] 최종 loss {loss.item():.4f}")
    return out


# %% [셀 6] 유전 알고리즘 진화 지표
GA_OPS = ["+", "-", "*", "/"]
TERM_NAMES = list(GA_TERMS.keys())


def random_formula(depth):
    if depth == 0 or random.random() < 0.3:
        return random.choice(TERM_NAMES)
    return (random.choice(GA_OPS), random_formula(depth - 1), random_formula(depth - 1))


def eval_formula(f):
    if isinstance(f, str):
        return GA_TERMS[f]
    op, left, right = f
    lv = eval_formula(left)
    rv = eval_formula(right)
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    safe = np.where(np.abs(rv) < 1e-6, np.nan, rv)    # 0으로 나누기 방지
    return lv / safe


def formula_str(f):
    if isinstance(f, str):
        return f
    return f"({formula_str(f[1])} {f[0]} {formula_str(f[2])})"


def formula_size(f):
    if isinstance(f, str):
        return 1
    return 1 + formula_size(f[1]) + formula_size(f[2])


def mutate(f):
    """30% 확률로 일부 가지를 새 수식으로 교체"""
    if isinstance(f, str) or random.random() < 0.3:
        return random_formula(2)
    if random.random() < 0.5:
        return (f[0], mutate(f[1]), f[2])
    return (f[0], f[1], mutate(f[2]))


def crossover(f1, f2):
    """부모 둘의 가지를 섞기"""
    if isinstance(f1, str) or isinstance(f2, str):
        return random.choice([f1, f2])
    return (f1[0], f1[1], f2[2])


def ga_fitness(f, tr_idx):
    try:
        x = eval_formula(f)[tr_idx]
        ic = calc_ic(x, FWD[H_MAIN][tr_idx])
        if np.isnan(ic):
            return -1.0
        return abs(ic) - 0.002 * formula_size(f)     # 복잡한 수식 감점(과최적화 방지)
    except Exception:
        return -1.0


def evolve(tr_idx):
    pop = [random_formula(GA_DEPTH) for _ in range(GA_POP)]
    scores = [ga_fitness(f, tr_idx) for f in pop]
    for gen in range(GA_GEN):
        order = sorted(range(GA_POP), key=lambda i: scores[i], reverse=True)
        elites = [pop[i] for i in order[:GA_POP // 5]]
        new_pop = list(elites)
        while len(new_pop) < GA_POP:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        scores = [ga_fitness(f, tr_idx) for f in pop]
    # 서로 다른 상위 2개 수식
    order = sorted(range(GA_POP), key=lambda i: scores[i], reverse=True)
    best, seen = [], set()
    for i in order:
        s = formula_str(pop[i])
        if s not in seen:
            seen.add(s)
            best.append((pop[i], scores[i]))
        if len(best) == 2:
            break
    return best


def build_indicators(a, b):
    """학습구간 [a, b) 로 모든 AI 지표 생성 → {이름: 전체기간 값}, 수식 설명"""
    # 정답(5일 뒤 수익률)이 검증구간에 걸치는 마지막 날들은 제외 (정보 누수 방지)
    tr_idx = np.arange(a + WARMUP, b - H_MAIN - 1)
    inds = {}
    if AI_OK:
        inds.update(ai_indicators(a, b, tr_idx))
    best = evolve(tr_idx)
    desc = {}
    for j, (f, sc) in enumerate(best):
        vals = eval_formula(f).astype(float)
        vals[~np.isfinite(vals)] = np.nan
        inds[f"ga_{j + 1}"] = vals
        desc[f"ga_{j + 1}"] = formula_str(f)
        print(f"    [GA] ga_{j + 1}: {formula_str(f)} | 학습 적합도 {sc:.3f}")
    return inds, desc, tr_idx


print("[5/9] AI 지표 생성기 준비 완료")


# %% [셀 7] 합성 점수 → 국면 판단 & 백테스트 엔진
def make_regime(inds, tr_idx):
    """
    학습 IC로 방향(부호)과 사용 여부 결정 → 표준화 후 평균 = 합성 점수
    합성 점수 상위 → 1(상승), 하위 → -1(하락), 나머지 0(중립)
    """
    used = []
    parts = []
    for name in inds:
        x = inds[name]
        ic_tr = calc_ic(x[tr_idx], FWD[H_MAIN][tr_idx])
        if np.isnan(ic_tr) or abs(ic_tr) < IC_MIN:
            continue
        mu = np.nanmean(x[tr_idx])
        sd = np.nanstd(x[tr_idx]) + 1e-8
        parts.append(np.sign(ic_tr) * np.clip((x - mu) / sd, -5, 5))
        used.append(name)
    regime = np.zeros(N, dtype=int)
    if len(parts) == 0:
        return regime, used, np.full(N, np.nan), (np.nan, np.nan)
    comp = np.nanmean(np.vstack(parts), axis=0)
    comp = pd.Series(comp).rolling(SMOOTH, min_periods=1).mean().values   # 과거만 사용
    up = np.nanquantile(comp[tr_idx], Q_LONG)
    lo = np.nanquantile(comp[tr_idx], Q_SHORT)
    regime[comp > up] = 1
    regime[comp < lo] = -1
    return regime, used, comp, (lo, up)


def backtest(a, b, regime, long_k, short_k):
    """
    a~b-1 구간. t일 종가에 국면 확인 → t+1 시가 체결
    - 상승: long_k 매수 / 하락: short_k 매수(None이면 현금) / 중립: 현금
    - 비중 = 해당 종목 변동성 타깃 비중
    - ATR 손절 후엔 국면이 바뀔 때까지 같은 방향 재진입 금지
    """
    n = b - a
    equity = np.ones(n)
    weights = np.zeros(n)
    cur = None          # 보유 종목 키
    stock = 0.0
    cash = 1.0
    order = None        # (종목키 or None, 목표비중)
    stop_px = 0.0
    locked = 0
    trades = 0

    for i in range(1, n):
        t = a + i
        # 1) 갭 반영
        if cur is not None and stock > 0:
            stock = stock * O[cur][t] / C[cur][t - 1]

        # 2) 시가 주문 체결
        if order is not None:
            k, w = order
            if cur is not None and stock > 0 and k != cur:      # 종목 교체 → 전량 매도
                cash = cash + stock * (1 - COST)
                stock = 0.0
                cur = None
            if k is not None and w > 0:
                total = stock + cash
                want = w * total
                trade = want - stock
                if stock == 0:
                    trades += 1
                    atr_prev = ATR[k][t - 1]
                    if STOP_ATR_K > 0 and not math.isnan(atr_prev):
                        stop_px = O[k][t] - STOP_ATR_K * atr_prev
                    else:
                        stop_px = 0.0
                cash = cash - trade - abs(trade) * COST
                stock = want
                cur = k
            order = None

        # 3) 장중: 손절 또는 시가→종가
        if cur is not None and stock > 0:
            if stop_px > 0 and L[cur][t] <= stop_px:
                exit_px = min(O[cur][t], stop_px)
                stock = stock * exit_px / O[cur][t]
                cash = cash + stock * (1 - COST)
                stock = 0.0
                locked = 1 if cur == long_k else -1
                cur = None
            else:
                stock = stock * C[cur][t] / O[cur][t]

        equity[i] = stock + cash
        cur_w = stock / equity[i] if equity[i] > 0 else 0.0
        weights[i] = cur_w

        # 4) 종가: 내일 목표 결정
        side = regime[t]
        if side != locked:
            locked = 0
        if side == 1:
            k = long_k
        elif side == -1 and short_k is not None:
            k = short_k
        else:
            k = None
        if locked != 0 and side == locked:
            k = None
        w = VOLW[k][t] if k is not None else 0.0
        if w < MIN_W:
            k = None
            w = 0.0

        if k is None:
            if stock > 0:
                order = (None, 0.0)
        elif k != cur or stock == 0:
            order = (k, w)
        elif abs(w - cur_w) > REBAL_BAND:
            order = (k, w)

    return equity, trades, weights


def metrics(equity, weights=None):
    ret = pd.Series(equity).pct_change().fillna(0)
    years = len(equity) / 252
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    expo = float(np.mean(weights)) if weights is not None else 1.0
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": mdd, "Exposure": expo}


print("[6/9] 합성/백테스트 엔진 준비 완료")


# %% [셀 8] 워크포워드 실행
ic_rows = []
fold_rows = []
oos = {name: {"eq": [1.0], "w": [0.0]} for name in CONFIGS}
start = 0
fold = 0
t0 = time.time()
while start + TRAIN_DAYS < N:
    a = start
    b = start + TRAIN_DAYS
    e = min(b + TEST_DAYS, N)
    fold += 1
    print(f"\n[폴드 {fold}] 학습 {DATES[a].date()}~{DATES[b - 1].date()} | 검증 {DATES[b].date()}~{DATES[e - 1].date()}")
    try:
        inds, desc, tr_idx = build_indicators(a, b)
        te_idx = np.arange(b, e)

        # --- 지표별 IC: 학습 부호를 검증에 그대로 적용 ---
        for name in inds:
            x = inds[name]
            ic_tr = calc_ic(x[tr_idx], FWD[H_MAIN][tr_idx])
            sgn = np.sign(ic_tr) if not np.isnan(ic_tr) and ic_tr != 0 else 1.0
            row = {"fold": fold, "indicator": name, "train_IC": ic_tr}
            for hz in IC_HORIZONS:
                row[f"test_IC_{hz}d"] = sgn * calc_ic(x[te_idx], FWD[hz][te_idx])
            ic_rows.append(row)

        # --- 합성 → 매매 ---
        regime, used, comp, (lo, up) = make_regime(inds, tr_idx)
        print(f"    사용 지표({len(used)}개): {used}")
        frow = {"fold": fold, "검증": f"{DATES[b].date()}~{DATES[e - 1].date()}", "사용지표수": len(used)}
        for name, (lk, sk) in CONFIGS.items():
            eq, tr, w = backtest(b - 1, e, regime, lk, sk)
            last = oos[name]["eq"][-1]
            oos[name]["eq"].extend([last * v for v in eq[1:]])
            oos[name]["w"].extend(list(w[1:]))
            frow[name] = round(eq[-1] - 1, 3)
        fold_rows.append(frow)
        print("    검증 수익: " + " | ".join(f"{k} {frow[k]:+.1%}" for k in CONFIGS))
    except Exception as ex:
        print("    [폴드 오류]", ex)
    start += TEST_DAYS

print(f"\n[7/9] 워크포워드 완료 ({time.time() - t0:.0f}초)")


# %% [셀 9] 결과 ① 지표 예측력(IC) 리포트
ic_df = pd.DataFrame(ic_rows)
pd.set_option("display.width", 220)
summary = []
for name, g in ic_df.groupby("indicator"):
    row = {"지표": name, "폴드수": len(g), "학습|IC|평균": g["train_IC"].abs().mean()}
    for hz in IC_HORIZONS:
        col = g[f"test_IC_{hz}d"].dropna()
        row[f"검증IC_{hz}d평균"] = col.mean() if len(col) > 0 else np.nan
        row[f"{hz}d방향적중률"] = (col > 0).mean() if len(col) > 0 else np.nan
        row[f"{hz}d_IR"] = col.mean() / (col.std() + 1e-8) if len(col) > 1 else np.nan
    m = row[f"검증IC_{H_MAIN}d평균"]
    hit = row[f"{H_MAIN}d방향적중률"]
    if m > 0.03 and hit >= 0.7:
        row["판정"] = "✅ 안정"
    elif m > 0.01 and hit >= 0.6:
        row["판정"] = "△ 약함"
    else:
        row["판정"] = "❌ 예측력 없음"
    summary.append(row)
ic_sum = pd.DataFrame(summary).sort_values(f"검증IC_{H_MAIN}d평균", ascending=False)
print("\n===== AI 지표 예측력 (검증구간 IC, 학습 방향 그대로 적용) =====")
print("  IC > 0 : 학습에서 찾은 방향이 검증에서도 맞음 / 보통 0.03 이상이면 의미 있는 수준")
print(ic_sum.round(3).to_string(index=False))


# %% [셀 10] 결과 ② 매매 방식 비교 & 차트
print("\n===== 폴드별 검증 수익 =====")
print(pd.DataFrame(fold_rows).to_string(index=False))

first = TRAIN_DAYS - 1
L_oos = len(oos[list(CONFIGS)[0]]["eq"])
idx = DATES[first:first + L_oos]
print("\n===== 표본외 성과 (검증구간만 이어붙임) =====")
for key, label in [("x1", "보유 229200(1배)"), ("x2", "보유 233740(2배)")]:
    bh = C[key][first:first + L_oos] / C[key][first]
    mb = metrics(bh)
    print(f"  {label:18s} CAGR {mb['CAGR']:7.1%} Sharpe {mb['Sharpe']:5.2f} MDD {mb['MDD']:7.1%}")
best_name, best_sharpe = None, -99
for name in CONFIGS:
    eq = np.array(oos[name]["eq"])
    m = metrics(eq, np.array(oos[name]["w"]))
    ok = "✅" if (m["Sharpe"] >= 1.5 and m["MDD"] >= -0.20) else "❌"
    print(f"  {name:18s} CAGR {m['CAGR']:7.1%} Sharpe {m['Sharpe']:5.2f} MDD {m['MDD']:7.1%} "
          f"평균비중 {m['Exposure']:.0%} | 목표(Sharpe≥1.5, MDD≤20%) {ok}")
    if m["Sharpe"] > best_sharpe:
        best_name, best_sharpe = name, m["Sharpe"]

try:
    fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [2, 1]})
    for name in CONFIGS:
        ax[0].plot(idx, oos[name]["eq"], label=name)
    ax[0].plot(idx, C["x1"][first:first + L_oos] / C["x1"][first], label="Buy&Hold 1x", alpha=0.4)
    ax[0].plot(idx, C["x2"][first:first + L_oos] / C["x2"][first], label="Buy&Hold 2x", alpha=0.4)
    ax[0].set_yscale("log")
    ax[0].set_title("Walk-forward out-of-sample equity (log)")
    ax[0].legend(fontsize=8)
    col = f"검증IC_{H_MAIN}d평균"
    s = ic_sum.set_index("지표")[col]
    ax[1].barh(s.index, s.values, color=["tab:green" if v > 0.03 else "tab:gray" for v in s.values])
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].axvline(0.03, color="tab:green", ls="--", lw=0.8)
    ax[1].set_title(f"OOS IC ({H_MAIN}d) by indicator")
    plt.tight_layout()
    plt.show()
except Exception as ex:
    print("[차트 오류]", ex)
print("[8/9] 결과 출력 완료")


# %% [셀 11] 오늘 기준 신호 (최근 3년으로 다시 학습)
try:
    a = N - TRAIN_DAYS
    b = N
    print(f"\n최근 3년({DATES[a].date()}~{DATES[-1].date()})으로 지표 재학습 중...")
    inds, desc, tr_idx = build_indicators(a, b)
    regime, used, comp, (lo, up) = make_regime(inds, tr_idx)
    print(f"\n===== {DATES[-1].date()} 기준 신호 =====")
    print(f"  사용 지표: {used}")
    for k in desc:
        print(f"    {k} = {desc[k]}")
    if len(used) == 0:
        print("  학습 IC 기준을 넘는 지표가 없음 → 관망")
    else:
        side = regime[-1]
        side_txt = {1: "상승", -1: "하락", 0: "중립"}[side]
        print(f"  합성 점수 {comp[-1]:.3f} (하락선 {lo:.3f} / 상승선 {up:.3f}) → {side_txt} 국면")
        for name, (lk, sk) in CONFIGS.items():
            if side == 1:
                k = lk
            elif side == -1 and sk is not None:
                k = sk
            else:
                k = None
            if k is None or VOLW[k][-1] < MIN_W:
                print(f"  [{name}] 내일 시가: 현금 보유")
            else:
                stop_txt = f", 손절가 약 {C[k][-1] - STOP_ATR_K * ATR[k][-1]:,.0f}원" if STOP_ATR_K > 0 else ""
                print(f"  [{name}] 내일 시가: {TICKERS[k]} 비중 {VOLW[k][-1]:.0%} 매수{stop_txt}")
    print(f"\n  ※ 표본외 Sharpe 기준 가장 나았던 방식: {best_name} ({best_sharpe:.2f})")
except Exception as ex:
    print("[오늘 신호 오류]", ex)

print("\n[9/9] 완료")
print("※ 과거 백테스트 결과이며 미래 수익을 보장하지 않습니다. 실거래 전 반드시 모의투자로 검증하세요.")
