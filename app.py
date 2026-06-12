"""
테마주 퀀트 대시보드 + 급등주 탐지 통합 앱
Streamlit Cloud 단독 실행 버전 (라이트모드)
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="테마주 퀀트 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 라이트모드 차트 색상 ─────────────────────────────────────────────────────
C = {
    "paper":   "white",
    "plot":    "#f8fafc",
    "font":    "#374151",
    "grid":    "#e5e7eb",
    "leg_bg":  "rgba(255,255,255,0.9)",
    "up":      "#16a34a",
    "down":    "#dc2626",
}

# ─── 테마 정의 ────────────────────────────────────────────────────────────────
THEMES: Dict[str, Dict] = {
    "ai_semiconductor": {
        "name": "AI/반도체",
        "stocks": {
            "005930": ("삼성전자",    "KS"),
            "000660": ("SK하이닉스",  "KS"),
            "042700": ("한미반도체",  "KS"),
            "336370": ("솔브레인",    "KQ"),
            "005290": ("동진쎄미켐",  "KS"),
        },
    },
    "battery": {
        "name": "2차전지",
        "stocks": {
            "373220": ("LG에너지솔루션", "KS"),
            "086520": ("에코프로",       "KQ"),
            "247540": ("에코프로비엠",   "KQ"),
            "003670": ("포스코퓨처엠",   "KS"),
            "066970": ("엘앤에프",       "KQ"),
        },
    },
    "bio": {
        "name": "바이오/제약",
        "stocks": {
            "068270": ("셀트리온",         "KS"),
            "207940": ("삼성바이오로직스", "KS"),
            "128940": ("한미약품",         "KQ"),
            "000100": ("유한양행",         "KS"),
            "185750": ("종근당",           "KS"),
        },
    },
    "defense": {
        "name": "방산",
        "stocks": {
            "012450": ("한화에어로스페이스", "KS"),
            "272210": ("한화시스템",         "KS"),
            "329180": ("현대로템",           "KS"),
            "079550": ("LIG넥스원",          "KS"),
            "010820": ("풍산",               "KS"),
        },
    },
    "nuclear": {
        "name": "원자력",
        "stocks": {
            "034020": ("두산에너빌리티", "KS"),
            "017890": ("한국전력기술",   "KS"),
            "298040": ("효성중공업",     "KS"),
            "028050": ("삼성엔지니어링", "KS"),
            "000720": ("현대건설",       "KS"),
        },
    },
    "shipbuilding": {
        "name": "조선",
        "stocks": {
            "009540": ("HD한국조선해양", "KS"),
            "010140": ("삼성중공업",     "KS"),
            "042660": ("한화오션",       "KS"),
            "267250": ("HD현대",         "KS"),
            "010620": ("현대미포조선",   "KS"),
        },
    },
    "game": {
        "name": "게임",
        "stocks": {
            "036570": ("엔씨소프트",   "KQ"),
            "251270": ("넷마블",       "KS"),
            "263750": ("펄어비스",     "KQ"),
            "293490": ("카카오게임즈", "KQ"),
            "112040": ("위메이드",     "KQ"),
        },
    },
}

# ─── 동적 테마 로딩 ──────────────────────────────────────────────────────────
_THEMES_UPDATED_AT: Optional[str] = None
THEMES_STAGES: Dict[str, int] = {}


def _try_load_dynamic_themes() -> None:
    global _THEMES_UPDATED_AT
    path = "themes_data.json"
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key, val in data.items():
            stocks = {c: tuple(v) for c, v in val.get("stocks", {}).items()}
            if key in THEMES:
                THEMES[key]["stocks"] = stocks
            else:
                THEMES[key] = {"name": val["name"], "stocks": stocks}
            if "stage" in val:
                THEMES_STAGES[key] = int(val["stage"])
        times = [v.get("updated_at", "") for v in data.values() if v.get("updated_at")]
        if times:
            _THEMES_UPDATED_AT = max(times)[:16].replace("T", " ")
    except Exception:
        pass


_try_load_dynamic_themes()

# ─── 상수 ────────────────────────────────────────────────────────────────────
STAGE_INFO = {
    0: {"name": "데이터 부족",   "en": "Insufficient",  "emoji": "❓", "color": "#6b7280"},
    1: {"name": "바닥 매집기",   "en": "Basing",        "emoji": "🧱", "color": "#6b7280"},
    2: {"name": "모멘텀 돌파기", "en": "Breakout",      "emoji": "🚀", "color": "#2563eb"},
    3: {"name": "주도 상승기",   "en": "Advancing",     "emoji": "📈", "color": "#16a34a"},
    4: {"name": "과열 분배기",   "en": "Distribution",  "emoji": "⚠️", "color": "#d97706"},
    5: {"name": "항복 투매기",   "en": "Declining",     "emoji": "📉", "color": "#dc2626"},
}

STAGE_BG = {
    1: "#f3f4f6", 2: "#dbeafe", 3: "#dcfce7",
    4: "#ffedd5", 5: "#fee2e2",
}

TIP = {
    "lss":       "LSS(대장주 점수): 테마 내 종목의 종합 주도력 점수(0~1). 높을수록 테마 상승 시 가장 먼저·가장 크게 움직이는 대장주입니다.",
    "roc":       "ROC(Rate of Change): 최근 20영업일 주가 등락률(%). 테마가 움직일 때 가장 탄력적으로 반응하는 종목을 찾는 데 사용합니다.",
    "tv":        "거래대금: 최근 20영업일 평균 (종가×거래량). 시장의 관심도와 유동성을 평가합니다.",
    "turnover":  "주식 거래 회전율: 최근 20일 총 거래량 ÷ 상장주식수. 손바뀜이 얼마나 활발히 일어났는지를 측정합니다.",
    "beta":      "베타(Beta): KOSPI 대비 개별 종목의 가격 민감도. 베타 2.0이면 시장이 1% 오를 때 약 2% 움직임을 의미합니다.",
    "wrs":       "가중 상대강도(WRS): 0.7×120일 수익률 + 0.3×240일 수익률. 6~12개월 장기 추세 강도를 평가합니다.",
    "ma200":     "MA200(200일 이동평균선): 장기 추세 기준선.",
    "slope":     "MA200 기울기: 200일선의 상승·하락 기울기. 0.005 이상이면 가파른 상승 추세입니다.",
    "rvol":      "RVOL(상대 거래량): 최근 5일 평균 ÷ 최근 40일 평균 거래량. 2.0 이상이면 거래량 폭발 신호.",
    "breadth":   "테마 참여율: 테마 내 종목 중 MA50 위에 있는 종목 비율(%). 70% 이상이면 광범위한 상승 확산.",
    "volatility":"20일 변동성: (고가-저가)/평균가의 20일 평균(%). 18% 초과 시 과열 주의.",
    "ma50":      "MA50(50일 이동평균선): 중기 추세 기준선.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 로딩
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_stock(ticker: str, exchange: str, start: str, end: str) -> Optional[pd.DataFrame]:
    required = ["Open", "High", "Low", "Close", "Volume"]

    def _dl(t):
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or df.empty: return None
            df = _flatten(df)
            df.index = pd.to_datetime(df.index)
            if any(c not in df.columns for c in required): return None
            df = df[required].copy()
            df["Volume"] = df["Volume"].fillna(0)
            return df.dropna(subset=["Close"]) if len(df) >= 60 else None
        except Exception:
            return None

    if ticker == "KS11":
        return _dl("^KS11")
    df = _dl(f"{ticker}.{exchange}")
    if df is None:
        alt = "KQ" if exchange == "KS" else "KS"
        df = _dl(f"{ticker}.{alt}")
    return df


def load_theme_data(theme_id: str) -> Dict[str, Any]:
    theme   = THEMES[theme_id]
    end_dt  = datetime.today().strftime("%Y-%m-%d")
    start_dt = (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d")
    benchmark = load_stock("KS11", "KS", start_dt, end_dt)
    stocks: Dict[str, pd.DataFrame] = {}
    fetch_status = []
    for ticker, (name, exch) in theme["stocks"].items():
        df = load_stock(ticker, exch, start_dt, end_dt)
        if df is not None:
            stocks[ticker] = df
            fetch_status.append({"ticker": ticker, "name": name, "ok": True,  "days": len(df)})
        else:
            fetch_status.append({"ticker": ticker, "name": name, "ok": False, "days": 0})
    return {
        "theme_id": theme_id, "theme_name": theme["name"],
        "benchmark": benchmark, "stocks": stocks,
        "fetch_status": fetch_status,
        "analysis_date": end_dt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LSS 엔진
# ═══════════════════════════════════════════════════════════════════════════════

def calc_roc(df, period=20):
    if len(df) < period + 1: return np.nan
    p0 = df["Close"].iloc[-(period+1)]
    return (df["Close"].iloc[-1] - p0) / p0 * 100 if p0 else np.nan

def calc_trading_value(df, period=20):
    if len(df) < period: return np.nan
    r = df.tail(period)
    return float((r["Close"] * r["Volume"]).mean())

def calc_turnover_ratio(df, period=20):
    if len(df) < period: return np.nan
    a20 = df["Volume"].tail(20).mean()
    al  = df["Volume"].tail(min(40, len(df))).mean()
    return float(a20 / al) if al > 0 else np.nan

def calc_beta(df, bm, period=60):
    if len(df) < period or bm is None or len(bm) < period: return np.nan
    s = df["Close"].pct_change().tail(period).dropna()
    b = bm["Close"].pct_change().tail(period).dropna()
    idx = s.index.intersection(b.index)
    if len(idx) < 30: return np.nan
    cov = np.cov(s.loc[idx].values, b.loc[idx].values)
    return float(cov[0,1] / cov[1,1]) if cov[1,1] != 0 else np.nan

def calc_wrs(df):
    p = df["Close"].iloc[-1]
    r120 = r240 = np.nan
    if len(df) >= 121:
        p0 = df["Close"].iloc[-121]
        r120 = (p-p0)/p0*100 if p0 else np.nan
    if len(df) >= 241:
        p0 = df["Close"].iloc[-241]
        r240 = (p-p0)/p0*100 if p0 else np.nan
    if pd.notna(r120) and pd.notna(r240): return 0.7*r120 + 0.3*r240
    return r120 if pd.notna(r120) else np.nan

def minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if mx == mn else (s-mn)/(mx-mn)

WEIGHTS = {"roc":0.30,"trading_value":0.30,"turnover_ratio":0.15,"beta":0.10,"wrs":0.15}

def calc_lss(stocks, benchmark) -> pd.DataFrame:
    records = []
    for tk, df in stocks.items():
        if df is None or len(df) < 60: continue
        records.append({
            "ticker": tk, "close": float(df["Close"].iloc[-1]),
            "roc":            calc_roc(df),
            "trading_value":  calc_trading_value(df),
            "turnover_ratio": calc_turnover_ratio(df),
            "beta":           calc_beta(df, benchmark),
            "wrs":            calc_wrs(df),
        })
    if not records: return pd.DataFrame()
    result = pd.DataFrame(records).set_index("ticker")
    for ind in WEIGHTS:
        col = result[ind].copy()
        norm = pd.Series(0.5, index=result.index)
        mask = col.notna()
        if mask.sum() > 1: norm[mask] = minmax(col[mask])
        result[f"{ind}_norm"] = norm
    result["lss"] = sum(result[f"{k}_norm"]*v for k,v in WEIGHTS.items())
    result = result.sort_values("lss", ascending=False).reset_index()
    result["rank"] = range(1, len(result)+1)
    result["is_leader"] = result["rank"] <= 3
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 생애주기 엔진
# ═══════════════════════════════════════════════════════════════════════════════

def build_theme_index(stocks) -> pd.DataFrame:
    p_list, h_list, l_list, v_list = [], [], [], []
    for tk, df in stocks.items():
        if df is None or len(df) < 200: continue
        base = df["Close"].iloc[0]
        if base == 0: continue
        f = 100.0 / base
        p_list.append((df["Close"]*f).rename(tk))
        h_list.append((df["High"] *f).rename(tk))
        l_list.append((df["Low"]  *f).rename(tk))
        v_list.append(df["Volume"].rename(tk))
    if not p_list: return pd.DataFrame()
    def avg(lst): return pd.concat(lst, axis=1).dropna(how="all").mean(axis=1)
    return pd.DataFrame({
        "Close": avg(p_list), "High": avg(h_list),
        "Low":   avg(l_list), "Volume": avg(v_list),
    }).dropna(subset=["Close"])

def calc_ma_slope(ma_s, window=15):
    if len(ma_s) < window: return 0.0
    y = ma_s.iloc[-window:].values.astype(float)
    slope, *_ = stats.linregress(np.arange(window, dtype=float), y)
    return float(slope / y[-1]) if y[-1] != 0 else 0.0

def calc_rvol(vol):
    if len(vol) < 40: return 1.0
    return float(vol.tail(5).mean() / vol.tail(40).mean()) if vol.tail(40).mean() > 0 else 1.0

def calc_breadth(stocks):
    above = total = 0
    for df in stocks.values():
        if df is None or len(df) < 50: continue
        total += 1
        if df["Close"].iloc[-1] > df["Close"].rolling(50).mean().iloc[-1]: above += 1
    return (above/total*100) if total > 0 else 0.0

def calc_volatility(df, period=20):
    if len(df) < period: return 0.0
    r = df.tail(period)
    avg = r["Close"].mean()
    return float(((r["High"]-r["Low"])/avg).mean()*100) if avg else 0.0

def determine_stage(theme_df, stocks) -> Dict:
    if theme_df.empty or len(theme_df) < 200:
        return {"stage": 0, "indicators": {}, "extra": {}, "conditions": {}}
    price   = float(theme_df["Close"].iloc[-1])
    ma200_s = theme_df["Close"].rolling(200).mean().dropna()
    ma50_s  = theme_df["Close"].rolling(50).mean().dropna()
    if ma200_s.empty:
        return {"stage": 0, "indicators": {}, "extra": {}, "conditions": {}}
    ma200   = float(ma200_s.iloc[-1])
    ma50    = float(ma50_s.iloc[-1]) if not ma50_s.empty else price
    slope   = calc_ma_slope(ma200_s)
    rvol    = calc_rvol(theme_df["Volume"])
    breadth = calc_breadth(stocks)
    volatility = calc_volatility(theme_df)
    pct     = (price/ma200 - 1)*100 if ma200 > 0 else 0.0
    rc = theme_df["Close"].tail(6).iloc[:-1]
    rm = ma200_s.tail(6).iloc[:-1]
    was_below = any(p < m for p, m in zip(rc.values, rm.values))
    indicators = {
        "price": round(price,2), "ma200": round(ma200,2), "ma50": round(ma50,2),
        "ma200_slope": round(slope,6), "rvol": round(rvol,2),
        "breadth": round(breadth,1), "volatility": round(volatility,2),
        "price_ma200_pct": round(pct,2),
    }
    if price < ma200 and slope < -0.005:
        stage=5; extra={"selling_climax": volatility>25 and rvol>2.5}
        cond={"가격 < MA200":True,"기울기 < -0.005":slope<-0.005}
    elif pct > 5 and slope > 0.002 and rvol >= 2.0 and breadth >= 50:
        stage=2; extra={"fresh_breakout":was_below}
        cond={"가격 > MA200+5%":pct>5,"기울기 > 0.002":slope>0.002,"RVOL ≥ 2.0":rvol>=2.0,"참여율 ≥ 50%":breadth>=50}
    elif price > ma200 and slope > 0.005 and breadth >= 70:
        stage=3; extra={}
        cond={"가격 > MA200":price>ma200,"기울기 > 0.005":slope>0.005,"참여율 ≥ 70%":breadth>=70}
    elif price > ma200 and abs(slope) <= 0.008 and breadth < 50 and volatility > 18:
        stage=4; extra={"divergence":True}
        cond={"가격 > MA200":price>ma200,"|기울기| ≤ 0.008":abs(slope)<=0.008,"참여율 < 50%":breadth<50,"변동성 > 18%":volatility>18}
    elif abs(pct) <= 5 and abs(slope) <= 0.005 and breadth < 50:
        stage=1; extra={}
        cond={"가격 ±5% 이내":abs(pct)<=5,"|기울기| ≤ 0.005":abs(slope)<=0.005,"참여율 < 50%":breadth<50}
    else:
        extra={"estimated":True}; cond={}
        if price < ma200:                     stage=5
        elif pct > 5 and slope > 0:           stage=2
        elif slope > 0.003 and breadth >= 60: stage=3
        elif price > ma200 and breadth < 60:  stage=4
        else:                                 stage=1
    return {"stage":stage,"indicators":indicators,"extra":extra,"conditions":cond}


# ═══════════════════════════════════════════════════════════════════════════════
# 급등주 탐지 엔진 (SurgeLogicEngine)
# ═══════════════════════════════════════════════════════════════════════════════

class SurgeLogicEngine:
    """5가지 급등 패턴 정량 탐지 엔진"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for span in [5, 10, 20, 60]:
            df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
        df["VOL_5"]  = df["Volume"].rolling(5).mean()
        df["VOL_20"] = df["Volume"].rolling(20).mean()
        df["BB_MID"]   = df["Close"].rolling(20).mean()
        df["BB_STD"]   = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + df["BB_STD"] * 2
        df["BB_LOWER"] = df["BB_MID"] - df["BB_STD"] * 2
        df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan)
        obv = [0]
        for i in range(1, len(df)):
            c, p = df["Close"].iloc[i], df["Close"].iloc[i-1]
            v = df["Volume"].iloc[i]
            obv.append(obv[-1] + v if c > p else (obv[-1] - v if c < p else obv[-1]))
        df["OBV"] = obv
        return df

    @classmethod
    def check_p1_ma_squeeze(cls, df: pd.DataFrame):
        """P1: 이평선 극한 수렴 + 거래량 씨마름"""
        if len(df) < 60: return False, 0
        latest = df.iloc[-1]
        emas = [latest[f"EMA_{s}"] for s in [5,10,20,60] if f"EMA_{s}" in df.columns]
        if not emas: return False, 0
        spread = (max(emas) - min(emas)) / min(emas) * 100 if min(emas) > 0 else 99
        is_squeeze = spread <= 3.0
        is_vol_dry = latest["VOL_5"] <= latest["VOL_20"] * 0.35 if latest["VOL_20"] > 0 else False
        score = 0
        if is_squeeze: score += 50
        if is_vol_dry: score += 40
        if spread <= 1.5: score += 10
        return score >= 70, min(score, 100)

    @classmethod
    def check_p2_accumulation(cls, df: pd.DataFrame):
        """P2: 세력 매집 기준봉 후 눌림목"""
        if len(df) < 30: return False, 0
        recent = df.tail(20).copy()
        target = recent[
            (recent["Close"] / recent["Open"].replace(0, np.nan) >= 1.15) &
            (recent["Volume"] >= recent["VOL_20"] * 5.0)
        ]
        if target.empty: return False, 0
        ref = target.iloc[-1]
        after = df.loc[ref.name + 1:] if ref.name + 1 <= df.index[-1] else pd.DataFrame()
        if after.empty: return False, 0
        center = (ref["High"] + ref["Low"]) / 2
        is_supported = after["Close"].min() >= center
        is_vol_down  = after["Volume"].mean() <= ref["Volume"] * 0.30
        score = 60
        if is_supported: score += 20
        if is_vol_down:  score += 20
        return score >= 70, min(score, 100)

    @classmethod
    def check_p3_obv_divergence(cls, df: pd.DataFrame):
        """P3: OBV 상승 다이버전스 & 쌍바닥"""
        if len(df) < 60: return False, 0
        recent = df.tail(60).reset_index(drop=True)
        mid = len(recent) // 2
        lo1_idx = recent["Close"].iloc[:mid].idxmin()
        lo2_idx = recent["Close"].iloc[mid:].idxmin() + mid
        price1, price2 = recent["Close"].iloc[lo1_idx], recent["Close"].iloc[lo2_idx]
        obv1,   obv2   = recent["OBV"].iloc[lo1_idx],   recent["OBV"].iloc[lo2_idx]
        price_lower = price2 <= price1 * 1.03
        obv_higher  = obv2 > obv1
        score = 0
        if price_lower and obv_higher:
            score += 60
            ratio = abs(obv2 - obv1) / (abs(obv1) + 1) * 100
            if ratio > 5:  score += 20
            if ratio > 15: score += 20
        return score >= 70, min(score, 100)

    @classmethod
    def check_p4_bb_squeeze(cls, df: pd.DataFrame):
        """P4: 볼린저밴드 극한 수렴 후 상단 돌파"""
        if len(df) < 60: return False, 0
        recent = df.tail(min(120, len(df)))
        current = df.iloc[-1]
        bw_series = recent["BB_WIDTH"].dropna()
        if bw_series.empty: return False, 0
        min_bw   = bw_series.min()
        cur_bw   = current["BB_WIDTH"] if pd.notna(current["BB_WIDTH"]) else 99
        is_squeeze   = cur_bw <= min_bw * 1.6
        broke_upper  = current["Close"] >= current["BB_UPPER"] * 0.97 if pd.notna(current["BB_UPPER"]) else False
        vol_surge    = current["VOL_5"] >= current["VOL_20"] * 1.5 if current["VOL_20"] > 0 else False
        score = 0
        if is_squeeze:  score += 35
        if broke_upper: score += 45
        if vol_surge:   score += 20
        return score >= 70, min(score, 100)

    @classmethod
    def check_p5_cup_handle(cls, df: pd.DataFrame):
        """P5: 컵앤핸들 패턴"""
        if len(df) < 90: return False, 0
        cup_data    = df.tail(90).head(60)
        handle_data = df.tail(30)
        cup_high  = cup_data["Close"].max()
        cup_low   = cup_data["Close"].min()
        cup_depth = (cup_high - cup_low) / cup_high * 100 if cup_high > 0 else 0
        handle_high = handle_data["Close"].max()
        handle_low  = handle_data["Close"].min()
        handle_depth = (handle_high - handle_low) / handle_high * 100 if handle_high > 0 else 99
        cur_price = df.iloc[-1]["Close"]
        near_rim  = cur_price >= cup_high * 0.94
        good_cup  = 12 <= cup_depth <= 55
        shallow_handle = handle_depth <= 15
        vol_contract   = handle_data["Volume"].mean() <= cup_data["Volume"].mean() * 0.75
        score = 0
        if good_cup:        score += 30
        if shallow_handle:  score += 25
        if near_rim:        score += 25
        if vol_contract:    score += 20
        return score >= 70, min(score, 100)

    @classmethod
    def scan(cls, df: pd.DataFrame) -> Dict:
        df = cls.calculate_indicators(df)
        patterns = {
            "P1 이평선 응축": cls.check_p1_ma_squeeze(df),
            "P2 매집봉 눌림목": cls.check_p2_accumulation(df),
            "P3 OBV 다이버전스": cls.check_p3_obv_divergence(df),
            "P4 볼린저 수렴돌파": cls.check_p4_bb_squeeze(df),
            "P5 컵앤핸들": cls.check_p5_cup_handle(df),
        }
        top_name = max(patterns, key=lambda k: patterns[k][1])
        top_score = patterns[top_name][1]
        matched = [name for name, (m, _) in patterns.items() if m]
        return {
            "patterns": {n: {"match": m, "score": s} for n, (m, s) in patterns.items()},
            "top_pattern": top_name,
            "top_score": top_score,
            "matched": matched,
            "is_signal": top_score >= 70,
        }


SENSITIVITY_THRESHOLD = {1: 0, 2: 55, 3: 68, 4: 82, 5: 91}
SURGE_STATUS = {
    (90, 101): "🔥 강력 매수",
    (80, 90):  "✅ 트리거 포착",
    (70, 80):  "👀 관찰 진입",
    (0,  70):  "─ 대기",
}

def _surge_status(score: int) -> str:
    for (lo, hi), label in SURGE_STATUS.items():
        if lo <= score < hi:
            return label
    return "─"


@st.cache_data(ttl=300, show_spinner=False)
def run_surge_scan(scope_key: str, sensitivity: int) -> List[Dict]:
    end_dt   = datetime.today().strftime("%Y-%m-%d")
    start_dt = (datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d")

    if scope_key == "_all_":
        universe = {}
        for theme in THEMES.values():
            for tk, info in theme["stocks"].items():
                universe[tk] = info
    else:
        universe = dict(THEMES.get(scope_key, {}).get("stocks", {}))

    threshold = SENSITIVITY_THRESHOLD[sensitivity]
    results = []

    for ticker, (name, exch) in universe.items():
        df = load_stock(ticker, exch, start_dt, end_dt)
        if df is None or len(df) < 60:
            continue
        try:
            res = SurgeLogicEngine.scan(df)
        except Exception:
            continue

        roc_1d = (
            (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            if len(df) > 1 else 0.0
        )
        score = res["top_score"]
        if score < threshold:
            continue

        results.append({
            "종목명":   name,
            "코드":     ticker,
            "패턴":     res["top_pattern"],
            "현재가":   int(df["Close"].iloc[-1]),
            "등락률":   f"{roc_1d:+.1f}%",
            "스코어":   score,
            "신호":     _surge_status(score),
            "_matched": res["matched"],
            "_df":      df,
        })

    results.sort(key=lambda x: x["스코어"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 자연어 해설
# ═══════════════════════════════════════════════════════════════════════════════

def generate_theme_commentary(lifecycle, lss_df, name_map, theme_name) -> str:
    stage = lifecycle["stage"]
    ind   = lifecycle["indicators"]
    info  = STAGE_INFO[stage]
    breadth    = ind.get("breadth", 0)
    slope      = ind.get("ma200_slope", 0)
    rvol       = ind.get("rvol", 1)
    volatility = ind.get("volatility", 0)

    STAGE_DESC = {
        1: "아직 시장의 주목을 받지 못하고 긴 횡보 구간에서 에너지를 비축하는 단계입니다.",
        2: "거래량 폭발과 함께 박스권을 강하게 돌파하는 초기 상승 신호가 포착됐습니다.",
        3: "추세가 완전히 확립된 강세 구간입니다. 대부분의 테마 종목이 일제히 상승하는 폭발적 랠리 단계입니다.",
        4: "고점 부근에서 변동성이 심화되고 있습니다. 후발 종목부터 약세 전환되는 다이버전스 발생 중입니다.",
        5: "추세가 완전히 무너진 하락 구간입니다.",
        0: "데이터가 부족하여 정확한 판단이 어렵습니다.",
    }
    lines = []
    lines.append(f"**{theme_name} 테마**는 현재 **{info['emoji']} Stage {stage} · {info['name']}** 단계입니다.")
    lines.append(STAGE_DESC.get(stage, ""))
    if breadth >= 70:
        lines.append(f"테마 참여율 **{breadth:.0f}%** — 거의 모든 종목이 MA50 위에 있어 광범위한 강세입니다.")
    elif breadth >= 50:
        lines.append(f"테마 참여율 **{breadth:.0f}%** — 절반 이상 종목이 MA50 위입니다.")
    else:
        lines.append(f"테마 참여율 **{breadth:.0f}%** — 소수 종목만 강세를 유지하는 선별적 장세입니다.")
    if slope > 0.005:
        lines.append(f"MA200 기울기({slope:.5f}) 가파르게 상승 중 — 장기 추세 살아있습니다.")
    elif slope < -0.005:
        lines.append(f"MA200 기울기({slope:.5f}) 명확히 하락 중 — 장기 추세 훼손 상태입니다.")
    else:
        lines.append(f"MA200 기울기({slope:.5f}) 거의 평탄 — 방향성 불명확합니다.")
    if volatility > 25:
        lines.append(f"⚠️ **변동성 {volatility:.1f}%** — 매우 높습니다. 포지션 크기 조절 필요.")
    elif volatility > 18:
        lines.append(f"변동성 {volatility:.1f}% — 다소 높아 단기 등락이 심할 수 있습니다.")
    if not lss_df.empty:
        top = lss_df.iloc[0]
        top_name = name_map.get(top["ticker"], top["ticker"])
        roc = top.get("roc")
        roc_str = f" 20일 수익률 **{roc:+.1f}%**를 기록하며" if pd.notna(roc) else ""
        lines.append(f"대장주는 **{top_name}(LSS {top['lss']:.4f})**으로,{roc_str} 테마를 선도하고 있습니다.")
    ADVICE = {
        1: "💡 거래량 증가와 MA200 상향 돌파를 확인한 후 대장주 중심으로 소량 관심 보유를 권합니다.",
        2: "💡 돌파 초기 단계. 대장주(LSS 1~2위) 중심으로 분할 매수 진입을 고려할 수 있습니다.",
        3: "💡 추세 추종이 유효한 구간. 대장주 비중을 높이되 변동성 확대 시 일부 익절하세요.",
        4: "💡 신규 매수 자제. 기존 보유분은 분할 매도를 고려하세요.",
        5: "💡 관망을 권합니다. 투매 클라이막스 이후 단기 반등 시점을 주시하세요.",
        0: "💡 데이터 확보 후 재분석이 필요합니다.",
    }
    lines.append(ADVICE.get(stage, ""))
    return "\n\n".join(lines)


def generate_stock_commentary(row, name, stage) -> str:
    rank = int(row["rank"])
    roc  = row.get("roc")
    beta = row.get("beta")
    wrs  = row.get("wrs")
    RANK = {1: "**테마 대장주**입니다. 테마가 오를 때 가장 먼저·가장 크게 움직입니다.",
            2: "**부대장주 2위**입니다. 대장주와 함께 테마를 이끄는 종목입니다.",
            3: "**부대장주 3위**입니다. 대장주 급등 이후 순환매가 들어오는 종목입니다."}
    lines = [RANK.get(rank, f"테마 내 **{rank}위** 종목입니다.")]
    if pd.notna(roc):
        lines.append(f"최근 20일 수익률 **{roc:+.1f}%**.")
    if pd.notna(beta):
        if beta > 1.5: lines.append(f"베타({beta:.2f}) 높음 — 테마 상승 시 증폭 수익 기대 가능.")
        elif beta < 0.8: lines.append(f"베타({beta:.2f}) 낮음 — 방어적 특성.")
    if pd.notna(wrs):
        lines.append(f"장기 WRS **{wrs:.1f}%**.")
    STAGE_STOCK = {1:"바닥 매집기 — 소량 관심 보유 수준.", 2:"돌파 초기 — 분할 매수 1순위 후보.",
                   3:"주도 상승 중 — 비중 적극 유지.", 4:"과열 구간 — 신규 진입 자제.",
                   5:"하락 구간 — 보유 중이면 손절 기준 재점검."}
    lines.append(f"📌 {STAGE_STOCK.get(stage, '')}")
    return " ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 차트 함수 (라이트모드)
# ═══════════════════════════════════════════════════════════════════════════════

def make_theme_chart(theme_df, theme_name, stage) -> go.Figure:
    if theme_df.empty: return go.Figure()
    ACCENT = {1:"#6b7280",2:"#2563eb",3:"#16a34a",4:"#d97706",5:"#dc2626"}
    accent = ACCENT.get(stage, "#2563eb")
    ma200_s = theme_df["Close"].rolling(200).mean()
    ma50_s  = theme_df["Close"].rolling(50).mean()
    plot    = theme_df.tail(250)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    r, g, b = int(accent[1:3],16), int(accent[3:5],16), int(accent[5:7],16)
    fig.add_trace(go.Scatter(x=plot.index, y=plot["Close"], name="테마지수",
        line=dict(color=accent, width=2), fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.08)"), row=1, col=1)
    m200 = ma200_s.tail(250)
    fig.add_trace(go.Scatter(x=m200.index, y=m200.values, name="MA200",
        line=dict(color="#b45309", width=2)), row=1, col=1)
    m50 = ma50_s.tail(250)
    fig.add_trace(go.Scatter(x=m50.index, y=m50.values, name="MA50",
        line=dict(color="#2563eb", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"], name="거래량",
        marker_color=accent, opacity=0.35), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{theme_name} 테마 지수</b>", font=dict(color=C["font"], size=15)),
        paper_bgcolor=C["paper"], plot_bgcolor=C["plot"], font=dict(color=C["font"]),
        xaxis_rangeslider_visible=False, height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor=C["leg_bg"]),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


def make_stock_chart(df, name) -> go.Figure:
    ma200_s = df["Close"].rolling(200).mean()
    ma50_s  = df["Close"].rolling(50).mean()
    plot    = df.tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=plot.index,
        open=plot["Open"], high=plot["High"], low=plot["Low"], close=plot["Close"],
        name=name, increasing_line_color=C["up"], decreasing_line_color=C["down"]),
        row=1, col=1)
    m200 = ma200_s.tail(200)
    fig.add_trace(go.Scatter(x=m200.index, y=m200.values, name="MA200",
        line=dict(color="#b45309", width=2)), row=1, col=1)
    m50 = ma50_s.tail(200)
    fig.add_trace(go.Scatter(x=m50.index, y=m50.values, name="MA50",
        line=dict(color="#2563eb", width=1.5, dash="dot")), row=1, col=1)
    vol_colors = [C["up"] if c >= o else C["down"]
                  for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"], name="거래량",
        marker_color=vol_colors, opacity=0.5), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{name}</b>", font=dict(color=C["font"], size=13)),
        paper_bgcolor=C["paper"], plot_bgcolor=C["plot"], font=dict(color=C["font"]),
        xaxis_rangeslider_visible=False, height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor=C["leg_bg"]),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


def make_radar(lss_df, name_map) -> go.Figure:
    cats = ["ROC", "거래대금", "회전율", "베타", "가중RS"]
    keys = ["roc_norm","trading_value_norm","turnover_ratio_norm","beta_norm","wrs_norm"]
    colors = ["#d97706","#9ca3af","#b45309"]
    fig = go.Figure()
    for i, row in lss_df[lss_df["rank"] <= 3].iterrows():
        vals = [float(row.get(k, 0) or 0) for k in keys] + [float(row.get(keys[0], 0) or 0)]
        c = colors[int(row["rank"])-1]
        r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(color=c, width=2),
            name=f"#{int(row['rank'])} {name_map.get(row['ticker'], row['ticker'])}",
        ))
    fig.update_layout(
        polar=dict(bgcolor=C["plot"],
            radialaxis=dict(visible=True, range=[0,1], gridcolor=C["grid"],
                            tickfont=dict(color=C["font"], size=9)),
            angularaxis=dict(gridcolor=C["grid"], tickfont=dict(color=C["font"]))),
        paper_bgcolor=C["paper"],
        legend=dict(font=dict(color=C["font"]), bgcolor=C["leg_bg"]),
        title=dict(text="<b>LSS 지표 레이더</b>", font=dict(color=C["font"])),
        height=320, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI 컴포넌트 — 테마주 분석
# ═══════════════════════════════════════════════════════════════════════════════

def render_stage_card(lifecycle):
    stage = lifecycle["stage"]
    info  = STAGE_INFO[stage]
    ind   = lifecycle["indicators"]
    extra = lifecycle["extra"]
    cond  = lifecycle["conditions"]
    bg    = STAGE_BG.get(stage, "#f3f4f6")
    color = info["color"]

    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:12px;'
        f'padding:18px;margin-bottom:14px;">'
        f'<div style="font-size:2rem">{info["emoji"]}</div>'
        f'<div style="font-size:1.4rem;font-weight:900;color:{color};">'
        f'Stage {stage} · {info["name"]}</div>'
        f'<div style="color:{color};opacity:0.7;font-size:0.83rem;">{info["en"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if extra.get("selling_climax"):
        st.error("⚡ 투매 클라이막스 감지!")
    if extra.get("fresh_breakout"):
        st.success("✅ 신선한 돌파 확인")
    if extra.get("divergence"):
        st.warning("🔀 다이버전스 감지")
    if extra.get("estimated"):
        st.info("⚠️ 근사 판별 적용")

    if not ind: return

    pct   = ind.get("price_ma200_pct", 0)
    slope = ind.get("ma200_slope", 0)
    brd   = ind.get("breadth", 0)
    rvol  = ind.get("rvol", 1)
    vlt   = ind.get("volatility", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("현재가 vs MA200", f"{ind.get('price',0):.1f}", f"{pct:+.1f}%", help=TIP["ma200"])
    c2.metric("MA200 기울기", f"{slope:.5f}",
              "상승↑" if slope>0.002 else "하락↓" if slope<-0.002 else "횡보→", help=TIP["slope"])
    c3.metric("테마 참여율", f"{brd:.1f}%",
              "확산" if brd>=70 else "수축" if brd<50 else "중립", help=TIP["breadth"])
    c4, c5, c6 = st.columns(3)
    c4.metric("RVOL", f"{rvol:.2f}x", "거래량 급증" if rvol>=2.0 else "보통", help=TIP["rvol"])
    c5.metric("20일 변동성", f"{vlt:.1f}%", "과열" if vlt>25 else "주의" if vlt>18 else "안정", help=TIP["volatility"])
    c6.metric("MA50", f"{ind.get('ma50',0):.1f}", help=TIP["ma50"])

    if cond:
        st.markdown("**판별 조건**")
        cols = st.columns(2)
        for i, (label, met) in enumerate(cond.items()):
            cols[i%2].markdown(f"{'✅' if met else '❌'} {label}")


def render_lss_cards(lss_df, name_map, stocks, stage):
    RANK_ICON  = {1:"👑",2:"🥈",3:"🥉"}
    RANK_LABEL = {1:"대장주",2:"부대장 1",3:"부대장 2"}
    for _, row in lss_df.iterrows():
        tk    = row["ticker"]
        name  = name_map.get(tk, tk)
        rank  = int(row["rank"])
        score = float(row["lss"])
        close = int(row["close"])
        is_top= bool(row["is_leader"])
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown(f"### {RANK_ICON.get(rank, f'#{rank}')} {name}")
                st.caption(f"`{tk}` | 현재가 **{close:,}원**" + (f" 🔥 **{RANK_LABEL.get(rank,'')}**" if is_top else ""))
            with h2:
                st.metric("LSS Score", f"{score:.4f}", help=TIP["lss"])
            st.progress(float(score), text=f"LSS {score:.4f}")
            st.markdown(f"> {generate_stock_commentary(row, name, stage)}")
            with st.expander(f"📊 {name} 상세 & 차트"):
                d1, d2, d3 = st.columns(3)
                roc = row.get("roc")
                d1.metric("ROC(20일)", f"{roc:.1f}%" if pd.notna(roc) else "─", help=TIP["roc"])
                tv = row.get("trading_value")
                d2.metric("거래대금", f"{tv/1e6:.0f}만원" if pd.notna(tv) else "─", help=TIP["tv"])
                beta = row.get("beta")
                d3.metric("베타", f"{beta:.2f}" if pd.notna(beta) else "─", help=TIP["beta"])
                d4, d5, _ = st.columns(3)
                tr = row.get("turnover_ratio")
                d4.metric("회전율", f"{tr:.4f}" if pd.notna(tr) else "─", help=TIP["turnover"])
                wrs = row.get("wrs")
                d5.metric("WRS", f"{wrs:.1f}%" if pd.notna(wrs) else "─", help=TIP["wrs"])
                if tk in stocks:
                    st.plotly_chart(make_stock_chart(stocks[tk], name), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UI 컴포넌트 — 급등주 탐지
# ═══════════════════════════════════════════════════════════════════════════════

def render_surge_detector(scope_key: str, sensitivity: int):
    st.markdown("## 🎯 급등주 탐지 시스템")
    scope_name = "전체 테마" if scope_key == "_all_" else THEMES.get(scope_key, {}).get("name", scope_key)
    stock_count = (
        sum(len(t["stocks"]) for t in THEMES.values())
        if scope_key == "_all_"
        else len(THEMES.get(scope_key, {}).get("stocks", {}))
    )
    st.caption(f"스캔 범위: **{scope_name}** ({stock_count}개 종목) | 엄격도: {sensitivity}/5")

    with st.spinner(f"📡 {stock_count}개 종목 패턴 분석 중... (30~90초 소요)"):
        results = run_surge_scan(scope_key, sensitivity)

    if not results:
        st.info("현재 설정 기준에 해당하는 종목이 없습니다. 엄격도를 낮춰 다시 시도해보세요.")
        return

    st.success(f"**{len(results)}개 종목** 탐지 완료")

    # 결과 요약 테이블
    display_df = pd.DataFrame([{
        "종목명": r["종목명"], "코드": r["코드"],
        "탐지 패턴": r["패턴"], "현재가": f"{r['현재가']:,}원",
        "등락률": r["등락률"], "스코어": r["스코어"], "신호": r["신호"],
    } for r in results])
    st.dataframe(
        display_df,
        column_config={"스코어": st.column_config.ProgressColumn(
            "매칭 스코어", min_value=0, max_value=100, format="%d점")},
        use_container_width=True, hide_index=True,
    )

    # 개별 종목 상세
    st.markdown("---")
    st.markdown("### 📋 종목별 상세")
    for r in results:
        score = r["스코어"]
        color = "#16a34a" if score >= 85 else ("#2563eb" if score >= 70 else "#6b7280")
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {r['신호']} {r['종목명']} `{r['코드']}`")
                st.caption(f"현재가 **{r['현재가']:,}원** | 등락률 {r['등락률']} | 탐지 패턴: {r['패턴']}")
                if r["_matched"]:
                    st.markdown("**매칭된 패턴:** " + " · ".join(r["_matched"]))
            with c2:
                st.markdown(
                    f'<div style="text-align:center;padding:10px;background:#f8fafc;'
                    f'border-radius:8px;border:2px solid {color};">'
                    f'<div style="font-size:1.8rem;font-weight:900;color:{color}">{score}</div>'
                    f'<div style="font-size:0.75rem;color:#6b7280">/ 100점</div></div>',
                    unsafe_allow_html=True,
                )
            with st.expander(f"📈 {r['종목명']} 차트 보기"):
                st.plotly_chart(make_stock_chart(r["_df"], r["종목명"]), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UI 컴포넌트 — Stage 2·3 필터 뷰
# ═══════════════════════════════════════════════════════════════════════════════

def render_stage_filter_view():
    st.markdown("## 🎯 Stage 2·3 핵심 테마")
    if not THEMES_STAGES:
        st.warning("아직 생애주기 데이터가 없습니다. 왼쪽 **🔄 테마 종목 재분석** 버튼을 먼저 실행해주세요.")
        return

    stage_2_3 = {k: THEMES[k] for k in THEMES_STAGES if THEMES_STAGES[k] in (2,3) and k in THEMES}
    all_st    = {k: THEMES[k] for k in THEMES_STAGES if k in THEMES}

    if stage_2_3:
        st.success(f"**{len(stage_2_3)}개 테마**가 현재 모멘텀 구간(Stage 2·3)입니다.")
        cols = st.columns(min(len(stage_2_3), 2))
        for i, (key, theme) in enumerate(stage_2_3.items()):
            stage = THEMES_STAGES[key]
            info  = STAGE_INFO[stage]
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(
                        f"### {info['emoji']} {theme['name']}  \n"
                        f"<span style='color:{info['color']};font-weight:700'>"
                        f"Stage {stage} · {info['name']}</span>",
                        unsafe_allow_html=True,
                    )
                    for j, (code, (name, _)) in enumerate(list(theme["stocks"].items())[:5]):
                        medal = {0:"👑",1:"🥈",2:"🥉"}.get(j, f"`{j+1}`")
                        st.markdown(f"{medal} {name} `{code}`")
                    st.caption(f"📅 {_THEMES_UPDATED_AT or '미확인'}")
    else:
        st.info("현재 Stage 2·3 테마가 없습니다. 전체 현황을 아래에서 확인하세요.")

    st.markdown("---")
    st.markdown("### 📊 전체 테마 생애주기 현황")
    rows = []
    for key, theme in all_st.items():
        stage = THEMES_STAGES.get(key, 0)
        info  = STAGE_INFO.get(stage, STAGE_INFO[0])
        top1  = list(theme["stocks"].values())
        rows.append({
            "테마": theme["name"],
            "단계": f"{info['emoji']} Stage {stage} · {info['name']}",
            "대장주": top1[0][0] if top1 else "─",
            "주목": "✅" if stage in (2,3) else ("⚠️" if stage==4 else "─"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"마지막 업데이트: {_THEMES_UPDATED_AT or '미실시'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:8px 0 14px">'
            '<div style="font-size:1.8rem">📊</div>'
            '<div style="font-size:1.05rem;font-weight:bold">테마주 퀀트 엔진</div>'
            '<div style="font-size:0.75rem;color:#6b7280">LSS & 생애주기 | 급등주 탐지</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        mode = st.radio(
            "모드 선택",
            ["📊 테마주 분석", "🎯 급등주 탐지"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.divider()

        # ── 테마주 분석 모드 ────────────────────────────────────────────────
        if mode == "📊 테마주 분석":
            theme_opts = {tid: t["name"] for tid, t in THEMES.items()}
            selected = st.selectbox("테마 선택", list(theme_opts.keys()),
                                    format_func=lambda x: theme_opts[x])
            with st.expander("📋 구성 종목"):
                for tk, (name, exch) in THEMES[selected]["stocks"].items():
                    st.markdown(f"• `{tk}` {name} ({exch})")

            st.divider()
            run = st.button("🔍 분석 실행", use_container_width=True, type="primary")

            st.divider()
            if _THEMES_UPDATED_AT:
                st.caption(f"🕐 마지막 재분석: {_THEMES_UPDATED_AT}")
            else:
                st.caption("📌 기본 테마 사용 중")
            refresh = st.button("🔄 테마 종목 재분석", use_container_width=True,
                                help="네이버 크롤링 → TF-IDF → LSS. 수 분 소요.")
            st.divider()
            stage_filter = st.button("🎯 Stage 2·3 핵심 테마만 보기", use_container_width=True)

            with st.expander("ℹ️ LSS란?"):
                st.markdown("""
**LSS**는 테마 내 **대장주** 식별 점수입니다.

생애주기 **Stage 2·3**에서 LSS **1~2위** 종목에 집중하는 것이 핵심 전략입니다.

가중치: ROC 30% | 거래대금 30% | 회전율 15% | WRS 15% | 베타 10%
""")

            return mode, selected, run, refresh, stage_filter, "_all_", 3, False

        # ── 급등주 탐지 모드 ────────────────────────────────────────────────
        else:
            scope_map = {"전체 테마 종목": "_all_"} | {t["name"]: k for k, t in THEMES.items()}
            scope_label = st.selectbox("스캔 범위", list(scope_map.keys()))
            scope_key   = scope_map[scope_label]

            st.markdown("**탐지 엄격도**")
            sensitivity = st.slider("", 1, 5, 3, label_visibility="collapsed")
            sens_labels = {1:"매우 느슨",2:"느슨",3:"보통",4:"엄격",5:"핵심만"}
            st.caption(f"현재: **{sens_labels[sensitivity]}** (스코어 {SENSITIVITY_THRESHOLD[sensitivity]}점↑)")

            st.divider()
            scan_btn = st.button("🔍 스캔 실행", use_container_width=True, type="primary")

            st.divider()
            with st.expander("ℹ️ 5가지 탐지 패턴"):
                st.markdown("""
**P1** 이평선 극한 수렴 + 거래량 씨마름
**P2** 세력 매집봉 출현 후 눌림목
**P3** OBV 상승 다이버전스 & 쌍바닥
**P4** 볼린저밴드 수렴 후 상단 돌파
**P5** 컵앤핸들 패턴
""")

            return mode, None, False, False, False, scope_key, sensitivity, scan_btn


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mode, selected, run, refresh, stage_filter, scope_key, sensitivity, scan_btn = render_sidebar()

    if "data" not in st.session_state:
        st.session_state.data = None

    # ── 급등주 탐지 모드 ────────────────────────────────────────────────────
    if mode == "🎯 급등주 탐지":
        if scan_btn:
            render_surge_detector(scope_key, sensitivity)
        else:
            st.markdown("""
            <div style="text-align:center;padding:80px 0">
                <div style="font-size:4rem">🎯</div>
                <h2>급등주 탐지 시스템</h2>
                <p style="color:#6b7280">왼쪽에서 스캔 범위와 엄격도를 설정하고 <b>스캔 실행</b>을 클릭하세요.</p>
                <p style="color:#9ca3af;font-size:0.9rem">5가지 기술적 패턴(이평선 수렴, 매집봉, OBV 다이버전스, 볼린저 수렴, 컵앤핸들)을 실시간 스캔합니다.</p>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── 테마 재분석 ─────────────────────────────────────────────────────────
    if refresh:
        st.markdown("## 🔄 테마 종목 재분석")
        st.info("네이버 크롤링 → TF-IDF 매칭 → 주가 로드 → LSS 선정 순으로 진행됩니다. 수 분 소요됩니다.")
        with st.status("재분석 진행 중...", expanded=True) as status:
            try:
                from crawler import update_themes as _run_crawler
                try:
                    _gh_token  = st.secrets.get("GITHUB_TOKEN", "")
                    _gh_repo   = st.secrets.get("GITHUB_REPO", "")
                    _gh_branch = st.secrets.get("GITHUB_BRANCH", "main")
                except Exception:
                    _gh_token = _gh_repo = _gh_branch = ""
                logs: List[str] = []
                log_box = st.empty()
                def _on_progress(msg):
                    logs.append(msg)
                    log_box.code("\n".join(logs[-30:]), language=None)
                result = _run_crawler(
                    progress_callback=_on_progress,
                    github_token=_gh_token,
                    github_repo=_gh_repo,
                    github_branch=_gh_branch or "main",
                )
                if result:
                    status.update(label=f"✅ 재분석 완료! ({len(result)}개 테마)", state="complete")
                    st.cache_data.clear()
                    time.sleep(1.5)
                    st.rerun()
                else:
                    status.update(label="⚠️ 결과 없음", state="error")
            except ImportError:
                status.update(label="❌ crawler.py 없음", state="error")
            except Exception as e:
                status.update(label=f"❌ 오류: {e}", state="error")
        return

    # ── Stage 2·3 필터 뷰 ──────────────────────────────────────────────────
    if stage_filter:
        render_stage_filter_view()
        return

    # ── 시작 화면 ───────────────────────────────────────────────────────────
    if not run and st.session_state.data is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 0">
            <div style="font-size:4rem">📊</div>
            <h2>테마주 퀀트 대시보드</h2>
            <p style="color:#6b7280">왼쪽에서 테마를 선택하고 <b>분석 실행</b>을 클릭하세요.</p>
            <p style="color:#9ca3af;font-size:0.9rem">각 지표 옆 ? 아이콘으로 용어 설명을 볼 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── 분석 실행 ───────────────────────────────────────────────────────────
    if run:
        st.cache_data.clear()
        with st.spinner("📡 데이터 수집 중... (최대 60초)"):
            raw = load_theme_data(selected)
        stocks, benchmark = raw["stocks"], raw["benchmark"]
        if len(stocks) < 2:
            st.error("분석 가능 종목이 부족합니다.")
            return
        lss_df    = calc_lss(stocks, benchmark)
        theme_idx = build_theme_index(stocks)
        lifecycle = determine_stage(theme_idx, stocks)
        name_map  = {tk: v[0] for tk, v in THEMES[selected]["stocks"].items()}
        st.session_state.data = {
            "raw": raw, "lss_df": lss_df, "theme_idx": theme_idx,
            "lifecycle": lifecycle, "name_map": name_map, "stocks": stocks,
        }

    d = st.session_state.data
    if d is None: return

    raw       = d["raw"]
    lss_df    = d["lss_df"]
    theme_idx = d["theme_idx"]
    lifecycle = d["lifecycle"]
    name_map  = d["name_map"]
    stocks    = d["stocks"]
    stage     = lifecycle["stage"]

    ok_cnt = sum(1 for s in raw["fetch_status"] if s["ok"])
    st.markdown(
        f"## {raw['theme_name']} 테마 분석  \n"
        f"<span style='color:#6b7280;font-size:0.88rem'>"
        f"분석일: {raw['analysis_date']} | 확보 종목: {ok_cnt}종목</span>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 종합 리포트", "🏆 LSS 대장주 순위", "🔬 데이터 상세"])

    with tab1:
        commentary = generate_theme_commentary(lifecycle, lss_df, name_map, raw["theme_name"])
        with st.container(border=True):
            st.markdown("### 🤖 AI 종합 분석 의견")
            st.markdown(commentary)
        st.markdown("---")
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 생애주기 단계")
            render_stage_card(lifecycle)
        with right:
            st.markdown("#### 테마 지수 차트 (최근 250일)")
            st.plotly_chart(make_theme_chart(theme_idx, raw["theme_name"], stage),
                            use_container_width=True)

    with tab2:
        if lss_df.empty:
            st.warning("LSS 분석 데이터가 없습니다.")
        else:
            st.info("**LSS 해석:** 이 점수는 테마가 오를 때 **가장 앞서 달리는 종목**을 찾는 지표입니다. Stage 2·3 구간에서 1~2위 종목에 집중하세요.")
            left2, right2 = st.columns([1, 1])
            with left2:
                render_lss_cards(lss_df, name_map, stocks, stage)
            with right2:
                st.plotly_chart(make_radar(lss_df, name_map), use_container_width=True)
                st.markdown("#### 전체 점수표")
                disp = lss_df[["ticker","rank","lss","roc","trading_value","beta","wrs"]].copy()
                disp["name"] = disp["ticker"].map(name_map)
                disp = disp[["rank","name","ticker","lss","roc","trading_value","beta","wrs"]]
                disp.columns = ["순위","종목명","티커","LSS","ROC(%)","거래대금","베타","WRS(%)"]
                disp["LSS"] = disp["LSS"].round(4)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### 데이터 수집 현황")
        fetch_df = pd.DataFrame([{
            "티커":s["ticker"],"종목명":s["name"],
            "상태":"✅ 성공" if s["ok"] else "❌ 실패","확보 일수":s["days"]}
            for s in raw["fetch_status"]])
        st.dataframe(fetch_df, use_container_width=True, hide_index=True)
        ind = lifecycle.get("indicators", {})
        if ind:
            st.markdown("#### 생애주기 핵심 지표")
            ind_df = pd.DataFrame([
                {"지표":"현재가(정규화)","값":ind.get("price"),"기준":"MA200 대비 ±5%"},
                {"지표":"MA200","값":ind.get("ma200"),"기준":"200일 이동평균"},
                {"지표":"MA200 기울기","값":ind.get("ma200_slope"),"기준":"0.005 / -0.005"},
                {"지표":"RVOL","값":ind.get("rvol"),"기준":"2.0 (돌파) / 2.5 (투매)"},
                {"지표":"테마 참여율(%)","값":ind.get("breadth"),"기준":"50% / 70%"},
                {"지표":"20일 변동성(%)","값":ind.get("volatility"),"기준":"18% / 25%"},
            ])
            st.dataframe(ind_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
