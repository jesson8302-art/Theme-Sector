"""
테마주 대장주 & 생애주기 판별 퀀트 대시보드
Streamlit Cloud 단독 실행 버전

실행: python -m streamlit run app.py
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

# ─── 테마 정의 ────────────────────────────────────────────────────────────────
THEMES: Dict[str, Dict] = {
    "ai_semiconductor": {
        "name": "AI/반도체",
        "stocks": {
            "005930": ("삼성전자",   "KS"),
            "000660": ("SK하이닉스", "KS"),
            "042700": ("한미반도체", "KS"),
            "336370": ("솔브레인",   "KQ"),
            "005290": ("동진쎄미켐", "KS"),
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
THEMES_STAGES: Dict[str, int] = {}   # {theme_key: stage_number}


def _try_load_dynamic_themes() -> None:
    """themes_data.json이 있으면 THEMES·THEMES_STAGES를 동적 데이터로 덮어씁니다."""
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

STAGE_INFO = {
    0: {"name": "데이터 부족",   "en": "Insufficient",  "emoji": "❓", "color": "#808080"},
    1: {"name": "바닥 매집기",   "en": "Basing",        "emoji": "🧱", "color": "#6c757d"},
    2: {"name": "모멘텀 돌파기", "en": "Breakout",      "emoji": "🚀", "color": "#0d6efd"},
    3: {"name": "주도 상승기",   "en": "Advancing",     "emoji": "📈", "color": "#198754"},
    4: {"name": "과열 분배기",   "en": "Distribution",  "emoji": "⚠️", "color": "#fd7e14"},
    5: {"name": "항복 투매기",   "en": "Declining",     "emoji": "📉", "color": "#dc3545"},
}

# 툴팁 텍스트 모음
TIP = {
    "lss":       "LSS(대장주 점수): 테마 내 종목의 종합 주도력 점수(0~1). 높을수록 테마 상승 시 가장 먼저·가장 크게 움직이는 대장주입니다. ROC·거래대금·회전율·베타·WRS 5개 지표를 가중 합산해 산출합니다.",
    "roc":       "ROC(Rate of Change): 최근 20영업일 주가 등락률(%). 테마가 움직일 때 가장 탄력적으로 반응하는 종목을 찾는 데 사용합니다.",
    "tv":        "거래대금: 최근 20영업일 평균 (종가×거래량). 시장의 관심도와 유동성을 평가합니다. 거래대금이 클수록 큰손의 개입이 활발함을 의미합니다.",
    "turnover":  "주식 거래 회전율: 최근 20일 총 거래량 ÷ 상장주식수. 손바뀜이 얼마나 활발히 일어났는지를 측정합니다.",
    "beta":      "베타(Beta): KOSPI 대비 개별 종목의 가격 민감도. 베타 2.0이면 시장이 1% 오를 때 이 종목은 약 2% 움직임을 의미합니다. 테마 상승 시 고베타 종목이 더 유리합니다.",
    "wrs":       "가중 상대강도(WRS): 0.7×120일 수익률 + 0.3×240일 수익률. 6~12개월 장기 추세 강도를 평가합니다. 높을수록 장기 상승 추세가 강합니다.",
    "ma200":     "MA200(200일 이동평균선): 장기 추세 기준선. 가격이 MA200 위이면 장기 강세, 아래이면 약세로 해석합니다.",
    "slope":     "MA200 기울기: 200일선의 상승·하락 기울기. 0.005 이상이면 가파른 상승, -0.005 이하면 명확한 하락 추세입니다.",
    "rvol":      "RVOL(상대 거래량): 최근 5일 평균 ÷ 최근 40일 평균 거래량. 2.0 이상이면 거래량 폭발로, 돌파 신호로 활용합니다.",
    "breadth":   "테마 참여율: 테마 내 종목 중 MA50(50일선) 위에 있는 종목 비율(%). 70% 이상이면 광범위한 상승 확산, 50% 미만이면 소수 종목만 강세.",
    "volatility":"20일 변동성: (고가-저가)/평균가의 20일 평균(%). 18% 초과 시 과열 주의, 25% 초과+거래량 급증이면 투매 클라이막스 신호.",
    "ma50":      "MA50(50일 이동평균선): 중기 추세 기준선. 가격이 MA50 위이면 중기 강세 구간입니다.",
}


# ─── 데이터 로딩 ──────────────────────────────────────────────────────────────

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
            if df is None or df.empty:
                return None
            df = _flatten(df)
            df.index = pd.to_datetime(df.index)
            if any(c not in df.columns for c in required):
                return None
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
    theme    = THEMES[theme_id]
    end_dt   = datetime.today().strftime("%Y-%m-%d")
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
        "fetch_status": fetch_status, "analysis_date": end_dt,
    }


# ─── LSS 엔진 ─────────────────────────────────────────────────────────────────

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
    result["rank"]      = range(1, len(result)+1)
    result["is_leader"] = result["rank"] <= 3
    return result


# ─── 생애주기 엔진 ────────────────────────────────────────────────────────────

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
    ma200      = float(ma200_s.iloc[-1])
    ma50       = float(ma50_s.iloc[-1]) if not ma50_s.empty else price
    slope      = calc_ma_slope(ma200_s)
    rvol       = calc_rvol(theme_df["Volume"])
    breadth    = calc_breadth(stocks)
    volatility = calc_volatility(theme_df)
    pct        = (price/ma200 - 1)*100 if ma200 > 0 else 0.0
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
        stage=5; climax=volatility>25 and rvol>2.5
        extra={"selling_climax":climax}
        cond={"가격 < MA200":True,"기울기 < -0.005":slope<-0.005,"변동성 > 25%":volatility>25,"RVOL > 2.5":rvol>2.5}
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


# ─── 자연어 해설 생성 ─────────────────────────────────────────────────────────

def generate_theme_commentary(lifecycle: Dict, lss_df: pd.DataFrame,
                               name_map: Dict, theme_name: str) -> str:
    stage = lifecycle["stage"]
    ind   = lifecycle["indicators"]
    info  = STAGE_INFO[stage]
    breadth    = ind.get("breadth", 0)
    slope      = ind.get("ma200_slope", 0)
    rvol       = ind.get("rvol", 1)
    volatility = ind.get("volatility", 0)
    pct        = ind.get("price_ma200_pct", 0)

    STAGE_DESC = {
        1: "아직 시장의 주목을 받지 못하고 긴 횡보 구간에서 에너지를 비축하는 단계입니다. 급등보다는 조용한 매집이 진행되는 시기로, 큰 상승이 나오기 전 선행 진입을 노리는 투자자에게 관심이 필요한 구간입니다.",
        2: "거래량 폭발과 함께 박스권을 강하게 돌파하는 초기 상승 신호가 포착됐습니다. 테마 모멘텀이 살아나는 구간으로, 대장주 중심의 초기 진입 기회일 수 있습니다. 단, 가짜 돌파(false breakout) 가능성도 있어 거래량 지속 여부 확인이 중요합니다.",
        3: "추세가 완전히 확립된 강세 구간입니다. 대부분의 테마 종목이 일제히 상승하는 폭발적 랠리 단계로, 추세 추종 전략이 가장 유효한 시기입니다. 대장주 중심으로 매수 비중을 높이는 것이 효과적입니다.",
        4: "고점 부근에서 변동성이 심화되고 있습니다. 지수나 대장주는 버티지만 후발 종목부터 약세로 전환되는 다이버전스가 발생 중입니다. 신규 진입보다는 기존 보유 비중 축소와 리스크 관리가 중요한 시점입니다.",
        5: "추세가 완전히 무너진 하락 구간입니다. 공포 심리에 의한 투매가 진행 중이며, 저점 매수보다는 관망이 안전합니다. 다만 투매 클라이막스(거래량 폭발+급락) 이후 단기 반등 기회를 노릴 수 있습니다.",
        0: "데이터가 부족하여 정확한 판단이 어렵습니다.",
    }

    lines = []
    lines.append(f"**{theme_name} 테마**는 현재 **{info['emoji']} Stage {stage} · {info['name']}** 단계입니다.")
    lines.append(STAGE_DESC.get(stage, ""))

    # 참여율
    if breadth >= 70:
        lines.append(f"테마 참여율이 **{breadth:.0f}%**로 거의 모든 종목이 MA50 위에 있어 광범위한 강세 흐름을 확인할 수 있습니다.")
    elif breadth >= 50:
        lines.append(f"테마 참여율이 **{breadth:.0f}%**로 절반 이상 종목이 MA50 위에 위치해 있습니다.")
    else:
        lines.append(f"테마 참여율이 **{breadth:.0f}%**로 소수 종목만 강세를 유지하는 선별적 장세입니다.")

    # 기울기
    if slope > 0.005:
        lines.append(f"MA200 기울기({slope:.5f})가 가파르게 상승 중으로, 장기 추세가 강하게 살아있습니다.")
    elif slope > 0:
        lines.append(f"MA200 기울기({slope:.5f})가 완만히 상승 중입니다.")
    elif slope < -0.005:
        lines.append(f"MA200 기울기({slope:.5f})가 명확하게 하락 중으로, 장기 추세가 훼손된 상태입니다.")
    else:
        lines.append(f"MA200 기울기({slope:.5f})가 거의 평탄해 방향성이 불명확합니다.")

    # 변동성
    if volatility > 25:
        lines.append(f"⚠️ **변동성이 {volatility:.1f}%로 매우 높습니다.** 단기 급등락 위험이 크므로 포지션 크기 조절이 필요합니다.")
    elif volatility > 18:
        lines.append(f"변동성({volatility:.1f}%)이 다소 높아 단기 등락이 심할 수 있습니다.")
    else:
        lines.append(f"변동성({volatility:.1f}%)은 안정적인 수준입니다.")

    # 대장주 요약
    if not lss_df.empty:
        top = lss_df.iloc[0]
        top_name = name_map.get(top["ticker"], top["ticker"])
        roc = top.get("roc")
        roc_str = f" 20일 수익률 **{roc:+.1f}%**를 기록하며" if pd.notna(roc) else ""
        lines.append(f"대장주는 **{top_name}(LSS {top['lss']:.4f})**으로,{roc_str} 테마를 선도하고 있습니다.")

    # 투자 제언
    ADVICE = {
        1: "💡 **투자 제언:** 아직 진입 시기는 아닙니다. 거래량 증가와 MA200 상향 돌파를 확인한 후 대장주 중심으로 소량 관심 보유를 권합니다.",
        2: "💡 **투자 제언:** 돌파 초기 단계입니다. 대장주(LSS 1~2위) 중심으로 분할 매수 진입을 고려할 수 있습니다. 손절선은 MA200 이하로 설정하세요.",
        3: "💡 **투자 제언:** 추세 추종 전략이 유효한 구간입니다. 대장주 비중을 높이되, 변동성 확대 시 일부 익절하며 리스크를 관리하세요.",
        4: "💡 **투자 제언:** 신규 매수는 자제하고 기존 보유분은 분할 매도를 고려하세요. 대장주가 MA200 아래로 이탈하면 전량 매도 검토가 필요합니다.",
        5: "💡 **투자 제언:** 관망을 권합니다. 투매 클라이막스(극단적 거래량+급락) 이후 단기 기술적 반등 시점을 노릴 수 있으나, 반등 이후 재하락 가능성도 염두에 두세요.",
        0: "💡 **투자 제언:** 데이터 확보 후 재분석이 필요합니다.",
    }
    lines.append(ADVICE.get(stage, ""))

    return "\n\n".join(lines)


def generate_stock_commentary(row: pd.Series, name: str, stage: int) -> str:
    roc  = row.get("roc")
    beta = row.get("beta")
    wrs  = row.get("wrs")
    tv   = row.get("trading_value")
    rank = int(row["rank"])

    lines = []

    # 순위 해석
    RANK_DESC = {
        1: f"**테마 대장주**입니다. 테마가 상승할 때 가장 먼저, 가장 크게 움직이는 핵심 종목으로, 테마 투자 시 1순위로 주목해야 할 종목입니다.",
        2: f"**테마 부대장주 2위**입니다. 대장주와 함께 테마 흐름을 이끄는 종목으로, 대장주 부재 시 대안으로 활용 가능합니다.",
        3: f"**테마 부대장주 3위**입니다. 대장주 급등 이후 순환 매수세가 들어오는 종목입니다.",
    }
    if rank in RANK_DESC:
        lines.append(RANK_DESC[rank])
    else:
        lines.append(f"테마 내 **{rank}위** 종목으로, 대장주 다음 차순위 상승 흐름에 합류하는 패턴을 보입니다.")

    # ROC
    if pd.notna(roc):
        if roc > 20:
            lines.append(f"최근 20일 수익률이 **+{roc:.1f}%**로 강력한 단기 모멘텀을 보유하고 있습니다.")
        elif roc > 5:
            lines.append(f"최근 20일 수익률 **+{roc:.1f}%**로 완만한 상승세를 유지하고 있습니다.")
        elif roc >= -5:
            lines.append(f"최근 20일 수익률 **{roc:.1f}%**로 중립적인 흐름입니다.")
        else:
            lines.append(f"최근 20일 수익률 **{roc:.1f}%**로 단기 약세 구간에 있어 주의가 필요합니다.")

    # Beta
    if pd.notna(beta):
        if beta > 1.5:
            lines.append(f"베타({beta:.2f})가 높아 테마 상승 시 **증폭된 수익**을 기대할 수 있습니다. 하락 시 손실도 큰 양날의 칼입니다.")
        elif beta > 0.8:
            lines.append(f"시장과 유사한 민감도(베타 {beta:.2f})를 가지고 있습니다.")
        else:
            lines.append(f"베타({beta:.2f})가 낮아 테마 급등 시 상대적으로 덜 오를 수 있으나, 방어적 특성을 지닙니다.")

    # WRS
    if pd.notna(wrs):
        if wrs > 30:
            lines.append(f"장기 가중 상대강도(WRS **{wrs:.1f}%**)가 매우 높아, 6~12개월 기준 강한 장기 상승 추세를 확인할 수 있습니다.")
        elif wrs > 0:
            lines.append(f"장기 추세(WRS {wrs:.1f}%)는 양호한 편입니다.")
        else:
            lines.append(f"장기 추세(WRS {wrs:.1f}%)는 아직 약세 구간입니다.")

    # 현재 stage와 조합한 한줄 제언
    STAGE_STOCK = {
        1: "현재 테마가 바닥 매집기이므로 소량 관심 보유 수준이 적절합니다.",
        2: "테마 돌파 초기이므로 이 종목은 분할 매수 1순위 후보입니다.",
        3: "테마 주도 상승 중이므로 이 종목의 비중을 적극 유지하세요.",
        4: "테마 과열 구간이므로 신규 진입보다 기존 수익 관리에 집중하세요.",
        5: "테마 하락 구간이므로 보유 중이라면 손절 기준을 재점검하세요.",
    }
    lines.append(f"📌 {STAGE_STOCK.get(stage, '')}")

    return " ".join(lines)


# ─── 차트 함수 ────────────────────────────────────────────────────────────────

def make_theme_chart(theme_df, theme_name, stage) -> go.Figure:
    if theme_df.empty: return go.Figure()
    COLORS = {1:"#6c757d",2:"#0d6efd",3:"#198754",4:"#fd7e14",5:"#dc3545"}
    accent = COLORS.get(stage, "#58a6ff")
    ma200_s = theme_df["Close"].rolling(200).mean()
    ma50_s  = theme_df["Close"].rolling(50).mean()
    plot    = theme_df.tail(250)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    r, g, b = int(accent[1:3],16), int(accent[3:5],16), int(accent[5:7],16)
    fig.add_trace(go.Scatter(x=plot.index, y=plot["Close"], name="테마지수",
        line=dict(color=accent, width=2), fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.1)"), row=1, col=1)
    m200 = ma200_s.tail(250)
    fig.add_trace(go.Scatter(x=m200.index, y=m200.values, name="MA200",
        line=dict(color="#e3b341", width=2)), row=1, col=1)
    m50 = ma50_s.tail(250)
    fig.add_trace(go.Scatter(x=m50.index, y=m50.values, name="MA50",
        line=dict(color="#58a6ff", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"], name="거래량",
        marker_color=accent, opacity=0.4), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{theme_name} 테마 지수</b>", font=dict(color="white", size=15)),
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22", font=dict(color="#c9d1d9"),
        xaxis_rangeslider_visible=False, height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    return fig


def make_stock_chart(df, name) -> go.Figure:
    ma200_s = df["Close"].rolling(200).mean()
    ma50_s  = df["Close"].rolling(50).mean()
    plot    = df.tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=plot.index,
        open=plot["Open"], high=plot["High"], low=plot["Low"], close=plot["Close"],
        name=name, increasing_line_color="#3fb950", decreasing_line_color="#f85149"),
        row=1, col=1)
    m200 = ma200_s.tail(200)
    fig.add_trace(go.Scatter(x=m200.index, y=m200.values, name="MA200",
        line=dict(color="#e3b341", width=2)), row=1, col=1)
    m50 = ma50_s.tail(200)
    fig.add_trace(go.Scatter(x=m50.index, y=m50.values, name="MA50",
        line=dict(color="#58a6ff", width=1.5, dash="dot")), row=1, col=1)
    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"], name="거래량",
        marker_color=vol_colors, opacity=0.6), row=2, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{name}</b>", font=dict(color="white", size=13)),
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22", font=dict(color="#c9d1d9"),
        xaxis_rangeslider_visible=False, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    return fig


def make_radar(lss_df, name_map) -> go.Figure:
    cats = ["ROC", "거래대금", "회전율", "베타", "가중RS"]
    keys = ["roc_norm","trading_value_norm","turnover_ratio_norm","beta_norm","wrs_norm"]
    colors = ["#ffd700","#c0c0c0","#cd7f32"]
    fig = go.Figure()
    for i, row in lss_df[lss_df["rank"] <= 3].iterrows():
        vals = [float(row.get(k, 0) or 0) for k in keys]
        vals += [vals[0]]
        c = colors[int(row["rank"])-1]
        r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color=c, width=2),
            name=f"#{int(row['rank'])} {name_map.get(row['ticker'], row['ticker'])}",
        ))
    fig.update_layout(
        polar=dict(bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#30363d",
                            tickfont=dict(color="#8b949e", size=9)),
            angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#c9d1d9"))),
        paper_bgcolor="#0e1117",
        legend=dict(font=dict(color="#c9d1d9"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="<b>LSS 지표 레이더</b>", font=dict(color="white")),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─── UI 컴포넌트 ──────────────────────────────────────────────────────────────

def render_stage_card(lifecycle):
    stage = lifecycle["stage"]
    info  = STAGE_INFO[stage]
    ind   = lifecycle["indicators"]
    extra = lifecycle["extra"]
    cond  = lifecycle["conditions"]

    STAGE_BG = {1:"#2d2d2d",2:"#0a2a5c",3:"#0d3320",4:"#4a2000",5:"#4a0000"}
    bg    = STAGE_BG.get(stage, "#2d2d2d")
    color = info["color"]

    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:14px;'
        f'padding:20px;margin-bottom:16px;">'
        f'<div style="font-size:2.2rem">{info["emoji"]}</div>'
        f'<div style="font-size:1.5rem;font-weight:900;color:{color};">'
        f'Stage {stage} · {info["name"]}</div>'
        f'<div style="color:{color};opacity:0.75;font-size:0.85rem;">{info["en"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if extra.get("selling_climax"):
        st.error("⚡ 투매 클라이막스 감지! (변동성 25%↑ + RVOL 2.5배↑)")
    if extra.get("fresh_breakout"):
        st.success("✅ 신선한 돌파 확인 (최근 5일 내 MA200 하회 이력)")
    if extra.get("divergence"):
        st.warning("🔀 다이버전스: 지수 상승 vs 종목 이탈 감지")
    if extra.get("estimated"):
        st.info("⚠️ 완전 조건 미충족 — 근사 판별 적용")

    if not ind:
        return

    pct   = ind.get("price_ma200_pct", 0)
    slope = ind.get("ma200_slope", 0)
    brd   = ind.get("breadth", 0)
    rvol  = ind.get("rvol", 1)
    vlt   = ind.get("volatility", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("현재가 vs MA200", f"{ind.get('price',0):.1f}",
              f"{pct:+.1f}%", help=TIP["ma200"])
    c2.metric("MA200 기울기", f"{slope:.5f}",
              "상승↑" if slope>0.002 else "하락↓" if slope<-0.002 else "횡보→",
              help=TIP["slope"])
    c3.metric("테마 참여율", f"{brd:.1f}%",
              "확산" if brd>=70 else "수축" if brd<50 else "중립",
              help=TIP["breadth"])

    c4, c5, c6 = st.columns(3)
    c4.metric("RVOL", f"{rvol:.2f}x",
              "거래량 급증" if rvol>=2.0 else "보통",
              help=TIP["rvol"])
    c5.metric("20일 변동성", f"{vlt:.1f}%",
              "과열" if vlt>25 else "주의" if vlt>18 else "안정",
              help=TIP["volatility"])
    c6.metric("MA50", f"{ind.get('ma50',0):.1f}", help=TIP["ma50"])

    if cond:
        st.markdown("**판별 조건 체크**")
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

        icon  = RANK_ICON.get(rank, f"#{rank}")
        label = RANK_LABEL.get(rank, "")

        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown(f"### {icon} {name}")
                st.caption(
                    f"`{tk}` &nbsp;|&nbsp; 현재가 **{close:,}원**"
                    + (f" &nbsp;🔥 **{label}**" if is_top else "")
                )
            with h2:
                st.metric("LSS Score", f"{score:.4f}",
                          help=TIP["lss"])
            st.progress(float(score), text=f"LSS {score:.4f}")

            # 자연어 해설
            commentary = generate_stock_commentary(row, name, stage)
            st.markdown(f"> {commentary}")

            # 지표 상세 + 차트
            with st.expander(f"📊 {name} 상세 지표 & 차트"):
                m = row
                d1, d2, d3 = st.columns(3)
                roc = m.get("roc")
                d1.metric("ROC (20일)",
                          f"{roc:.1f}%" if pd.notna(roc) else "─",
                          help=TIP["roc"])
                tv = m.get("trading_value")
                d2.metric("거래대금(평균)",
                          f"{tv/1e6:.0f}만원" if pd.notna(tv) else "─",
                          help=TIP["tv"])
                beta = m.get("beta")
                d3.metric("베타", f"{beta:.2f}" if pd.notna(beta) else "─",
                          help=TIP["beta"])
                d4, d5, _ = st.columns(3)
                tr = m.get("turnover_ratio")
                d4.metric("회전율", f"{tr:.4f}" if pd.notna(tr) else "─",
                          help=TIP["turnover"])
                wrs = m.get("wrs")
                d5.metric("가중상대강도(WRS)",
                          f"{wrs:.1f}%" if pd.notna(wrs) else "─",
                          help=TIP["wrs"])

                # 개별 종목 차트
                if tk in stocks:
                    st.plotly_chart(
                        make_stock_chart(stocks[tk], name),
                        use_container_width=True,
                    )


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 18px">
            <div style="font-size:2rem">📊</div>
            <div style="font-size:1.1rem;font-weight:bold;color:#e6edf3">테마주 퀀트 엔진</div>
            <div style="font-size:0.78rem;color:#8b949e">LSS & 생애주기 분석</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        theme_opts = {tid: t["name"] for tid, t in THEMES.items()}
        selected   = st.selectbox("테마 선택", list(theme_opts.keys()),
                                  format_func=lambda x: theme_opts[x])
        with st.expander("📋 구성 종목"):
            for tk, (name, exch) in THEMES[selected]["stocks"].items():
                st.markdown(f"• `{tk}` {name} ({exch})")

        st.divider()
        run = st.button("🔍 분석 실행", use_container_width=True, type="primary")

        st.divider()
        # 업데이트 시각 표시
        if _THEMES_UPDATED_AT:
            st.caption(f"🕐 마지막 재분석: {_THEMES_UPDATED_AT}")
        else:
            st.caption("📌 기본 테마 사용 중 (재분석 미실시)")
        refresh = st.button(
            "🔄 테마 종목 재분석",
            use_container_width=True,
            help="네이버 증권 크롤링 → TF-IDF 매칭 → LSS 선정. 수 분 소요.",
        )

        st.divider()
        with st.expander("ℹ️ LSS란 무엇인가요?"):
            st.markdown("""
**LSS(Leader Stock Score)**는 테마 내에서 **'대장주'가 누구인지** 찾는 점수입니다.

- **매수 타이밍 지표가 아닙니다**
- 테마가 상승할 때 **가장 먼저·가장 크게** 움직이는 종목을 찾습니다
- 5개 지표를 0~1로 정규화 후 가중 합산합니다

**활용법:**
생애주기 **Stage 2~3** 구간에서 LSS **1~2위** 종목에 집중하는 전략이 효과적입니다.
""")
        st.markdown("""
<div style="font-size:0.76rem;color:#8b949e;line-height:1.8">
<b>LSS 가중치</b><br>
▪ ROC(모멘텀) · 30%<br>
▪ 거래대금 · 30%<br>
▪ 주식회전율 · 15%<br>
▪ 테마베타 · 10%<br>
▪ 가중상대강도 · 15%<br><br>
<b>생애주기</b><br>
🧱 Stage 1 · 바닥 매집기<br>
🚀 Stage 2 · 모멘텀 돌파기<br>
📈 Stage 3 · 주도 상승기<br>
⚠️ Stage 4 · 과열 분배기<br>
📉 Stage 5 · 항복 투매기
</div>
""", unsafe_allow_html=True)

        st.divider()
        stage_filter = st.button(
            "🎯 Stage 2·3 핵심 테마만 보기",
            use_container_width=True,
            help="모멘텀 돌파기·주도 상승기 테마만 필터링. 재분석 후 즉시 반영됩니다.",
        )

        return selected, run, refresh, stage_filter


# ─── Stage 2·3 필터 뷰 ────────────────────────────────────────────────────────

def render_stage_filter_view():
    """Stage 2·3 핵심 테마 전용 뷰"""
    st.markdown("## 🎯 Stage 2·3 핵심 테마")

    if not THEMES_STAGES:
        st.warning(
            "아직 생애주기 데이터가 없습니다.\n\n"
            "왼쪽 **🔄 테마 종목 재분석** 버튼을 먼저 실행해주세요. "
            "재분석 완료 후 생애주기가 자동으로 계산됩니다."
        )
        return

    stage_2_3 = {k: THEMES[k] for k in THEMES_STAGES
                 if THEMES_STAGES[k] in (2, 3) and k in THEMES}
    all_stages = {k: THEMES[k] for k in THEMES_STAGES if k in THEMES}

    if stage_2_3:
        st.success(
            f"**{len(stage_2_3)}개 테마**가 현재 모멘텀 구간(Stage 2·3)입니다. "
            "이 테마의 LSS 1~2위 종목을 주목하세요."
        )
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
                    st.markdown("**구성 종목 (LSS 순위)**")
                    for j, (code, (name, _)) in enumerate(
                        list(theme["stocks"].items())[:5]
                    ):
                        medal = {0: "👑", 1: "🥈", 2: "🥉"}.get(j, f"`{j+1}`")
                        st.markdown(f"{medal} {name} &nbsp;`{code}`",
                                    unsafe_allow_html=True)
                    st.caption(f"📅 {_THEMES_UPDATED_AT or '업데이트 시각 미확인'}")
    else:
        st.info(
            "현재 Stage 2·3에 해당하는 테마가 없습니다.\n\n"
            "전체 테마 현황을 아래 표에서 확인하세요."
        )

    # 전체 테마 현황 요약표
    st.markdown("---")
    st.markdown("### 📊 전체 테마 생애주기 현황")
    rows = []
    for key, theme in all_stages.items():
        stage = THEMES_STAGES.get(key, 0)
        info  = STAGE_INFO.get(stage, STAGE_INFO[0])
        top1  = list(theme["stocks"].values())
        top1_name = top1[0][0] if top1 else "─"
        rows.append({
            "테마":    theme["name"],
            "단계":    f"{info['emoji']} Stage {stage} · {info['name']}",
            "대장주":  top1_name,
            "주목":    "✅" if stage in (2, 3) else ("⚠️" if stage == 4 else "─"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"마지막 업데이트: {_THEMES_UPDATED_AT or '미실시'}")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    selected, run, refresh, stage_filter = render_sidebar()

    if "data" not in st.session_state:
        st.session_state.data = None

    # ── 테마 재분석 ────────────────────────────────────────────────────────
    if refresh:
        st.markdown("## 🔄 테마 종목 재분석")
        st.info(
            "네이버 증권 크롤링 → TF-IDF 매칭 → 주가 로드 → LSS 선정 순으로 진행됩니다.\n"
            "종목 수에 따라 **3~10분** 소요될 수 있습니다. 창을 닫지 마세요."
        )
        with st.status("재분석 진행 중...", expanded=True) as status:
            try:
                from crawler import update_themes as _run_crawler

                # GitHub 자동 커밋 설정 (Streamlit secrets에서 읽기)
                try:
                    _gh_token  = st.secrets.get("GITHUB_TOKEN", "")
                    _gh_repo   = st.secrets.get("GITHUB_REPO", "")
                    _gh_branch = st.secrets.get("GITHUB_BRANCH", "main")
                except Exception:
                    _gh_token = _gh_repo = _gh_branch = ""

                logs: List[str] = []
                log_box = st.empty()

                def _on_progress(msg: str) -> None:
                    logs.append(msg)
                    log_box.code("\n".join(logs[-30:]), language=None)

                result = _run_crawler(
                    progress_callback=_on_progress,
                    github_token=_gh_token,
                    github_repo=_gh_repo,
                    github_branch=_gh_branch or "main",
                )

                if result:
                    status.update(
                        label=f"✅ 재분석 완료! ({len(result)}개 테마 업데이트)",
                        state="complete",
                    )
                    st.cache_data.clear()
                    time.sleep(1.5)
                    st.rerun()
                else:
                    status.update(label="⚠️ 결과 없음 — 네이버 접속 문제일 수 있습니다.", state="error")
            except ImportError:
                status.update(label="❌ crawler.py 파일을 찾을 수 없습니다.", state="error")
                st.error("같은 폴더에 crawler.py 파일이 있는지 확인하세요.")
            except Exception as e:
                status.update(label=f"❌ 오류: {e}", state="error")
        return

    # ── Stage 2·3 필터 뷰 ──────────────────────────────────────────────────
    if stage_filter:
        render_stage_filter_view()
        return

    if not run and st.session_state.data is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 0">
            <div style="font-size:5rem">📊</div>
            <h2 style="color:#e6edf3">테마주 퀀트 대시보드</h2>
            <p style="color:#8b949e;font-size:1.05rem">
                왼쪽에서 테마를 선택하고 <b>분석 실행</b>을 클릭하세요.
            </p>
            <p style="color:#8b949e;font-size:0.9rem">
                각 지표 옆 <b>?</b> 아이콘에 마우스를 올리면 용어 설명을 볼 수 있습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if run:
        st.cache_data.clear()
        with st.spinner("📡 데이터 수집 중... (최대 60초)"):
            raw = load_theme_data(selected)
        stocks, benchmark = raw["stocks"], raw["benchmark"]
        if len(stocks) < 2:
            st.error("분석 가능 종목이 부족합니다. 잠시 후 다시 시도해주세요.")
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
    if d is None:
        return

    raw       = d["raw"]
    lss_df    = d["lss_df"]
    theme_idx = d["theme_idx"]
    lifecycle = d["lifecycle"]
    name_map  = d["name_map"]
    stocks    = d["stocks"]
    stage     = lifecycle["stage"]

    ok_cnt = sum(1 for s in raw["fetch_status"] if s["ok"])
    st.markdown(
        f"## {raw['theme_name']} 테마 분석\n"
        f"<span style='color:#8b949e;font-size:0.88rem'>"
        f"분석일: {raw['analysis_date']} &nbsp;|&nbsp; 확보 종목: {ok_cnt}종목 &nbsp;|&nbsp; "
        f"지표 옆 ? 아이콘으로 용어 설명 확인 가능</span>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 종합 리포트", "🏆 LSS 대장주 순위", "🔬 데이터 상세"])

    # ── Tab 1: 종합 리포트 ───────────────────────────────────────────────────
    with tab1:
        # 자연어 테마 해설
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
            st.plotly_chart(
                make_theme_chart(theme_idx, raw["theme_name"], stage),
                use_container_width=True,
            )

    # ── Tab 2: LSS 대장주 ───────────────────────────────────────────────────
    with tab2:
        if lss_df.empty:
            st.warning("LSS 분석 데이터가 없습니다.")
        else:
            # LSS 설명 배너
            st.info(
                "**LSS(대장주 점수) 해석법:** 이 점수는 '지금 사야 할 종목'이 아니라 "
                "**'테마가 오를 때 가장 앞서 달리는 말'**을 찾는 지표입니다. "
                "생애주기 Stage 2·3 구간에서 LSS 1~2위 종목에 집중하는 것이 핵심 전략입니다."
            )

            left2, right2 = st.columns([1, 1])
            with left2:
                st.markdown("#### 종목별 LSS 순위")
                render_lss_cards(lss_df, name_map, stocks, stage)

            with right2:
                st.plotly_chart(make_radar(lss_df, name_map), use_container_width=True)

                # 점수표
                st.markdown("#### 전체 점수표")
                disp = lss_df[["ticker","rank","lss","roc","trading_value","beta","wrs"]].copy()
                disp["name"] = disp["ticker"].map(name_map)
                disp = disp[["rank","name","ticker","lss","roc","trading_value","beta","wrs"]]
                disp.columns = ["순위","종목명","티커","LSS","ROC(%)","거래대금","베타","WRS(%)"]
                disp["LSS"] = disp["LSS"].round(4)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Tab 3: 데이터 상세 ──────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 데이터 수집 현황")
        fetch_df = pd.DataFrame([
            {"티커":s["ticker"],"종목명":s["name"],
             "상태":"✅ 성공" if s["ok"] else "❌ 실패","확보 일수":s["days"]}
            for s in raw["fetch_status"]
        ])
        st.dataframe(fetch_df, use_container_width=True, hide_index=True)

        ind = lifecycle.get("indicators", {})
        if ind:
            st.markdown("#### 생애주기 핵심 지표 원본값")
            ind_df = pd.DataFrame([
                {"지표":"현재가(정규화)","값":ind.get("price"),"임계 기준":"MA200 대비 ±5%"},
                {"지표":"MA200","값":ind.get("ma200"),"임계 기준":"200일 이동평균"},
                {"지표":"MA50","값":ind.get("ma50"),"임계 기준":"50일 이동평균"},
                {"지표":"MA200 기울기","값":ind.get("ma200_slope"),"임계 기준":"0.005 / -0.005"},
                {"지표":"RVOL","값":ind.get("rvol"),"임계 기준":"2.0 (돌파) / 2.5 (투매)"},
                {"지표":"테마 참여율(%)","값":ind.get("breadth"),"임계 기준":"50% / 70%"},
                {"지표":"20일 변동성(%)","값":ind.get("volatility"),"임계 기준":"18% / 25%"},
                {"지표":"가격/MA200 괴리(%)","값":ind.get("price_ma200_pct"),"임계 기준":"±5%"},
            ])
            st.dataframe(ind_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
