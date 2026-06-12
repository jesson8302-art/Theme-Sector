"""
SurgeLogicEngine — 급등주 패턴 탐지 엔진
app.py 와 crawler.py 공유 모듈
"""

from typing import Dict, List
import numpy as np
import pandas as pd

# ─── 상수 ────────────────────────────────────────────────────────────────────

SENSITIVITY_THRESHOLD = {1: 0, 2: 55, 3: 68, 4: 82, 5: 91}

_SURGE_STATUS_MAP = [
    (90, "🔥 강력 매수"),
    (80, "✅ 트리거 포착"),
    (70, "👀 관찰 진입"),
    (0,  "─ 대기"),
]


def get_surge_status(score: int) -> str:
    for threshold, label in _SURGE_STATUS_MAP:
        if score >= threshold:
            return label
    return "─"


# ─── 엔진 ────────────────────────────────────────────────────────────────────

class SurgeLogicEngine:
    """5가지 급등 패턴 정량 탐지 엔진 (Pandas 기반)"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for span in [5, 10, 20, 60]:
            df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
        df["VOL_5"]    = df["Volume"].rolling(5).mean()
        df["VOL_20"]   = df["Volume"].rolling(20).mean()
        df["BB_MID"]   = df["Close"].rolling(20).mean()
        df["BB_STD"]   = df["Close"].rolling(20).std()
        df["BB_UPPER"] = df["BB_MID"] + df["BB_STD"] * 2
        df["BB_LOWER"] = df["BB_MID"] - df["BB_STD"] * 2
        df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"].replace(0, np.nan)
        obv = [0]
        for i in range(1, len(df)):
            c, p, v = df["Close"].iloc[i], df["Close"].iloc[i - 1], df["Volume"].iloc[i]
            obv.append(obv[-1] + v if c > p else (obv[-1] - v if c < p else obv[-1]))
        df["OBV"] = obv
        return df

    # ── P1: 이평선 극한 수렴 + 거래량 씨마름 ─────────────────────────────────
    @classmethod
    def check_p1_ma_squeeze(cls, df: pd.DataFrame):
        if len(df) < 60:
            return False, 0
        latest = df.iloc[-1]
        emas = [latest[f"EMA_{s}"] for s in [5, 10, 20, 60] if f"EMA_{s}" in df.columns]
        if not emas:
            return False, 0
        min_e = min(emas)
        spread = (max(emas) - min_e) / min_e * 100 if min_e > 0 else 99.0
        is_squeeze = spread <= 3.0
        is_vol_dry = (latest["VOL_5"] <= latest["VOL_20"] * 0.35
                      if latest["VOL_20"] > 0 else False)
        score = 0
        if is_squeeze:  score += 50
        if is_vol_dry:  score += 40
        if spread <= 1.5: score += 10
        return score >= 70, min(score, 100)

    # ── P2: 세력 매집봉 출현 후 눌림목 ──────────────────────────────────────
    @classmethod
    def check_p2_accumulation(cls, df: pd.DataFrame):
        if len(df) < 30:
            return False, 0
        recent = df.tail(20).copy()
        target = recent[
            (recent["Close"] / recent["Open"].replace(0, np.nan) >= 1.15) &
            (recent["Volume"] >= recent["VOL_20"] * 5.0)
        ]
        if target.empty:
            return False, 0
        ref = target.iloc[-1]
        after = df.loc[ref.name + 1:] if ref.name + 1 <= df.index[-1] else pd.DataFrame()
        if after.empty:
            return False, 0
        center = (ref["High"] + ref["Low"]) / 2
        is_supported = after["Close"].min() >= center
        is_vol_down  = after["Volume"].mean() <= ref["Volume"] * 0.30
        score = 60
        if is_supported: score += 20
        if is_vol_down:  score += 20
        return score >= 70, min(score, 100)

    # ── P3: OBV 상승 다이버전스 & 쌍바닥 ─────────────────────────────────────
    @classmethod
    def check_p3_obv_divergence(cls, df: pd.DataFrame):
        if len(df) < 60:
            return False, 0
        recent = df.tail(60).reset_index(drop=True)
        mid = len(recent) // 2
        lo1 = recent["Close"].iloc[:mid].idxmin()
        lo2 = recent["Close"].iloc[mid:].idxmin() + mid
        price1, price2 = recent["Close"].iloc[lo1], recent["Close"].iloc[lo2]
        obv1,   obv2   = recent["OBV"].iloc[lo1],   recent["OBV"].iloc[lo2]
        price_lower = price2 <= price1 * 1.03
        obv_higher  = obv2 > obv1
        score = 0
        if price_lower and obv_higher:
            score += 60
            ratio = abs(obv2 - obv1) / (abs(obv1) + 1) * 100
            if ratio > 5:  score += 20
            if ratio > 15: score += 20
        return score >= 70, min(score, 100)

    # ── P4: 볼린저밴드 극한 수렴 후 상단 돌파 ───────────────────────────────
    @classmethod
    def check_p4_bb_squeeze(cls, df: pd.DataFrame):
        if len(df) < 60:
            return False, 0
        recent  = df.tail(min(120, len(df)))
        current = df.iloc[-1]
        bw_s = recent["BB_WIDTH"].dropna()
        if bw_s.empty:
            return False, 0
        min_bw = bw_s.min()
        cur_bw = current["BB_WIDTH"] if pd.notna(current["BB_WIDTH"]) else 99.0
        is_squeeze  = cur_bw <= min_bw * 1.6
        broke_upper = (current["Close"] >= current["BB_UPPER"] * 0.97
                       if pd.notna(current["BB_UPPER"]) else False)
        vol_surge   = (current["VOL_5"] >= current["VOL_20"] * 1.5
                       if current["VOL_20"] > 0 else False)
        score = 0
        if is_squeeze:  score += 35
        if broke_upper: score += 45
        if vol_surge:   score += 20
        return score >= 70, min(score, 100)

    # ── P5: 컵앤핸들 패턴 ────────────────────────────────────────────────────
    @classmethod
    def check_p5_cup_handle(cls, df: pd.DataFrame):
        if len(df) < 90:
            return False, 0
        cup    = df.tail(90).head(60)
        handle = df.tail(30)
        cup_high  = cup["Close"].max()
        cup_low   = cup["Close"].min()
        cup_depth = (cup_high - cup_low) / cup_high * 100 if cup_high > 0 else 0.0
        h_high    = handle["Close"].max()
        h_low     = handle["Close"].min()
        h_depth   = (h_high - h_low) / h_high * 100 if h_high > 0 else 99.0
        cur_price      = df.iloc[-1]["Close"]
        near_rim       = cur_price >= cup_high * 0.94
        good_cup       = 12 <= cup_depth <= 55
        shallow_handle = h_depth <= 15
        vol_contract   = handle["Volume"].mean() <= cup["Volume"].mean() * 0.75
        score = 0
        if good_cup:       score += 30
        if shallow_handle: score += 25
        if near_rim:       score += 25
        if vol_contract:   score += 20
        return score >= 70, min(score, 100)

    # ── 통합 스캔 ─────────────────────────────────────────────────────────────
    @classmethod
    def scan(cls, df: pd.DataFrame) -> Dict:
        df = cls.calculate_indicators(df)
        patterns = {
            "P1 이평선 응축":    cls.check_p1_ma_squeeze(df),
            "P2 매집봉 눌림목":  cls.check_p2_accumulation(df),
            "P3 OBV 다이버전스": cls.check_p3_obv_divergence(df),
            "P4 볼린저 수렴돌파":cls.check_p4_bb_squeeze(df),
            "P5 컵앤핸들":       cls.check_p5_cup_handle(df),
        }
        top_name  = max(patterns, key=lambda k: patterns[k][1])
        top_score = patterns[top_name][1]
        matched   = [n for n, (m, _) in patterns.items() if m]
        return {
            "patterns": {n: {"match": m, "score": s} for n, (m, s) in patterns.items()},
            "top_pattern": top_name,
            "top_score":   top_score,
            "matched":     matched,
            "is_signal":   top_score >= 70,
        }
