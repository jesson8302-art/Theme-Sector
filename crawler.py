"""
테마 데이터 자동 수집 크롤러
네이버 금융 전체 테마 크롤링 → LSS + 생애주기 계산 → JSON 저장
GitHub Actions 또는 Streamlit 버튼에서 실행 가능
"""

import base64
import json
import os
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy import stats

# ─── 네이버 크롤링 ────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_naver_theme_list() -> pd.DataFrame:
    """네이버 금융 전체 테마 목록 크롤링"""
    url = "https://finance.naver.com/sise/theme.nhn"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = []
        for a in soup.select("td.col_type1 a"):
            href = a.get("href", "")
            if "no=" in href:
                try:
                    no = int(href.split("no=")[-1].split("&")[0])
                    name = a.get_text(strip=True)
                    if name:
                        rows.append({"no": no, "name": name})
                except ValueError:
                    continue
        df = pd.DataFrame(rows).drop_duplicates("no").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[ERROR] 테마 목록 크롤링 실패: {e}")
        return pd.DataFrame()


def get_naver_theme_stocks(theme_no: int) -> List[Dict]:
    """특정 테마의 종목 목록 크롤링"""
    url = (
        f"https://finance.naver.com/sise/sise_group_detail.naver"
        f"?type=theme&no={theme_no}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
        stocks = []
        for a in soup.select("td.name a"):
            href = a.get("href", "")
            if "code=" in href:
                code = href.split("code=")[-1].split("&")[0].strip()
                name = a.get_text(strip=True)
                if code and len(code) == 6 and code.isdigit():
                    stocks.append({"code": code, "name": name})
        return stocks
    except Exception as e:
        print(f"[ERROR] 테마 {theme_no} 종목 크롤링 실패: {e}")
        return []


# ─── 주가 데이터 로딩 ─────────────────────────────────────────────────────────

def load_stock_with_exchange(code: str, start: str, end: str):
    """KS → KQ 순서로 시도, (DataFrame, exchange) 반환"""

    def _dl(suffix):
        try:
            df = yf.download(
                f"{code}.{suffix}", start=start, end=end,
                progress=False, auto_adjust=True
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            required = ["Open", "High", "Low", "Close", "Volume"]
            if any(c not in df.columns for c in required):
                return None
            df = df[required].copy()
            df["Volume"] = df["Volume"].fillna(0)
            df = df.dropna(subset=["Close"])
            return df if len(df) >= 60 else None
        except Exception:
            return None

    df = _dl("KS")
    if df is not None:
        return df, "KS"
    df = _dl("KQ")
    if df is not None:
        return df, "KQ"
    return None, None


def load_benchmark(start: str, end: str) -> Optional[pd.DataFrame]:
    """KOSPI 벤치마크 로딩"""
    try:
        df = yf.download("^KS11", start=start, end=end,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df if not df.empty else None
    except Exception:
        return None


# ─── LSS 계산 ─────────────────────────────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if mx == mn else (s - mn) / (mx - mn)


def calc_lss(stocks_data: Dict, benchmark) -> pd.DataFrame:
    """종목별 LSS(대장주 점수) 계산"""
    records = []
    for tk, df in stocks_data.items():
        if df is None or len(df) < 60:
            continue

        roc = np.nan
        if len(df) >= 21:
            p0 = df["Close"].iloc[-21]
            if p0:
                roc = (df["Close"].iloc[-1] - p0) / p0 * 100

        tv = float((df["Close"] * df["Volume"]).tail(20).mean())

        a20 = df["Volume"].tail(20).mean()
        al = df["Volume"].tail(min(40, len(df))).mean()
        turn = float(a20 / al) if al > 0 else np.nan

        beta = np.nan
        if benchmark is not None and len(df) >= 60 and len(benchmark) >= 60:
            s = df["Close"].pct_change().tail(60).dropna()
            b = benchmark["Close"].pct_change().tail(60).dropna()
            idx = s.index.intersection(b.index)
            if len(idx) >= 30:
                cov = np.cov(s.loc[idx].values, b.loc[idx].values)
                if cov[1, 1] != 0:
                    beta = float(cov[0, 1] / cov[1, 1])

        p = df["Close"].iloc[-1]
        r120 = r240 = np.nan
        if len(df) >= 121:
            p0 = df["Close"].iloc[-121]
            if p0:
                r120 = (p - p0) / p0 * 100
        if len(df) >= 241:
            p0 = df["Close"].iloc[-241]
            if p0:
                r240 = (p - p0) / p0 * 100
        if pd.notna(r120) and pd.notna(r240):
            wrs = 0.7 * r120 + 0.3 * r240
        elif pd.notna(r120):
            wrs = r120
        else:
            wrs = np.nan

        records.append({
            "ticker": tk, "roc": roc, "trading_value": tv,
            "turnover_ratio": turn, "beta": beta, "wrs": wrs,
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("ticker")
    weights = {"roc": 0.30, "trading_value": 0.30,
               "turnover_ratio": 0.15, "beta": 0.10, "wrs": 0.15}
    for ind in weights:
        col = result[ind]
        norm = pd.Series(0.5, index=result.index)
        mask = col.notna()
        if mask.sum() > 1:
            norm[mask] = _minmax(col[mask])
        result[f"{ind}_norm"] = norm

    result["lss"] = sum(result[f"{k}_norm"] * v for k, v in weights.items())
    result = result.sort_values("lss", ascending=False).reset_index()
    result["rank"] = range(1, len(result) + 1)
    return result


# ─── 생애주기 계산 ────────────────────────────────────────────────────────────

def _build_theme_index(stocks_data: Dict) -> pd.DataFrame:
    price_frames, vol_frames = [], []
    for tk, df in stocks_data.items():
        if df is None or len(df) < 200:
            continue
        base = df["Close"].iloc[0]
        if base == 0:
            continue
        price_frames.append((df["Close"] * 100 / base).rename(tk))
        vol_frames.append(df["Volume"].rename(tk))

    if not price_frames:
        return pd.DataFrame()

    price_idx = pd.concat(price_frames, axis=1).dropna(how="all").mean(axis=1)
    vol_idx = pd.concat(vol_frames, axis=1).dropna(how="all").mean(axis=1)
    return pd.DataFrame({
        "Close": price_idx, "Volume": vol_idx
    }).dropna(subset=["Close"])


def compute_lifecycle_stage(stocks_data: Dict) -> int:
    """생애주기 Stage 판별 (0~5)"""
    theme_df = _build_theme_index(stocks_data)
    if theme_df.empty or len(theme_df) < 200:
        return 0

    price = float(theme_df["Close"].iloc[-1])
    ma200_s = theme_df["Close"].rolling(200).mean().dropna()
    if ma200_s.empty:
        return 0

    ma200 = float(ma200_s.iloc[-1])

    window = min(15, len(ma200_s))
    y = ma200_s.iloc[-window:].values.astype(float)
    slope_raw, *_ = stats.linregress(np.arange(window, dtype=float), y)
    slope = float(slope_raw / y[-1]) if y[-1] != 0 else 0.0

    vol = theme_df["Volume"].dropna()
    rvol = (
        float(vol.tail(5).mean() / vol.tail(40).mean())
        if len(vol) >= 40 and vol.tail(40).mean() > 0
        else 1.0
    )

    above = total = 0
    for df in stocks_data.values():
        if df is None or len(df) < 50:
            continue
        total += 1
        ma50 = df["Close"].rolling(50).mean().iloc[-1]
        if df["Close"].iloc[-1] > ma50:
            above += 1
    breadth = (above / total * 100) if total > 0 else 0.0

    if len(theme_df) >= 20:
        r = theme_df["Close"].tail(20)
        avg = r.mean()
        volatility = float((r.max() - r.min()) / avg * 100) if avg else 0.0
    else:
        volatility = 0.0

    pct = (price / ma200 - 1) * 100 if ma200 > 0 else 0.0

    if price < ma200 and slope < -0.005:
        return 5
    elif pct > 5 and slope > 0.002 and rvol >= 2.0 and breadth >= 50:
        return 2
    elif price > ma200 and slope > 0.005 and breadth >= 70:
        return 3
    elif price > ma200 and abs(slope) <= 0.008 and breadth < 50 and volatility > 18:
        return 4
    elif abs(pct) <= 5 and abs(slope) <= 0.005 and breadth < 50:
        return 1
    else:
        if price < ma200:
            return 5
        elif pct > 5 and slope > 0:
            return 2
        elif slope > 0.003 and breadth >= 60:
            return 3
        elif price > ma200 and breadth < 60:
            return 4
        else:
            return 1


# ─── GitHub API 커밋 (Streamlit 버튼용) ──────────────────────────────────────

def commit_to_github(
    content: str,
    token: str,
    repo: str,
    file_path: str = "themes_data.json",
    branch: str = "main",
) -> bool:
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    sha = None
    try:
        r = requests.get(url, headers=headers, params={"ref": branch}, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass

    payload = {
        "message": f"🔄 테마 데이터 수동 업데이트 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": base64.b64encode(content.encode("utf-8")).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(url, headers=headers, json=payload, timeout=15)
        return r.status_code in (200, 201)
    except Exception:
        return False


# ─── 메인 파이프라인 ──────────────────────────────────────────────────────────

def update_themes(
    output_path: str = "themes_data.json",
    max_themes: Optional[int] = None,
    max_stocks_per_theme: int = 10,
    progress_callback: Optional[Callable] = None,
    github_token: str = "",
    github_repo: str = "",
    github_branch: str = "main",
) -> Dict:
    """전체 테마 크롤링 → 분석 → JSON 저장 파이프라인"""

    def log(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # Step 1: 네이버 전체 테마 목록
    log("📡 [1/6] 네이버 금융 테마 목록 크롤링 중...")
    theme_list = get_naver_theme_list()
    if theme_list.empty:
        log("❌ 테마 목록 크롤링 실패")
        return {}
    if max_themes:
        theme_list = theme_list.head(max_themes)
    log(f"✅ {len(theme_list)}개 테마 확인")

    # Step 2: 테마별 종목 수집
    log("📋 [2/6] 테마별 종목 수집 중...")
    theme_stocks_raw: Dict[int, Dict] = {}
    for i, (_, row) in enumerate(theme_list.iterrows()):
        stocks = get_naver_theme_stocks(int(row["no"]))
        if len(stocks) >= 2:
            theme_stocks_raw[int(row["no"])] = {
                "name": row["name"], "stocks": stocks
            }
        if (i + 1) % 30 == 0:
            log(f"  ↳ {i+1}/{len(theme_list)} 처리 완료...")
        time.sleep(0.25)

    log(f"✅ {len(theme_stocks_raw)}개 테마 종목 확보 (2종목 이상)")

    # Step 3: 고유 종목 코드 수집
    all_codes: Dict[str, str] = {}
    for td in theme_stocks_raw.values():
        for s in td["stocks"]:
            all_codes[s["code"]] = s["name"]
    log(f"📊 [3/6] 고유 종목 {len(all_codes)}개 확인")

    # Step 4: 주가 데이터 로드
    log(f"📈 [4/6] 주가 데이터 수집 중 ({len(all_codes)}개)...")
    end_dt = datetime.today().strftime("%Y-%m-%d")
    start_dt = (datetime.today() - timedelta(days=420)).strftime("%Y-%m-%d")

    benchmark = load_benchmark(start_dt, end_dt)
    stock_dfs: Dict[str, pd.DataFrame] = {}
    stock_exch: Dict[str, str] = {}

    for i, (code, name) in enumerate(all_codes.items()):
        df, exch = load_stock_with_exchange(code, start_dt, end_dt)
        if df is not None:
            stock_dfs[code] = df
            stock_exch[code] = exch
        if (i + 1) % 100 == 0:
            log(f"  ↳ {i+1}/{len(all_codes)} 로드 완료 (성공: {len(stock_dfs)}개)...")
        time.sleep(0.05)

    log(f"✅ {len(stock_dfs)}/{len(all_codes)} 종목 데이터 확보")

    # Step 5-6: 테마별 LSS + 생애주기 계산
    log("🔢 [5/6] LSS 및 생애주기 계산 중...")
    result: Dict[str, Dict] = {}
    now_str = datetime.now().isoformat()

    for i, (theme_no, td) in enumerate(theme_stocks_raw.items()):
        stocks_in_theme = {
            s["code"]: stock_dfs[s["code"]]
            for s in td["stocks"]
            if s["code"] in stock_dfs
        }
        if len(stocks_in_theme) < 2:
            continue

        lss_df = calc_lss(stocks_in_theme, benchmark)
        if not lss_df.empty:
            top_codes = lss_df.head(max_stocks_per_theme)["ticker"].tolist()
        else:
            top_codes = list(stocks_in_theme.keys())[:max_stocks_per_theme]

        stage = compute_lifecycle_stage(stocks_in_theme)

        key = f"naver_{theme_no}"
        result[key] = {
            "name": td["name"],
            "stocks": {
                code: [all_codes.get(code, code), stock_exch.get(code, "KS")]
                for code in top_codes
            },
            "stage": stage,
            "updated_at": now_str,
        }

        if (i + 1) % 30 == 0:
            log(f"  ↳ {i+1}/{len(theme_stocks_raw)} 테마 계산 완료...")

    log(f"✅ {len(result)}개 테마 분석 완료")

    # Step 7: JSON 저장
    log("💾 [6/6] 결과 저장 중...")
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    log(f"✅ {output_path} 저장 완료 ({len(result)}개 테마)")

    # GitHub API 커밋 (Streamlit 버튼에서 실행 시)
    if github_token and github_repo:
        log("🚀 GitHub API 커밋 중...")
        ok = commit_to_github(json_str, github_token, github_repo,
                              "themes_data.json", github_branch)
        log("✅ GitHub 커밋 완료" if ok else "⚠️ GitHub 커밋 실패 (계속 진행)")

    return result


# ─── 단독 실행 엔트리포인트 (GitHub Actions) ─────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 테마 데이터 자동 업데이트 시작")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    result = update_themes(
        output_path="themes_data.json",
        max_themes=None,           # None = 네이버 전체 테마
        max_stocks_per_theme=10,   # 테마별 LSS 상위 10종목
        progress_callback=print,
        # GitHub Actions는 git 명령어로 직접 커밋 → API 토큰 불필요
        github_token="",
        github_repo="",
    )

    print("=" * 60)
    print(f"✅ 완료: {len(result)}개 테마 업데이트")
    print("=" * 60)
