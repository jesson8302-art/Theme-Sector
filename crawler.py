"""
테마주 퀀트 대시보드 — 동적 테마 크롤러
========================================
파이프라인:
  1) 네이버 증권 테마 목록 크롤링
  2) TF-IDF + 코사인 유사도로 우리 테마 ↔ 네이버 테마 매칭
  3) 매칭된 네이버 테마의 구성 종목 수집
  4) yfinance 주가 로드 (KS/KQ 자동 탐지)
  5) LSS 점수 계산 → 상위 종목 선정
  6) 생애주기(Stage) 계산
  7) themes_data.json 저장 + GitHub 자동 커밋

단독 실행:  python crawler.py
앱 내 호출: from crawler import update_themes
"""

import base64
import json
import re
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf


# ─── 테마 키워드 정의 ─────────────────────────────────────────────────────────
THEME_DEFINITIONS: Dict[str, Dict] = {
    "ai_semiconductor": {
        "name": "AI/반도체",
        "keywords": (
            "AI 인공지능 반도체 GPU HBM 메모리 파운드리 칩 DRAM NAND "
            "팹리스 시스템반도체 반도체소재 반도체장비 AI반도체 NPU"
        ),
    },
    "battery": {
        "name": "2차전지",
        "keywords": (
            "배터리 2차전지 리튬 양극재 음극재 전해질 분리막 ESS "
            "배터리셀 LFP NCM 전기차배터리 리튬이온 배터리소재"
        ),
    },
    "bio": {
        "name": "바이오/제약",
        "keywords": (
            "바이오 제약 신약 임상 의약품 항체 세포치료 유전자 "
            "CMO CRO 바이오시밀러 신약개발 바이오텍 제약바이오"
        ),
    },
    "defense": {
        "name": "방산",
        "keywords": (
            "방산 무기 미사일 전투기 함정 장갑차 방위산업 군수 "
            "탄약 K방산 레이더 방위 방어체계 무기체계"
        ),
    },
    "nuclear": {
        "name": "원자력",
        "keywords": (
            "원자력 원전 핵발전 SMR 우라늄 핵융합 원자로 "
            "방사성 핵에너지 원자력발전 소형원자로"
        ),
    },
    "shipbuilding": {
        "name": "조선",
        "keywords": (
            "조선 선박 LNG운반선 컨테이너선 해양플랜트 "
            "조선소 선박엔진 해양 드릴십"
        ),
    },
    "game": {
        "name": "게임",
        "keywords": (
            "게임 온라인게임 모바일게임 게임사 게임개발 "
            "e스포츠 메타버스게임 게임콘텐츠"
        ),
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Referer": "https://finance.naver.com/",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 네이버 증권 크롤링
# ═══════════════════════════════════════════════════════════════════════════════

def get_naver_theme_list() -> pd.DataFrame:
    """네이버 증권 전체 테마 목록 크롤링"""
    url = "https://finance.naver.com/sise/theme.nhn"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  ❌ 테마 목록 크롤링 실패: {e}")
        return pd.DataFrame()

    themes, seen = [], set()
    for a in soup.select("a[href*='sise_group_detail']"):
        href = a.get("href", "")
        m = re.search(r"no=(\d+)", href)
        if m:
            no, name = m.group(1), a.text.strip()
            if no not in seen and name:
                seen.add(no)
                themes.append({"no": no, "name": name})

    return pd.DataFrame(themes)


def get_naver_theme_stocks(theme_no: str) -> List[Dict]:
    """특정 네이버 테마의 구성 종목 목록 크롤링"""
    url = (
        "https://finance.naver.com/sise/sise_group_detail.naver"
        f"?type=theme&no={theme_no}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return []

    stocks, seen = [], set()
    for a in soup.select("a[href*='code=']"):
        m = re.search(r"code=(\d{6})", a.get("href", ""))
        if m:
            code, name = m.group(1), a.text.strip()
            if code not in seen and name:
                seen.add(code)
                stocks.append({"code": code, "name": name})
    return stocks


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TF-IDF 테마 매칭
# ═══════════════════════════════════════════════════════════════════════════════

def match_themes_tfidf(
    naver_df: pd.DataFrame,
    top_k: int = 3,
) -> Dict[str, List[str]]:
    """TF-IDF + 코사인 유사도 → 네이버 테마 번호 매핑"""
    if naver_df.empty:
        return {}

    naver_names  = naver_df["name"].tolist()
    our_keys     = list(THEME_DEFINITIONS.keys())
    our_keywords = [THEME_DEFINITIONS[k]["keywords"] for k in our_keys]

    vec   = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    tfidf = vec.fit_transform(naver_names + our_keywords)

    naver_vecs = tfidf[: len(naver_names)]
    our_vecs   = tfidf[len(naver_names):]
    sim_matrix = cosine_similarity(our_vecs, naver_vecs)

    result: Dict[str, List[str]] = {}
    for i, key in enumerate(our_keys):
        sims    = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:top_k]
        matched_nos   = [naver_df.iloc[j]["no"]   for j in top_idx if sims[j] > 0.05]
        matched_names = [naver_df.iloc[j]["name"] for j in top_idx if sims[j] > 0.05]
        result[key] = matched_nos
        print(f"  {THEME_DEFINITIONS[key]['name']:10s} → {matched_names}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. yfinance 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_stock_with_exchange(
    code: str,
    days: int = 420,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """KS → KQ 순서로 거래소 자동 탐지 후 주가 데이터 로드"""
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    required = ["Open", "High", "Low", "Close", "Volume"]

    for exch in ("KS", "KQ"):
        try:
            df = yf.download(
                f"{code}.{exch}", start=start, end=end,
                progress=False, auto_adjust=True,
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if any(c not in df.columns for c in required):
                continue
            df = df[required].copy()
            df["Volume"] = df["Volume"].fillna(0)
            df = df.dropna(subset=["Close"])
            if len(df) >= 60:
                return df, exch
        except Exception:
            continue
    return None, None


def load_benchmark(days: int = 420) -> Optional[pd.DataFrame]:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = yf.download("^KS11", start=start, end=end,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Close"]].dropna()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LSS 점수 계산
# ═══════════════════════════════════════════════════════════════════════════════

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if mx == mn else (s - mn) / (mx - mn)


LSS_WEIGHTS = {"roc": 0.30, "tv": 0.30, "turnover": 0.15, "beta": 0.10, "wrs": 0.15}


def calc_lss(
    stocks_data: Dict[str, pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
) -> pd.DataFrame:

    def roc(df, p=20):
        if len(df) < p + 1: return np.nan
        p0 = df["Close"].iloc[-(p + 1)]
        return (df["Close"].iloc[-1] - p0) / p0 * 100 if p0 else np.nan

    def tv(df, p=20):
        if len(df) < p: return np.nan
        r = df.tail(p)
        return float((r["Close"] * r["Volume"]).mean())

    def turnover(df, p=20):
        if len(df) < p: return np.nan
        a20 = df["Volume"].tail(20).mean()
        al  = df["Volume"].tail(min(40, len(df))).mean()
        return float(a20 / al) if al > 0 else np.nan

    def beta(df, bm, p=60):
        if len(df) < p or bm is None or len(bm) < p: return np.nan
        s   = df["Close"].pct_change().tail(p).dropna()
        b   = bm["Close"].pct_change().tail(p).dropna()
        idx = s.index.intersection(b.index)
        if len(idx) < 30: return np.nan
        cov = np.cov(s.loc[idx].values, b.loc[idx].values)
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan

    def wrs(df):
        p = df["Close"].iloc[-1]
        r120 = r240 = np.nan
        if len(df) >= 121:
            p0 = df["Close"].iloc[-121]
            r120 = (p - p0) / p0 * 100 if p0 else np.nan
        if len(df) >= 241:
            p0 = df["Close"].iloc[-241]
            r240 = (p - p0) / p0 * 100 if p0 else np.nan
        if pd.notna(r120) and pd.notna(r240):
            return 0.7 * r120 + 0.3 * r240
        return r120 if pd.notna(r120) else np.nan

    records = []
    for code, df in stocks_data.items():
        if df is None or len(df) < 60: continue
        records.append({
            "code":     code,
            "close":    float(df["Close"].iloc[-1]),
            "roc":      roc(df),
            "tv":       tv(df),
            "turnover": turnover(df),
            "beta":     beta(df, benchmark),
            "wrs":      wrs(df),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("code")
    for ind in LSS_WEIGHTS:
        col  = result[ind].copy()
        norm = pd.Series(0.5, index=result.index)
        mask = col.notna()
        if mask.sum() > 1:
            norm[mask] = _minmax(col[mask])
        result[f"{ind}_norm"] = norm

    result["lss"] = sum(result[f"{k}_norm"] * v for k, v in LSS_WEIGHTS.items())
    result = result.sort_values("lss", ascending=False).reset_index()
    result["rank"] = range(1, len(result) + 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 생애주기(Stage) 계산
# ═══════════════════════════════════════════════════════════════════════════════

def _build_theme_index(stocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    p_list, h_list, l_list, v_list = [], [], [], []
    for tk, df in stocks.items():
        if df is None or len(df) < 200: continue
        base = df["Close"].iloc[0]
        if base == 0: continue
        f = 100.0 / base
        p_list.append((df["Close"] * f).rename(tk))
        h_list.append((df["High"]  * f).rename(tk))
        l_list.append((df["Low"]   * f).rename(tk))
        v_list.append(df["Volume"].rename(tk))
    if not p_list:
        return pd.DataFrame()

    def avg(lst):
        return pd.concat(lst, axis=1).dropna(how="all").mean(axis=1)

    return pd.DataFrame({
        "Close": avg(p_list), "High": avg(h_list),
        "Low":   avg(l_list), "Volume": avg(v_list),
    }).dropna(subset=["Close"])


def _ma_slope(ma_s: pd.Series, window: int = 15) -> float:
    if len(ma_s) < window: return 0.0
    y = ma_s.iloc[-window:].values.astype(float)
    slope, *_ = stats.linregress(np.arange(window, dtype=float), y)
    return float(slope / y[-1]) if y[-1] != 0 else 0.0


def _rvol(vol: pd.Series) -> float:
    if len(vol) < 40: return 1.0
    return float(vol.tail(5).mean() / vol.tail(40).mean()) if vol.tail(40).mean() > 0 else 1.0


def _breadth(stocks: Dict[str, pd.DataFrame]) -> float:
    above = total = 0
    for df in stocks.values():
        if df is None or len(df) < 50: continue
        total += 1
        if df["Close"].iloc[-1] > df["Close"].rolling(50).mean().iloc[-1]:
            above += 1
    return (above / total * 100) if total > 0 else 0.0


def _volatility(df: pd.DataFrame, period: int = 20) -> float:
    if len(df) < period: return 0.0
    r   = df.tail(period)
    avg = r["Close"].mean()
    return float(((r["High"] - r["Low"]) / avg).mean() * 100) if avg else 0.0


def compute_lifecycle_stage(stocks_data: Dict[str, pd.DataFrame]) -> int:
    """테마 구성 종목 데이터로 Stage 1~5 반환 (0=데이터 불충분)"""
    theme_df = _build_theme_index(stocks_data)
    if theme_df.empty or len(theme_df) < 200:
        return 0

    price   = float(theme_df["Close"].iloc[-1])
    ma200_s = theme_df["Close"].rolling(200).mean().dropna()
    if ma200_s.empty:
        return 0

    ma200      = float(ma200_s.iloc[-1])
    slope      = _ma_slope(ma200_s)
    rvol       = _rvol(theme_df["Volume"])
    breadth    = _breadth(stocks_data)
    volatility = _volatility(theme_df)
    pct        = (price / ma200 - 1) * 100 if ma200 > 0 else 0.0

    # app.py의 determine_stage()와 동일한 판별 트리
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
        # 근사 판별
        if price < ma200:                      return 5
        elif pct > 5 and slope > 0:            return 2
        elif slope > 0.003 and breadth >= 60:  return 3
        elif price > ma200 and breadth < 60:   return 4
        else:                                  return 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GitHub 자동 커밋
# ═══════════════════════════════════════════════════════════════════════════════

def commit_to_github(
    content: str,
    token: str,
    repo: str,
    file_path: str = "themes_data.json",
    branch: str = "main",
) -> bool:
    """
    GitHub API로 파일 생성/업데이트 커밋.
    token: Personal Access Token (repo 권한 필요)
    repo:  "username/repo-name" 형식
    """
    if not token or not repo:
        return False

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 기존 파일 SHA 조회 (업데이트 시 필수)
    sha = None
    try:
        resp = requests.get(url, headers=headers,
                            params={"ref": branch}, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass

    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    now_str     = datetime.now().strftime("%Y-%m-%d %H:%M KST")
    payload: Dict = {
        "message": f"chore: 테마 데이터 자동 업데이트 ({now_str})",
        "content": content_b64,
        "branch":  branch,
    }
    if sha:
        payload["sha"] = sha  # 업데이트 시 기존 SHA 포함

    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=15)
        return resp.status_code in (200, 201)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════════════════

def update_themes(
    output_path:       str = "themes_data.json",
    max_stocks:        int = 10,
    top_naver_themes:  int = 3,
    progress_callback: Optional[Callable[[str], None]] = None,
    github_token:      str = "",
    github_repo:       str = "",
    github_branch:     str = "main",
) -> Dict:
    """
    전체 파이프라인 실행 → themes_data.json 저장 + GitHub 커밋.

    github_token / github_repo 가 설정된 경우 자동으로 GitHub에 push.
    재분석 시 기존 파일을 완전히 덮어씁니다.
    """

    def log(msg: str) -> None:
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 벤치마크 ────────────────────────────────────────────────────────────
    log("📡 KOSPI 벤치마크 로드 중...")
    benchmark = load_benchmark()
    if benchmark is None:
        log("  ⚠️ 벤치마크 로드 실패 (베타 계산 제외)")

    # ── Step 1: 네이버 테마 목록 ────────────────────────────────────────────
    log("\n① 네이버 증권 테마 목록 크롤링 중...")
    naver_df = get_naver_theme_list()
    if naver_df.empty:
        log("  ❌ 크롤링 실패 — 기존 데이터를 유지합니다.")
        return {}
    log(f"  → {len(naver_df)}개 테마 발견")

    # ── Step 2: TF-IDF 매칭 ─────────────────────────────────────────────────
    log("\n② TF-IDF 테마 매칭 중...")
    theme_to_naver = match_themes_tfidf(naver_df, top_k=top_naver_themes)

    # ── Step 3: 구성 종목 수집 ──────────────────────────────────────────────
    log("\n③ 매칭 테마의 종목 리스트 수집 중...")
    theme_raw: Dict[str, List[Dict]] = {}
    for theme_key, naver_nos in theme_to_naver.items():
        all_stocks: List[Dict] = []
        for no in naver_nos:
            all_stocks.extend(get_naver_theme_stocks(no))
            time.sleep(0.4)

        seen: set = set()
        uniq: List[Dict] = []
        for s in all_stocks:
            if s["code"] not in seen:
                seen.add(s["code"])
                uniq.append(s)

        theme_raw[theme_key] = uniq
        name = THEME_DEFINITIONS[theme_key]["name"]
        log(f"  {name}: {len(uniq)}개 종목 후보")

    # ── Step 4: 주가 데이터 로드 ────────────────────────────────────────────
    all_codes: set = set()
    for stocks in theme_raw.values():
        all_codes.update(s["code"] for s in stocks)

    log(f"\n④ 주가 데이터 로드 중 (총 {len(all_codes)}개)...")
    log("  ⏳ 종목 수에 따라 수 분 소요됩니다.")

    code_to_df:   Dict[str, pd.DataFrame] = {}
    code_to_exch: Dict[str, str]          = {}

    for i, code in enumerate(sorted(all_codes)):
        df, exch = load_stock_with_exchange(code)
        if df is not None and exch is not None:
            code_to_df[code]   = df
            code_to_exch[code] = exch
        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(all_codes)} 완료 ({len(code_to_df)}개 확보)")
        time.sleep(0.05)

    log(f"  → 최종 {len(code_to_df)}개 종목 확보")

    # ── Step 5 & 6: LSS + 생애주기 계산 ────────────────────────────────────
    log("\n⑤ LSS 점수 계산 및 생애주기 판별 중...")
    final_data: Dict = {}

    for theme_key, raw_stocks in theme_raw.items():
        name     = THEME_DEFINITIONS[theme_key]["name"]
        name_map = {s["code"]: s["name"] for s in raw_stocks}
        avail    = {c: code_to_df[c] for c in name_map if c in code_to_df}

        if len(avail) < 3:
            log(f"  {name}: 데이터 부족 ({len(avail)}개) — 건너뜀")
            continue

        # LSS 계산 → 상위 종목 선정
        lss_df = calc_lss(avail, benchmark)
        if lss_df.empty:
            log(f"  {name}: LSS 계산 실패")
            continue

        top = lss_df.head(max_stocks)
        stocks_out: Dict[str, List] = {}
        for _, row in top.iterrows():
            code = row["code"]
            stocks_out[code] = [
                name_map.get(code, code),
                code_to_exch.get(code, "KS"),
            ]

        # 생애주기 Stage 계산 (상위 종목들로)
        top_avail = {c: code_to_df[c] for c in stocks_out if c in code_to_df}
        stage = compute_lifecycle_stage(top_avail)

        final_data[theme_key] = {
            "name":       name,
            "stocks":     stocks_out,
            "stage":      stage,
            "updated_at": datetime.now().isoformat(),
        }
        stage_names = {0:"데이터부족",1:"바닥매집",2:"모멘텀돌파",
                       3:"주도상승",4:"과열분배",5:"항복투매"}
        log(f"  ✅ {name}: {len(stocks_out)}개 선정 | Stage {stage} ({stage_names.get(stage,'')})")

    # ── Step 7: JSON 저장 ───────────────────────────────────────────────────
    if not final_data:
        log("\n⚠️ 선정된 종목이 없어 파일을 저장하지 않았습니다.")
        return {}

    json_content = json.dumps(final_data, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_content)
    log(f"\n💾 로컬 저장 완료 → {output_path}")

    # ── Step 8: GitHub 자동 커밋 (토큰 설정 시) ────────────────────────────
    if github_token and github_repo:
        log(f"☁️  GitHub({github_repo})에 저장 중...")
        ok = commit_to_github(
            content=json_content,
            token=github_token,
            repo=github_repo,
            file_path=output_path,
            branch=github_branch,
        )
        if ok:
            log("  ✅ GitHub 저장 완료! (앱 재시작 후에도 데이터 유지)")
        else:
            log("  ⚠️ GitHub 저장 실패 — 토큰·저장소 설정을 확인하세요.")
    else:
        log("ℹ️  GitHub 토큰 미설정 — 로컬 저장만 완료.")

    return final_data


# ─── 단독 실행 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("테마주 동적 크롤러")
    print("=" * 60)
    result = update_themes()
    if result:
        print("\n[선정 결과 요약]")
        for key, val in result.items():
            top3 = list(val["stocks"].values())[:3]
            print(f"  Stage{val['stage']} {val['name']:10s}: {[v[0] for v in top3]}")
    print("=" * 60)
