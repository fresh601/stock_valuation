# -*- coding: utf-8 -*-
"""
네이버(와이즈리포트) 자동 수집 + 적정주가 계산 최종본 (현재가는 수동 입력)
- encparam/id 자동 획득(Selenium) → main/fs/profit/value 자동 수집
- EPS/BPS/EBITDA/FCF₀/순부채/발행주식수 추출
  · EBITDA: '당기/최근/TTM' 우선 → (E) → 최근 실적, 그리고 '마진(%)' 행은 제외
  · FCF₀: main_wide의 FCF(자유/잉여현금흐름) 우선 → value의 FCF → CFO±CAPEX 파생(부호 자동 보정)
- 모든 표 단위(UNIT)를 '원' 기준으로 환산
- DCF(3구간 성장 + TV) + 상대가치(PER/PBR/EV/EBITDA) + MIX(가중치)
- 시나리오 버튼(보수/기준/낙관)으로 성장률·할인율·안전마진 자동 세팅
- 현재가 수동 입력 → 현재가/적정가/상승여력 카드 표시
- Plotly graph_objects 기반 안전 차트(safe_bar_go)
- 결과 엑셀에 META 기록(선택열/출처/파라미터/가중치 등)

실행:
  pip install -r requirements.txt
  streamlit run streamlit_valuation_final.py
"""

import io
import re
import time
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from collections import defaultdict
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

st.set_page_config(page_title="적정주가 계산기 · 최종본", layout="wide")

# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────
UNIT_MAP = {
    '원': 1.0,
    '천원': 1e3,
    '만원': 1e4,
    '백만원': 1e6,   # 공급 포맷 혼재 대비(표 단위 문자열을 기준으로 환산)
    '억원': 1e8,
    '십억원': 1e9,
    '백억원': 1e10,
    '천억원': 1e11,
    '조원': 1e12,
}

def to_number(s):
    if s is None:
        return None
    s = str(s).strip()
    if s in ("", "-"):
        return None
    s = s.replace(",", "")
    m = re.fullmatch(r"\(([-+]?\d*\.?\d+)\)", s)  # (1,234) → -1234
    if m:
        return -float(m.group(1))
    try:
        return float(s)
    except Exception:
        return None

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").replace("\xa0", " ").strip())

def scale_by_unit(df: pd.DataFrame, unit_col: str = '단위') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if unit_col not in df.columns:
        num_cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
        df[num_cols] = df[num_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors='coerce')
        return df
    unit_str = str(df[unit_col].iloc[0])
    mul = 1.0
    for k, v in UNIT_MAP.items():
        if k in unit_str:
            mul = v
            break
    num_cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    df[num_cols] = df[num_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors='coerce') * mul
    return df

def pick_prefer_current_then_estimate(row: pd.Series):
    """열 선택 우선순위: 당기/최근/TTM/12M → (E)/Estimate/예상/FWD → 일반 오른쪽값"""
    cols = list(row.index)
    prefer_now = [i for i, c in enumerate(cols) if re.search(r'당기|최근|TTM|12M', str(c), re.I)]
    for i in reversed(prefer_now):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v), cols[i], 'current'
    prefer_est = [i for i, c in enumerate(cols) if re.search(r'\(E\)|Estimate|예상|FWD|Forward', str(c), re.I)]
    for i in reversed(prefer_est):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v), cols[i], 'estimate'
    for i in range(len(cols) - 1, -1, -1):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v), cols[i], 'actual'
    return None, None, None

# ──────────────────────────────────────────────────────────────
# 안전 차트(Plotly graph_objects)
# ──────────────────────────────────────────────────────────────

import pandas as pd
import plotly.graph_objects as go

def safe_bar_go(df: pd.DataFrame, x: str, y: str, title: str = None, eps: float = 1e-9):
    """
    안전하게 막대 그래프를 만드는 함수
    - df: 데이터프레임
    - x : x축 컬럼명
    - y : y축 컬럼명
    - title : 그래프 제목
    - eps : 임계치(이 값보다 작은 절댓값은 0으로 간주하고 숨김)
    """
    df2 = df.copy()
    df2[y] = pd.to_numeric(df2[y], errors='coerce')
    # 거의 0에 수렴하는 값은 NaN 처리 → 막대 숨김
    df2.loc[df2[y].abs() < eps, y] = pd.NA
    df2 = df2.dropna(subset=[y])
    if df2.empty:
        return None

    x_vals = df2[x].astype(str).tolist()
    y_vals = df2[y].astype(float).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        text=[f"{v:,.2f}" for v in y_vals],
        textposition="auto"
    ))
    fig.update_layout(
        title=title or "",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_white"
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Selenium: encparam / id
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_encparam_and_id(cmp_cd: str, page_key: str) -> dict:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        url = f"https://navercomp.wisereport.co.kr/v2/company/{page_key}.aspx?cmp_cd={cmp_cd}"
        driver.get(url)
        time.sleep(2.2)
        html = driver.page_source
        enc_match = re.search(r"encparam\s*:\s*['\"]?([a-zA-Z0-9+/=]+)['\"]?", html)
        id_match = re.search(r"cmp_cd\s*=\s*['\"]?([0-9]+)['\"]?", html)
        return {"cmp_cd": cmp_cd, "encparam": enc_match.group(1) if enc_match else None, "id": id_match.group(1) if id_match else None}
    finally:
        driver.quit()

# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────

def fetch_main_table(cmp_cd: str, encparam: str, cmp_id: str):
    url = "https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx"
    headers = {'Accept': 'application/json, text/html, */*; q=0.01','User-Agent': 'Mozilla/5.0','X-Requested-With': 'XMLHttpRequest','Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={cmp_cd}'}
    params = {'cmp_cd': cmp_cd,'fin_typ': '0','freq_typ': 'Y','encparam': encparam,'id': cmp_id}
    res = requests.get(url, headers=headers, params=params, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    tables = soup.select("table.gHead01.all-width")
    target = next((tb for tb in tables if "연간" in clean_text(tb.get_text(" ")) or re.search(r"20\d\d", tb.get_text(" "))), None)
    if not target:
        raise ValueError("연간 주요재무정보 테이블을 찾지 못했습니다.")
    thead_rows = target.select("thead tr")
    year_cells = thead_rows[-1].find_all(["th", "td"]) if thead_rows else []
    year_counter = defaultdict(int)
    years = []
    for th in year_cells:
        t = clean_text(th.get_text(" "))
        if t and not re.search(r"주요재무정보|구분", t):
            year_counter[t] += 1
            suffix = f"_{year_counter[t]}" if year_counter[t] > 1 else ""
            years.append(t + suffix)
    rows = []
    for tr in target.select("tbody tr"):
        th = tr.find("th")
        if not th:
            continue
        metric = clean_text(th.get_text(" "))
        tds = tr.find_all("td")
        values = []
        for i in range(len(years)):
            if i < len(tds):
                raw = tds[i].get("title") or clean_text(tds[i].get_text(" "))
                values.append(to_number(raw))
            else:
                values.append(None)
        rows.append([metric] + values)
    df_wide = pd.DataFrame(rows, columns=["지표"] + years).set_index("지표")
    return df_wide


def parse_json_table(js: dict) -> pd.DataFrame:
    data = js.get("DATA", [])
    labels_raw = js.get("YYMM", [])
    unit = js.get("UNIT", "")
    if not data:
        return pd.DataFrame()
    labels = [re.sub(r"<br\s*/?>", " ", l).strip() for l in labels_raw]
    year_keys = sorted([k for k in data[0] if re.match(r"^DATA\d+$", k)], key=lambda x: int(x[4:]))
    if len(labels) < len(year_keys):
        labels += [f"DATA{i+1}" for i in range(len(labels), len(year_keys))]
    rows = [[r.get("ACC_NM", "")] + [r.get(k, "") for k in year_keys] for r in data]
    df = pd.DataFrame(rows, columns=["항목"] + labels[:len(year_keys)])
    df.insert(1, "단위", unit)
    return df


def fetch_json_mode(cmp_cd: str, mode: str, encparam: str) -> pd.DataFrame:
    url = "https://navercomp.wisereport.co.kr/v2/company/cF3002.aspx" if mode == "fs" else "https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx"
    rpt_map = {"fs": "1", "profit": "1", "value": "5"}
    headers = {'Accept': 'application/json, text/html, */*; q=0.01','User-Agent': 'Mozilla/5.0','X-Requested-With': 'XMLHttpRequest','Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={cmp_cd}'}
    params = {'cmp_cd': cmp_cd, 'frq': '0', 'rpt': rpt_map[mode], 'finGubun': 'MAIN', 'frqTyp': '0', 'cn': '', 'encparam': encparam}
    res = requests.get(url, params=params, headers=headers, timeout=20)
    res.raise_for_status()
    try:
        js = res.json()
    except json.JSONDecodeError:
        return pd.DataFrame()
    return parse_json_table(js)

# ──────────────────────────────────────────────────────────────
# 값 추출 헬퍼
# ──────────────────────────────────────────────────────────────

def infer_from_main(df_main_wide: pd.DataFrame, patterns: list[str]):
    if df_main_wide is None or df_main_wide.empty:
        return None, None, None
    idx = df_main_wide.index.astype(str)
    for p in patterns:
        mask = idx.str.contains(p, case=False, regex=True)
        if mask.any():
            row = df_main_wide.loc[mask].iloc[0]
            return pick_prefer_current_then_estimate(row)
    return None, None, None


def pick_latest_from_table(df: pd.DataFrame, patterns: list[str]):
    if df is None or df.empty:
        return None, None, None
    cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    for p in patterns:
        mask = df["항목"].astype(str).str.contains(p, case=False, regex=True, na=False)
        if mask.any():
            row = df.loc[mask].iloc[0][cols]
            return pick_prefer_current_then_estimate(row)
    return None, None, None


def pick_from_table_with_exclude(df: pd.DataFrame, include_patterns: list[str], exclude_patterns: list[str]):
    if df is None or df.empty:
        return None, None, None
    cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    s = df["항목"].astype(str)
    mask_inc = None
    for p in include_patterns:
        m = s.str.contains(p, case=False, regex=True, na=False)
        mask_inc = m if mask_inc is None else (mask_inc | m)
    if mask_inc is None or not mask_inc.any():
        return None, None, None
    if exclude_patterns:
        for q in exclude_patterns:
            excl = s.str.contains(q, case=False, regex=True, na=False)
            mask_inc = mask_inc & (~excl)
    if not mask_inc.any():
        return None, None, None
    row = df.loc[mask_inc].iloc[0][cols]
    return pick_prefer_current_then_estimate(row)

# ── FCF/CFO/CAPEX 탐색 + 조합 ─────────────────────────────────

def find_cfo_any(df_value, df_fs):
    val, col, _ = pick_latest_from_table(df_value, [r'영업활동.*현금흐름|영업활동으로인한현금흐름|^CFO$|CFO<당기>|CFO＜당기＞'])
    if val is not None:
        return float(val), f"value:{col}"
    val, col, _ = pick_latest_from_table(df_fs, [r'영업활동.*현금흐름|영업활동으로인한현금흐름|^CFO$'])
    if val is not None:
        return float(val), f"fs:{col}"
    return None, None

def find_capex_any(df_value, df_fs):
    val, col, _ = pick_latest_from_table(df_fs, [r'\*?CAPEX|유형자산의\s*취득|설비투자|유형자산.*취득'])
    if val is not None:
        return float(val), f"fs:{col}"
    val, col, _ = pick_latest_from_table(df_value, [r'\*?CAPEX|유형자산의\s*취득|설비투자|유형자산.*취득'])
    if val is not None:
        return float(val), f"value:{col}"
    return None, None

def combine_fcf(cfo, capex):
    if cfo is None or capex is None:
        return None
    return float(cfo - abs(capex)) if capex >= 0 else float(cfo + capex)


def find_fcf_any(df_main, df_value, df_fs):
    # 1) main_wide의 FCF/자유·잉여현금흐름
    if df_main is not None and not df_main.empty:
        idx = df_main.index.astype(str)
        mask = idx.str.contains(r'FCF|자유현금흐름|잉여현금흐름|Free\s*Cash\s*Flow', case=False, regex=True)
        if mask.any():
            row = df_main.loc[mask].iloc[0]
            val, col, typ = pick_prefer_current_then_estimate(row)
            if val is not None:
                return float(val), f"main:{col}", "direct-main"
    # 2) value의 FCF
    fcf0, fcf_col, _ = pick_latest_from_table(df_value, [r'FCF|Free\s*Cash\s*Flow|자유현금흐름|잉여현금흐름'])
    if fcf0 is not None:
        return float(fcf0), f"value:{fcf_col}", "direct-value"
    # 3) CFO ± CAPEX 파생
    cfo, cfo_col = find_cfo_any(df_value, df_fs)
    capex, capex_col = find_capex_any(df_value, df_fs)
    if cfo is not None and capex is not None:
        fcf0 = combine_fcf(cfo, capex)
        sign = '-' if capex >= 0 else '+'
        return float(fcf0), f"CFO[{cfo_col}] {sign} CAPEX[{capex_col}]", "derived"
    return None, None, None

# ──────────────────────────────────────────────────────────────
# 핵심 값 패키징
# ──────────────────────────────────────────────────────────────

def extract_core_numbers(df_main, df_fs, df_profit, df_value):
    # 발행주식수
    shares, shares_col, shares_used = infer_from_main(df_main, [r"발행주식수|주식수|보통주수|총발행주식"])
    # 순부채
    net_debt, nd_col, nd_used = pick_latest_from_table(df_fs, [r"^\*?순부채", r"Net\s*Debt"])
    # EPS/BPS
    eps, eps_col, eps_type = pick_latest_from_table(df_value, [r"EPS"]) or (None, None, None)
    if eps is None:
        eps, eps_col, eps_type = infer_from_main(df_main, [r"EPS"])  
    bps, bps_col, bps_type = pick_latest_from_table(df_value, [r"BPS"]) or (None, None, None)
    if bps is None:
        bps, bps_col, bps_type = infer_from_main(df_main, [r"BPS"])  
    # EBITDA — 마진 제외 + 당기 우선
    ebitda, e_col, e_type = pick_from_table_with_exclude(
        df_profit,
        include_patterns=[r'^\s*EBITDA\s*$', r'EBITDA\s*\(.*\)$', r'\bEBITDA\b'],
        exclude_patterns=[r'마진|margin|%']
    )
    if ebitda is None:
        ebitda, e_col, e_type = infer_from_main(df_main, [r"^\s*EBITDA\s*$", r"\bEBITDA\b", r"EBITDA\s*\(.*\)$"])
    # 백업: 영업이익 + 상각비
    if ebitda is None:
        op_inc, op_col, _ = pick_from_table_with_exclude(df_profit, [r'영업이익|Operating\s*Income|OP'], [r'율|마진|margin|%'])
        da, da_col, _ = pick_from_table_with_exclude(df_profit, [r'감가상각비|상각비|Depreciation|Amortization|D&A|DA'], [r'율|마진|margin|%'])
        if op_inc is not None and da is not None:
            ebitda = float(op_inc) + float(da)
            e_col = f"영업이익[{op_col}] + 상각비[{da_col}]"
            e_type = "derived"
    # FCF₀ — main → value → 파생(CFO±CAPEX)
    fcf0, fcf_col, fcf_type = find_fcf_any(df_main, df_value, df_fs)

    meta_cols = {
        'shares_col': shares_col, 'shares_type': shares_used,
        'net_debt_col': nd_col, 'net_debt_type': nd_used,
        'eps_col': eps_col, 'eps_type': eps_type,
        'bps_col': bps_col, 'bps_type': bps_type,
        'ebitda_col': e_col, 'ebitda_type': e_type,
        'fcf_col': fcf_col, 'fcf_type': fcf_type,
    }
    return {"shares": shares, "net_debt": net_debt, "eps": eps, "bps": bps, "ebitda": ebitda, "fcf0": fcf0, "meta_cols": meta_cols}

# ──────────────────────────────────────────────────────────────
# Valuation 엔진
# ──────────────────────────────────────────────────────────────

def dcf_fair_price(fcf0, g_high, g_mid, g_low, g_tv, r, shares, net_debt, safety=0.0):
    if not shares or shares <= 0 or not r or r <= 0 or fcf0 is None:
        return None, None, None, None
    years = list(range(1, 11))
    growths = [g_high if y <= 3 else (g_mid if y <= 6 else g_low) for y in years]
    fcfs, last = [], float(fcf0)
    for g in growths:
        last = last * (1.0 + g)
        fcfs.append(last)
    disc = [(1.0 / ((1.0 + r) ** t)) for t in years]
    pv_fcfs = [f * d for f, d in zip(fcfs, disc)]
    tv = fcfs[-1] * (1.0 + g_tv) / (r - g_tv) if r > g_tv else np.nan
    pv_tv = tv * disc[-1] if np.isfinite(tv) else 0.0
    ev = float(np.nansum(pv_fcfs) + pv_tv)
    equity = ev - (net_debt or 0.0)
    per_share = equity / shares
    target = per_share * (1.0 - (safety or 0.0))
    tv_fcf_display = fcfs[-1] * (1.0 + g_tv) if np.isfinite(tv) else np.nan
    detail = pd.DataFrame({"Year": years + ["TV"], "FCF": fcfs + [tv_fcf_display], "Discount": disc + [disc[-1]], "PV": pv_fcfs + [pv_tv]})
    return float(target), float(ev), float(equity), detail

def per_price(eps, per, safety=0.0):
    if eps is None or per is None:
        return None
    return float(eps * per * (1.0 - (safety or 0.0)))

def pbr_price(bps, pbr, safety=0.0):
    if bps is None or pbr is None:
        return None
    return float(bps * pbr * (1.0 - (safety or 0.0)))

def evebitda_price(ebitda, ev_ebitda, shares, net_debt, safety=0.0):
    if ebitda is None or ev_ebitda is None or not shares:
        return None
    ev = ebitda * ev_ebitda
    equity = ev - (net_debt or 0.0)
    per_share = equity / shares
    return float(per_share * (1.0 - (safety or 0.0)))


def fair_price_ev_ebitda(ev_multiple: float,
                         ebitda: float | None,
                         net_debt: float | None,
                         shares_out: float | None) -> float | None:
    """EBITDA 없거나 ≤0, 주식수 ≤0이면 계산 중단."""
    if ebitda is None or not np.isfinite(ebitda) or ebitda <= 0:
        return None
    if net_debt is None or not np.isfinite(net_debt):
        net_debt = 0.0
    if shares_out is None or not np.isfinite(shares_out) or shares_out <= 0:
        return None
    ev_fair = float(ev_multiple) * float(ebitda)
    equity_fair = ev_fair - float(net_debt)
    px = equity_fair / float(shares_out)
    return float(px) if np.isfinite(px) else None



def _pick_latest_numeric_row(row, prefer_col: str | None = None):
    if prefer_col and prefer_col in row.index:
        v = pd.to_numeric(row.get(prefer_col), errors='coerce')
        if pd.notna(v):
            return float(v)
    for col in reversed(row.index[1:]):  # 오른쪽(최신)부터
        v = pd.to_numeric(row.get(col), errors='coerce')
        if pd.notna(v):
            return float(v)
    return np.nan

def resolve_fcf0(tbl_cashflow: pd.DataFrame | None,
                 tbl_profit: pd.DataFrame | None = None,
                 prefer_col: str | None = None) -> float | None:
    """
    1) 현금흐름표에서 FCF(자유/잉여현금흐름/FCF) 직접 탐색
    2) 없으면 근사: (영업활동현금흐름 - 유형자산취득(|절댓값|))
    """
    if tbl_cashflow is not None and not tbl_cashflow.empty:
        df = tbl_cashflow.copy()
        df.columns = [str(c).strip() for c in df.columns]
        label_col = df.columns[0]

        pat_fcf = r'(free\s*cash\s*flow|fcf|자유\s*현금\s*흐름|잉여\s*현금\s*흐름)'
        m_fcf = df[label_col].astype(str).str.contains(pat_fcf, flags=re.I, regex=True, na=False)
        if m_fcf.any():
            v = df.loc[m_fcf].apply(lambda r: _pick_latest_numeric_row(r, prefer_col), axis=1).dropna()
            if not v.empty and np.isfinite(v.iloc[0]):
                return float(v.iloc[0])

        pat_cfo   = r'(영업활동현금흐름|영업활동으로인한현금흐름|CFO\b)'
        pat_capex = r'(유형자산[의]*\s*취득|CAPEX|유형자산의취득|유형자산취득)'
        m_cfo   = df[label_col].astype(str).str.contains(pat_cfo, flags=re.I, regex=True, na=False)
        m_capex = df[label_col].astype(str).str.contains(pat_capex, flags=re.I, regex=True, na=False)

        def _pick(mask):
            if not mask.any(): return np.nan
            row = df.loc[mask].iloc[0]
            return _pick_latest_numeric_row(row, prefer_col)

        cfo   = _pick(m_cfo)
        capex = _pick(m_capex)
        if np.isfinite(cfo):
            capex = abs(capex) if np.isfinite(capex) else np.nan
            return float(cfo - (capex if np.isfinite(capex) else 0.0))

    if tbl_profit is not None and not tbl_profit.empty:
        df = tbl_profit.copy()
        df.columns = [str(c).strip() for c in df.columns]
        label_col = df.columns[0]
        m = df[label_col].astype(str).str.contains(
            r'(free\s*cash\s*flow|fcf|자유\s*현금\s*흐름|잉여\s*현금\s*흐름)',
            flags=re.I, regex=True, na=False
        )
        if m.any():
            v = df.loc[m].apply(lambda r: _pick_latest_numeric_row(r, prefer_col), axis=1).dropna()
            if not v.empty and np.isfinite(v.iloc[0]):
                return float(v.iloc[0])

    return None



# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

st.title("📈 적정주가 계산기 · 최종본")
st.caption("현재가는 직접 입력하고, EBITDA는 '당기' 금액 우선·마진 제외, FCF는 main→value→파생 순으로 결정합니다.")

with st.sidebar:
    st.header("입력")
    cmp_cd = st.text_input("종목코드 (6자리)", value="066570")
    current_price = st.number_input("현재가 (원)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
    run = st.button("자동 수집 → 계산", type="primary")

    st.header("시나리오")
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = '기준'
    c1, c2, c3 = st.columns(3)
    if c1.button("보수"): st.session_state['scenario'] = '보수'
    if c2.button("기준"): st.session_state['scenario'] = '기준'
    if c3.button("낙관"): st.session_state['scenario'] = '낙관'
    st.write(f"선택된 시나리오: **{st.session_state['scenario']}**")

    def scenario_params(name: str):
        if name == '보수':
            return dict(g_high=0.08, g_mid=0.05, g_low=0.03, g_tv=0.02, r=0.10, safety=0.35)
        if name == '낙관':
            return dict(g_high=0.18, g_mid=0.12, g_low=0.06, g_tv=0.035, r=0.08, safety=0.25)
        return dict(g_high=0.15, g_mid=0.10, g_low=0.05, g_tv=0.03, r=0.09, safety=0.30)

    par = scenario_params(st.session_state['scenario'])

    st.subheader("DCF 파라미터")
    g_high = st.number_input("고성장률 (Y1~Y3)", value=par['g_high'], step=0.005)
    g_mid  = st.number_input("중간성장률 (Y4~Y6)", value=par['g_mid'], step=0.005)
    g_low  = st.number_input("저성장률 (Y7~Y10)", value=par['g_low'], step=0.005)
    g_tv   = st.number_input("장기성장률 g (TV)", value=par['g_tv'], step=0.005)
    r      = st.number_input("할인율 r (WACC)", value=par['r'], step=0.005)
    safety = st.number_input("안전마진", value=par['safety'], step=0.05)

    st.subheader("상대가치 배수")
    per_mult = st.number_input("업종 PER", value=12.0, step=0.5)
    pbr_mult = st.number_input("업종 PBR", value=1.2, step=0.1)
    ev_mult  = st.number_input("EV/EBITDA", value=7.0, step=0.5)

    st.subheader("MIX 가중치")
    w_dcf = st.slider("DCF", 0.0, 1.0, 0.4, 0.05)
    w_per = st.slider("PER", 0.0, 1.0, 0.2, 0.05)
    w_pbr = st.slider("PBR", 0.0, 1.0, 0.2, 0.05)
    w_ev  = st.slider("EV/EBITDA", 0.0, 1.0, 0.2, 0.05)

if run:
    if not re.fullmatch(r"\d{6}", cmp_cd):
        st.error("종목코드는 6자리 숫자여야 합니다. 예: 005930 / 066570")
        st.stop()

    page_key = "c1010001"
    with st.spinner("토큰 획득 중..."):
        tk = get_encparam_and_id(cmp_cd, page_key)
    encparam, cmp_id = tk.get("encparam"), tk.get("id")
    cA, cB, cC = st.columns(3)
    cA.metric("종목코드", cmp_cd)
    cB.metric("encparam", (encparam[:10] + "…") if encparam else "없음")
    cC.metric("id", cmp_id or "없음")
    if not encparam or not cmp_id:
        st.warning("토큰 추출 실패. 잠시 후 재시도.")
        st.stop()

    with st.spinner("데이터 수집(main/fs/profit/value)..."):
        df_main   = fetch_main_table(cmp_cd, encparam, cmp_id)
        df_fs     = fetch_json_mode(cmp_cd, "fs", encparam)
        df_profit = fetch_json_mode(cmp_cd, "profit", encparam)
        df_value  = fetch_json_mode(cmp_cd, "value", encparam)

    # 단위 환산(원)
    df_fs = scale_by_unit(df_fs)
    df_profit = scale_by_unit(df_profit)
    df_value = scale_by_unit(df_value)

    with st.spinner("핵심 값 추출 중..."):
        core = extract_core_numbers(df_main, df_fs, df_profit, df_value)

    st.subheader("🔑 추출된 핵심 입력값")
    core_view = pd.DataFrame([{
        '발행주식수(주)': core['shares'],
        '순부채(원)': core['net_debt'],
        'EPS(예상/실적)': core['eps'],
        'BPS(예상/실적)': core['bps'],
        'EBITDA(원)': core['ebitda'],
        'FCF₀(원)': core['fcf0'],
    }])
    st.dataframe(core_view, use_container_width=True)

    with st.expander("선택된 열/출처(META)"):
        st.json(core['meta_cols'])

    # DCF 가드: 왜 비활성인지 즉시 표기
    missing = []
    if core.get("fcf0") is None: missing.append("FCF₀")
    if core.get("shares") in (None, 0, np.nan): missing.append("주식수")
    if r in (None, 0, np.nan): missing.append("할인율 r")
    if missing:
      st.warning("DCF 계산이 비활성화된 이유: " + ", ".join(missing))
    if core.get("fcf0") is None:
      core["fcf0"] = resolve_fcf0(tbl_cashflow=df_fs, tbl_profit=df_profit, prefer_col=None)

    # Valuation
    # --- Valuation (정규화 + 가드 버전) ---------------------------------------

    def _to_float(x, default=None):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return default
            if isinstance(x, str):
                x = x.replace(",", "").strip()
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default
    
    def _rate(x):
        """퍼센트로 들어오면 소수로 변환(예: 9 → 0.09), 이미 소수면 그대로."""
        v = _to_float(x, default=None)
        if v is None:
            return None
        return v/100.0 if v > 1.0 else v
    
    def _clamp_r_gt_g(r0, g0, eps=1e-4):
        """DCF 성립 조건(r > g_tv) 보장."""
        if r0 is None or g0 is None:
            return r0, g0
        return (r0, min(g0, r0 - eps)) if r0 <= g0 else (r0, g0)
    
    # 0) 코어 입력 정규화
    fcf0     = _to_float(core.get("fcf0"))
    ebitda   = _to_float(core.get("ebitda"))
    shares   = _to_float(core.get("shares"))
    net_debt = _to_float(core.get("net_debt"), default=0.0)
    
    # 1) 성장률/할인율 정규화(퍼센트 → 소수) + r>g_tv 보정
    r_n      = _rate(r)
    gH, gM, gL = _rate(g_high), _rate(g_mid), _rate(g_low)
    g_tv_n   = _rate(g_tv)
    r_n, g_tv_n = _clamp_r_gt_g(r_n, g_tv_n)
    
    # 2) EV/EBITDA 배수 정규화
    ev_mult_n = _to_float(ev_mult)
    ev_mult_n = ev_mult_n if (ev_mult_n is not None and ev_mult_n > 0) else None
    
    # 3) DCF 계산 (기존 함수 사용)
    px_dcf, ev, equity, dcf_detail = dcf_fair_price(
        fcf0=fcf0,
        g_high=gH, g_mid=gM, g_low=gL,
        g_tv=g_tv_n, r=r_n,
        shares=shares, net_debt=net_debt, safety=safety
    )
    
    # 4) PER/PBR (기존 그대로)
    px_per = per_price(core["eps"], per_mult, safety=safety)
    px_pbr = pbr_price(core["bps"], pbr_mult, safety=safety)
    
    # 5) EV/EBITDA 가드 버전 사용(이미 4-2로 추가해 둔 함수)
    px_ev  = fair_price_ev_ebitda(
        ev_multiple=ev_mult_n,
        ebitda=ebitda,
        net_debt=net_debt,
        shares_out=shares
    )
    
    # 6) MIX 재가중(값 있는 항목만)
    pairs = [(px_dcf, w_dcf), (px_per, w_per), (px_pbr, w_pbr), (px_ev, w_ev)]
    use_wsum = sum(w for px, w in pairs if px is not None) or 1.0
    mix_price = (float(np.nansum([px * (w / use_wsum) for px, w in pairs if px is not None]))
                 if any(px is not None for px, _ in pairs) else None)
    
    # (선택) 디버그 캡션: 계산 직전 값 점검
    st.caption(
        f"DBG ▶ shares={shares}, r={r_n}, g_tv={g_tv_n}, ev_mult={ev_mult_n}, "
        f"FCF0={fcf0}, EBITDA={ebitda}, DCF_px={px_dcf}, EV/EBITDA_px={px_ev}"
    )
# -------------------------------------------------------------------------


    st.subheader("📌 적정주가 요약")
    summary = pd.DataFrame({"방법": ["DCF", "PER", "PBR", "EV/EBITDA", "MIX(가중)"], "적정주가": [px_dcf, px_per, px_pbr, px_ev, mix_price]})
    fig = safe_bar_go(summary, "방법", "적정주가", title="방법별 적정주가")
    if fig is None:
        st.info("표시할 적정주가 값이 없습니다.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("현재가", f"{current_price:,.2f} 원")
    col2.metric("적정가(MIX)", f"{(mix_price or 0):,.2f} 원")
    up = None if not current_price or not mix_price else (mix_price / current_price - 1.0) * 100.0
    col3.metric("상승여력", f"{up:.2f}%" if up is not None else "-")

    st.subheader("DCF 세부내역")
    if dcf_detail is not None:
        st.dataframe(dcf_detail, use_container_width=True)
    else:
        st.info("DCF 계산을 위해 FCF₀/주식수/할인율이 필요합니다.")

    # 결과 엑셀 생성
    meta = {
        'ticker': cmp_cd,
        'scenario': st.session_state['scenario'],
        'params': dict(g_high=g_high, g_mid=g_mid, g_low=g_low, g_tv=g_tv, r=r, safety=safety),
        'multiples': dict(PER=per_mult, PBR=pbr_mult, EV_EBITDA=ev_mult),
        'weights': dict(DCF=w_dcf, PER=w_per, PBR=w_pbr, EVEBITDA=w_ev),
        'selected_cols': core['meta_cols'],
    }

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        summary.to_excel(wr, sheet_name="SUMMARY", index=False)
        pd.DataFrame([{'현재가': current_price, '적정가(MIX)': mix_price, '상승여력%': up}]).to_excel(wr, sheet_name="PRICE", index=False)
        pd.DataFrame([{'발행주식수(주)': core['shares'], '순부채(원)': core['net_debt'], 'EPS': core['eps'], 'BPS': core['bps'], 'EBITDA(원)': core['ebitda'], 'FCF₀(원)': core['fcf0']}]).to_excel(wr, sheet_name="CORE_INPUTS", index=False)
        try:
            df_main.reset_index().to_excel(wr, sheet_name="MAIN_SNAPSHOT", index=False)
            df_fs.to_excel(wr, sheet_name="FS_SNAPSHOT", index=False)
            df_profit.to_excel(wr, sheet_name="PROFIT_SNAPSHOT", index=False)
            df_value.to_excel(wr, sheet_name="VALUE_SNAPSHOT", index=False)
        except Exception:
            pass
        pd.DataFrame([meta]).to_excel(wr, sheet_name="META", index=False)

    st.download_button("결과 엑셀 다운로드", data=out.getvalue(), file_name=f"{cmp_cd}_valuation_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("좌측에서 종목코드·현재가를 입력하고 ‘자동 수집 → 계산’을 눌러주세요.")
