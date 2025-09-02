# -*- coding: utf-8 -*-
"""
네이버(와이즈리포트) 자동 수집 + 적정주가 계산 최종본 (현재가는 수동 입력)
- encparam/id 자동 획득(Selenium) → main/fs/profit/value 자동 수집
- EPS/BPS/EBITDA/FCF₀/순부채/발행주식수 추출
  · (E)/Estimate/예상/FWD 표기가 있는 열을 최우선으로 선택, 없으면 가장 오른쪽 실적 열 사용
  · 모든 표 단위(UNIT)를 '원' 기준으로 환산
- DCF(3구간 성장 + 터미널) + 상대가치(PER/PBR/EV/EBITDA) + MIX(가중치)
- 시나리오 버튼(보수/기준/낙관)으로 성장률·할인율·안전마진 자동 세팅
- 현재가는 사용자가 직접 입력 → 현재가/적정가/상승여력 카드 표시
- Plotly 4.x/5.x 호환( text_auto 미사용, 안전 헬퍼 사용 )
- 결과 엑셀에 META(선택된 열, 단위, 시나리오, 가중치, 기준시점 라벨) 기록

필수 설치:
  pip install streamlit selenium beautifulsoup4 lxml html5lib pandas requests openpyxl plotly numpy

실행:
  streamlit run streamlit_valuation_final.py
"""

import io
import re
import time
import json
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from collections import defaultdict

# Selenium (encparam/id 추출)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

st.set_page_config(page_title="적정주가 계산기 · 최종본", layout="wide")

# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
UNIT_MAP = {
    '원': 1.0,
    '천원': 1e3,
    '만원': 1e4,
    '백만원': 1e8,  # 일부 표는 '백만원'이 실제로 1e6로 표기되나, 와이즈리포트 JSON은 보통 원 단위 문자열을 반환 → 아래에서 숫자화하며 재확인 필요
    '억원': 1e8,
    '십억원': 1e9,
    '백억원': 1e10,
    '천억원': 1e11,
    '조원': 1e12,
}


def nvl(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return x
    except Exception:
        return default


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


def extract_year_label(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    m = re.search(r"(20\d{2})(?:[./-]?(?:0?[1-9]|1[0-2]))?", x)
    return m.group(1) if m else x


def rightmost_value(row: pd.Series):
    for v in row[::-1]:
        if pd.notna(v):
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                try:
                    return pd.to_numeric(v, errors="coerce")
                except Exception:
                    pass
    return np.nan


def scale_by_unit(df: pd.DataFrame, unit_col: str = '단위') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if unit_col not in df.columns:
        # 숫자형으로 통일
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


def pick_latest_estimate(row: pd.Series):
    """(E)/Estimate/예상/FWD 라벨이 있는 열을 뒤에서 앞으로 우선 탐색 → 없으면 일반 열을 뒤에서 앞으로.
    Returns: (value: float|None, used_col_name: str|None, used_type: 'estimate'|'actual'|None)
    """
    cols = list(row.index)
    # 1) 예상 라벨이 있는 열을 뒤에서 앞으로 탐색
    prefer_idx = [i for i, c in enumerate(cols) if re.search(r"\(E\)|Estimate|예상|FWD|Forward", str(c), re.I)]
    for i in reversed(prefer_idx):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v), cols[i], 'estimate'
    # 2) 일반 열: 가장 오른쪽 비결측
    for i in range(len(cols) - 1, -1, -1):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v), cols[i], 'actual'
    return None, None, None

# ──────────────────────────────────────────────────────────────
# 안전한 차트 헬퍼 (Plotly 4.x/5.x 공용)
# ──────────────────────────────────────────────────────────────

def safe_bar(df: pd.DataFrame, x: str, y: str, color: str = None, textfmt: str = '.2f', title: str = None):
    df2 = df.dropna(subset=[y]).copy()
    if df2.empty:
        return None
    df2[y] = pd.to_numeric(df2[y], errors='coerce')
    fig = px.bar(df2, x=x, y=y, color=color)
    fig.update_traces(texttemplate=f"%{{y:{textfmt}}}", textposition="outside")
    fig.update_layout(
        uniformtext_minsize=8, uniformtext_mode="show",
        margin=dict(t=40, r=20, l=20, b=50), title=title or ""
    )
    return fig


def safe_line(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = None):
    df2 = df.dropna(subset=[y]).copy()
    if df2.empty:
        return None
    df2[y] = pd.to_numeric(df2[y], errors='coerce')
    fig = px.line(df2, x=x, y=y, color=color, markers=True)
    fig.update_layout(margin=dict(t=40, r=20, l=20, b=50), title=title or "")
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
        return {
            "cmp_cd": cmp_cd,
            "encparam": enc_match.group(1) if enc_match else None,
            "id": id_match.group(1) if id_match else None,
        }
    finally:
        driver.quit()

# ──────────────────────────────────────────────────────────────
# MAIN(HTML), JSON(fs/profit/value)
# ──────────────────────────────────────────────────────────────

def fetch_main_table(cmp_cd: str, encparam: str, cmp_id: str):
    url = "https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx"
    cookies = {'setC1010001': '%5B%7B...%7D%5D'}
    headers = {
        'Accept': 'application/json, text/html, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={cmp_cd}',
    }
    params = {'cmp_cd': cmp_cd, 'fin_typ': '0', 'freq_typ': 'Y', 'encparam': encparam, 'id': cmp_id}
    res = requests.get(url, headers=headers, cookies=cookies, params=params, timeout=20)
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
    # 숫자화는 scale_by_unit에서 일괄 처리
    return df


def fetch_json_mode(cmp_cd: str, mode: str, encparam: str) -> pd.DataFrame:
    url = "https://navercomp.wisereport.co.kr/v2/company/cF3002.aspx" if mode == "fs" else \
          "https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx"
    rpt_map = {"fs": "1", "profit": "1", "value": "5"}
    headers = {
        'Accept': 'application/json, text/html, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={cmp_cd}',
    }
    cookies = {'setC1040001': '%5B%7B...%7D%5D'}
    params = {'cmp_cd': cmp_cd, 'frq': '0', 'rpt': rpt_map[mode], 'finGubun': 'MAIN', 'frqTyp': '0', 'cn': '', 'encparam': encparam}
    res = requests.get(url, params=params, headers=headers, cookies=cookies, timeout=20)
    res.raise_for_status()
    try:
        js = res.json()
    except json.JSONDecodeError:
        return pd.DataFrame()
    return parse_json_table(js)

# ──────────────────────────────────────────────────────────────
# 핵심 값 추출
# ──────────────────────────────────────────────────────────────

def infer_from_main(df_main_wide: pd.DataFrame, patterns: list[str]):
    if df_main_wide is None or df_main_wide.empty:
        return None, None, None
    idx = df_main_wide.index.astype(str)
    for p in patterns:
        mask = idx.str.contains(p, case=False, regex=True)
        if mask.any():
            row = df_main_wide.loc[mask].iloc[0]
            val, col, used = pick_latest_estimate(row)
            return val, col, used
    return None, None, None


def pick_latest_from_table(df: pd.DataFrame, patterns: list[str]):
    if df is None or df.empty:
        return None, None, None
    cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    for p in patterns:
        mask = df["항목"].astype(str).str.contains(p, case=False, regex=True, na=False)
        if mask.any():
            row = df.loc[mask].iloc[0][cols]
            return pick_latest_estimate(row)
    return None, None, None


def extract_core_numbers(df_main, df_fs, df_profit, df_value):
    # 발행주식수 (가능하면 main에서)
    shares, shares_col, shares_used = infer_from_main(df_main, [r"발행주식수|주식수|보통주수|총발행주식"])  # 단위가 '주'인지 확인 필요(보통 '주')

    # 순부채 (*순부채 최우선)
    net_debt, nd_col, nd_used = pick_latest_from_table(df_fs, [r"^\*?순부채", r"Net\s*Debt"])  # '원' 기준 환산됨

    # EPS/BPS/EBITDA (value/profit/main 순)
    eps, eps_col, eps_type = pick_latest_from_table(df_value,  [r"EPS"])
    if eps is None:
        eps, eps_col, eps_type = infer_from_main(df_main, [r"EPS"])  # 주당 값

    bps, bps_col, bps_type = pick_latest_from_table(df_value,  [r"BPS"])
    if bps is None:
        bps, bps_col, bps_type = infer_from_main(df_main, [r"BPS"])  # 주당 값

    ebitda, e_col, e_type = pick_latest_from_table(df_profit, [r"EBITDA"])
    if ebitda is None:
        ebitda, e_col, e_type = infer_from_main(df_main, [r"EBITDA"])

    # FCF₀: value의 FCF 우선, 없으면 fs에서 CFO/CAPEX로 계산
    fcf0, fcf_col, fcf_type = pick_latest_from_table(df_value, [r"FCF|Free\s*Cash\s*Flow"])
    if fcf0 is None:
        cfo, cfo_col, _ = pick_latest_from_table(df_fs, [r"영업활동.*현금흐름|영업활동으로인한현금흐름|CFO"])
        capex, capex_col, _ = pick_latest_from_table(df_fs, [r"유형자산의\s*취득|CAPEX|설비투자|유형자산.*취득"])
        if cfo is not None and capex is not None:
            # CAPEX가 음수(유출)면 그대로 더하는 형태가 자연스러움
            fcf0 = float(cfo) + float(capex)
            fcf_col = f"CFO[{cfo_col}] + CAPEX[{capex_col}]"
            fcf_type = 'derived'

    meta_cols = {
        'shares_col': shares_col, 'shares_type': shares_used,
        'net_debt_col': nd_col, 'net_debt_type': nd_used,
        'eps_col': eps_col, 'eps_type': eps_type,
        'bps_col': bps_col, 'bps_type': bps_type,
        'ebitda_col': e_col, 'ebitda_type': e_type,
        'fcf_col': fcf_col, 'fcf_type': fcf_type,
    }

    return {
        "shares": shares,
        "net_debt": net_debt,
        "eps": eps,
        "bps": bps,
        "ebitda": ebitda,
        "fcf0": fcf0,
        "meta_cols": meta_cols,
    }

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
    detail = pd.DataFrame({"Year": years + ["TV"], "FCF": fcfs + [np.nan], "Discount": disc + [disc[-1]], "PV": pv_fcfs + [pv_tv]})
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

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

st.title("📈 적정주가 계산기 · 최종본")
st.caption("현재가는 직접 입력하고, 나머지는 자동으로 수집·계산합니다. 예상치가 있으면 우선 적용합니다.")

with st.sidebar:
    st.header("입력")
    cmp_cd = st.text_input("종목코드 (6자리)", value="066570")
    current_price = st.number_input("현재가 (원)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
    run = st.button("자동 수집 → 계산", type="primary")

    st.header("시나리오")
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = '기준'

    colS1, colS2, colS3 = st.columns(3)
    if colS1.button("보수"):
        st.session_state['scenario'] = '보수'
    if colS2.button("기준"):
        st.session_state['scenario'] = '기준'
    if colS3.button("낙관"):
        st.session_state['scenario'] = '낙관'
    st.write(f"선택된 시나리오: **{st.session_state['scenario']}**")

    # 기본 파라미터 (시나리오에 따라 값 세팅)
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

    # 1) 토큰
    page_key = "c1010001"
    with st.spinner("토큰 획득 중..."):
        tk = get_encparam_and_id(cmp_cd, page_key)
    encparam, cmp_id = tk.get("encparam"), tk.get("id")
    c1, c2, c3 = st.columns(3)
    c1.metric("종목코드", cmp_cd)
    c2.metric("encparam", (encparam[:10] + "…") if encparam else "없음")
    c3.metric("id", cmp_id or "없음")

    if not encparam or not cmp_id:
        st.warning("토큰 추출 실패. 잠시 후 다시 시도 또는 IP/봇 차단 이슈 확인.")
        st.stop()

    # 2) 수집
    with st.spinner("데이터 수집(main/fs/profit/value)..."):
        df_main = fetch_main_table(cmp_cd, encparam, cmp_id)
        df_fs   = fetch_json_mode(cmp_cd, "fs", encparam)
        df_profit = fetch_json_mode(cmp_cd, "profit", encparam)
        df_value  = fetch_json_mode(cmp_cd, "value", encparam)

    # 3) 단위 스케일 통일 (원 기준)
    df_fs = scale_by_unit(df_fs)
    df_profit = scale_by_unit(df_profit)
    df_value = scale_by_unit(df_value)

    # 4) 핵심 값 추출
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

    with st.expander("선택된 열/라벨(META)"):
        st.json(core['meta_cols'])

    # 5) Valuation
    px_dcf, ev, equity, dcf_detail = dcf_fair_price(
        fcf0=core["fcf0"], g_high=g_high, g_mid=g_mid, g_low=g_low, g_tv=g_tv,
        r=r, shares=core["shares"], net_debt=core["net_debt"], safety=safety
    )
    px_per = per_price(core["eps"], per_mult, safety=safety)
    px_pbr = pbr_price(core["bps"], pbr_mult, safety=safety)
    px_ev  = evebitda_price(core["ebitda"], ev_mult, core["shares"], core["net_debt"], safety=safety)

    wsum = (w_dcf + w_per + w_pbr + w_ev) or 1.0
    parts = []
    for px, w in [(px_dcf,w_dcf),(px_per,w_per),(px_pbr,w_pbr),(px_ev,w_ev)]:
        if px is not None:
            parts.append(px * (w / wsum))
    mix_price = float(np.nansum(parts)) if parts else None

    st.subheader("📌 적정주가 요약")
    summary = pd.DataFrame({
        "방법": ["DCF", "PER", "PBR", "EV/EBITDA", "MIX(가중)"],
        "적정주가": [px_dcf, px_per, px_pbr, px_ev, mix_price],
    })
    fig = safe_bar(summary, "방법", "적정주가", title="방법별 적정주가")
    if fig is None:
        st.info("표시할 적정주가 값이 없습니다.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    # 6) 현재가 카드 및 상승여력
    colK1, colK2, colK3 = st.columns(3)
    colK1.metric("현재가", f"{current_price:,.2f} 원")
    colK2.metric("적정가(MIX)", f"{(mix_price or 0):,.2f} 원")
    up = None if not current_price or not mix_price else (mix_price / current_price - 1.0) * 100.0
    colK3.metric("상승여력", f"{up:.2f}%" if up is not None else "-")

    st.subheader("DCF 세부내역")
    if dcf_detail is not None:
        st.dataframe(dcf_detail, use_container_width=True)
    else:
        st.info("DCF 계산을 위해 FCF₀/주식수/할인율이 필요합니다.")

    # 7) 결과 엑셀 (재현성 메타 포함)
    meta = {
        'ticker': cmp_cd,
        'scenario': st.session_state['scenario'],
        'params': dict(g_high=g_high, g_mid=g_mid, g_low=g_low, g_tv=g_tv, r=r, safety=safety),
        'multiples': dict(PER=per_mult, PBR=pbr_mult, EV_EBITDA=ev_mult),
        'weights': dict(DCF=w_dcf, PER=w_per, PBR=w_pbr, EVEBITDA=w_ev, sum=wsum),
        'selected_cols': core['meta_cols'],
    }

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        summary.to_excel(wr, sheet_name="SUMMARY", index=False)
        pd.DataFrame([{'현재가': current_price, '적정가(MIX)': mix_price, '상승여력%': up}]).to_excel(wr, sheet_name="PRICE", index=False)
        core_view.to_excel(wr, sheet_name="CORE_INPUTS", index=False)
        if dcf_detail is not None:
            dcf_detail.to_excel(wr, sheet_name="DCF_DETAIL", index=False)
        try:
            df_main.reset_index().to_excel(wr, sheet_name="MAIN_SNAPSHOT", index=False)
            df_fs.to_excel(wr, sheet_name="FS_SNAPSHOT", index=False)
            df_profit.to_excel(wr, sheet_name="PROFIT_SNAPSHOT", index=False)
            df_value.to_excel(wr, sheet_name="VALUE_SNAPSHOT", index=False)
        except Exception:
            pass
        pd.DataFrame([meta]).to_excel(wr, sheet_name="META", index=False)

    st.download_button(
        "결과 엑셀 다운로드",
        data=out.getvalue(),
        file_name=f"{cmp_cd}_valuation_final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("좌측에서 종목코드·현재가를 입력하고 ‘자동 수집 → 계산’을 눌러주세요.")
