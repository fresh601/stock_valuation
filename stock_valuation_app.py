# -*- coding: utf-8 -*-
"""
네이버(와이즈리포트) 자동 수집 + 적정주가 계산기 (Final Version)
- 개선점: 현재가 자동 수집, 단위 중복 계산 수정, Selenium 안정성 강화, 중복 함수 제거
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
from selenium.webdriver.common.by import By

st.set_page_config(page_title="적정주가 계산기 · Final", layout="wide")

# ──────────────────────────────────────────────────────────────
# 1. 유틸리티 & 단위 변환
# ──────────────────────────────────────────────────────────────
UNIT_MAP = {
    '원': 1.0, '천원': 1e3, '만원': 1e4,
    '백만원': 1e6, '억원': 1e8, '십억원': 1e9,
    '백억원': 1e10, '천억원': 1e11, '조원': 1e12,
}

def to_number(s):
    if s is None: return None
    s = str(s).strip()
    if s in ("", "-"): return None
    s = s.replace(",", "")
    m = re.fullmatch(r"\(([-+]?\d*\.?\d+)\)", s)
    if m: return -float(m.group(1))
    try: return float(s)
    except: return None

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").replace("\xa0", " ").strip())

def scale_by_unit(df: pd.DataFrame, unit_col: str = '단위') -> pd.DataFrame:
    """표의 '단위' 컬럼을 감지하여 모든 숫자를 '원' 단위로 변환"""
    if df is None or df.empty: return df
    
    # 숫자형 컬럼 식별
    num_cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    
    if unit_col not in df.columns:
        # 단위가 없으면 콤마만 제거하고 반환
        df[num_cols] = df[num_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors='coerce')
        return df
        
    unit_str = str(df[unit_col].iloc[0])
    mul = 1.0
    for k, v in UNIT_MAP.items():
        if k in unit_str:
            mul = v
            break
            
    df[num_cols] = df[num_cols].replace(",", "", regex=True).apply(pd.to_numeric, errors='coerce') * mul
    return df

def pick_prefer_current_then_estimate(row: pd.Series):
    """열 선택 우선순위: 당기/최근/TTM -> (E)/예상 -> 가장 오른쪽(최근)"""
    cols = list(row.index)
    # 1. 확정 실적 (당기/최근/TTM)
    prefer_now = [i for i, c in enumerate(cols) if re.search(r'당기|최근|TTM|12M', str(c), re.I)]
    for i in reversed(prefer_now):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v): return float(v), cols[i], 'current'
    # 2. 컨센서스 (E)
    prefer_est = [i for i, c in enumerate(cols) if re.search(r'\(E\)|Estimate|예상|FWD', str(c), re.I)]
    for i in reversed(prefer_est):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v): return float(v), cols[i], 'estimate'
    # 3. 그 외 가장 최신 데이터
    for i in range(len(cols) - 1, -1, -1):
        v = pd.to_numeric(str(row.iloc[i]).replace(',', ''), errors='coerce')
        if pd.notna(v): return float(v), cols[i], 'actual'
    return None, None, None

# ──────────────────────────────────────────────────────────────
# 2. Selenium & 크롤링 (현재가 자동 수집 추가)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_encparam_id_price(cmp_cd: str, page_key: str) -> dict:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--single-process") # 메모리 부족 방지
    
    driver = webdriver.Chrome(options=chrome_options)
    current_price = 0.0
    try:
        url = f"https://navercomp.wisereport.co.kr/v2/company/{page_key}.aspx?cmp_cd={cmp_cd}"
        driver.get(url)
        time.sleep(2.0) # 페이지 로딩 대기
        
        html = driver.page_source
        
        # 1. 암호화 토큰 추출
        enc_match = re.search(r"encparam\s*:\s*['\"]?([a-zA-Z0-9+/=]+)['\"]?", html)
        id_match = re.search(r"cmp_cd\s*=\s*['\"]?([0-9]+)['\"]?", html)
        
        # 2. 현재가 추출 (WiseReport 상단 배너 or 네이버 금융 구조)
        try:
            # WiseReport 팝업 내 상단 현재가 위치 (.cny_head .no_today .blind)
            price_elem = driver.find_element(By.CSS_SELECTOR, ".cny_head .no_today .blind")
            if price_elem:
                current_price = float(price_elem.text.replace(",", ""))
        except:
            current_price = 0.0
            
        return {
            "cmp_cd": cmp_cd,
            "encparam": enc_match.group(1) if enc_match else None,
            "id": id_match.group(1) if id_match else None,
            "current_price": current_price
        }
    finally:
        driver.quit()

def fetch_main_table(cmp_cd: str, encparam: str, cmp_id: str):
    url = "https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx"
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={cmp_cd}'}
    params = {'cmp_cd': cmp_cd, 'fin_typ': '0', 'freq_typ': 'Y', 'encparam': encparam, 'id': cmp_id}
    res = requests.get(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()
    
    soup = BeautifulSoup(res.text, 'html.parser')
    tables = soup.select("table.gHead01.all-width")
    target = next((tb for tb in tables if "연간" in clean_text(tb.get_text(" ")) or re.search(r"20\d\d", tb.get_text(" "))), None)
    
    if not target: return pd.DataFrame()
    
    # 헤더(연도) 파싱
    thead_rows = target.select("thead tr")
    year_cells = thead_rows[-1].find_all(["th", "td"]) if thead_rows else []
    years = []
    for th in year_cells:
        t = clean_text(th.get_text(" "))
        if t and not re.search(r"주요재무정보|구분", t):
            years.append(t)
            
    # 데이터 파싱
    rows = []
    for tr in target.select("tbody tr"):
        th = tr.find("th")
        if not th: continue
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
        
    return pd.DataFrame(rows, columns=["지표"] + years).set_index("지표")

def fetch_json_mode(cmp_cd: str, mode: str, encparam: str) -> pd.DataFrame:
    """fs(재무상태표), profit(손익계산서), value(투자지표) JSON 수집"""
    base_url = "https://navercomp.wisereport.co.kr/v2/company/cF3002.aspx" if mode == "fs" else "https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx"
    rpt_map = {"fs": "1", "profit": "1", "value": "5"}
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': f'https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={cmp_cd}'}
    params = {'cmp_cd': cmp_cd, 'frq': '0', 'rpt': rpt_map[mode], 'finGubun': 'MAIN', 'frqTyp': '0', 'encparam': encparam}
    
    res = requests.get(base_url, params=params, headers=headers, timeout=10)
    try: js = res.json()
    except: return pd.DataFrame()
    
    data = js.get("DATA", [])
    labels = [re.sub(r"<br\s*/?>", " ", l).strip() for l in js.get("YYMM", [])]
    unit = js.get("UNIT", "")
    
    if not data: return pd.DataFrame()
    
    # DATA1, DATA2... 키 매핑
    year_keys = sorted([k for k in data[0] if re.match(r"^DATA\d+$", k)], key=lambda x: int(x[4:]))
    rows = [[r.get("ACC_NM", "")] + [r.get(k, "") for k in year_keys] for r in data]
    
    df = pd.DataFrame(rows, columns=["항목"] + labels[:len(year_keys)])
    df.insert(1, "단위", unit)
    return df

# ──────────────────────────────────────────────────────────────
# 3. 핵심 지표 추출 (Helper Functions)
# ──────────────────────────────────────────────────────────────
def infer_from_main(df_main_wide, patterns):
    if df_main_wide is None or df_main_wide.empty: return None, None, None
    idx = df_main_wide.index.astype(str)
    for p in patterns:
        mask = idx.str.contains(p, case=False, regex=True)
        if mask.any():
            return pick_prefer_current_then_estimate(df_main_wide.loc[mask].iloc[0])
    return None, None, None

def pick_latest_from_table(df, patterns):
    if df is None or df.empty: return None, None, None
    cols = [c for c in df.columns if c not in ("항목", "단위", "전년대비 (YoY, %)")]
    for p in patterns:
        mask = df["항목"].astype(str).str.contains(p, case=False, regex=True, na=False)
        if mask.any():
            return pick_prefer_current_then_estimate(df.loc[mask].iloc[0][cols])
    return None, None, None

def extract_core_numbers(df_main, df_fs, df_profit, df_value):
    # 1. 발행주식수
    shares, _, _ = infer_from_main(df_main, [r"발행주식수|주식수"])
    
    # 2. 순부채 (Net Debt)
    net_debt, _, _ = pick_latest_from_table(df_fs, [r"^\*?순부채", r"Net\s*Debt"])
    
    # 3. EPS / BPS
    eps, _, _ = pick_latest_from_table(df_value, [r"EPS"])
    if eps is None: eps, _, _ = infer_from_main(df_main, [r"EPS"])
    
    bps, _, _ = pick_latest_from_table(df_value, [r"BPS"])
    if bps is None: bps, _, _ = infer_from_main(df_main, [r"BPS"])
    
    # 4. EBITDA (마진 제외)
    ebitda = None
    if df_profit is not None and not df_profit.empty:
        # EBITDA 항목 찾기 (율, 마진 제외)
        cols = [c for c in df_profit.columns if c not in ("항목", "단위")]
        mask = df_profit["항목"].str.contains(r"EBITDA", case=False, na=False) & \
               ~df_profit["항목"].str.contains(r"율|마진|%", case=False, na=False)
        if mask.any():
            ebitda, _, _ = pick_prefer_current_then_estimate(df_profit.loc[mask].iloc[0][cols])
            
    if ebitda is None:
        ebitda, _, _ = infer_from_main(df_main, [r"^\s*EBITDA\s*$"])

    # 5. FCF (Main -> Value -> CFO-CAPEX)
    fcf0 = None
    # 5-1. Main
    val, _, _ = infer_from_main(df_main, [r"FCF|자유현금흐름|잉여현금흐름"])
    if val is not None: fcf0 = val
    
    # 5-2. Value Table
    if fcf0 is None:
        val, _, _ = pick_latest_from_table(df_value, [r"FCF|자유현금흐름"])
        if val is not None: fcf0 = val
        
    # 5-3. CFO - CAPEX
    if fcf0 is None:
        cfo, _, _ = pick_latest_from_table(df_value, [r"영업활동.*현금흐름|CFO"])
        if cfo is None: cfo, _, _ = pick_latest_from_table(df_fs, [r"영업활동.*현금흐름|CFO"])
        
        capex, _, _ = pick_latest_from_table(df_fs, [r"CAPEX|유형자산.*취득|설비투자"])
        if capex is None: capex, _, _ = pick_latest_from_table(df_value, [r"CAPEX|유형자산.*취득"])
        
        if cfo is not None and capex is not None:
            # CAPEX가 양수로 표기되어 있으면 빼주고, 음수면 더해줌(보통 현금유출은 음수표기지만 양수표기인 경우도 있음)
            # 여기서는 안전하게 절대값을 뺌
            fcf0 = float(cfo) - abs(float(capex))

    return {"shares": shares, "net_debt": net_debt, "eps": eps, "bps": bps, "ebitda": ebitda, "fcf0": fcf0}

# ──────────────────────────────────────────────────────────────
# 4. Valuation 로직 (통합됨)
# ──────────────────────────────────────────────────────────────
def calculate_dcf(fcf0, g_high, g_mid, g_low, g_tv, r, shares, net_debt, safety):
    if not all([shares, r, fcf0]): return None, None, None, None
    
    years = range(1, 11)
    growths = [g_high]*3 + [g_mid]*3 + [g_low]*4
    
    fcfs = []
    last = fcf0
    for g in growths:
        last *= (1 + g)
        fcfs.append(last)
        
    disc_factors = [1 / ((1 + r) ** t) for t in years]
    pv_fcfs = [f * d for f, d in zip(fcfs, disc_factors)]
    
    # Terminal Value
    term_val = fcfs[-1] * (1 + g_tv) / (r - g_tv) if r > g_tv else 0
    pv_tv = term_val * disc_factors[-1]
    
    ev = sum(pv_fcfs) + pv_tv
    equity = ev - (net_debt or 0)
    price = (equity / shares) * (1 - safety)
    
    detail = pd.DataFrame({"Year": list(years) + ["TV"], "FCF": fcfs + [term_val], "PV": pv_fcfs + [pv_tv]})
    return price, ev, equity, detail

def calculate_multiple_price(metric, multiple, shares=None, net_debt=0, kind='PER', safety=0.0):
    """PER, PBR, EV/EBITDA 통합 계산"""
    if metric is None or multiple is None: return None
    
    val = 0.0
    if kind in ['PER', 'PBR']:
        val = metric * multiple
    elif kind == 'EV/EBITDA':
        if not shares: return None
        ev = metric * multiple
        equity = ev - (net_debt or 0)
        val = equity / shares
        
    return val * (1 - safety)

# ──────────────────────────────────────────────────────────────
# 5. UI (Streamlit)
# ──────────────────────────────────────────────────────────────
st.title("📈 적정주가 계산기 v2.0")
st.caption("네이버 증권 데이터 기반 자동 수집 및 멀티플/DCF 적정주가 산출 (안전마진 적용)")

# Session State 초기화
if 'fetched_price' not in st.session_state: st.session_state.fetched_price = 0.0
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False

with st.sidebar:
    st.header("1. 종목 선택")
    cmp_cd = st.text_input("종목코드 (6자리)", value="005930") # 삼성전자 기본
    
    # 현재가 입력 (자동 수집된 값이 있으면 그것을 기본값으로)
    default_price = st.session_state.fetched_price if st.session_state.fetched_price > 0 else 0.0
    current_price_input = st.number_input("현재가 (원, 0이면 자동)", value=default_price, step=100.0, format="%.0f")
    
    btn_run = st.button("데이터 가져오기 & 분석", type="primary")
    
    st.divider()
    st.header("2. 시나리오 설정")
    scenario = st.radio("시장 관점", ["보수적", "중립적", "낙관적"], index=1, horizontal=True)
    
    # 시나리오별 파라미터 매핑
    if scenario == '보수적':
        p = {'g_h': 0.05, 'g_m': 0.03, 'g_l': 0.02, 'g_tv': 0.01, 'r': 0.10, 'safe': 0.35}
    elif scenario == '낙관적':
        p = {'g_h': 0.15, 'g_m': 0.10, 'g_l': 0.05, 'g_tv': 0.03, 'r': 0.08, 'safe': 0.20}
    else: # 중립
        p = {'g_h': 0.10, 'g_m': 0.06, 'g_l': 0.03, 'g_tv': 0.02, 'r': 0.09, 'safe': 0.30}
        
    with st.expander("DCF 상세 변수 수정"):
        g_high = st.number_input("고성장(1-3년)", value=p['g_h'], format="%.3f")
        g_mid  = st.number_input("중성장(4-6년)", value=p['g_m'], format="%.3f")
        g_low  = st.number_input("저성장(7-10년)", value=p['g_l'], format="%.3f")
        g_tv   = st.number_input("영구성장(TV)", value=p['g_tv'], format="%.3f")
        r      = st.number_input("할인율(WACC)", value=p['r'], format="%.3f")
        safety = st.number_input("안전마진", value=p['safe'], format="%.2f")

    st.header("3. 가중치(MIX)")
    w_dcf = st.slider("DCF 비중", 0.0, 1.0, 0.4)
    w_per = st.slider("PER 비중", 0.0, 1.0, 0.2)
    w_pbr = st.slider("PBR 비중", 0.0, 1.0, 0.2)
    w_ev  = st.slider("EV/EBITDA 비중", 0.0, 1.0, 0.2)
    
    st.subheader("멀티플 가정")
    per_m = st.number_input("Target PER", value=10.0)
    pbr_m = st.number_input("Target PBR", value=1.2)
    ev_m  = st.number_input("Target EV/EBITDA", value=6.0)

# 메인 로직
if btn_run:
    st.session_state.run_analysis = True
    # 1. 토큰 및 현재가 수집
    with st.spinner("네이버 증권 접속 중..."):
        tk = get_encparam_id_price(cmp_cd, "c1010001")
        
    if not tk['encparam']:
        st.error("토큰 정보를 가져오지 못했습니다. 종목코드를 확인하세요.")
        st.stop()
        
    # 현재가 업데이트 (세션 상태 저장하여 리프레시 후에도 유지)
    if tk['current_price'] > 0:
        st.session_state.fetched_price = tk['current_price']
        
    # 데이터 수집
    with st.spinner("재무제표 긁어오는 중..."):
        df_main = fetch_main_table(cmp_cd, tk['encparam'], tk['id'])
        df_fs = fetch_json_mode(cmp_cd, "fs", tk['encparam'])
        df_pf = fetch_json_mode(cmp_cd, "profit", tk['encparam'])
        df_vl = fetch_json_mode(cmp_cd, "value", tk['encparam'])
        
    # 단위 변환 (중요: 여기서만 변환 수행)
    df_fs = scale_by_unit(df_fs)
    df_pf = scale_by_unit(df_pf)
    df_vl = scale_by_unit(df_vl)
    
    # 핵심 지표 추출
    core = extract_core_numbers(df_main, df_fs, df_pf, df_vl)
    st.session_state.core_data = core # 데이터 저장
    
    st.rerun() # 데이터를 다 가져왔으면 UI 갱신을 위해 재실행

if st.session_state.run_analysis and 'core_data' in st.session_state:
    core = st.session_state.core_data
    # 현재가 결정 (사용자 입력 우선, 없으면 자동 수집값)
    final_current_price = current_price_input if current_price_input > 0 else st.session_state.fetched_price

    # 1. 입력값 확인 섹션
    st.subheader(f"📊 {cmp_cd} 핵심 재무 데이터 (단위: 원)")
    
    # 보기 좋게 DataFrame 생성
    disp_df = pd.DataFrame([core]).T
    disp_df.columns = ["값"]
    disp_df["설명"] = ["발행주식수", "순부채 ((-)는 순현금)", "주당순이익(EPS)", "주당순자산(BPS)", "EBITDA", "잉여현금흐름(FCF)"]
    st.dataframe(disp_df, use_container_width=True)
    
    if core['net_debt'] and core['net_debt'] < 0:
        st.info(f"💡 순부채가 {core['net_debt']:,.0f}원으로 음수입니다. 이는 기업이 빚보다 현금이 많은 '순현금' 상태임을 의미하며, 적정주가를 높이는 요인이 됩니다.")

    # 2. 적정주가 계산
    # 데이터 정제
    shares = core['shares']
    net_debt = core['net_debt'] if core['net_debt'] is not None else 0
    
    # (A) DCF
    px_dcf, ev_dcf, eq_dcf, df_detail = calculate_dcf(
        core['fcf0'], g_high, g_mid, g_low, g_tv, r, shares, net_debt, safety
    )
    
    # (B) Relative
    px_per = calculate_multiple_price(core['eps'], per_m, kind='PER', safety=safety)
    px_pbr = calculate_multiple_price(core['bps'], pbr_m, kind='PBR', safety=safety)
    px_ev  = calculate_multiple_price(core['ebitda'], ev_m, shares, net_debt, kind='EV/EBITDA', safety=safety)
    
    # (C) MIX
    prices = {'DCF': px_dcf, 'PER': px_per, 'PBR': px_pbr, 'EV/EBITDA': px_ev}
    weights = {'DCF': w_dcf, 'PER': w_per, 'PBR': w_pbr, 'EV/EBITDA': w_ev}
    
    valid_prices = []
    valid_weights = []
    
    for k, v in prices.items():
        if v is not None and v > 0:
            valid_prices.append(v)
            valid_weights.append(weights[k])
            
    if valid_prices:
        final_w = np.array(valid_weights) / sum(valid_weights)
        mix_price = np.dot(valid_prices, final_w)
    else:
        mix_price = 0
        
    # 3. 결과 시각화
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 주가", f"{final_current_price:,.0f} 원")
    c2.metric("적정 주가 (MIX)", f"{mix_price:,.0f} 원", delta=f"{mix_price - final_current_price:,.0f} 원")
    
    upside = ((mix_price / final_current_price) - 1) * 100 if final_current_price > 0 else 0
    c3.metric("상승 여력", f"{upside:.2f} %", delta_color="normal" if upside > 0 else "inverse")
    
    # 차트
    res_df = pd.DataFrame({
        "Method": list(prices.keys()) + ["MIX"],
        "Price": [p if p else 0 for p in prices.values()] + [mix_price]
    })
    
    fig = go.Figure(data=[
        go.Bar(x=res_df["Method"], y=res_df["Price"], text=res_df["Price"].apply(lambda x: f"{x:,.0f}"), textposition='auto', marker_color=['#e0e0e0']*4 + ['#ff4b4b'])
    ])
    fig.add_hline(y=final_current_price, line_dash="dot", annotation_text="현재가", annotation_position="bottom right")
    fig.update_layout(title="Valuation Summary", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    # DCF 상세 다운로드
    if df_detail is not None:
        with st.expander("DCF 상세 계산 내역 보기"):
            st.dataframe(df_detail)
            csv = df_detail.to_csv(index=False).encode('utf-8-sig')
            st.download_button("DCF 엑셀 다운로드", csv, "dcf_detail.csv", "text/csv")

else:
    st.info("좌측 사이드바에서 종목코드를 입력하고 '데이터 가져오기'를 눌러주세요.")
