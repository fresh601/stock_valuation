# -*- coding: utf-8 -*-
"""
네이버(와이즈리포트) 자동수집 + 적정주가 계산 · 풀버전
- encparam/id 자동 획득(Selenium)
- main/fs/profit/value 수집
- EPS/BPS/EBITDA/FCF₀/순부채/발행주식수 추출
- FCF₀: 당기/최근/TTM → 최근 실적 → 현재연도에 가장 가까운 (E)
- EBITDA: 금액 행만(마진/%) 제외. 백업: 영업이익 + 감가상각비
- 단위(원/백만원/억원/조원 등) 자동 환산
- DCF + PER + PBR + EV/EBITDA + MIX
- 출처(META) 표시 + 엑셀 다운로드
"""

import re, io, time, json
import numpy as np
import pandas as pd
import requests, streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import plotly.graph_objects as go

st.set_page_config(page_title="적정주가 계산기 · 풀버전", layout="wide")

# ───────────────────────────────
# 공통 유틸
UNIT_MAP = {
    '원':1,'천원':1e3,'만원':1e4,'백만원':1e6,'억원':1e8,'십억원':1e9,'백억원':1e10,'천억원':1e11,'조원':1e12
}
EST_PAT = re.compile(r"\(E\)|\bE\b|Estimate|예상|FWD|Forward", re.I)

def to_number(x):
    if x in (None,"","-"): return None
    s = str(x).strip().replace(",","")
    m = re.fullmatch(r"\(([-+]?\d*\.?\d+)\)", s)
    if m: return -float(m.group(1))
    try: return float(s)
    except: return None

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").replace("\xa0"," ").strip())

def _miss(v):
    try:
        if v is None: return True
        if isinstance(v,(float,np.floating)): return np.isnan(v) or np.isinf(v)
        if isinstance(v,(int,)): return False
        return False
    except: return True

# ───────────────────────────────
# Selenium 토큰
@st.cache_data(show_spinner=False)
def get_token(cmp_cd, page="c1010001"):
    opts=Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    d=webdriver.Chrome(options=opts)
    try:
        d.get(f"https://navercomp.wisereport.co.kr/v2/company/{page}.aspx?cmp_cd={cmp_cd}")
        time.sleep(2.0)
        h=d.page_source
        enc=re.search(r"encparam\s*:\s*['\"]?([a-zA-Z0-9+/=]+)['\"]?",h)
        cid=re.search(r"cmp_cd\s*=\s*['\"]?([0-9]+)['\"]?",h)
        return (enc.group(1) if enc else None),(cid.group(1) if cid else None)
    finally:
        d.quit()

# ───────────────────────────────
# main_wide 수집 + 단위 스케일 감지
def detect_unit_multiplier_from_html(html_table: str) -> float:
    txt = clean_text(html_table or "")
    for k,v in UNIT_MAP.items():
        if k in txt: return v
    return 1.0

def fetch_main(cmp_cd, enc, cid):
    """
    main_wide(연간 주요재무정보) 테이블을 가져온다.
    - 정식 헤더/리퍼러 사용
    - 연간 테이블 탐색 로직을 유연화
    - 실패 시 예외를 던지지 않고 (빈 df, 1.0) 반환
    """
    url = "https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={cmp_cd}",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    params = {
        "cmp_cd": cmp_cd,
        "fin_typ": "0",   # 연결
        "freq_typ": "Y",  # 연간
        "encparam": enc,
        "id": cid,
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
        res.raise_for_status()
    except Exception:
        # 실패 시 빈 df 반환
        return pd.DataFrame(), 1.0

    soup = BeautifulSoup(res.text, "html.parser")

    # 후보 테이블 전부 수집
    cands = soup.select("table.gHead01.all-width") or soup.find_all("table")
    target = None

    def looks_like_year_table(tb):
        txt = clean_text(tb.get_text(" "))
        # 연도/연간 신호가 있으면 가점
        if re.search(r"20\d{2}", txt):  # 2019, 2020, ...
            return True
        if "연간" in txt:
            return True
        # 헤더 열이 여러 개면 가점
        thead = tb.select_one("thead")
        if thead:
            ths = thead.find_all(["th", "td"])
            if len(ths) >= 5:
                return True
        return False

    for tb in cands:
        if looks_like_year_table(tb):
            target = tb
            break

    if target is None:
        # 못 찾으면 빈 df 반환 (예외 X)
        return pd.DataFrame(), 1.0

    # 표 단위(원/억원/조원 등) 감지
    main_mul = detect_unit_multiplier_from_html(str(target))

    # 헤더 추출(중복 컬럼명 방지)
    thead_rows = target.select("thead tr")
    year_cells = thead_rows[-1].find_all(["th", "td"]) if thead_rows else []
    years, counter = [], {}
    for th in year_cells:
        t = clean_text(th.get_text(" "))
        if t and not re.search(r"구분|주요재무정보", t):
            counter[t] = counter.get(t, 0) + 1
            years.append(t + (f"_{counter[t]}" if counter[t] > 1 else ""))

    # 본문
    rows = []
    for tr in target.select("tbody tr"):
        th = tr.find("th")
        if not th:
            continue
        metric = clean_text(th.get_text(" "))
        tds = tr.find_all("td")
        vals = []
        for i in range(len(years)):
            cell = tds[i] if i < len(tds) else None
            raw = (cell.get("title") if cell else None) or (clean_text(cell.get_text(" ")) if cell else None)
            vals.append(to_number(raw))
        rows.append([metric] + vals)

    if not rows or not years:
        # 구조는 있으나 실데이터가 비어있는 경우도 방어
        return pd.DataFrame(), main_mul

    df = pd.DataFrame(rows, columns=["지표"] + years).set_index("지표")
    return df, main_mul


# ───────────────────────────────
# JSON 모드 수집 + 단위 스케일 적용
def parse_json_table(js:dict)->pd.DataFrame:
    data=js.get("DATA",[]); labels=js.get("YYMM",[]); unit_txt=js.get("UNIT","")
    if not data: return pd.DataFrame()
    labels=[re.sub(r"<br\s*/?>"," ", str(l)).strip() for l in labels]
    year_keys=sorted([k for k in data[0] if k.startswith("DATA")],key=lambda x:int(x[4:]))
    rows=[[r.get("ACC_NM","")]+[r.get(k,"") for k in year_keys] for r in data]
    df=pd.DataFrame(rows, columns=["항목"]+labels[:len(year_keys)])
    num_cols=[c for c in df.columns if c != "항목"]
    df[num_cols]=df[num_cols].replace(",","",regex=True).apply(pd.to_numeric,errors="coerce")
    mul=1.0
    for k,v in UNIT_MAP.items():
        if k in unit_txt: mul=v; break
    df[num_cols]=df[num_cols]*mul
    return df

def fetch_json(cmp_cd, mode, enc):
    url="https://navercomp.wisereport.co.kr/v2/company/cF3002.aspx" if mode=="fs" else "https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx"
    rpt={"fs":"1","profit":"1","value":"5"}[mode]
    r=requests.get(url,params={'cmp_cd':cmp_cd,'frq':'0','rpt':rpt,'finGubun':'MAIN','frqTyp':'0','encparam':enc},timeout=20)
    r.raise_for_status()
    try: js=r.json()
    except json.JSONDecodeError: return pd.DataFrame()
    return parse_json_table(js)

# ───────────────────────────────
# 값 선택 규칙
def pick_prefer_current_then_estimate(row: pd.Series):
    cols=list(row.index)
    # 1) 당기/최근/TTM/12M
    for i in reversed([i for i,c in enumerate(cols) if re.search(r"당기|최근|TTM|12M",str(c),re.I)]):
        v = pd.to_numeric(str(row.iloc[i]).replace(",",""), errors="coerce")
        if pd.notna(v): return float(v), cols[i], "current"
    # 2) (E)/예상
    for i in reversed([i for i,c in enumerate(cols) if EST_PAT.search(str(c))]):
        v = pd.to_numeric(str(row.iloc[i]).replace(",",""), errors="coerce")
        if pd.notna(v): return float(v), cols[i], "estimate"
    # 3) 일반 오른쪽값
    for i in range(len(cols)-1,-1,-1):
        v = pd.to_numeric(str(row.iloc[i]).replace(",",""), errors="coerce")
        if pd.notna(v): return float(v), cols[i], "actual"
    return None, None, None

def pick_fcf_actual_then_nearest(row: pd.Series):
    cols=list(row.index)
    def val(i): return pd.to_numeric(str(row.iloc[i]).replace(",",""), errors="coerce")
    # 1) 당기/최근/TTM
    for i in reversed([i for i,c in enumerate(cols) if re.search(r"당기|최근|TTM|12M",str(c),re.I)]):
        v=val(i)
        if pd.notna(v): return float(v), cols[i], "current"
    # 2) 최근 실적 연도
    for i in reversed([i for i,c in enumerate(cols) if re.search(r"20\d{2}",str(c)) and not EST_PAT.search(str(c))]):
        v=val(i)
        if pd.notna(v): return float(v), cols[i], "actual"
    # 3) 현재연도에 가장 가까운 (E)
    cand=[]
    for i,c in enumerate(cols):
        if EST_PAT.search(str(c)):
            m=re.search(r"(20\d{2})",str(c))
            if m: cand.append((i,int(m.group(1))))
    if cand:
        cur_y=pd.Timestamp.today().year
        cand.sort(key=lambda t:abs(t[1]-cur_y))
        for i,_ in cand:
            v=val(i)
            if pd.notna(v): return float(v), cols[i], "estimate-near"
    return None, None, None

def infer_from_main(df_main, patterns, mul=1.0):
    if df_main is None or df_main.empty: return None, None, None
    idx=df_main.index.astype(str)
    for p in patterns:
        m = idx.str.contains(p, case=False, regex=True)
        if m.any():
            row=df_main.loc[m].iloc[0]
            v,col,typ=pick_prefer_current_then_estimate(row)
            if v is not None and mul!=1.0: v=float(v)*float(mul)
            return v,col,typ
    return None, None, None

def pick_from_table(df, include_patterns, exclude_patterns=None):
    if df is None or df.empty: return None, None, None
    cols=[c for c in df.columns if c!="항목"]
    s=df["항목"].astype(str)
    inc=None
    for p in include_patterns:
        cur=s.str.contains(p, case=False, regex=True, na=False)
        inc = cur if inc is None else (inc|cur)
    if inc is None or not inc.any(): return None, None, None
    if exclude_patterns:
        for q in exclude_patterns:
            exc=s.str.contains(q, case=False, regex=True, na=False)
            inc = inc & (~exc)
        if not inc.any(): return None, None, None
    row=df.loc[inc].iloc[0][cols]
    return pick_prefer_current_then_estimate(row)

# FCF: main → value → (CFO ± CAPEX)
def find_fcf_any(df_main, df_value, df_fs, main_mul: float):
    # 1) main_wide
    if df_main is not None and not df_main.empty:
        mask = df_main.index.astype(str).str.contains(r"FCF|자유현금흐름|잉여현금흐름|Free\s*Cash\s*Flow", case=False, regex=True)
        if mask.any():
            row=df_main.loc[mask].iloc[0]
            v,col,typ=pick_fcf_actual_then_nearest(row)
            if v is not None:
                return float(v)*float(main_mul), f"main:{col}", "direct-main"
    # 2) value
    if df_value is not None and not df_value.empty:
        m = df_value["항목"].astype(str).str.contains(r"FCF|자유현금흐름|잉여현금흐름|Free\s*Cash\s*Flow", case=False, regex=True, na=False)
        if m.any():
            row = df_value.loc[m].iloc[0].drop(labels=["항목"])
            v,col,typ=pick_fcf_actual_then_nearest(row)
            if v is not None:
                return float(v), f"value:{col}", "direct-value"
    # 3) 파생: CFO ± CAPEX
    def _pick(df, pats):
        if df is None or df.empty: return None, None
        cols=[c for c in df.columns if c!="항목"]
        for p in pats:
            mm = df["항목"].astype(str).str.contains(p, case=False, regex=True, na=False)
            if mm.any():
                row=df.loc[mm].iloc[0][cols]
                v,col,_=pick_prefer_current_then_estimate(row)
                if v is not None: return float(v), col
        return None, None
    cfo,_ = _pick(df_value, [r"영업활동.*현금흐름|영업활동으로인한현금흐름|^CFO$|CFO<당기>|CFO＜당기＞"])
    if cfo is None:
        cfo,_ = _pick(df_fs, [r"영업활동.*현금흐름|영업활동으로인한현금흐름|^CFO$"])
    capex,_ = _pick(df_fs, [r"\*?CAPEX|유형자산의\s*취득|설비투자|유형자산.*취득"])
    if capex is None:
        capex,_ = _pick(df_value, [r"\*?CAPEX|유형자산의\s*취득|설비투자|유형자산.*취득"])
    if cfo is not None and capex is not None:
        fcf0 = cfo - abs(capex) if capex>=0 else cfo + capex
        return float(fcf0), "derived:CFO±CAPEX", "derived"
    return None, None, None

# EBITDA 추출(마진% 제외, 금액만)
def find_ebitda(df_profit, df_main, main_mul: float):
    ebitda, e_col, e_typ = pick_from_table(
        df_profit,
        include_patterns=[r'^\s*EBITDA\s*$', r'\bEBITDA\b', r'EBITDA\s*\(.*\)$'],
        exclude_patterns=[r'마진|margin|%']
    )
    if ebitda is not None:
        return float(ebitda), f"profit:{e_col}", e_typ
    # main_wide 백업
    ebitda, e_col, e_typ = infer_from_main(
        df_main,
        [r'^\s*EBITDA\s*$', r'\bEBITDA\b', r'EBITDA\s*\(.*\)$'],
        mul=main_mul
    )
    if ebitda is not None: return float(ebitda), f"main:{e_col}", e_typ
    # 파생: 영업이익 + 감가상각비
    op, op_col, _ = pick_from_table(df_profit, [r'영업이익|Operating\s*Income|OP'], exclude_patterns=[r'율|마진|margin|%'])
    da, da_col, _ = pick_from_table(df_profit, [r'감가상각비|상각비|Depreciation|Amortization|D&A|DA'], exclude_patterns=[r'율|마진|margin|%'])
    if op is not None and da is not None:
        return float(op)+float(da), f"profit:영업이익[{op_col}] + 상각비[{da_col}]", "derived"
    return None, None, None

# ───────────────────────────────
# Valuation
def dcf_price(fcf0, r, shares, g1, g2, g3, g_tv, safety=0.3, net_debt=0.0):
    if _miss(fcf0) or _miss(shares) or not r or r<=0: return None, None, None, None
    years=list(range(1,11))
    gs=[g1]*3+[g2]*3+[g3]*4
    fcfs=[]; last=float(fcf0)
    for g in gs:
        last *= (1+g)
        fcfs.append(last)
    disc=[1/(1+r)**t for t in years]
    pv_fcfs=[a*b for a,b in zip(fcfs,disc)]
    tv = fcfs[-1]*(1+g_tv)/(r-g_tv) if r>g_tv else np.nan
    last_disc = disc[-1] if np.isfinite(disc[-1]) else 1.0
    pv_tv = (tv*last_disc) if np.isfinite(tv) else 0.0
    ev = float(np.nansum(pv_fcfs) + pv_tv)
    equity = ev - (net_debt or 0.0)
    per_share = equity / shares
    target = per_share * (1.0 - (safety or 0.0))
    detail = pd.DataFrame({"Year":years+["TV"],"FCF":fcfs+[np.nan],"Discount":disc+[last_disc],"PV":pv_fcfs+[pv_tv]})
    return float(target), float(ev), float(equity), detail

def per_price(eps, per, safety=0.3):
    if _miss(eps) or _miss(per): return None
    return float(eps*per*(1-safety))

def pbr_price(bps, pbr, safety=0.3):
    if _miss(bps) or _miss(pbr): return None
    return float(bps*pbr*(1-safety))

def ev_ebitda_price(ebitda, ev_mult, shares, net_debt=0.0, safety=0.3):
    if _miss(ebitda) or _miss(ev_mult) or _miss(shares) or shares<=0: return None
    ev = float(ebitda)*float(ev_mult)
    equity = ev - (net_debt or 0.0)
    return float((equity/shares)*(1-safety))

def safe_bar(df, x, y, title=""):
    df2=df.dropna(subset=[y])
    if df2.empty: return None
    fig=go.Figure([go.Bar(x=df2[x].astype(str), y=df2[y].astype(float),
                          text=[f"{v:,.0f}" if pd.notna(v) else "" for v in df2[y]],
                          textposition="outside")])
    fig.update_layout(title=title, margin=dict(t=40,r=20,l=20,b=40), xaxis_title=x, yaxis_title=y)
    return fig

# ───────────────────────────────
# UI
st.title("📊 적정주가 계산기 · 풀버전")
st.caption("FCF₀: 당기/최근/TTM → 최근 실적 → 현재연도에 가장 가까운 (E) · main_wide/JSON 단위 자동 환산")

with st.sidebar:
    cmp = st.text_input("종목코드 (6자리)", value="005930")
    current_price = st.number_input("현재가(원)", min_value=0.0, value=70000.0, step=10.0)
    st.markdown("---")
    st.subheader("DCF 파라미터")
    g1 = st.number_input("고성장률 (Y1~3)", value=0.15, step=0.005)
    g2 = st.number_input("중간성장률 (Y4~6)", value=0.10, step=0.005)
    g3 = st.number_input("저성장률 (Y7~10)", value=0.05, step=0.005)
    g_tv = st.number_input("장기성장률 g (TV)", value=0.03, step=0.005)
    r    = st.number_input("할인율 r (WACC)", value=0.09, step=0.005)
    safety = st.number_input("안전마진", value=0.30, step=0.05)
    st.markdown("---")
    st.subheader("상대가치 배수")
    per_mult = st.number_input("업종 PER", value=12.0, step=0.5)
    pbr_mult = st.number_input("업종 PBR", value=1.2, step=0.1)
    ev_mult  = st.number_input("EV/EBITDA", value=7.0, step=0.5)
    st.markdown("---")
    st.subheader("MIX 가중치")
    w_dcf = st.slider("DCF", 0.0, 1.0, 0.4, 0.05)
    w_per = st.slider("PER", 0.0, 1.0, 0.2, 0.05)
    w_pbr = st.slider("PBR", 0.0, 1.0, 0.2, 0.05)
    w_ev  = st.slider("EV/EBITDA", 0.0, 1.0, 0.2, 0.05)
    run = st.button("수집 → 계산", type="primary")

if run:
    if not re.fullmatch(r"\d{6}", cmp):
        st.error("종목코드는 6자리 숫자여야 합니다. 예) 005930 / 066570")
        st.stop()

    with st.spinner("토큰 획득 중..."):
        enc, cid = get_token(cmp)
    if not enc or not cid:
        st.error("토큰 추출 실패")
        st.stop()

    with st.spinner("데이터 수집(main/fs/profit/value)..."):
        main, main_mul = fetch_main(cmp, enc, cid)
        fs    = fetch_json(cmp, "fs", enc)
        prof  = fetch_json(cmp, "profit", enc)
        value = fetch_json(cmp, "value", enc)

    if main is None or main.empty:
        st.warning("main_wide(연간 주요재무정보) 표를 찾지 못했습니다. FCF₀ 1차 소스(main)는 생략하고 value/파생으로 시도합니다.")
        # main이 없어도 진행 가능한 파트(DCF/EV/...)는 value/prof/fs로 계산 유지


    # 핵심 값 추출 + META
    meta = {"main_mul": main_mul}

    # 발행주식수(주) — mul 적용하지 않음
    shares, shares_col, shares_typ = infer_from_main(main, [r"발행주식수|주식수|보통주수|총발행주식"], mul=1.0)
    meta["shares_col"]=shares_col; meta["shares_type"]=shares_typ

    # 시가총액/현재가로 백업 주식수 (필요 시)
    if (_miss(shares) or shares==0) and current_price>0:
        mcap, mcap_col, _ = infer_from_main(main, [r"시가총액|Market\s*Cap"], mul=main_mul)
        if not _miss(mcap):
            shares = float(mcap)/float(current_price)
            meta["shares_col"]=f"derived:mcap[{mcap_col}]/price"; meta["shares_type"]="derived"

    # 순부채: fs → value
    net_debt, nd_col, nd_typ = pick_from_table(fs, [r"^\*?순부채|Net\s*Debt"], exclude_patterns=None)
    if _miss(net_debt):
        net_debt, nd_col, nd_typ = pick_from_table(value, [r"^\*?순부채|Net\s*Debt"], exclude_patterns=None)
    if _miss(net_debt): net_debt = 0.0
    meta["net_debt_col"]=nd_col; meta["net_debt_type"]=nd_typ

    # EPS / BPS
    eps, eps_col, eps_typ = pick_from_table(value, [r"^EPS$|EPS\b"], exclude_patterns=[r'마진|%'])
    if _miss(eps):  # main_wide 백업(필요 시)
        eps, eps_col, eps_typ = infer_from_main(main, [r"^EPS$|EPS\b"], mul=1.0)
    bps, bps_col, bps_typ = pick_from_table(value, [r"^BPS$|BPS\b"], exclude_patterns=[r'마진|%'])
    if _miss(bps):
        bps, bps_col, bps_typ = infer_from_main(main, [r"^BPS$|BPS\b"], mul=1.0)
    meta["eps_col"]=eps_col; meta["eps_type"]=eps_typ
    meta["bps_col"]=bps_col; meta["bps_type"]=bps_typ

    # EBITDA
    ebitda, e_col, e_typ = find_ebitda(prof, main, main_mul)
    meta["ebitda_col"]=e_col; meta["ebitda_type"]=e_typ

    # FCF₀
    fcf0, fcf_col, fcf_typ = find_fcf_any(main, value, fs, main_mul)
    meta["fcf_col"]=fcf_col; meta["fcf_type"]=fcf_typ

    # 핵심값 표
    st.subheader("🔑 핵심 입력값")
    core_df = pd.DataFrame([{
        "발행주식수(주)": shares,
        "순부채(원)": net_debt,
        "EPS": eps,
        "BPS": bps,
        "EBITDA(원)": ebitda,
        "FCF₀(원)": fcf0
    }])
    st.dataframe(core_df, use_container_width=True)

    with st.expander("출처(META)"):
        st.json(meta)

    # Valuation 계산
    px_dcf, ev, equity, dcf_detail = dcf_price(fcf0, r, shares, g1, g2, g3, g_tv, safety, net_debt)
    px_per = per_price(eps, per_mult, safety)
    px_pbr = pbr_price(bps, pbr_mult, safety)
    px_ev  = ev_ebitda_price(ebitda, ev_mult, shares, net_debt, safety)

    # MIX
    wsum = max(w_dcf + w_per + w_pbr + w_ev, 1e-9)
    parts = [x*w/wsum for x,w in [(px_dcf,w_dcf),(px_per,w_per),(px_pbr,w_pbr),(px_ev,w_ev)] if x is not None]
    mix_price = float(np.nansum(parts)) if parts else None

    st.subheader("📌 적정주가 요약")
    summary = pd.DataFrame({
        "방법":["DCF","PER","PBR","EV/EBITDA","MIX(가중)"],
        "적정주가":[px_dcf, px_per, px_pbr, px_ev, mix_price]
    })
    st.dataframe(summary, use_container_width=True)
    fig = safe_bar(summary, "방법", "적정주가", title="방법별 적정주가")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("현재가", f"{current_price:,.0f} 원")
    col2.metric("적정가(MIX)", f"{(mix_price or 0):,.0f} 원")
    up = None if _miss(mix_price) or current_price<=0 else (mix_price/current_price-1)*100
    col3.metric("상승여력", f"{up:.2f}%" if up is not None else "-")

    st.subheader("DCF 세부내역")
    if dcf_detail is not None:
        st.dataframe(dcf_detail, use_container_width=True)
    else:
        st.info("DCF 계산을 위해 FCF₀/주식수/할인율이 필요합니다.")

    # 엑셀 다운로드
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        summary.to_excel(wr, sheet_name="SUMMARY", index=False)
        pd.DataFrame([{
            "현재가": current_price, "적정가(MIX)": mix_price, "상승여력%": up
        }]).to_excel(wr, sheet_name="PRICE", index=False)
        core_df.to_excel(wr, sheet_name="CORE_INPUTS", index=False)
        try:
            main.reset_index().to_excel(wr, sheet_name="MAIN_SNAPSHOT", index=False)
            fs.to_excel(wr, sheet_name="FS_SNAPSHOT", index=False)
            prof.to_excel(wr, sheet_name="PROFIT_SNAPSHOT", index=False)
            value.to_excel(wr, sheet_name="VALUE_SNAPSHOT", index=False)
        except Exception:
            pass
        pd.DataFrame([meta]).to_excel(wr, sheet_name="META", index=False)

    st.download_button(
        "⬇️ 결과 엑셀 다운로드",
        data=out.getvalue(),
        file_name=f"{cmp}_valuation_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
