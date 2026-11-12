# %% [markdown]
# ## 2) สเปก 3 บริษัท (แหล่งความจริงชุดเดียว)
# ตัวเลขนโยบายทั้งหมด (วันลา เงินเดือนจ่ายวันไหน วงเงินประกัน ฯลฯ) อยู่ใน `_SPEC` ชุดเดียว
# แล้วใช้ทั้งตอนสร้างฐานข้อมูลและตอนเขียนเอกสารนโยบาย ทำให้ "ตัวเลขใน DB ตรงกับตัวเลขในเอกสาร" เสมอ
# และ 3 บริษัทมีนโยบายต่างกันจริง จึงเห็นประโยชน์ของ RAG ที่ต้องดึงเฉพาะของบริษัทตัวเอง

# %%
COMPANIES = _SPEC                                   # code -> dict
COMPANY_BY_ID = {c["id"]: c for c in COMPANIES.values()}
CODE_BY_ID = {c["id"]: c["code"] for c in COMPANIES.values()}


def vacation_days_for_tenure(code: str, years: float) -> int:
    """สิทธิ์พักร้อนต่อปีตามอายุงาน (ปี) ของบริษัท code"""
    days = 0
    for t in sorted(COMPANIES[code]["leave"]["vacation_tiers"], key=lambda t: t["min_years"]):
        if years >= t["min_years"]:
            days = t["days"]
    return days


def tenure_years(hire_date, as_of=AS_OF) -> float:
    if isinstance(hire_date, str):
        hire_date = date.fromisoformat(hire_date)
    return (as_of - hire_date).days / 365.25


# ตารางเปรียบเทียบตัวเลขสำคัญของ 3 บริษัท (ไว้โชว์ว่านโยบายต่างกัน)
def company_comparison() -> pd.DataFrame:
    rows = []
    for c in COMPANIES.values():
        L, P, B = c["leave"], c["pay"], c["benefits"]
        rows.append({
            "บริษัท": c["name_th"], "อุตสาหกรรม": c["industry_th"],
            "ลาป่วย (วัน)": L["sick_days"], "ใบรับรองแพทย์เมื่อลา ≥ (วัน)": L["sick_cert_after_days"],
            "พักร้อน (ปีแรก→อาวุโส)": f'{L["vacation_tiers"][0]["days"]}→{L["vacation_tiers"][-1]["days"]}',
            "ลากิจ (วัน)": L["personal_days"], "เงินเดือนออกวันที่": P["payday_day"],
            "ประกัน IPD (บาท)": f'{B["ipd_limit"]:,}', "ระบบ HR": c["hr_system"],
        })
    return pd.DataFrame(rows)


display(company_comparison())


def _smoke_02():
    assert set(COMPANIES) == {"TT", "SF", "AR"}
    assert vacation_days_for_tenure("TT", 0.5) == 8 and vacation_days_for_tenure("TT", 1) == 12
    assert vacation_days_for_tenure("TT", 10) == 18 and vacation_days_for_tenure("SF", 3.2) == 10
    assert COMPANY_BY_ID[2]["code"] == "SF"
