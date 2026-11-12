# %% [markdown]
# ## 7) Tools + ระบบสิทธิ์ (RBAC) + Login
# ข้อมูลส่วนตัว (วันลา เงินเดือน ผลประเมิน ฯลฯ) **ไม่ให้ LLM เขียน SQL เอง** แต่ให้เรียก "tool" ที่เขียนไว้ล่วงหน้า (parameterized)
# และตรวจสิทธิ์ **ในโค้ด** ทุกครั้งก่อนรัน — ไม่พึ่ง prompt ซึ่งโดนหลอกได้
#
# หลักสำคัญ 3 ข้อ
# 1. `user` มาจาก session ที่ login แล้วเท่านั้น (tool ของตัวเอง เช่น `get_my_leave_balance` ไม่มีพารามิเตอร์ emp_id ให้ใครแก้)
# 2. ทุก query ผูก `company_id` ของผู้ใช้ — ข้ามบริษัทไม่ได้
# 3. ถูกปฏิเสธ = ตอบสุภาพ และไม่เปิดเผยว่ามีข้อมูลอยู่หรือไม่

# %%
import contextlib

_ALL = set(ROLES)
PERMISSIONS = {
    "get_my_profile": _ALL, "get_my_leave_balance": _ALL, "get_my_leave_requests": _ALL,
    "get_my_salary": _ALL, "get_holidays": _ALL, "get_employee_info": _ALL, "search_policy": _ALL,
    "get_team_leave": {"manager", "hr"}, "get_team_performance": {"manager", "hr"},
    "get_employee_salary": {"hr"}, "get_salary_summary": {"hr", "executive"},
}
DENY_MESSAGES = {
    "get_team_leave": "ข้อมูลวันลาของพนักงานท่านอื่นเปิดให้เฉพาะหัวหน้างาน (เฉพาะลูกทีมของตน) และ HR",
    "get_team_performance": "ผลการประเมินของพนักงานท่านอื่นเปิดให้เฉพาะหัวหน้างาน (เฉพาะลูกทีมของตน) และ HR",
    "get_employee_salary": "ข้อมูลเงินเดือนของพนักงานท่านอื่นเปิดเผยได้เฉพาะฝ่าย HR เท่านั้น",
    "get_salary_summary": "ข้อมูลสรุปเงินเดือนเปิดให้เฉพาะ HR และผู้บริหาร",
}
# รายละเอียด tool ที่ส่งให้ router (LLM) เลือกใช้ — เขียนเป็นอังกฤษให้โมเดลเข้าใจแม่น
TOOLS = {
    "get_my_profile": {"desc": "The logged-in user's own profile: position, department, manager, hire date, contact details.", "args": {}},
    "get_my_leave_balance": {"desc": "Remaining/used vacation, sick and personal leave days of the logged-in user.", "args": {}},
    "get_my_leave_requests": {"desc": "Recent leave requests and their status (approved/pending/rejected) of the logged-in user.", "args": {}},
    "get_my_salary": {"desc": "The logged-in user's own salary, allowance, bonus and payday.", "args": {}},
    "get_holidays": {"desc": "Company public-holiday calendar for this year.", "args": {}},
    "get_employee_info": {"desc": "Look up another employee's directory info (position, department, work email, extension, office; more fields only if permitted).", "args": {"target": "name, nickname or employee id"}},
    "search_policy": {"desc": "Search company policy documents (rules, benefits, procedures, how-to, contacts, FAQ).", "args": {"query": "standalone Thai search query"}},
    "get_team_leave": {"desc": "Who in the team is on leave / leave balances of team members (managers: direct reports, HR: whole company).", "args": {"target": "optional person name"}},
    "get_team_performance": {"desc": "Performance review results of team members (managers: direct reports, HR: whole company).", "args": {"target": "optional person name"}},
    "get_employee_salary": {"desc": "Salary of ANOTHER employee.", "args": {"target": "name, nickname or employee id"}},
    "get_salary_summary": {"desc": "Aggregate salary statistics by department (average/min/max).", "args": {}},
}
for _n, _t in TOOLS.items():
    _t["roles"] = sorted(PERMISSIONS[_n])

LEAVE_TH = {"vacation": "ลาพักร้อน", "sick": "ลาป่วย", "personal": "ลากิจ"}
STATUS_TH = {"approved": "อนุมัติแล้ว", "pending": "รออนุมัติ", "rejected": "ไม่อนุมัติ"}
PUBLIC_FIELDS = ["emp_id", "name_th", "name_en", "nickname_th", "nickname_en", "position_th", "position_en",
                 "department_th", "department_en", "email_work", "phone_ext", "office_location_th", "office_location_en"]
CONTACT_FIELDS = ["phone_mobile", "line_id", "chat_handle"]
SENSITIVE_FIELDS = ["birth_date", "hire_date", "emergency_name", "emergency_relation_th", "emergency_phone"]


def _records(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _exec(sql, params=()):
    with contextlib.closing(db()) as con:
        cur = con.execute(sql, params)
        con.commit()
        return cur.fetchall()


# ---------- Login ----------
MAX_FAILS = 5


def authenticate(emp_id: str, password: str):
    """คืน (user dict | None, ข้อความไทย) — ผิดครบ 5 ครั้งล็อกบัญชี"""
    emp_id = (emp_id or "").strip().upper()
    if not emp_id or not password:
        return None, "กรุณากรอกรหัสพนักงานและรหัสผ่าน"
    rows = _exec("select password_hash, failed_attempts, locked from users where emp_id=?", (emp_id,))
    if not rows:
        return None, "รหัสพนักงานหรือรหัสผ่านไม่ถูกต้อง"
    pw_hash, fails, locked = rows[0]
    if locked:
        return None, "บัญชีนี้ถูกล็อกเพราะกรอกรหัสผิดหลายครั้ง กรุณาติดต่อ HR"
    if hash_pw(emp_id, password) != pw_hash:
        fails += 1
        _exec("update users set failed_attempts=?, locked=? where emp_id=?", (fails, int(fails >= MAX_FAILS), emp_id))
        if fails >= MAX_FAILS:
            return None, "กรอกรหัสผิดครบ 5 ครั้ง บัญชีถูกล็อก กรุณาติดต่อ HR"
        return None, f"รหัสพนักงานหรือรหัสผ่านไม่ถูกต้อง (ผิด {fails}/{MAX_FAILS} ครั้ง)"
    _exec("update users set failed_attempts=0 where emp_id=?", (emp_id,))
    user = load_user(emp_id)
    return user, f"ยินดีต้อนรับ {user['name_th']}"


def load_user(emp_id: str) -> dict:
    """สร้าง user dict จากรหัสพนักงาน (ไม่ตรวจรหัสผ่าน — ใช้ภายในระบบ/ชุดทดสอบเท่านั้น)"""
    e = q("select e.*, c.code as company_code, c.name_th as company_name_th, c.name_en as company_name_en "
          "from employees e join companies c on c.company_id=e.company_id where e.emp_id=?", (emp_id,)).iloc[0]
    user = {k: (None if pd.isna(e[k]) else e[k]) for k in
            ["emp_id", "company_id", "company_code", "company_name_th", "company_name_en", "name_th", "name_en",
             "nickname_th", "position_th", "position_en", "department_th", "department_en", "role", "manager_id"]}
    user["company_id"] = int(user["company_id"])
    return user


# ---------- หาตัวพนักงานจากชื่อ ----------
_TITLES = ["นางสาว", "น.ส.", "นาย", "นาง", "คุณ", "น้อง", "พี่", "ท่าน", "mr.", "ms.", "mrs.", "k."]


def find_employees(user: dict, target: str) -> pd.DataFrame:
    """ค้นพนักงาน 'ในบริษัทเดียวกัน' จากรหัส/ชื่อไทย/ชื่ออังกฤษ/ชื่อเล่น"""
    t = (target or "").strip()
    low = t.lower()
    for p in _TITLES:
        if low.startswith(p):
            t, low = t[len(p):].strip(), low[len(p):].strip()
            break
    df = q("select * from employees where company_id=?", (user["company_id"],))
    if not t:
        return df.iloc[0:0]
    by_id = df[df["emp_id"].str.lower() == low]
    if len(by_id):
        return by_id
    first_th = df["name_th"].str.split().str[0]
    mask = (df["name_th"].str.contains(t, regex=False) | df["name_en"].str.lower().str.contains(low, regex=False)
            | (df["nickname_th"] == t) | (df["nickname_en"].str.lower() == low) | (first_th == t))
    return df[mask].head(5)


def _team_ids(user):
    return q("select emp_id from employees where manager_id=? and company_id=?", (user["emp_id"], user["company_id"]))["emp_id"].tolist()


def _is_direct_report(user, emp_row):
    return emp_row["manager_id"] == user["emp_id"]


def _ok(tool, title, data, message=""):
    return {"tool": tool, "status": "ok", "title": title, "data": data, "message": message}


def _nf(tool, message):
    return {"tool": tool, "status": "not_found", "title": "ไม่พบข้อมูล", "data": None, "message": message}


# ---------- Tools (ข้อมูลของตัวเอง) ----------
def t_get_my_profile(user, args):
    e = q("select * from employees where emp_id=?", (user["emp_id"],)).iloc[0].to_dict()
    mgr = q("select name_th, position_th, email_work, phone_ext from employees where emp_id=?", (e["manager_id"],)) if e.get("manager_id") else pd.DataFrame()
    d = {k: (None if pd.isna(v) else v) for k, v in e.items()}
    d["manager"] = _records(mgr)[0] if len(mgr) else None
    return _ok("get_my_profile", "ข้อมูลส่วนตัวของคุณ", d)


def t_get_my_leave_balance(user, args):
    df = q("select leave_type, entitled, carried_over, used, remaining from leave_balances where emp_id=? and year=?", (user["emp_id"], YEAR))
    if df.empty:
        return _nf("get_my_leave_balance", "ไม่พบข้อมูลวันลาของคุณ")
    df["ประเภท"] = df["leave_type"].map(LEAVE_TH)
    emp = q("select hire_date from employees where emp_id=?", (user["emp_id"],)).iloc[0]
    pend = q("select leave_type, start_date, end_date, days from leave_requests where emp_id=? and status='pending'", (user["emp_id"],))
    return _ok("get_my_leave_balance", f"วันลาคงเหลือปี {YEAR} (ณ {AS_OF.isoformat()})",
               {"balances": _records(df), "tenure_years": round(tenure_years(emp["hire_date"]), 1),
                "pending_requests": _records(pend)})


def t_get_my_leave_requests(user, args):
    df = q("select request_id, leave_type, start_date, end_date, days, status, reason_th from leave_requests "
           "where emp_id=? order by start_date desc limit 10", (user["emp_id"],))
    if df.empty:
        return _nf("get_my_leave_requests", "ยังไม่มีประวัติการลา")
    df["ประเภท"] = df["leave_type"].map(LEAVE_TH)
    df["สถานะ"] = df["status"].map(STATUS_TH)
    return _ok("get_my_leave_requests", "ใบลาล่าสุดของคุณ", _records(df))


def t_get_my_salary(user, args):
    df = q("select p.*, c.payday_th from payroll p join employees e on e.emp_id=p.emp_id "
           "join companies c on c.company_id=e.company_id where p.emp_id=?", (user["emp_id"],))
    if df.empty:
        return _nf("get_my_salary", "ไม่พบข้อมูลเงินเดือน")
    return _ok("get_my_salary", "เงินเดือนและรายได้ของคุณ", _records(df)[0])


def t_get_holidays(user, args):
    df = q("select holiday_date, name_th, name_en, note_th from holidays where company_id=? order by holiday_date", (user["company_id"],))
    df["upcoming"] = df["holiday_date"] >= AS_OF.isoformat()
    spec = COMPANY_BY_ID[user["company_id"]]
    return _ok("get_holidays", f"วันหยุดบริษัทปี {YEAR}",
               {"holidays": _records(df), "upcoming_count": int(df["upcoming"].sum()),
                "work_note": spec["holidays"].get("work_note_th", "")})


# ---------- Tools (ข้อมูลคนอื่น ต้องมีสิทธิ์) ----------
def t_get_employee_info(user, args):
    found = find_employees(user, args.get("target", ""))
    if found.empty:
        return _nf("get_employee_info", "ไม่พบพนักงานตามชื่อที่ระบุในบริษัทของคุณ")
    people = []
    for _, e in found.iterrows():
        fields, hidden = list(PUBLIC_FIELDS), []
        me, report = e["emp_id"] == user["emp_id"], _is_direct_report(user, e)
        if me or user["role"] == "hr":
            fields += CONTACT_FIELDS + SENSITIVE_FIELDS
        elif user["role"] == "manager" and report:
            fields += ["phone_mobile", "line_id"]
            hidden = SENSITIVE_FIELDS + ["chat_handle"]
        else:
            hidden = CONTACT_FIELDS + SENSITIVE_FIELDS
        people.append({**{k: (None if pd.isna(e[k]) else e[k]) for k in fields if k in e.index}, "ข้อมูลที่ไม่เปิดเผย": hidden})
    msg = "พบหลายคนที่ชื่อใกล้เคียง กรุณาระบุชื่อ-นามสกุลหรือรหัสพนักงานเพื่อความชัดเจน" if len(people) > 1 else ""
    return _ok("get_employee_info", "ข้อมูลพนักงาน", people, msg)


def t_get_team_leave(user, args):
    ids = _team_ids(user) if user["role"] == "manager" else q("select emp_id from employees where company_id=?", (user["company_id"],))["emp_id"].tolist()
    if args.get("target"):
        f = find_employees(user, args["target"])
        ids = [i for i in ids if i in set(f["emp_id"])]
    if not ids:
        return _nf("get_team_leave", "ไม่พบสมาชิกในทีมตามเงื่อนไข")
    ph = ",".join("?" * len(ids))
    people = q(f"select emp_id, name_th, position_th from employees where emp_id in ({ph})", tuple(ids))
    bal = q(f"select emp_id, remaining from leave_balances where year=? and leave_type='vacation' and emp_id in ({ph})", (YEAR, *ids))
    req = q(f"select emp_id, leave_type, start_date, end_date, days, status from leave_requests "
            f"where status in ('approved','pending') and end_date >= ? and emp_id in ({ph}) order by start_date", (AS_OF.isoformat(), *ids))
    req["ประเภท"] = req["leave_type"].map(LEAVE_TH)
    req["สถานะ"] = req["status"].map(STATUS_TH)
    out = []
    for _, p in people.iterrows():
        r = req[req["emp_id"] == p["emp_id"]]
        b = bal[bal["emp_id"] == p["emp_id"]]
        out.append({"ชื่อ": p["name_th"], "ตำแหน่ง": p["position_th"],
                    "พักร้อนคงเหลือ": None if b.empty else float(b["remaining"].iloc[0]),
                    "ใบลาที่กำลังจะถึง/อยู่ระหว่างลา": _records(r.drop(columns=["emp_id", "leave_type", "status"]))})
    return _ok("get_team_leave", "สถานะการลาของทีม" if user["role"] == "manager" else "สถานะการลาของพนักงานบริษัท", out)


def t_get_team_performance(user, args):
    ids = _team_ids(user) if user["role"] == "manager" else q("select emp_id from employees where company_id=?", (user["company_id"],))["emp_id"].tolist()
    if args.get("target"):
        f = find_employees(user, args["target"])
        ids = [i for i in ids if i in set(f["emp_id"])]
    if not ids:
        return _nf("get_team_performance", "ไม่พบสมาชิกในทีมตามเงื่อนไข")
    ph = ",".join("?" * len(ids))
    df = q(f"select e.name_th as ชื่อ, e.position_th as ตำแหน่ง, r.year as ปี, r.rating as คะแนน, r.grade as เกรด, r.comment_th as ความเห็น "
           f"from performance_reviews r join employees e on e.emp_id=r.emp_id where r.emp_id in ({ph}) order by e.name_th", tuple(ids))
    return _ok("get_team_performance", "ผลการประเมินประจำปี", _records(df))


def t_get_employee_salary(user, args):
    found = find_employees(user, args.get("target", ""))
    if found.empty:
        return _nf("get_employee_salary", "ไม่พบพนักงานตามชื่อที่ระบุในบริษัทของคุณ")
    ph = ",".join("?" * len(found))
    df = q(f"select e.name_th as ชื่อ, e.position_th as ตำแหน่ง, p.base_salary as เงินเดือน, p.allowance as เงินเพิ่ม, "
           f"p.allowance_note_th as หมายเหตุเงินเพิ่ม, p.bonus_months_last_year as โบนัสปีที่แล้วเดือน from payroll p "
           f"join employees e on e.emp_id=p.emp_id where p.emp_id in ({ph})", tuple(found["emp_id"]))
    return _ok("get_employee_salary", "ข้อมูลเงินเดือน (เฉพาะ HR)", _records(df))


def t_get_salary_summary(user, args):
    df = q("select e.department_th as แผนก, p.base_salary from payroll p join employees e on e.emp_id=p.emp_id where e.company_id=?", (user["company_id"],))
    g = df.groupby("แผนก")["base_salary"].agg(["count", "mean", "min", "max"]).reset_index()
    rows = []
    for _, r in g.iterrows():
        if user["role"] == "executive" and r["count"] < 3:          # k-anonymity: กลุ่มเล็กเกินไปจะระบุตัวตนได้
            rows.append({"แผนก": r["แผนก"], "จำนวนคน": int(r["count"]), "หมายเหตุ": "ไม่แสดง (กลุ่มน้อยกว่า 3 คน เพื่อปกป้องความเป็นส่วนตัว)"})
        else:
            rows.append({"แผนก": r["แผนก"], "จำนวนคน": int(r["count"]), "เฉลี่ย": round(r["mean"]), "ต่ำสุด": int(r["min"]), "สูงสุด": int(r["max"])})
    rows.append({"แผนก": "ทั้งบริษัท", "จำนวนคน": int(len(df)), "เฉลี่ย": round(df["base_salary"].mean())})
    return _ok("get_salary_summary", "สรุปเงินเดือนตามแผนก", rows)


_IMPL = {n[2:]: f for n, f in globals().items() if n.startswith("t_")}


def run_tool(user: dict, name: str, args: dict = None) -> dict:
    """ตรวจสิทธิ์ในโค้ด แล้วรัน tool; คืน dict {tool,status,title,data,message}; search_policy ถูกจัดการใน agent"""
    args = args or {}
    if name not in PERMISSIONS:
        return {"tool": name, "status": "error", "title": "ไม่รู้จักเครื่องมือ", "data": None, "message": f"unknown tool {name}"}
    if user["role"] not in PERMISSIONS[name]:
        return {"tool": name, "status": "denied", "title": "ไม่มีสิทธิ์",
                "data": None, "message": DENY_MESSAGES.get(name, "คุณไม่มีสิทธิ์เข้าถึงข้อมูลนี้")}
    if name == "search_policy":
        return {"tool": name, "status": "ok", "title": "ค้นนโยบาย", "data": None, "message": ""}
    if name in ("get_employee_info", "get_employee_salary") and not str(args.get("target", "")).strip():
        return _nf(name, "ไม่ได้ระบุชื่อพนักงาน กรุณาระบุชื่อหรือรหัสพนักงาน")
    try:
        return _IMPL[name](user, args)
    except Exception as e:                      # ไม่ให้ error รั่วเป็นข้อมูลภายใน
        return {"tool": name, "status": "error", "title": "เกิดข้อผิดพลาด", "data": None, "message": f"{type(e).__name__}: {e}"}


# ---------- มุมมองตารางสำหรับ Data Explorer ----------
_TABLES = ["companies", "employees", "leave_balances", "leave_requests", "payroll", "performance_reviews",
           "holidays", "audit_log", "users", "chunks"]


def list_tables():
    have = set(q("select name from sqlite_master where type='table'")["name"])
    return [t for t in _TABLES if t in have]


def scoped_df(table: str, user: dict, scope: bool = False, company_code: str = None) -> pd.DataFrame:
    """scope=False: ข้อมูลดิบ (โหมด demo/admin) | scope=True: เฉพาะแถว/คอลัมน์ที่ผู้ใช้นี้มีสิทธิ์เห็น"""
    if table not in _TABLES:
        raise ValueError(f"unknown table {table}")
    df = q(f"select * from {table}")
    if not scope:
        if company_code in COMPANIES:
            cid = COMPANIES[company_code]["id"]
            if "company_id" in df.columns:
                df = df[df["company_id"] == cid]
            elif "emp_id" in df.columns:
                df = df[df["emp_id"].str.startswith(company_code)]
        return df.reset_index(drop=True)
    role, me, cid = user["role"], user["emp_id"], user["company_id"]
    team = set(_team_ids(user)) if role == "manager" else set()
    if table == "users":
        return pd.DataFrame({"หมายเหตุ": ["ตาราง users ถูกซ่อนในมุมมองตามสิทธิ์"]})
    if table == "companies":
        return df[df["company_id"] == cid].reset_index(drop=True)
    if table == "holidays":
        return df[df["company_id"] == cid].reset_index(drop=True)
    if table == "chunks":
        keep = df["company_id"].eq(cid) & df["allowed_roles"].apply(lambda r: role in json.loads(r))
        return df[keep].reset_index(drop=True)
    if table == "employees":
        df = df[df["company_id"] == cid]
        if role == "hr":
            return df.reset_index(drop=True)
        if role == "executive":
            return df.drop(columns=CONTACT_FIELDS + SENSITIVE_FIELDS, errors="ignore").reset_index(drop=True)
        rows = df[df["emp_id"].isin({me} | team)].copy()
        mask_team = rows["emp_id"] != me
        for c in SENSITIVE_FIELDS + ["chat_handle"]:
            rows.loc[mask_team, c] = None
        return rows.reset_index(drop=True)
    if table == "audit_log":
        return (df[df["company_id"] == cid] if role == "hr" else df[df["emp_id"] == me]).reset_index(drop=True)
    # ตารางที่มี emp_id
    company_ids = set(q("select emp_id from employees where company_id=?", (cid,))["emp_id"])
    if role == "hr":
        allowed = company_ids
    elif role == "manager" and table in ("leave_balances", "leave_requests", "performance_reviews"):
        allowed = {me} | team
    else:
        allowed = {me}
    return df[df["emp_id"].isin(allowed)].reset_index(drop=True)


# ---------- ทดสอบ (ใช้ DB จำลองขนาดเล็กตามสัญญา ไม่พึ่งการ build จริง) ----------
def _make_fixture_db(path):
    import tempfile
    rows, E = [], []
    for code, c in COMPANIES.items():
        cid = c["id"]
        def emp(n, role, dept, mgr, name, hire="2022-01-01"):
            return {"emp_id": f"{code}{1000 + n}", "company_id": cid, "name_th": name, "name_en": name, "nickname_th": f"น{n}",
                    "nickname_en": f"N{n}", "gender": "F", "birth_date": "1990-01-01", "position_th": "ตำแหน่ง", "position_en": "Pos",
                    "department_th": dept, "department_en": dept, "level": "x", "role": role, "manager_id": mgr, "hire_date": hire,
                    "employment_type_th": "พนักงานประจำ", "status_th": "ทำงานอยู่", "email_work": f"e{n}@{c['domain']}",
                    "phone_ext": f"{c['contacts']['hr']['ext'][0]}{n:03d}", "phone_mobile": "081-000-000" + str(n % 10),
                    "line_id": f"line{n}", "chat_handle": f"@h{n}", "office_location_th": "ชั้น 1", "office_location_en": "F1",
                    "emergency_name": "แม่", "emergency_relation_th": "แม่", "emergency_phone": "089-111-1111"}
        E += [emp(1, "executive", "บริหาร", None, f"ผู้บริหาร{code}"),
              emp(2, "hr", "บุคคล", f"{code}1001", f"เอชอาร์{code}"),
              emp(4, "manager", "วิศวกรรม", f"{code}1001", f"หัวหน้าเอ{code}"),
              emp(5, "manager", "ขาย", f"{code}1001", f"หัวหน้าบี{code}"),
              emp(7, "employee", "วิศวกรรม", f"{code}1004", f"สมชาย{code}"),
              emp(8, "employee", "วิศวกรรม", f"{code}1004", f"สมหญิง{code}"),
              emp(9, "employee", "ขาย", f"{code}1005", f"สมศรี{code}")]
    emp_df = pd.DataFrame(E)
    comp_df = pd.DataFrame([{"company_id": c["id"], "code": k, "name_th": c["name_th"], "name_en": c["name_en"], "industry_th": "x",
                             "industry_en": "x", "domain": c["domain"], "hr_system": c["hr_system"], "hr_email": c["contacts"]["hr"]["email"],
                             "hr_ext": c["contacts"]["hr"]["ext"], "hr_line_oa": "@x", "payday_th": c["pay"]["payday_rule_th"]} for k, c in COMPANIES.items()])
    users = pd.DataFrame([{"emp_id": i, "salt": i.lower() + "-hr", "password_hash": hash_pw(i, "1234"), "failed_attempts": 0, "locked": 0} for i in emp_df["emp_id"]])
    bal = pd.DataFrame([{"emp_id": i, "year": YEAR, "leave_type": t, "entitled": 10, "carried_over": 0, "used": 3, "remaining": 7}
                        for i in emp_df["emp_id"] for t in ("vacation", "sick", "personal")])
    req = pd.DataFrame([{"request_id": n, "emp_id": i, "leave_type": "vacation", "start_date": "2026-11-20", "end_date": "2026-11-21", "days": 2,
                         "status": "pending", "reason_th": "พักผ่อน", "approver_id": None, "created_at": "2026-11-01"} for n, i in enumerate(emp_df["emp_id"])])
    pay = pd.DataFrame([{"emp_id": i, "base_salary": 30000 + 1000 * n, "allowance": 0, "allowance_note_th": "", "bonus_months_last_year": 1.5,
                         "bank_last4": "1234", "provident_pct": 5} for n, i in enumerate(emp_df["emp_id"])])
    perf = pd.DataFrame([{"emp_id": i, "year": 2025, "rating": 4.0, "grade": "B", "comment_th": "ดี", "reviewer_id": None} for i in emp_df["emp_id"]])
    hol = pd.DataFrame([{"company_id": c["id"], "holiday_date": d, "name_th": "วันหยุด", "name_en": "H", "note_th": ""} for c in COMPANIES.values() for d in ("2026-01-01", "2026-12-10")])
    with contextlib.closing(sqlite3.connect(path)) as con:
        for name, df in [("companies", comp_df), ("employees", emp_df), ("users", users), ("leave_balances", bal), ("leave_requests", req),
                         ("payroll", pay), ("performance_reviews", perf), ("holidays", hol)]:
            df.to_sql(name, con, index=False)
        con.execute("create table audit_log(log_id integer primary key autoincrement, ts, emp_id, company_id, role, question, decision, tools_json, blocked int)")
        con.commit()


def _smoke_07():
    import tempfile
    global DB_PATH
    real = DB_PATH
    DB_PATH = Path(tempfile.mkdtemp()) / "fixture.db"
    try:
        _make_fixture_db(DB_PATH)
        users = {r: authenticate(f"TT100{n}", "1234")[0] for r, n in
                 {"executive": 1, "hr": 2, "manager": 4, "employee": 7}.items()}
        assert all(users.values()) and users["manager"]["role"] == "manager" and users["employee"]["company_code"] == "TT"
        # login
        assert authenticate("TT1007", "wrong")[0] is None and authenticate("NOPE", "1234")[0] is None
        for _ in range(5):
            authenticate("TT1008", "x")
        assert authenticate("TT1008", "1234")[0] is None                       # ถูกล็อกแล้ว
        # เมทริกซ์สิทธิ์: เฉลยเขียนตายตัวตาม docs/CONTRACT.md §11 (ห้ามอ้างอิง PERMISSIONS เพื่อไม่ให้เป็นการเทียบตัวเอง)
        expected_allowed = {
            "get_my_profile": "employee manager hr executive", "get_my_leave_balance": "employee manager hr executive",
            "get_my_leave_requests": "employee manager hr executive", "get_my_salary": "employee manager hr executive",
            "get_holidays": "employee manager hr executive", "get_employee_info": "employee manager hr executive",
            "get_team_leave": "manager hr", "get_team_performance": "manager hr",
            "get_employee_salary": "hr", "get_salary_summary": "hr executive"}
        assert set(expected_allowed) == set(PERMISSIONS) - {"search_policy"}
        for tool, allowed in expected_allowed.items():
            for role, u in users.items():
                r = run_tool(u, tool, {"target": "สมหญิงTT"})
                assert (r["status"] != "denied") == (role in allowed.split()), (tool, role, r["status"])
        # ข้อมูลตัวเองถูกต้อง
        bal = run_tool(users["employee"], "get_my_leave_balance")
        assert bal["status"] == "ok" and {b["leave_type"] for b in bal["data"]["balances"]} == {"vacation", "sick", "personal"}
        # หัวหน้าเห็นแค่ลูกทีมของตัวเอง (TT1004 ดูแล TT1007,TT1008 ไม่รวม TT1009)
        tl = run_tool(users["manager"], "get_team_leave")
        names = {p["ชื่อ"] for p in tl["data"]}
        assert names == {"สมชายTT", "สมหญิงTT"}, names
        # HR เห็นทั้งบริษัท และไม่ปนบริษัทอื่น
        hr_names = {p["ชื่อ"] for p in run_tool(users["hr"], "get_team_leave")["data"]}
        assert len(hr_names) == 7 and not any("SF" in n or "AR" in n for n in hr_names)
        # ข้ามบริษัท: หาชื่อพนักงาน SF จาก TT ต้องไม่เจอ
        assert run_tool(users["hr"], "get_employee_salary", {"target": "สมชายSF"})["status"] == "not_found"
        assert run_tool(users["hr"], "get_employee_salary", {"target": "คุณสมชายTT"})["status"] == "ok"
        # ฟิลด์ติดต่อ: พนักงานทั่วไปไม่เห็นเบอร์มือถือคนอื่น / หัวหน้าเห็นของลูกทีม / HR เห็นหมด
        e = run_tool(users["employee"], "get_employee_info", {"target": "สมหญิงTT"})["data"][0]
        assert "phone_mobile" not in e and "phone_ext" in e and "phone_mobile" in e["ข้อมูลที่ไม่เปิดเผย"]
        m = run_tool(users["manager"], "get_employee_info", {"target": "สมหญิงTT"})["data"][0]
        assert "phone_mobile" in m and "birth_date" not in m
        m2 = run_tool(users["manager"], "get_employee_info", {"target": "สมศรีTT"})["data"][0]      # ไม่ใช่ลูกทีม
        assert "phone_mobile" not in m2
        h = run_tool(users["hr"], "get_employee_info", {"target": "สมหญิงTT"})["data"][0]
        assert "birth_date" in h and "emergency_phone" in h
        # ผู้บริหาร: สรุปเงินเดือน ซ่อนกลุ่ม < 3 คน
        s = run_tool(users["executive"], "get_salary_summary")["data"]
        assert any("ไม่แสดง" in str(r.get("หมายเหตุ", "")) for r in s)
        assert all("ต่ำสุด" in r for r in run_tool(users["hr"], "get_salary_summary")["data"] if r["แผนก"] != "ทั้งบริษัท")
        # มุมมองตามสิทธิ์
        assert set(scoped_df("payroll", users["employee"], True)["emp_id"]) == {"TT1007"}
        assert set(scoped_df("payroll", users["manager"], True)["emp_id"]) == {"TT1004"}
        assert set(scoped_df("leave_balances", users["manager"], True)["emp_id"]) == {"TT1004", "TT1007", "TT1008"}
        assert len(scoped_df("payroll", users["hr"], True)) == 7
        assert len(scoped_df("payroll", users["executive"], True)) == 1
        assert "phone_mobile" not in scoped_df("employees", users["executive"], True).columns
        assert len(scoped_df("employees", users["employee"], False)) == 21 and len(scoped_df("employees", users["employee"], False, "SF")) == 7
        assert "users" in list_tables() and "หมายเหตุ" in scoped_df("users", users["employee"], True).columns
        assert run_tool(users["employee"], "drop_table")["status"] == "error"
    finally:
        DB_PATH = real
