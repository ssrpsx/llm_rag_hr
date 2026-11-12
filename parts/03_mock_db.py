# %% [markdown]
# ## 3) ฐานข้อมูลตัวอย่าง
# สร้างข้อมูลบุคลากรสมมติสำหรับสาธิต HR RAG โดยยึดตัวเลขจากสเปกบริษัท
# ข้อมูลทั้งหมดสร้างซ้ำได้ด้วย seed เดิมและใช้โดเมน/เบอร์โทรสมมติ

# %%
import hashlib
import random
import re
import sqlite3
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

DEMO_PASSWORD = "1234"

_FALLBACK_SEEDS = {
    "male_first": [{"th": t, "en": e} for t, e in [("สมชาย", "Somchai"), ("อนันต์", "Anan"), ("กิตติ", "Kitti"), ("ธนา", "Thana"), ("วิชัย", "Wichai"), ("ปกรณ์", "Pakorn"), ("ณรงค์", "Narong"), ("ภาคิน", "Phakin"), ("ธีรภัทร", "Teerapat"), ("ชาญ", "Chan"), ("เอก", "Ek"), ("ภัทร", "Phat")]],
    "female_first": [{"th": t, "en": e} for t, e in [("สุดา", "Suda"), ("พิมพ์ชนก", "Pimchanok"), ("ศิริพร", "Siriporn"), ("มาลี", "Malee"), ("กมล", "Kamon"), ("นภา", "Napa"), ("อรทัย", "Orathai"), ("วราภรณ์", "Waraporn"), ("ชนากานต์", "Chanakan"), ("ปวีณา", "Paweena"), ("รัตนา", "Rattana"), ("ธิดา", "Thida")]],
    "surnames": [{"th": t, "en": e} for t, e in [("ใจดี", "Jaidee"), ("ศรีสุข", "Srisuk"), ("แสงทอง", "Saengthong"), ("บุญมี", "Boonmee"), ("วงศ์ไทย", "Wongthai"), ("รักษา", "Raksa"), ("พูนผล", "Poonphon"), ("เกษมสุข", "Kasemsuk"), ("ธรรมดี", "Thamdee"), ("เจริญ", "Charoen"), ("มีชัย", "Meechai"), ("เพิ่มพูน", "Phoempoon")]],
    "nicknames": {"M": [{"th": t, "en": e} for t, e in [("โอ๋", "Oh"), ("ต้น", "Ton"), ("บอย", "Boy"), ("นัท", "Nut"), ("บอล", "Ball"), ("เอก", "Ek"), ("ป๊อป", "Pop"), ("โจ", "Joe"), ("อาร์ต", "Art"), ("กอล์ฟ", "Golf"), ("วิน", "Win"), ("เคน", "Ken")]], "F": [{"th": t, "en": e} for t, e in [("นุ่น", "Noon"), ("แอน", "Ann"), ("ฝน", "Fon"), ("เมย์", "May"), ("ฟ้า", "Fah"), ("อ้อม", "Om"), ("พลอย", "Ploy"), ("แพร", "Prae"), ("มิ้น", "Mint"), ("กิ๊ฟ", "Gift"), ("ดาว", "Dao"), ("แป้ง", "Pang")]]},
    "relations_th": ["แม่", "พ่อ", "คู่สมรส", "พี่น้อง"],
    "leave_reasons_th": {"vacation": ["พักผ่อนประจำปี", "เดินทางส่วนตัว"], "sick": ["พักรักษาสุขภาพ", "พบแพทย์"], "personal": ["จัดการธุระส่วนตัว", "ติดต่อหน่วยงาน"]},
    "rejected_reasons_th": ["ไม่อนุมัติเนื่องจากช่วง High Season ตามนโยบาย"],
    "review_comments_th": {str(i): ["ผลงานเป็นไปตามเป้าหมายและร่วมมือกับทีมได้ดี"] for i in range(1, 6)},
}


# %%
def hash_pw(emp_id, pw):
    salt = emp_id.lower() + "-hr"
    return hashlib.sha256(f"{salt}:{pw}".encode("utf-8")).hexdigest()


def _pick_pool(seeds, key, gender=None):
    pool = seeds.get(key, [])
    if gender and isinstance(pool, dict):
        pool = pool.get(gender, [])
    return pool or _FALLBACK_SEEDS[key][gender] if gender else (pool or _FALLBACK_SEEDS[key])


def _iso(d):
    return d.isoformat()


def _base_holidays():
    path = Path("docs/BASE_HOLIDAYS_2026.md")
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"(\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$", line)
            if m:
                rows.append(m.groups())
    return rows


def build_database(seed=42):
    rng = random.Random(seed)
    companies_spec = COMPANIES
    seeds = _SEEDS if isinstance(_SEEDS, dict) and _SEEDS.get("male_first") else _FALLBACK_SEEDS
    company_rows, employee_rows, user_rows, payroll_rows, review_rows = [], [], [], [], []
    leave_rows, request_rows, employees_by_company, holiday_rows = [], [], {}, []
    used_names = set()
    for code, spec in companies_spec.items():
        cid = spec["id"]
        contacts = spec["contacts"]
        company_rows.append({"company_id": cid, "code": code, "name_th": spec["name_th"], "name_en": spec["name_en"], "industry_th": spec["industry_th"], "industry_en": spec["industry_en"], "domain": spec["domain"], "hr_system": spec["hr_system"], "hr_email": contacts["hr"]["email"], "hr_ext": contacts["hr"]["ext"], "hr_line_oa": contacts["hr"].get("line_oa"), "payday_th": spec["pay"]["payday_rule_th"]})
        skipped = set(spec.get("holidays", {}).get("skip", []))
        for mmdd, name_th, name_en in _base_holidays():
            if mmdd not in skipped:
                holiday_rows.append({"company_id": cid, "holiday_date": f"{YEAR}-{mmdd}", "name_th": name_th, "name_en": name_en, "note_th": "วันหยุดตามปฏิทินตัวอย่าง"})
        for extra in spec.get("holidays", {}).get("extra", []):
            holiday_rows.append({"company_id": cid, "holiday_date": extra["date"], "name_th": extra["name_th"], "name_en": extra["name_en"], "note_th": extra.get("note_th", "")})
        people = []
        n = 1
        for org in spec["org"]:
            for _ in range(org["count"]):
                emp_id = f"{code}{1000+n}"
                n += 1
                gender = rng.choice(["M", "F"])
                first = rng.choice(_pick_pool(seeds, "male_first" if gender == "M" else "female_first"))
                surname = rng.choice(_pick_pool(seeds, "surnames"))
                while (first["th"], surname["th"]) in used_names:
                    surname = rng.choice(_pick_pool(seeds, "surnames"))
                used_names.add((first["th"], surname["th"]))
                nick = rng.choice(_pick_pool(seeds, "nicknames", gender))
                people.append({"emp_id": emp_id, "org": org, "gender": gender, "first": first, "surname": surname, "nick": nick})
        exec_id = people[0]["emp_id"]
        managers = {p["org"]["dept_en"]: p["emp_id"] for p in people if p["org"]["role"] == "manager"}
        probation = [p for p in people if p["org"]["role"] == "employee"][-2:]
        long_tenure = [p for p in people if p["org"]["role"] == "employee"][:2]
        for idx, p in enumerate(people):
            o = p["org"]
            if p in probation:
                hire = date(2026, rng.randint(7, 8), rng.randint(1, 28))
            elif p in long_tenure:
                hire = date(rng.randint(2016, 2020), rng.randint(1, 12), rng.randint(1, 28))
            else:
                hire = date(rng.randint(2021, 2026), rng.randint(1, 8), rng.randint(1, 28))
            manager_id = None if o["role"] == "executive" else (managers.get(o["dept_en"], exec_id) if o["role"] == "employee" else exec_id)
            full_en = f"{p['first']['en']} {p['surname']['en']}"
            prefix = re.sub(r"[^a-z]", "", p["first"]["en"].lower()) + "." + p["surname"]["en"][0].lower()
            domain = spec["domain"]
            ext_prefix = str(contacts["hr"]["ext"])
            phone_ext = (ext_prefix[:1] + f"{rng.randrange(0,1000):03d}") if len(ext_prefix) > 3 else f"{ext_prefix}{rng.randrange(0,10**(4-len(ext_prefix))):0{4-len(ext_prefix)}d}"
            mobile = f"0{rng.choice([8,9])}{rng.randrange(0,10):01d}-{rng.randrange(100,1000):03d}-{rng.randrange(1000,10000):04d}"
            row = {"emp_id": p["emp_id"], "company_id": cid, "name_th": p["first"]["th"]+" "+p["surname"]["th"], "name_en": full_en, "nickname_th": p["nick"]["th"], "nickname_en": p["nick"]["en"], "gender": p["gender"], "birth_date": _iso(date(rng.randint(1970, 2002), rng.randint(1,12), rng.randint(1,28))), "position_th": o["position_th"], "position_en": o["position_en"], "department_th": o["dept_th"], "department_en": o["dept_en"], "level": o["level"], "role": o["role"], "manager_id": manager_id, "hire_date": _iso(hire), "employment_type_th": "ทดลองงาน" if (AS_OF-hire).days < spec["hr_process"]["probation_days"] else "พนักงานประจำ", "status_th": "ทำงานอยู่", "email_work": f"{prefix}@{domain}", "phone_ext": phone_ext, "phone_mobile": mobile, "line_id": "@"+p["emp_id"].lower()+"-line", "chat_handle": "@"+p["emp_id"].lower() if code == "TT" else None, "office_location_th": contacts["hr"].get("location_th", spec["headquarters"]["name_th"]), "office_location_en": contacts["hr"].get("location_en", spec["headquarters"]["name_en"]), "emergency_name": "สมใจ "+p["surname"]["th"], "emergency_relation_th": rng.choice(_pick_pool(seeds,"relations_th")), "emergency_phone": mobile}
            employee_rows.append(row); employees_by_company.setdefault(code, []).append(row)
            band = o["band"]; low = ((band[0]+499)//500)*500; high = (band[1]//500)*500
            salary = rng.randrange(low//500, high//500+1)*500
            allowance = spec.get("work",{}).get("night_shift_allowance",0) * (2 if code == "SF" and o["level"] == "Operator" else 0)
            if code == "AR": allowance = rng.randrange(500,1501,100)
            if code == "TT" and o["role"] == "manager": allowance = 3000
            payroll_rows.append({"emp_id":p["emp_id"],"base_salary":salary,"allowance":allowance,"allowance_note_th":"ค่าเวร/Service Charge/ค่าตำแหน่งตามสเปก" if allowance else "ไม่มี", "bonus_months_last_year":rng.choice([0,0.5,1,1.5,2]),"bank_last4":f"{rng.randrange(10000):04d}","provident_pct":spec["pay"]["provident_employee_pct"]})
            review_rows.append({"emp_id":p["emp_id"],"year":2025,"rating":rng.choice([1,1.5,2,2.5,3,3.5,4,4.5,5]),"grade":rng.choice(["A","B","C","D"]),"comment_th":"ผลงานเป็นไปตามเป้าหมายและร่วมมือกับทีมได้ดี","reviewer_id":manager_id or p["emp_id"]})
            user_rows.append({"emp_id":p["emp_id"],"salt":p["emp_id"].lower()+"-hr","password_hash":hash_pw(p["emp_id"],DEMO_PASSWORD),"failed_attempts":0,"locked":0})
        # Balances first; approved requests are split into weekday segments that exactly sum to used.
        for person_index, (p, erow) in enumerate(zip(people, employees_by_company[code])):
            tenure = (AS_OF-date.fromisoformat(erow["hire_date"])).days/365.25
            for typ in ("vacation","sick","personal"):
                if typ == "vacation":
                    entitled = vacation_days_for_tenure(code, tenure)
                    if tenure < 1: entitled = max(1, round(entitled * max(1, (AS_OF.month-date.fromisoformat(erow["hire_date"]).month+1)/12)))
                else: entitled = spec["leave"]["sick_days"] if typ == "sick" else spec["leave"]["personal_days"]
                carry = rng.randint(0,spec["leave"].get("vacation_carry_over_max",0)) if typ == "vacation" and tenure >= 1.5 else 0
                used = rng.randint(0,min(entitled+carry,3))
                leave_rows.append({"emp_id":erow["emp_id"],"year":YEAR,"leave_type":typ,"entitled":entitled,"carried_over":carry,"used":used,"remaining":entitled+carry-used})
                left=used; rid=len(request_rows)+1
                while left:
                    days=min(left,rng.randint(1,3)); d=AS_OF-timedelta(days=rng.randint(5,220))
                    while d.weekday()>=5: d-=timedelta(days=1)
                    request_rows.append({"request_id":rid,"emp_id":erow["emp_id"],"leave_type":typ,"start_date":_iso(d),"end_date":_iso(d+timedelta(days=days-1)),"days":days,"status":"approved","reason_th":rng.choice(seeds.get("leave_reasons_th",{}).get(typ, _FALLBACK_SEEDS["leave_reasons_th"][typ])),"approver_id":erow["manager_id"] or erow["emp_id"],"created_at":_iso(d-timedelta(days=2))}); rid+=1; left-=days
            # Approximately one quarter of each company has a future vacation request.
            if person_index % 4 == 0:
                d = date(YEAR, rng.choice([11, 12]), rng.randint(1, 20))
                request_rows.append({"request_id":len(request_rows)+1,"emp_id":erow["emp_id"],"leave_type":"vacation","start_date":_iso(d),"end_date":_iso(d+timedelta(days=1)),"days":2,"status":"pending","reason_th":"วางแผนพักผ่อนล่วงหน้า","approver_id":erow["manager_id"] or erow["emp_id"],"created_at":_iso(AS_OF)})
    ar_people = employees_by_company.get("AR", [])
    for erow in ar_people[:2]:
        d = date(YEAR, 12, 20)
        request_rows.append({"request_id":len(request_rows)+1,"emp_id":erow["emp_id"],"leave_type":"vacation","start_date":_iso(d),"end_date":_iso(d+timedelta(days=1)),"days":2,"status":"rejected","reason_th":"ไม่อนุมัติเนื่องจาก High Season blackout ตามนโยบาย","approver_id":erow["manager_id"] or erow["emp_id"],"created_at":_iso(AS_OF)})
    with db() as conn:
        for table, rows in [("companies",company_rows),("employees",employee_rows),("users",user_rows),("payroll",payroll_rows),("performance_reviews",review_rows),("leave_balances",leave_rows),("leave_requests",request_rows),("holidays",holiday_rows)]:
            pd.DataFrame(rows).to_sql(table,conn,if_exists="replace",index=False)
        conn.execute("DROP TABLE IF EXISTS audit_log")
        conn.execute("CREATE TABLE audit_log (log_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, emp_id TEXT, company_id INTEGER, role TEXT, question TEXT, decision TEXT, tools_json TEXT, blocked INTEGER)")
        conn.commit()


# %%
def _smoke_03():
    build_database()
    with db() as conn:
        assert conn.execute("select count(*) from companies").fetchone()[0] == 3
        emp = pd.read_sql_query("select * from employees", conn)
        assert 57 <= len(emp) <= 63 and emp.emp_id.is_unique and emp.name_th.is_unique and emp.name_en.is_unique
        assert emp.manager_id.dropna().isin(emp.emp_id).all()
        assert all(hash_pw(e,"1234") == conn.execute("select password_hash from users where emp_id=?",(e,)).fetchone()[0] for e in emp.emp_id)
        assert conn.execute("select count(*) from holidays").fetchone()[0] > 0
        mismatch = conn.execute("select b.emp_id from leave_balances b left join leave_requests r on r.emp_id=b.emp_id and r.leave_type=b.leave_type and r.status='approved' and substr(r.start_date,1,4)=cast(b.year as text) group by b.emp_id,b.leave_type,b.used having b.used != coalesce(sum(r.days),0)").fetchall()
        assert not mismatch, mismatch
        assert conn.execute("select count(*) from leave_requests where status='pending' and leave_type='vacation' and start_date between '2026-11-01' and '2026-12-31'").fetchone()[0] >= len(emp)*0.20
        assert conn.execute("select count(*) from leave_requests where status='rejected' and emp_id like 'AR%' and reason_th like '%High Season%'").fetchone()[0] >= 2
        payroll = pd.read_sql_query("select e.emp_id,p.base_salary from employees e join payroll p using(emp_id)", conn)
        for row in payroll.itertuples(index=False):
            org = next(o for o in COMPANIES[row.emp_id[:2]]["org"] if o["position_en"] == emp.set_index("emp_id").loc[row.emp_id,"position_en"])
            assert org["band"][0] <= row.base_salary <= org["band"][1] and row.base_salary % 500 == 0
        for code in COMPANIES:
            recent = conn.execute("select count(*) from employees where emp_id like ? and role='employee' and hire_date>'2026-07-14'",(code+"%",)).fetchone()[0]
            assert recent >= 2


# %%
build_database()
_row_counts = q("SELECT name AS table_name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
_row_counts["rows"] = [_ for _ in [q(f'SELECT COUNT(*) AS n FROM "{t}"').iloc[0, 0] for t in _row_counts["table_name"]]]
display(_row_counts)
