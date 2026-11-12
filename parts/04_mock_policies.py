# %% [markdown]
# ## 4) ไฟล์นโยบาย `.md` (Mock) — จัดไฟล์ให้เป็นระเบียบ
# นโยบายแต่ละเรื่องเป็นไฟล์ Markdown แยกกัน เก็บเป็นโฟลเดอร์ตามบริษัท `policies/<TT|SF|AR>/<doc_id>.md`
# แต่ละไฟล์มี **front matter (YAML)** เก็บ metadata สำหรับกรองตอน retrieve:
# `company_id`, `allowed_roles` (ใครอ่านได้), `status` (active/superseded), `effective_date`, `category`
#
# มีทั้งเอกสารทั่วไป เอกสารจำกัดสิทธิ์ (เช่น โครงสร้างเงินเดือน เฉพาะ HR) และ **เอกสารฉบับเก่าที่ถูกยกเลิก**
# เพื่อทดสอบว่าระบบไม่หยิบนโยบายเก่ามาตอบ

# %%
def write_policies() -> list:
    """เขียนไฟล์นโยบายทั้งหมดลง POL (ลบของเก่าก่อน) แล้วคืนรายการ Path"""
    if POL.exists():
        shutil.rmtree(POL)
    paths = []
    for rel, text in POLICY_FILES.items():
        p = POL / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


_written = write_policies()
print(f"เขียนไฟล์นโยบาย {len(_written)} ไฟล์ -> {POL}")


def policy_inventory() -> pd.DataFrame:
    """สรุปจำนวนไฟล์ตามบริษัท / สถานะ / กลุ่มผู้อ่าน (อ่านจาก front matter)"""
    import yaml
    rows = []
    for p in sorted(POL.glob("*/*.md")):
        m = re.match(r"^---\n(.*?)\n---\n", p.read_text(encoding="utf-8"), re.S)
        fm = yaml.safe_load(m.group(1)) if m else {}
        rows.append({"company": p.parent.name, "doc_id": p.stem, "status": fm.get("status"),
                     "category": fm.get("category"), "allowed_roles": ", ".join(fm.get("allowed_roles", []))})
    return pd.DataFrame(rows)


_inv = policy_inventory()
if len(_inv):
    display(_inv.groupby(["company", "category"]).size().unstack(fill_value=0))
