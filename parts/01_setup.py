# %% [markdown]
# # 🤖 HR RAG Assistant — ผู้ช่วย HR อัตโนมัติ (3 บริษัท)
# Notebook นี้เป็นตัวอย่างระบบ **RAG สำหรับตอบคำถามพนักงานซ้ำ ๆ เพื่อลดภาระ HR** ข้อมูลทั้งหมดเป็นข้อมูลสมมติ (Mock)
#
# ```
# คำถาม → Input Guard → Router → ตรวจสิทธิ์ → ┬ SQL Tools (ข้อมูลส่วนตัว: วันลา เงินเดือน)
#                                              └ Retrieval (นโยบาย .md: กรอง metadata → Dense+BM25 → RRF → Rerank)
#       → LLM เรียบเรียงคำตอบ → Output Guard → Audit log
# ```
# **ก่อนรัน:** เลือก Runtime → Change runtime type → **T4 GPU** แล้วกด Run all (ครั้งแรกโหลดโมเดลราว 5–10 นาที)
#
# | ส่วน | เนื้อหา |
# |---|---|
# | 1–2 | ติดตั้ง ตั้งค่า โหลดสเปก 3 บริษัท |
# | 3–4 | Mock ฐานข้อมูล SQLite และไฟล์นโยบาย .md |
# | 5–6 | แบ่ง chunk + BGE-M3 + Retriever (hybrid + rerank) |
# | 7–8 | Tools/สิทธิ์ และ Agent (guardrail) |
# | 9–10 | Visualization และ Evaluation |
# | 11–12 | Gradio UI (login + ดูข้อมูล + trace) |

# %% [markdown]
# ## 1) ติดตั้งและตั้งค่า

# %%
# @gpu
!pip -q install "gradio>=5,<6" sentence-transformers pythainlp bitsandbytes accelerate plotly scikit-learn pyyaml pandas

# %%
import os, re, json, math, time, sqlite3, hashlib, random, shutil
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_default_base = "/content/hr_rag" if Path("/content").exists() else "./hr_rag"
BASE = Path(os.environ.get("HR_RAG_BASE", _default_base))
DATA = BASE / "data"
POL = DATA / "policies"      # ไฟล์นโยบาย .md (เขียนลงดิสก์ที่ส่วน 4)
IDX = DATA / "index"         # embeddings + chunks
DB_PATH = DATA / "hr.db"     # SQLite
for _d in (DATA, POL, IDX):
    _d.mkdir(parents=True, exist_ok=True)

AS_OF = date(2026, 11, 10)   # "วันนี้" ของระบบจำลอง (ล็อกไว้ให้ผลซ้ำได้)
YEAR = 2026
ROLES = ["employee", "manager", "hr", "executive"]
ROLE_TH = {"employee": "พนักงาน", "manager": "หัวหน้างาน", "hr": "HR", "executive": "ผู้บริหาร"}


def db() -> sqlite3.Connection:
    """เปิด connection ใหม่ของ SQLite"""
    return sqlite3.connect(DB_PATH)


def q(sql: str, params=()) -> pd.DataFrame:
    """รัน SQL แล้วคืน DataFrame"""
    con = db()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def _smoke_01():
    assert BASE.exists() and POL.exists() and IDX.exists()
    with db() as con:
        con.execute("create table if not exists _t(x)")
        con.execute("drop table _t")
    assert isinstance(q("select 1 as a"), pd.DataFrame)
