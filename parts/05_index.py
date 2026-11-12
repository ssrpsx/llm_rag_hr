# %% [markdown]
# ## 5) แบ่ง chunk และสร้างดัชนี (BGE-M3)
# ขั้นตอน: อ่านไฟล์ .md → แยก front matter → **แบ่ง chunk ตามหัวข้อ `##`** (แต่ละ chunk ติด header "[ชื่อเอกสาร] หัวข้อ")
# → แปลงเป็นเวกเตอร์ด้วย **BGE-M3** (รองรับภาษาไทย) → เก็บ metadata ไว้ในตาราง `chunks` ของ SQLite และเวกเตอร์ใน `.npy`
#
# metadata ที่เก็บต่อ chunk: `company_id`, `doc_id`, `allowed_roles`, `status`, `effective_date`, `category`
# ใช้กรอง **ก่อน** ค้นหา เพื่อให้เอกสารของบริษัทอื่น / เกินสิทธิ์ / ฉบับเก่า ไม่เคยถูกหยิบมาเลย

# %%
# โหลดโมเดล embedding (BGE-M3) และ reranker (BGE-Reranker-v2-m3) ขึ้น GPU
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

EMB_ID = "BAAI/bge-m3"
RERANK_ID = "BAAI/bge-reranker-v2-m3"
_dev = "cuda" if torch.cuda.is_available() else "cpu"

_emb_model = SentenceTransformer(EMB_ID, device=_dev)
_emb_model.max_seq_length = 512
_reranker = CrossEncoder(RERANK_ID, device=_dev, max_length=512)
if _dev == "cuda":
    _emb_model.half()
    _reranker.model.half()


def embed_fn(texts, batch_size: int = 16):
    """list[str] -> np.ndarray (n, 1024) normalised"""
    return _emb_model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def rerank_fn(query: str, docs):
    """คืนคะแนนความเกี่ยวข้อง 0..1 ของแต่ละ doc ต่อ query"""
    s = np.asarray(_reranker.predict([(query, d) for d in docs], batch_size=8, show_progress_bar=False), dtype=np.float64)
    if len(s) and (s.min() < 0 or s.max() > 1):
        s = 1.0 / (1.0 + np.exp(-s))
    return s.tolist()


print("embedding + reranker พร้อมบน", _dev)

# %%
import yaml


def parse_front_matter(text: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def split_sections(body: str):
    """แยกเนื้อหาเป็น [(heading, text)] ตามหัวข้อ '## ' (ส่วนก่อนหัวข้อแรกคือ 'ภาพรวม')"""
    parts = re.split(r"^## ", body, flags=re.M)
    out = []
    intro = re.sub(r"^# .*\n?", "", parts[0].strip(), count=1).strip()
    if intro:
        out.append(("ภาพรวม", intro))
    for p in parts[1:]:
        head, _, rest = p.partition("\n")
        if rest.strip():
            out.append((head.strip(), rest.strip()))
    return out


def split_long(text: str, limit: int = 900):
    """ถ้าหัวข้อยาวเกิน limit ตัวอักษร แบ่งตามย่อหน้า (ไม่ตัดกลางตาราง/ย่อหน้า)"""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


CHUNK_COLUMNS = ["chunk_id", "company_id", "company_code", "doc_id", "title", "title_en", "heading",
                 "allowed_roles", "status", "effective_date", "version", "category", "source_file", "n_chars", "text"]


def make_chunks(policy_dir: Path = None) -> pd.DataFrame:
    policy_dir = policy_dir or POL
    rows = []
    for p in sorted(policy_dir.glob("*/*.md")):
        fm, body = parse_front_matter(p.read_text(encoding="utf-8"))
        if not fm:
            continue
        for heading, sec in split_sections(body):
            for k, piece in enumerate(split_long(sec)):
                h = heading if k == 0 else f"{heading} (ต่อ {k + 1})"
                text = f"[{fm['title']}] {h}\n{piece}"
                rows.append({
                    "chunk_id": f"{fm['company_code']}:{fm['doc_id']}:{len(rows):03d}",
                    "company_id": int(fm["company_id"]), "company_code": fm["company_code"],
                    "doc_id": fm["doc_id"], "title": fm["title"], "title_en": fm.get("title_en", ""),
                    "heading": h, "allowed_roles": list(fm["allowed_roles"]), "status": fm["status"],
                    "effective_date": str(fm["effective_date"]), "version": int(fm["version"]),
                    "category": fm["category"], "source_file": f"{p.parent.name}/{p.name}",
                    "n_chars": len(text), "text": text})
    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)


def build_index() -> None:
    """chunk → embed → บันทึก chunks (SQLite) + embeddings.npy"""
    chunks = make_chunks()
    if chunks.empty:
        print("ไม่พบไฟล์นโยบาย — ข้ามการสร้างดัชนี")
        return
    t0 = time.time()
    E = embed_fn(chunks["text"].tolist())
    np.save(IDX / "embeddings.npy", E)
    out = chunks.copy()
    out["allowed_roles"] = out["allowed_roles"].apply(lambda r: json.dumps(r, ensure_ascii=False))
    with db() as con:
        out.to_sql("chunks", con, if_exists="replace", index=False)
    (IDX / "meta.json").write_text(json.dumps({"n_chunks": len(chunks), "dim": int(E.shape[1])}), encoding="utf-8")
    print(f"สร้างดัชนี {len(chunks)} chunks, dim={E.shape[1]} ใช้เวลา {time.time() - t0:.1f}s")


def load_index():
    """คืน (chunks DataFrame ที่ allowed_roles เป็น list, embeddings ndarray)"""
    ch = q("select * from chunks")
    ch["allowed_roles"] = ch["allowed_roles"].apply(json.loads)
    E = np.load(IDX / "embeddings.npy")
    assert len(ch) == len(E), "chunks กับ embeddings จำนวนไม่ตรงกัน — รัน build_index() ใหม่"
    return ch, E


build_index()
_ch = make_chunks()
if len(_ch):
    display(_ch.groupby("company_code")["n_chars"].describe()[["count", "mean", "max"]].round(0))
    display(_ch[["chunk_id", "heading", "status", "allowed_roles", "n_chars"]].head(8))
