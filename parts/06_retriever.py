# %% [markdown]
# ## 6) Retriever — กรอง metadata → Hybrid (Dense + BM25) → RRF → Rerank
# ```
# chunk ทั้งหมด ─► กรองบริษัท ─► กรองสิทธิ์ (role) ─► กรองเวอร์ชันเก่า
#              ─► Dense (BGE-M3)  ┐
#              ─► BM25 (คำไทย)    ┴► RRF รวมอันดับ ─► Top-12 ─► Reranker ─► Top-4 (เกินเกณฑ์) ─► LLM
# ```
# - **กรองก่อนค้นหา (pre-filter)**: เอกสารที่ไม่มีสิทธิ์ไม่เคยเข้าไปในผู้สมัคร จึงรั่วไม่ได้ แม้ถูกหลอกด้วย prompt
# - **BM25 + Dense**: Dense เก่งเรื่องความหมาย ส่วน BM25 เก่งคำเฉพาะ (รหัส ชื่อระบบ ตัวเลข)
# - **Reranker**: อ่านคำถามคู่กับ chunk พร้อมกัน แม่นกว่า จึงใช้จัดอันดับขั้นสุดท้าย
# - ถ้าคะแนน rerank ของทุก chunk ต่ำกว่าเกณฑ์ → "ไม่พบข้อมูล" แล้วแนะนำติดต่อ HR (ไม่เดา)

# %%
from pythainlp.tokenize import word_tokenize as _wt

RERANK_MIN = 0.05          # คะแนน rerank ขั้นต่ำที่ยอมรับเป็นหลักฐาน
RRF_K = 60

_TOKEN_RE = re.compile(r"[฀-๿a-zA-Z0-9]+")


def tokenize_th(text: str):
    """ตัดคำไทย/อังกฤษสำหรับ BM25"""
    out = []
    for t in _wt(text.lower(), engine="newmm", keep_whitespace=False):
        t = t.strip()
        if not t or not _TOKEN_RE.search(t):
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)
    return out


class BM25:
    """BM25 Okapi แบบย่อ (ไม่พึ่งไลบรารีภายนอก)"""

    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.N = k1, b, len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.dl.mean()) if self.N else 0.0
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, qtoks):
        s = np.zeros(self.N)
        for t in set(qtoks):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avg or 1)))
        return s


class Retriever:
    def __init__(self, chunks: pd.DataFrame, E: np.ndarray, embed_fn, rerank_fn):
        self.chunks = chunks.reset_index(drop=True)
        self.E = E
        self.embed_fn, self.rerank_fn = embed_fn, rerank_fn
        self.tokens = [tokenize_th(t) for t in self.chunks["text"]]
        self._roles = self.chunks["allowed_roles"].tolist()
        self.doc_of = dict(zip(self.chunks["chunk_id"], self.chunks["doc_id"]))

    def search(self, query: str, user: dict, k_fused: int = 12, k_final: int = 4, min_score: float = None) -> dict:
        min_score = RERANK_MIN if min_score is None else min_score
        ch = self.chunks
        t_all = time.perf_counter()
        N = len(ch)
        in_co = ch["company_id"].values == user["company_id"]
        role_ok = np.array([user["role"] in r for r in self._roles], dtype=bool)
        active = ch["status"].values == "active"
        m_role, m_all = in_co & role_ok, in_co & role_ok & active
        doc_ids, chunk_ids = ch["doc_id"].values, ch["chunk_id"].values
        funnel = [["ทุก chunk ในดัชนี (3 บริษัท)", int(N)],
                  ["หลังกรองบริษัท", int(in_co.sum())],
                  ["หลังกรองสิทธิ์ตามตำแหน่ง", int(m_role.sum())],
                  ["หลังกรองเอกสารฉบับเก่า", int(m_all.sum())]]
        res = {"query": query, "funnel": funnel, "candidates": [], "final": [],
               "accessible_docs": sorted(set(doc_ids[m_all])),
               "hidden_by_role": sorted(set(doc_ids[in_co & ~role_ok])),
               "hidden_by_version": sorted(set(doc_ids[m_role & ~active])),
               "stage_orders": {"dense": [], "bm25": [], "fused": [], "rerank": []},
               "_q_emb": None, "_idx": np.where(m_all)[0], "timing": {}}
        idx = res["_idx"]
        if len(idx) == 0:
            funnel += [["ผู้สมัครหลัง Hybrid (Dense+BM25→RRF)", 0], ["หลัง Rerank (ผ่านเกณฑ์)", 0]]
            return res

        t = time.perf_counter()
        q_emb = self.embed_fn([query])[0]
        res["_q_emb"] = q_emb
        sims = self.E[idx] @ q_emb
        d_order = np.argsort(-sims)
        dense_rank = np.empty(len(idx), dtype=int)
        dense_rank[d_order] = np.arange(1, len(idx) + 1)
        res["timing"]["dense_ms"] = round((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        bm = BM25([self.tokens[i] for i in idx])
        bscores = bm.scores(tokenize_th(query))
        bm_order = [j for j in np.argsort(-bscores) if bscores[j] > 0]
        bm_rank = np.zeros(len(idx), dtype=int)
        for r, j in enumerate(bm_order, 1):
            bm_rank[j] = r
        res["timing"]["bm25_ms"] = round((time.perf_counter() - t) * 1000)

        rrf = 1.0 / (RRF_K + dense_rank) + np.where(bm_rank > 0, 1.0 / (RRF_K + np.maximum(bm_rank, 1)), 0.0)
        f_order = np.argsort(-rrf)
        cand = f_order[:k_fused]
        funnel.append(["ผู้สมัครหลัง Hybrid (Dense+BM25→RRF)", int(len(cand))])

        t = time.perf_counter()
        rr = np.asarray(self.rerank_fn(query, [ch["text"].iloc[idx[j]] for j in cand]), dtype=float)
        res["timing"]["rerank_ms"] = round((time.perf_counter() - t) * 1000)
        r_pos = np.argsort(-rr)                       # ตำแหน่งใน cand เรียงตามคะแนน rerank
        rerank_rank = np.empty(len(cand), dtype=int)
        rerank_rank[r_pos] = np.arange(1, len(cand) + 1)
        selected = {int(p) for p in r_pos[:k_final] if rr[p] >= min_score}

        for pos, j in enumerate(cand):
            gi = idx[j]
            res["candidates"].append({
                "chunk_id": chunk_ids[gi], "doc_id": doc_ids[gi], "title": ch["title"].iloc[gi],
                "heading": ch["heading"].iloc[gi], "dense": float(sims[j]), "bm25": float(bscores[j]),
                "rrf": float(rrf[j]), "dense_rank": int(dense_rank[j]),
                "bm25_rank": int(bm_rank[j]) if bm_rank[j] else None, "fused_rank": pos + 1,
                "rerank": float(rr[pos]), "rerank_rank": int(rerank_rank[pos]), "selected": pos in selected})
        for p in r_pos:
            if int(p) in selected:
                gi = idx[cand[p]]
                res["final"].append({"chunk_id": chunk_ids[gi], "doc_id": doc_ids[gi], "title": ch["title"].iloc[gi],
                                     "heading": ch["heading"].iloc[gi], "rerank": float(rr[p]),
                                     "text": ch["text"].iloc[gi]})
        funnel.append(["หลัง Rerank (ผ่านเกณฑ์)", len(res["final"])])
        res["stage_orders"] = {"dense": [chunk_ids[idx[j]] for j in d_order],
                               "bm25": [chunk_ids[idx[j]] for j in bm_order],
                               "fused": [chunk_ids[idx[j]] for j in f_order],
                               "rerank": [chunk_ids[idx[cand[p]]] for p in r_pos]}
        res["timing"]["total_ms"] = round((time.perf_counter() - t_all) * 1000)
        return res

    def rank_of_docs(self, stage_chunk_ids, expect_docs):
        """อันดับ (1-based) แรกที่พบเอกสารใน expect_docs ในรายการ chunk_id ของแต่ละ stage; None ถ้าไม่พบ"""
        for r, cid in enumerate(stage_chunk_ids, 1):
            if self.doc_of.get(cid) in expect_docs:
                return r
        return None


retriever = Retriever(*load_index(), embed_fn, rerank_fn)
print(f"Retriever พร้อม: {len(retriever.chunks)} chunks")

# %% [markdown]
# **ทดลอง 1 คำถาม** — พนักงานบริษัท TT ถามเรื่องใบรับรองแพทย์ ดูตารางผู้สมัครและคะแนนแต่ละขั้น

# %%
_u = {"company_id": 1, "role": "employee"}
_r = retriever.search("ลาป่วยต้องมีใบรับรองแพทย์ตั้งแต่กี่วัน", _u)
print(" → ".join(f"{a}: {b}" for a, b in _r["funnel"]))
display(pd.DataFrame(_r["candidates"])[["fused_rank", "rerank_rank", "doc_id", "heading", "dense", "bm25", "rerank", "selected"]].round(3))
