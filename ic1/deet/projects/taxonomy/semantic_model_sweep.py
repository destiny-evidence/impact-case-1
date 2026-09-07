from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.project import DeetProject
from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.cli_helpers import prepare_documents
from deet.extractors.keyword.semantic_keyword_extractor import SemanticKeywordDataExtractor

cfg = DataExtractionConfig.from_yaml(Path("configs/semantic.yaml"))
proj = DeetProject.load(); proc = proj.process_data()
proc.populate_custom_prompts(method=CustomPromptPopulationMethod.TAXONOMY, config=cfg)
strat = proj.load_evaluation_strategy(); proc.filter_documents_by_ids(strat.get_active_ids(proj))
docs,_ = prepare_documents(proc.documents, cfg, linked_document_path=proj.linked_documents_path,
                           pdf_dir=proj.pdf_dir_abspath, link_map_path=proj.link_map_path)
ex = SemanticKeywordDataExtractor.__new__(SemanticKeywordDataExtractor)  # skip model load
from deet.extractors.keyword.base_keyword_extractor import BaseKeywordDataExtractor
BaseKeywordDataExtractor.__init__(ex, cfg)
attrs = proc.attributes
phrases_per = ex._phrases_per_attribute(attrs)
flat, owner = [], []
for i,ph in enumerate(phrases_per):
    for p in ph: flat.append(p); owner.append(i)
owner=np.array(owner)
gold = pd.read_csv("data-extraction-experiments/2026-09-03_09-50-22_semantic/goldstandard_llm_comparison.csv")
goldmap = {(r.attribute_id,r.document_id): bool(r.human_extraction) for r in gold.itertuples()}
doc_sents=[]
for doc in docs:
    doc.set_abstract_context()
    s = ex._split_into_sentences(doc.context)
    doc_sents.append((int(doc.safe_identity.document_id), s))

def sweep(model_name):
    m = SentenceTransformer(model_name)
    pe = m.encode(flat, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    rows=[]
    for did,sents in doc_sents:
        if not sents: continue
        se = m.encode(sents, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        sim = pe @ se.T
        best = sim.max(axis=1)
        for i,a in enumerate(attrs):
            mm = best[owner==i]; mx=float(mm.max()) if mm.size else 0.0
            g = goldmap.get((a.attribute_id,did))
            if g is None: continue
            rows.append((a.attribute_id,mx,g))
    S=pd.DataFrame(rows,columns=["attr","sim","gold"])
    S=S[S.groupby("attr").gold.transform("max")>0]
    print(f"\n### {model_name}  (scoreable pairs {len(S)}, gold+ {int(S.gold.sum())})")
    print(f"{'thr':>5}{'P':>7}{'R':>7}{'F1':>7}")
    best_f=(-1,0)
    for t in np.arange(0.25,0.72,0.05):
        pred=S.sim>=t
        tp=int((pred&S.gold).sum());fp=int((pred&~S.gold).sum());fn=int((~pred&S.gold).sum())
        P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F=2*P*R/(P+R) if P+R else 0
        if F>best_f[0]: best_f=(F,t)
        print(f"{t:5.2f}{P:7.3f}{R:7.3f}{F:7.3f}")
    print(f"  BEST F1 {best_f[0]:.3f} @ thr {best_f[1]:.2f}")

for mdl in ["sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "NeuML/pubmedbert-base-embeddings"]:
    try: sweep(mdl)
    except Exception as e: print(f"\n### {mdl} FAILED: {e}")
