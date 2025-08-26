# nick_prepare.py
import os, pickle, re
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

DATA_FILE = "names.csv"          # <— was TSV; now your CSV
OUT_DIR = "prep_out"
KGE_DIM = 128
TEXT_MODEL = "bert-base-uncased"

def norm_name(s: str) -> str:
    s = str(s).strip().strip("'\"`.,:;()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s

def load_nick_pairs(path: str) -> pd.DataFrame:
    """
    Accepts CSV or TSV. Expected header columns:
      - with relationship: name1, relationship, name2
      - without relationship: name1, name2
    If 'relationship' exists, we keep only rows where it contains 'nickname'.
    """
    # Try to auto-infer delimiter and header
    df = pd.read_csv(path, sep=None, engine="python")
    # If the columns aren’t as expected, retry without header
    if not set(df.columns).issuperset({"name1"}) or df.shape[1] not in (2, 3):
        df = pd.read_csv(path, sep=None, engine="python", header=None)
    # Normalize to standard column names
    if df.shape[1] == 3:
        df.columns = ["name1", "relationship", "name2"]
    elif df.shape[1] == 2:
        df.columns = ["name1", "name2"]
        df["relationship"] = "has_nickname"
    else:
        raise ValueError("Expected 2 or 3 columns: name1[, relationship], name2")

    # Filter on relationship if present
    if "relationship" in df.columns:
        df = df[df["relationship"].astype(str).str.lower().str.contains("nickname")]

    for c in ["name1", "name2"]:
        df[c] = df[c].map(norm_name)
    df = df[(df["name1"] != "") & (df["name2"] != "")]
    return df.drop_duplicates()

def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df.iterrows():
        a, b = r["name1"], r["name2"]
        if not G.has_node(a): G.add_node(a, kind="PERSON")
        if not G.has_node(b): G.add_node(b, kind="PERSON")
        if G.has_edge(a, b): G[a][b]["w"] += 1
        else: G.add_edge(a, b, w=1, rel="has_nickname")
    return G

def train_node2vec(G: nx.Graph, dim=128):
    n2v = Node2Vec(G, dimensions=dim, walk_length=80, num_walks=10,
                   p=1.0, q=1.0, weight_key="w", workers=4, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=256)
    return {n: model.wv.get_vector(str(n)).astype("float32") for n in G.nodes()}

def embed_names(names):
    emb = TransformerWordEmbeddings(model=TEXT_MODEL, layers="-1")
    name2vec = {}
    for n in names:
        sent = Sentence(n)
        emb.embed(sent)
        toks = [t.embedding.cpu().numpy() for t in sent]
        name2vec[n] = (np.mean(toks, axis=0) if toks else np.zeros(768, dtype=np.float32)).astype("float32")
    return name2vec

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_nick_pairs(DATA_FILE)
    G = build_graph(df)
    print(f"KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    kge = train_node2vec(G, dim=KGE_DIM)
    text = embed_names(list(G.nodes()))

    with open(os.path.join(OUT_DIR, "Dict_embed_node2vec.obj"), "wb") as f:
        pickle.dump(kge, f)
    with open(os.path.join(OUT_DIR, "name_text_embeddings.pkl"), "wb") as f:
        pickle.dump(text, f)
    with open(os.path.join(OUT_DIR, "nick_kg.gpickle"), "wb") as f:
        pickle.dump(G, f)
    print("Saved Node2Vec + text embeddings.")

if __name__ == "__main__":
    main()
