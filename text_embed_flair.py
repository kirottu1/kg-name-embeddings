from typing import List, Dict
import numpy as np
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

# BERT-base via Flair, word-level embeddings
EMB = TransformerWordEmbeddings(model="bert-base-uncased", layers="-1", pooling_operation="first")

def embed_name(name: str) -> np.ndarray:
    # single word/phrase embedding (average over tokens)
    sent = Sentence(name)
    EMB.embed(sent)
    vecs = [t.embedding.cpu().numpy() for t in sent]
    return np.mean(vecs, axis=0) if vecs else np.zeros(768, dtype=np.float32)

def embed_name_with_context(name: str, contexts: List[str]) -> np.ndarray:
    if not contexts:
        return embed_name(name)
    vecs = []
    # embed the name itself plus a few short contexts
    vecs.append(embed_name(name))
    for c in contexts[:5]:
        sent = Sentence(c[:256])
        EMB.embed(sent)
        toks = [t.embedding.cpu().numpy() for t in sent]
        if toks:
            vecs.append(np.mean(toks, axis=0))
    return np.mean(vecs, axis=0)

if __name__ == "__main__":
    import json, sys, pickle
    # Expect a JSON mapping: { "Name": ["context1","context2", ...], ... }
    ctx_path = sys.argv[1] if len(sys.argv) > 1 else None
    if ctx_path:
        name2ctx = json.load(open(ctx_path, "r", encoding="utf-8"))
        names = list(name2ctx.keys())
    else:
        import networkx as nx
        G = nx.read_gpickle("nick_kg.gpickle")
        names = list(G.nodes()); name2ctx = {n: [] for n in names}

    name2vec = {n: embed_name_with_context(n, name2ctx.get(n, [])) for n in names}
    with open("text_emb_flair.pkl", "wb") as f:
        pickle.dump(name2vec, f)
    print(f"Text embeddings saved for {len(name2vec)} names")
