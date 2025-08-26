# nick_build_dataset.py
import os, pickle, random
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

DATA_FILE = "names.csv"          # <— was TSV; now your CSV
PREP_DIR = "prep_out"
OUT_DIR = "testset_trainset"
SEED = 42
NEG_PER_POS = 5

def load_all():
    with open(os.path.join(PREP_DIR, "nick_kg.gpickle"), "rb") as f:
        G = pickle.load(f)
    kge = pickle.load(open(os.path.join(PREP_DIR, "Dict_embed_node2vec.obj"), "rb"))
    txt = pickle.load(open(os.path.join(PREP_DIR, "name_text_embeddings.pkl"), "rb"))
    return G, kge, txt

def load_pairs():
    # Reuse the same flexible loader logic as in nick_prepare
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    if not set(df.columns).issuperset({"name1"}) or df.shape[1] not in (2, 3):
        df = pd.read_csv(DATA_FILE, sep=None, engine="python", header=None)
    if df.shape[1] == 3:
        df.columns = ["name1", "relationship", "name2"]
        df = df[df["relationship"].astype(str).str.lower().str.contains("nickname")]
    elif df.shape[1] == 2:
        df.columns = ["name1", "name2"]
    else:
        raise ValueError("Expected 2 or 3 columns: name1[, relationship], name2")
    return list(map(tuple, df[["name1","name2"]].drop_duplicates().values))

def build_knn(vectors: dict, n_neighbors=30):
    names = list(vectors.keys())
    X = np.stack([vectors[n] for n in names])
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, max(2, len(names)-1)), metric="cosine").fit(X)
    return names, X, nn

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    G, kge, txt = load_all()
    positives = load_pairs()

    names_txt, Xtxt, nn_txt = build_knn(txt, 30)
    names_kge, Xkge, nn_kge = build_knn(kge, 30)
    itxt = {n:i for i,n in enumerate(names_txt)}
    kidx = {n:i for i,n in enumerate(names_kge)}

    rng = random.Random(SEED)
    X_list, y_list = [], []

    def hard_negs(anchor, k=10):
        res = set()
        if anchor in itxt:
            _, idxs = nn_txt.kneighbors(Xtxt[itxt[anchor]:itxt[anchor]+1], n_neighbors=min(k+1, len(names_txt)))
            res.update(names_txt[j] for j in idxs[0] if names_txt[j] != anchor)
        if anchor in kidx:
            _, idxs = nn_kge.kneighbors(Xkge[kidx[anchor]:kidx[anchor]+1], n_neighbors=min(k+1, len(names_kge)))
            res.update(names_kge[j] for j in idxs[0] if names_kge[j] != anchor)
        return list(res)

    for a, b in positives:
        if a not in txt or b not in txt or a not in kge or b not in kge:
            continue
        X_list.append(np.hstack([kge[b], txt[a]])); y_list.append(1.0)
        X_list.append(np.hstack([kge[a], txt[b]])); y_list.append(1.0)

        pool = [z for z in hard_negs(b, 10) if z not in (a, b) and not G.has_edge(a, z)]
        rng.shuffle(pool)
        for z in pool[:NEG_PER_POS]:
            if z in kge and z in txt:
                X_list.append(np.hstack([kge[z], txt[a]])); y_list.append(0.0)

        pool = [z for z in hard_negs(a, 10) if z not in (a, b) and not G.has_edge(b, z)]
        rng.shuffle(pool)
        for z in pool[:NEG_PER_POS]:
            if z in kge and z in txt:
                X_list.append(np.hstack([kge[z], txt[b]])); y_list.append(0.0)

    X = torch.from_numpy(np.stack(X_list).astype("float64"))  # match their nn.py dtype
    y = torch.tensor(y_list, dtype=torch.float64).reshape(-1, 1)

    n = len(y); n_tr = int(0.9 * n)
    idx = np.arange(n); rng.shuffle(idx)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    trainset = TensorDataset(X[tr_idx], y[tr_idx])
    testset  = TensorDataset(X[te_idx], y[te_idx])

    # Save with their expected filenames (so A.11 can be used as-is)
    os.makedirs("x_y/BC5CDR", exist_ok=True)
    import pickle
    pickle.dump([trainset.tensors[0]], open("x_y/BC5CDR/x_tensors_full_train_name_BC5CDR_node2vec_name.obj","wb"))
    pickle.dump(trainset.tensors[1].tolist(), open("x_y/BC5CDR/y_label_full_train_name_BC5CDR_node2vec_name.obj","wb"))
    pickle.dump([testset.tensors[0]], open("x_y/BC5CDR/x_tensors_full_test_name_BC5CDR_node2vec_name.obj","wb"))
    pickle.dump(testset.tensors[1].tolist(), open("x_y/BC5CDR/y_label_full_test_name_BC5CDR_node2vec_name.obj","wb"))

    os.makedirs(OUT_DIR, exist_ok=True)
    pickle.dump(trainset, open(os.path.join(OUT_DIR, "trainset_name_BC5CDR_node2vec_name.obj"), "wb"))
    pickle.dump(testset,  open(os.path.join(OUT_DIR, "testset_name_BC5CDR_node2vec_name.obj"), "wb"))
    print(f"Saved train/test tensors: train {len(tr_idx)}, test {len(te_idx)}")

if __name__ == "__main__":
    main()
