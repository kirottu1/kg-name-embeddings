# nick_infer.py  — strict & precise
import os, json, pickle, numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from siamese_nn import SiameseNetwork

PREP_DIR = "prep_out"
MODEL    = "models/model_name_siamese_nick_node2vec_name.pth"
TAU_PATH = "models/model_name_siamese_nick_node2vec_name.threshold.txt"

# Conservative knobs
K_CANDIDATES = 15            # TEXT kNN only
DEFAULT_TAU  = 0.85          # fallback if no saved threshold

def load_all():
    kge = pickle.load(open(os.path.join(PREP_DIR,"Dict_embed_node2vec.obj"),"rb"))
    txt = pickle.load(open(os.path.join(PREP_DIR,"name_text_embeddings.pkl"),"rb"))
    return kge, txt

def knn_text_index(text_vecs: dict, k=25):
    names = list(text_vecs.keys())
    X = np.stack([text_vecs[n] for n in names])
    nn = NearestNeighbors(n_neighbors=min(k, max(2, len(names)-1)), metric="cosine").fit(X)
    return names, X, nn

def score_pair(net, kge_vec, txt_vec):
    # model expects float64 (matches nn.py)
    import torch
    x1 = torch.from_numpy(kge_vec.astype("float64"))[None, :]
    x2 = torch.from_numpy(txt_vec.astype("float64"))[None, :]
    with torch.no_grad():
        p = net(x1, x2).cpu().numpy()[0][0]
    return float(p)

def main():
    os.makedirs("out", exist_ok=True)

    # threshold
    TAU = DEFAULT_TAU
    if os.path.exists(TAU_PATH):
        try: TAU = float(open(TAU_PATH).read().strip())
        except Exception: pass
    print(f"Using threshold τ={TAU:.2f}")

    # embeddings
    kge, txt = load_all()
    # consider only names that have BOTH vectors
    names_all = sorted(set(txt.keys()) & set(kge.keys()))
    if not names_all:
        print("No names with both KGE and TEXT embeddings. Check prep stage.")
        return

    # text kNN
    names, Xtxt, nn = knn_text_index({n: txt[n] for n in names_all}, k=K_CANDIDATES)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = SiameseNetwork()
    net.load_state_dict(torch.load(MODEL, map_location=device))
    net.eval()

    # 1) pick top-1 candidate above τ for each name (TEXT-only candidates)
    best_of = {}  # name -> (best_cand, score)
    for i, s in enumerate(names):
        # get nearest text neighbors
        _, idxs = nn.kneighbors(Xtxt[i:i+1], n_neighbors=min(K_CANDIDATES, len(names)))
        cands = [names[j] for j in idxs[0] if names[j] != s]
        best_c, best_p = None, -1.0
        for c in cands:
            p = score_pair(net, kge[c], txt[s])
            if p > best_p:
                best_p, best_c = p, c
        if best_p >= TAU and best_c is not None:
            best_of[s] = (best_c, best_p)

    # 2) mutual top-1 merges only
    parent = {n:n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    merges = 0
    for s,(c,ps) in best_of.items():
        bc = best_of.get(c)
        if bc is not None and bc[0] == s:
            union(s, c); merges += 1

    # 3) clusters (≥2 members)
    clusters = {}
    for n in names:
        r = find(n)
        clusters.setdefault(r, []).append(n)
    merged = {r: sorted(v) for r,v in clusters.items() if len(v) > 1}

    with open("out/alias_merged.json","w",encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Names considered: {len(names)}")
    print(f"Merges (mutual top-1 @ τ={TAU:.2f}): {merges}")
    print(f"Wrote out/alias_merged.json with {len(merged)} merged clusters")

if __name__ == "__main__":
    main()
