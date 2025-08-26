# apply_aliases_from_cluster_json.py
import os, json, csv
import networkx as nx

IN_DIR = "out"
ALIAS_JSON = os.path.join(IN_DIR, "alias_merged.json")
NODES_CSV  = os.path.join(IN_DIR, "nodes.csv")
EDGES_CSV  = os.path.join(IN_DIR, "edges.csv")
OUT_PREFIX = os.path.join(IN_DIR, "merged")

# 1) load clusters
clusters = json.load(open(ALIAS_JSON, "r", encoding="utf-8"))

# 2) load original graph from CSVs
G = nx.Graph()
# nodes.csv: id,kind,degree
with open(NODES_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        node_id = row["id"]
        kind = row.get("kind", "")
        deg  = int(row.get("degree", "0") or 0)
        G.add_node(node_id, kind=kind, degree=deg)

# edges.csv: src,dst,weight
with open(EDGES_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        w = row.get("weight", "1")
        try:
            w = float(w)
        except:
            w = 1.0
        G.add_edge(row["src"], row["dst"], w=w)

# helper: degree lookup (fallback if nodes.csv lacked degree)
deg_lookup = {n: d for n, d in G.degree()}

def choose_canonical(members):
    """
    Pick a representative per cluster:
    1) prefer names with a space (likely have last name),
    2) then higher degree in the graph,
    3) then longer surface form.
    """
    return sorted(
        members,
        key=lambda n: (
            - (1 if " " in n else 0),
            - deg_lookup.get(n, 0),
            - len(n),
        ),
    )[0]

# 3) build alias->canonical map
alias2canon = {}
for _, members in clusters.items():
    canon = choose_canonical(members)
    for m in members:
        alias2canon[m] = canon

# 4) collapse graph
H = nx.Graph()
# nodes
for n, data in G.nodes(data=True):
    c = alias2canon.get(n, n)
    if not H.has_node(c):
        H.add_node(c, **data)
    else:
        # preserve PERSON kind if any alias had it
        if data.get("kind") == "PERSON":
            H.nodes[c]["kind"] = "PERSON"
# edges (sum weights)
for u, v, data in G.edges(data=True):
    cu, cv = alias2canon.get(u, u), alias2canon.get(v, v)
    if cu == cv:
        continue  # self-loop after merge
    w = data.get("w") if "w" in data else data.get("weight", 1.0)
    if H.has_edge(cu, cv):
        H[cu][cv]["w"] = H[cu][cv].get("w", 0.0) + (w or 1.0)
    else:
        H.add_edge(cu, cv, w=(w or 1.0))

# 5) write outputs for audit & viz
os.makedirs(IN_DIR, exist_ok=True)
# mapping CSV
with open(f"{OUT_PREFIX}_alias_map.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["alias", "canonical"])
    for a, c in sorted(alias2canon.items()):
        if a != c:
            w.writerow([a, c])

# merged nodes/edges CSVs
with open(f"{OUT_PREFIX}_nodes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "kind", "degree"])
    for n, d in H.nodes(data=True):
        w.writerow([n, d.get("kind",""), H.degree(n)])
with open(f"{OUT_PREFIX}_edges.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["src","dst","weight"])
    for u, v, d in H.edges(data=True):
        w.writerow([u, v, d.get("w", 1.0)])

# GEXF for Gephi
nx.write_gexf(H, f"{OUT_PREFIX}.gexf")

print(f"Original: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
print(f"Merged:   {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")
print(f"Wrote: {OUT_PREFIX}_alias_map.csv, {OUT_PREFIX}_nodes.csv, {OUT_PREFIX}_edges.csv, {OUT_PREFIX}.gexf")
