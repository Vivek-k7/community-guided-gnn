import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from community import community_louvain
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DATA_DIR = Path("git_web_ml")
ARTIFACT_DIR = Path("artifacts/frontend")
EDGES_PATH = DATA_DIR / "musae_git_edges.csv"
TARGET_PATH = DATA_DIR / "musae_git_target.csv"
FEATURES_PATH = DATA_DIR / "musae_git_features.json"

ALGORITHMS = ["louvain", "label_propagation", "spectral", "kmeans", "gnn"]
ALGO_DISPLAY = {
    "louvain": "Louvain",
    "label_propagation": "Label Propagation",
    "spectral": "Spectral Clustering",
    "kmeans": "K-Means (Features)",
    "gnn": "GraphSAGE + K-Means",
}


def algo_dir(algo: str) -> Path:
    return ARTIFACT_DIR / algo


def algo_artifacts_exist(algo: str) -> bool:
    d = algo_dir(algo)
    return all(
        (d / f).exists()
        for f in [
            "node_community.csv",
            "community_summary.csv",
            "community_representatives.json",
            "global_stats.json",
        ]
    )


def _load_raw() -> tuple[nx.Graph, pd.DataFrame, int]:
    edges = pd.read_csv(EDGES_PATH)
    target = pd.read_csv(TARGET_PATH)
    n_nodes = int(target["id"].max()) + 1
    graph = nx.from_pandas_edgelist(edges, "id_1", "id_2", create_using=nx.Graph)
    return graph, target, n_nodes


def load_features_matrix(n_nodes: int) -> sp.csr_matrix:
    cache = ARTIFACT_DIR / "features_matrix.npz"
    if cache.exists():
        return sp.load_npz(str(cache))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_PATH) as f:
        raw = json.load(f)
    row_idx, col_idx = [], []
    for node_str, feat_list in raw.items():
        nid = int(node_str)
        for fi in feat_list:
            row_idx.append(nid)
            col_idx.append(int(fi))
    mat = sp.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(n_nodes, 4005),
    )
    sp.save_npz(str(cache), mat)
    return mat


def _remap(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(np.unique(labels))
    m = {old: new for new, old in enumerate(uniq)}
    return np.array([m[c] for c in labels], dtype=int)


def _save_artifacts(
    algo: str,
    labels: np.ndarray,
    graph: nx.Graph,
    target: pd.DataFrame,
    modularity: Optional[float] = None,
) -> None:
    d = algo_dir(algo)
    d.mkdir(parents=True, exist_ok=True)
    n_nodes = len(labels)
    degree_dict = dict(graph.degree())

    node_df = pd.DataFrame(
        {
            "id": np.arange(n_nodes, dtype=int),
            "community_id": labels,
            "degree": [degree_dict.get(i, 0) for i in range(n_nodes)],
        }
    ).merge(target[["id", "name"]], how="left", on="id")
    node_df["name"] = node_df["name"].fillna("")
    node_df.to_csv(d / "node_community.csv", index=False)

    comm_to_nodes: dict[int, list[int]] = {}
    for nid, cid in enumerate(labels):
        comm_to_nodes.setdefault(int(cid), []).append(nid)

    rows, reps = [], {}
    for cid, nodes in comm_to_nodes.items():
        degs = [degree_dict.get(n, 0) for n in nodes]
        sub = graph.subgraph(nodes)
        n_sub, m_sub = sub.number_of_nodes(), sub.number_of_edges()
        dens = (2.0 * m_sub) / (n_sub * (n_sub - 1)) if n_sub > 1 else 0.0
        rows.append(
            {
                "community_id": cid,
                "size": len(nodes),
                "internal_density": dens,
                "avg_degree": float(np.mean(degs)),
                "max_degree": int(np.max(degs)),
            }
        )
        top = sorted(nodes, key=lambda x: degree_dict.get(x, 0), reverse=True)[:10]
        reps[str(cid)] = (
            node_df[node_df["id"].isin(top)][["id", "name", "degree"]]
            .sort_values("degree", ascending=False)
            .to_dict(orient="records")
        )

    pd.DataFrame(rows).sort_values("size", ascending=False).to_csv(
        d / "community_summary.csv", index=False
    )
    with open(d / "community_representatives.json", "w") as f:
        json.dump(reps, f, indent=2)

    if modularity is None:
        part = {i: int(labels[i]) for i in range(n_nodes)}
        try:
            modularity = community_louvain.modularity(part, graph)
        except Exception:
            modularity = 0.0

    with open(d / "global_stats.json", "w") as f:
        json.dump(
            {
                "num_nodes": n_nodes,
                "num_edges": graph.number_of_edges(),
                "num_communities": len(comm_to_nodes),
                "modularity": float(modularity),
                "density": float(nx.density(graph)),
                "algorithm": algo,
            },
            f,
            indent=2,
        )


# ── Algorithm builders ──────────────────────────────────────────────────────

def _build_louvain(graph: nx.Graph, target: pd.DataFrame, n_nodes: int) -> None:
    best_part, best_mod = None, -1.0
    for seed in [0, 1, 2, 42, 99]:
        part = community_louvain.best_partition(graph, random_state=seed)
        mod = community_louvain.modularity(part, graph)
        if mod > best_mod:
            best_mod, best_part = mod, part
    labels = _remap(np.array([best_part.get(i, 0) for i in range(n_nodes)], dtype=int))
    _save_artifacts("louvain", labels, graph, target, best_mod)


def _build_label_propagation(graph: nx.Graph, target: pd.DataFrame, n_nodes: int) -> None:
    communities = list(nx.algorithms.community.label_propagation_communities(graph))
    part: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            part[node] = cid
    labels = _remap(np.array([part.get(i, 0) for i in range(n_nodes)], dtype=int))
    _save_artifacts("label_propagation", labels, graph, target)


def _build_spectral(
    graph: nx.Graph, target: pd.DataFrame, n_nodes: int, n_clusters: int
) -> None:
    # Normalized adjacency spectral embedding via randomized SVD → K-Means
    rows_e, cols_e = [], []
    for u, v in graph.edges():
        rows_e.extend([u, v])
        cols_e.extend([v, u])
    adj = sp.csr_matrix(
        (np.ones(len(rows_e)), (rows_e, cols_e)), shape=(n_nodes, n_nodes)
    )
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    svd = TruncatedSVD(n_components=n_clusters, random_state=42, n_iter=7)
    emb = normalize(svd.fit_transform(A_norm), norm="l2")

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = _remap(km.fit_predict(emb))
    _save_artifacts("spectral", labels, graph, target)


def _build_kmeans(
    graph: nx.Graph,
    target: pd.DataFrame,
    features_matrix: sp.csr_matrix,
    n_nodes: int,
    n_clusters: int,
) -> None:
    n_comp = min(64, features_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
    emb = svd.fit_transform(features_matrix)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = _remap(km.fit_predict(emb))
    _save_artifacts("kmeans", labels, graph, target)


def _build_gnn(
    graph: nx.Graph,
    target: pd.DataFrame,
    features_matrix: sp.csr_matrix,
    n_nodes: int,
    n_clusters: int,
) -> None:
    # GraphSAGE-style: reduce features → 2-hop mean aggregation → K-Means
    n_comp = min(64, features_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
    feat_reduced = svd.fit_transform(features_matrix)  # (n_nodes, 64) dense

    # Row-normalized adjacency with self-loops for mean aggregation
    rows_e = list(range(n_nodes))
    cols_e = list(range(n_nodes))
    for u, v in graph.edges():
        rows_e.extend([u, v])
        cols_e.extend([v, u])
    adj = sp.csr_matrix(
        (np.ones(len(rows_e)), (rows_e, cols_e)), shape=(n_nodes, n_nodes)
    )
    deg = np.array(adj.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / deg)
    A_norm = D_inv @ adj

    h = A_norm @ feat_reduced   # Layer 1: sparse @ dense = dense
    h = A_norm @ h              # Layer 2

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = _remap(km.fit_predict(h))
    _save_artifacts("gnn", labels, graph, target)


# ── Public API ────────────────────────────────────────────────────────────────

def build_algorithm(algo: str) -> None:
    """Build and cache artifacts for the given algorithm (builds Louvain first if needed)."""
    if not algo_artifacts_exist("louvain"):
        graph, target, n_nodes = _load_raw()
        _build_louvain(graph, target, n_nodes)

    if algo_artifacts_exist(algo):
        return

    graph, target, n_nodes = _load_raw()

    n_clusters = 37
    stats_path = algo_dir("louvain") / "global_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            n_clusters = json.load(f)["num_communities"]

    if algo == "louvain":
        pass  # already built above
    elif algo == "label_propagation":
        _build_label_propagation(graph, target, n_nodes)
    elif algo == "spectral":
        _build_spectral(graph, target, n_nodes, n_clusters)
    elif algo == "kmeans":
        features_matrix = load_features_matrix(n_nodes)
        _build_kmeans(graph, target, features_matrix, n_nodes, n_clusters)
    elif algo == "gnn":
        features_matrix = load_features_matrix(n_nodes)
        _build_gnn(graph, target, features_matrix, n_nodes, n_clusters)


def load_algorithm_data(algo: str) -> dict:
    if not algo_artifacts_exist(algo):
        build_algorithm(algo)
    d = algo_dir(algo)
    node_df = pd.read_csv(d / "node_community.csv")
    community_summary = pd.read_csv(d / "community_summary.csv")
    with open(d / "community_representatives.json") as f:
        reps = json.load(f)
    with open(d / "global_stats.json") as f:
        stats = json.load(f)
    return {
        "node_df": node_df,
        "community_summary": community_summary,
        "community_representatives": reps,
        "global_stats": stats,
    }


def load_comparison_stats() -> pd.DataFrame:
    rows = []
    for algo in ALGORITHMS:
        p = algo_dir(algo) / "global_stats.json"
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            rows.append(
                {
                    "Algorithm": ALGO_DISPLAY[algo],
                    "Communities": s["num_communities"],
                    "Modularity": round(s["modularity"], 4),
                    "Avg Community Size": round(s["num_nodes"] / s["num_communities"], 1),
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_edges() -> pd.DataFrame:
    return pd.read_csv(EDGES_PATH)


def lookup_user(query: str, node_df: pd.DataFrame) -> Optional[pd.Series]:
    query = query.strip()
    if not query:
        return None
    if query.isdigit():
        rows = node_df[node_df["id"] == int(query)]
        return None if rows.empty else rows.iloc[0]
    rows = node_df[node_df["name"].str.lower() == query.lower()]
    return None if rows.empty else rows.iloc[0]


def feature_similar_users(
    user_row: pd.Series,
    node_df: pd.DataFrame,
    features_matrix: sp.csr_matrix,
    top_k: int,
) -> pd.DataFrame:
    comm_id = int(user_row["community_id"])
    uid = int(user_row["id"])
    candidates = node_df[
        (node_df["community_id"] == comm_id) & (node_df["id"] != uid)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["id", "name", "degree", "similarity_score"])
    cand_ids = candidates["id"].to_numpy()
    sims = cosine_similarity(features_matrix[uid], features_matrix[cand_ids]).flatten()
    candidates["similarity_score"] = sims
    return (
        candidates.sort_values("similarity_score", ascending=False)
        .head(top_k)[["id", "name", "degree", "similarity_score"]]
        .reset_index(drop=True)
    )


def toy_similar_users(
    user_row: pd.Series, node_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    comm_id = int(user_row["community_id"])
    uid = int(user_row["id"])
    udeg = int(user_row["degree"])
    candidates = node_df[
        (node_df["community_id"] == comm_id) & (node_df["id"] != uid)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["id", "name", "degree", "similarity_score"])
    denom = np.maximum(candidates["degree"].to_numpy(), udeg) + 1.0
    candidates["similarity_score"] = 1.0 - (
        np.abs(candidates["degree"].to_numpy() - udeg) / denom
    )
    return (
        candidates.sort_values("similarity_score", ascending=False)
        .head(top_k)[["id", "name", "degree", "similarity_score"]]
        .reset_index(drop=True)
    )


def community_subgraph(
    community_id: int,
    node_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_nodes: int = 120,
    ensure_node_id: int | None = None,
) -> nx.Graph:
    comm_nodes = node_df[node_df["community_id"] == community_id][["id", "degree"]].copy()
    if comm_nodes.empty:
        return nx.Graph()
    comm_nodes = comm_nodes.sort_values("degree", ascending=False).head(max_nodes)
    keep = set(comm_nodes["id"].tolist())
    # Always include the searched user even if they rank below max_nodes by degree
    if ensure_node_id is not None:
        keep.add(ensure_node_id)
    mask = edges_df["id_1"].isin(keep) & edges_df["id_2"].isin(keep)
    g = nx.from_pandas_edgelist(
        edges_df.loc[mask, ["id_1", "id_2"]], "id_1", "id_2", create_using=nx.Graph
    )
    g.add_nodes_from(keep)  # isolated nodes still appear
    return g


# Palette of 40 distinct hex colours for community colouring
_PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
    "#bfef45","#fabed4","#469990","#dcbeff","#9a6324","#fffac8","#800000","#aaffc3",
    "#808000","#ffd8b1","#000075","#a9a9a9","#e6beff","#008080","#ff4500","#2e8b57",
    "#8b0000","#00ced1","#ff69b4","#cd853f","#4682b4","#da70d6","#32cd32","#ff6347",
    "#7b68ee","#20b2aa","#ff1493","#00fa9a","#ffa500","#c71585","#00bfff","#696969",
]


_TOOLTIP_CSS = """
<style>
.vis-tooltip {
  background: #1e1e2e !important;
  border: 1px solid #555 !important;
  border-radius: 8px !important;
  padding: 10px 14px !important;
  color: #e0e0e0 !important;
  font-family: 'Segoe UI', Arial, sans-serif !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  white-space: pre-line !important;
  max-width: 240px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
  pointer-events: none !important;
}
</style>
"""

_CLICK_JS = """
<script>
(function waitReady() {
  if (typeof network === 'undefined' || typeof nodes === 'undefined') {
    setTimeout(waitReady, 150);
    return;
  }
  network.on('click', function(params) {
    if (params.nodes.length === 0) return;
    var node = nodes.get(params.nodes[0]);
    if (node && node.github) {
      window.open('https://github.com/' + node.github, '_blank');
    }
  });
  network.on('hoverNode', function() {
    document.getElementById('mynetwork').style.cursor = 'pointer';
  });
  network.on('blurNode', function() {
    document.getElementById('mynetwork').style.cursor = 'default';
  });
})();
</script>
"""


def build_pyvis_graph(
    g: nx.Graph,
    node_df: pd.DataFrame,
    node_to_comm: dict[int, int] | None = None,
    height: str = "520px",
    highlight_id: int | None = None,
) -> str:
    """Return pyvis HTML string for an interactive graph.

    Hover shows a styled tooltip (username, ID, degree, community, GitHub URL).
    Click a node to open their GitHub profile in a new tab.
    highlight_id draws that node in gold.
    """
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="white")
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 120}},
      "interaction": {"hover": true, "tooltipDelay": 80},
      "edges": {"color": {"color": "#444"}, "smooth": false}
    }
    """)

    id_to_row = node_df.set_index("id")

    for node in g.nodes():
        nid = int(node)
        comm = node_to_comm.get(nid, -1) if node_to_comm else -1

        if nid in id_to_row.index:
            row = id_to_row.loc[nid]
            name = str(row["name"]).strip() or ""
            degree = int(row["degree"])
            community_id = int(row["community_id"]) if "community_id" in row.index else comm
        else:
            name = ""
            degree = g.degree(node)
            community_id = comm

        display_name = name or f"ID {nid}"
        color = "#FFD700" if nid == highlight_id else _PALETTE[community_id % len(_PALETTE)]
        size = max(8, min(30, 6 + degree // 4))
        label = name if name else str(nid)

        has_github = bool(name)
        tooltip_lines = [
            display_name,
            f"ID: {nid}   Degree: {degree}",
            f"Community: {community_id}",
        ]
        if has_github:
            tooltip_lines.append(f"github.com/{name}")
            tooltip_lines.append("Click to open GitHub \u2197")

        tooltip = "\n".join(tooltip_lines)

        net.add_node(
            nid,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            github=name if has_github else "",
        )

    for u, v in g.edges():
        net.add_edge(int(u), int(v))

    html = net.generate_html()
    # Inject tooltip styling and GitHub click handler
    html = html.replace("</head>", _TOOLTIP_CSS + "</head>")
    html = html.replace("</body>", _CLICK_JS + "</body>")
    return html


def sampled_global_graph(
    node_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    nodes_per_community: int = 5,
    max_communities: int = 20,
    seed: int = 42,
) -> tuple[nx.Graph, dict[int, int]]:
    comm_sizes = (
        node_df.groupby("community_id", as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(max_communities)
    )
    selected = comm_sizes["community_id"].astype(int).tolist()
    rng = np.random.default_rng(seed)
    sampled: list[int] = []
    node_to_comm: dict[int, int] = {}
    for cid in selected:
        group = node_df[node_df["community_id"] == cid][["id", "degree"]].copy()
        if group.empty:
            continue
        pool = group.sort_values("degree", ascending=False).head(
            max(25, nodes_per_community * 3)
        )
        k = min(nodes_per_community, len(pool))
        chosen = rng.choice(pool["id"].to_numpy(), size=k, replace=False).tolist()
        sampled.extend(chosen)
        for nid in chosen:
            node_to_comm[int(nid)] = int(cid)
    keep = set(sampled)
    mask = edges_df["id_1"].isin(keep) & edges_df["id_2"].isin(keep)
    g = nx.from_pandas_edgelist(
        edges_df.loc[mask, ["id_1", "id_2"]], "id_1", "id_2", create_using=nx.Graph
    )
    g.add_nodes_from(sampled)
    return g, node_to_comm
