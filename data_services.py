import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

DATA_DIR = Path("git_web_ml")
ARTIFACT_DIR = Path("artifacts/frontend")

EDGES_PATH = DATA_DIR / "musae_git_edges.csv"
TARGET_PATH = DATA_DIR / "musae_git_target.csv"

NODE_COMMUNITY_PATH = ARTIFACT_DIR / "node_community.csv"
COMMUNITY_SUMMARY_PATH = ARTIFACT_DIR / "community_summary.csv"
COMMUNITY_REPRESENTATIVES_PATH = ARTIFACT_DIR / "community_representatives.json"
GLOBAL_STATS_PATH = ARTIFACT_DIR / "global_stats.json"


def artifacts_exist() -> bool:
    required = [
        NODE_COMMUNITY_PATH,
        COMMUNITY_SUMMARY_PATH,
        COMMUNITY_REPRESENTATIVES_PATH,
        GLOBAL_STATS_PATH,
    ]
    return all(p.exists() for p in required)


def build_artifacts() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(EDGES_PATH)
    target = pd.read_csv(TARGET_PATH)
    n_nodes = int(target["id"].max()) + 1

    graph = nx.from_pandas_edgelist(edges, "id_1", "id_2", create_using=nx.Graph)
    degree_dict = dict(graph.degree())

    candidate_seeds = [0, 1, 2, 42, 99]
    best_partition = None
    best_modularity = -1.0
    for seed in candidate_seeds:
        part = community_louvain.best_partition(graph, random_state=seed)
        mod = community_louvain.modularity(part, graph)
        if mod > best_modularity:
            best_modularity = mod
            best_partition = part

    partition = best_partition
    labels = np.array([partition[i] for i in range(n_nodes)], dtype=int)
    unique_comms = sorted(np.unique(labels))
    remap = {old: new for new, old in enumerate(unique_comms)}
    labels = np.array([remap[c] for c in labels], dtype=int)

    node_df = pd.DataFrame(
        {
            "id": np.arange(n_nodes, dtype=int),
            "community_id": labels,
            "degree": np.array([degree_dict.get(i, 0) for i in range(n_nodes)], dtype=int),
        }
    )
    node_df = node_df.merge(target[["id", "name"]], how="left", on="id")
    node_df["name"] = node_df["name"].fillna("")
    node_df.to_csv(NODE_COMMUNITY_PATH, index=False)

    comm_to_nodes = {}
    for node_id, comm_id in enumerate(labels):
        comm_to_nodes.setdefault(int(comm_id), []).append(node_id)

    def internal_density(nodes: list[int]) -> float:
        sub = graph.subgraph(nodes)
        n = sub.number_of_nodes()
        m = sub.number_of_edges()
        return 0.0 if n < 2 else (2.0 * m) / (n * (n - 1))

    rows = []
    representatives = {}
    for comm_id, nodes in comm_to_nodes.items():
        node_degrees = [degree_dict.get(n, 0) for n in nodes]
        rows.append(
            {
                "community_id": int(comm_id),
                "size": int(len(nodes)),
                "internal_density": float(internal_density(nodes)),
                "avg_degree": float(np.mean(node_degrees)),
                "max_degree": int(np.max(node_degrees)),
            }
        )

        reps = sorted(nodes, key=lambda n: degree_dict.get(n, 0), reverse=True)[:10]
        representatives[str(comm_id)] = (
            node_df[node_df["id"].isin(reps)][["id", "name", "degree"]]
            .sort_values("degree", ascending=False)
            .to_dict(orient="records")
        )

    pd.DataFrame(rows).sort_values("size", ascending=False).to_csv(COMMUNITY_SUMMARY_PATH, index=False)

    with open(COMMUNITY_REPRESENTATIVES_PATH, "w", encoding="utf-8") as f:
        json.dump(representatives, f, indent=2)

    global_stats = {
        "num_nodes": int(n_nodes),
        "num_edges": int(graph.number_of_edges()),
        "num_communities": int(len(comm_to_nodes)),
        "modularity": float(best_modularity),
        "density": float(nx.density(graph)),
    }
    with open(GLOBAL_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)


def load_data() -> dict:
    if not artifacts_exist():
        build_artifacts()

    node_df = pd.read_csv(NODE_COMMUNITY_PATH)
    community_summary = pd.read_csv(COMMUNITY_SUMMARY_PATH)
    with open(COMMUNITY_REPRESENTATIVES_PATH, "r", encoding="utf-8") as f:
        representatives = json.load(f)
    with open(GLOBAL_STATS_PATH, "r", encoding="utf-8") as f:
        global_stats = json.load(f)

    return {
        "node_df": node_df,
        "community_summary": community_summary,
        "community_representatives": representatives,
        "global_stats": global_stats,
    }


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


def toy_similar_users(user_row: pd.Series, node_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    comm_id = int(user_row["community_id"])
    uid = int(user_row["id"])
    udeg = int(user_row["degree"])
    candidates = node_df[(node_df["community_id"] == comm_id) & (node_df["id"] != uid)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["id", "name", "community_id", "degree", "similarity_score"])

    denom = np.maximum(candidates["degree"].to_numpy(), udeg) + 1.0
    candidates["similarity_score"] = 1.0 - (np.abs(candidates["degree"].to_numpy() - udeg) / denom)
    return candidates.sort_values("similarity_score", ascending=False).head(top_k)[
        ["id", "name", "community_id", "degree", "similarity_score"]
    ]


def community_subgraph(
    community_id: int, node_df: pd.DataFrame, edges_df: pd.DataFrame, max_nodes: int = 120
) -> nx.Graph:
    comm_nodes = node_df[node_df["community_id"] == community_id][["id", "degree"]].copy()
    if comm_nodes.empty:
        return nx.Graph()

    comm_nodes = comm_nodes.sort_values("degree", ascending=False).head(max_nodes)
    keep_ids = set(comm_nodes["id"].tolist())
    mask = edges_df["id_1"].isin(keep_ids) & edges_df["id_2"].isin(keep_ids)
    sub_edges = edges_df.loc[mask, ["id_1", "id_2"]]
    return nx.from_pandas_edgelist(sub_edges, "id_1", "id_2", create_using=nx.Graph)


def sampled_global_graph(
    node_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    nodes_per_community: int = 5,
    max_communities: int = 20,
    seed: int = 42,
) -> tuple[nx.Graph, dict[int, int]]:
    # Pick largest communities for readability, then sample a small node set from each.
    comm_sizes = (
        node_df.groupby("community_id", as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(max_communities)
    )
    selected_comms = comm_sizes["community_id"].astype(int).tolist()

    rng = np.random.default_rng(seed)
    sampled_nodes: list[int] = []
    node_to_comm: dict[int, int] = {}

    for comm_id in selected_comms:
        group = node_df[node_df["community_id"] == comm_id][["id", "degree"]].copy()
        if group.empty:
            continue
        # Bias toward informative nodes while keeping randomness.
        top_pool = group.sort_values("degree", ascending=False).head(max(25, nodes_per_community * 3))
        k = min(nodes_per_community, len(top_pool))
        chosen = rng.choice(top_pool["id"].to_numpy(), size=k, replace=False).tolist()
        sampled_nodes.extend(chosen)
        for nid in chosen:
            node_to_comm[int(nid)] = int(comm_id)

    keep = set(sampled_nodes)
    mask = edges_df["id_1"].isin(keep) & edges_df["id_2"].isin(keep)
    sub_edges = edges_df.loc[mask, ["id_1", "id_2"]]
    graph = nx.from_pandas_edgelist(sub_edges, "id_1", "id_2", create_using=nx.Graph)
    graph.add_nodes_from(sampled_nodes)
    return graph, node_to_comm
