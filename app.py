import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from data_services import community_subgraph
from data_services import load_data
from data_services import load_edges
from data_services import lookup_user
from data_services import sampled_global_graph
from data_services import toy_similar_users

st.set_page_config(page_title="GitHub Developer Insights", layout="wide")


@st.cache_data(show_spinner=False)
def cached_data():
    return load_data()


@st.cache_data(show_spinner=False)
def cached_edges():
    return load_edges()


data = cached_data()
node_df = data["node_df"]
community_summary = data["community_summary"]
community_representatives = data["community_representatives"]
global_stats = data["global_stats"]
edges_df = cached_edges()

st.title("Community-Aware Developer Insights")
st.markdown(
    "Louvain-based community analytics from the real GitHub graph. "
    "Developer similarity remains a toy baseline until GNN embeddings are trained."
)

with st.sidebar:
    st.header("Developer Search")
    user_query = st.text_input("Username or numeric User ID", placeholder="e.g., shawflying or 1")
    k_value = st.slider("Top-K Similar Developers", min_value=1, max_value=20, value=5)
    submit_button = st.button("Analyze Developer")

st.subheader("Global Graph Overview")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Nodes", f"{global_stats['num_nodes']:,}")
g2.metric("Edges", f"{global_stats['num_edges']:,}")
g3.metric("Communities", f"{global_stats['num_communities']:,}")
g4.metric("Modularity", f"{global_stats['modularity']:.3f}")
g5.metric("Density", f"{global_stats['density']:.6f}")

st.divider()

v1, v2 = st.columns(2)
with v1:
    st.markdown("**Top Communities by Size**")
    top_size = community_summary.head(15).copy()
    top_size = top_size.set_index(top_size["community_id"].astype(str))[["size"]]
    st.bar_chart(top_size)

with v2:
    st.markdown("**Community Size vs Internal Density**")
    st.scatter_chart(
        community_summary.rename(columns={"size": "x_size", "internal_density": "y_density"}),
        x="x_size",
        y="y_density",
    )

st.divider()

st.subheader("Sampled Global Community Graph")
sg1, sg2 = st.columns(2)
with sg1:
    sample_comms = st.slider("Communities to include", min_value=5, max_value=40, value=20, step=1)
with sg2:
    sample_nodes = st.slider("Nodes per community", min_value=2, max_value=12, value=4, step=1)

sample_g, node_to_comm = sampled_global_graph(
    node_df=node_df,
    edges_df=edges_df,
    nodes_per_community=sample_nodes,
    max_communities=sample_comms,
    seed=42,
)

if sample_g.number_of_nodes() == 0:
    st.info("No sampled nodes available for global graph view.")
else:
    # One community -> one color.
    comm_values = sorted({node_to_comm.get(int(n), -1) for n in sample_g.nodes()})
    cmap = plt.cm.get_cmap("hsv", max(len(comm_values), 1))
    comm_to_color = {c: cmap(i) for i, c in enumerate(comm_values)}
    node_colors = [comm_to_color[node_to_comm.get(int(n), -1)] for n in sample_g.nodes()]

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(sample_g, seed=42)
    nx.draw_networkx(
        sample_g,
        pos=pos,
        with_labels=False,
        node_size=55,
        width=0.5,
        edge_color="#BFC9CA",
        node_color=node_colors,
        ax=ax,
    )
    ax.set_title(
        f"Sampled Community Graph ({sample_g.number_of_nodes()} nodes, {sample_g.number_of_edges()} edges)"
    )
    ax.axis("off")
    st.pyplot(fig)

st.divider()

st.subheader("Community Explorer")
cids = community_summary["community_id"].astype(int).tolist()
selected_cid = st.selectbox("Select Community ID", cids, index=0)

community_stats = community_summary[community_summary["community_id"] == selected_cid].iloc[0]

tab_graph, tab_details = st.tabs(["Subgraph View", "Community Details"])

with tab_graph:
    max_nodes = st.slider("Max nodes in subgraph", min_value=30, max_value=200, value=100, step=10)
    sub_g = community_subgraph(selected_cid, node_df, edges_df, max_nodes=max_nodes)
    if sub_g.number_of_nodes() == 0:
        st.info("No nodes available for this community.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(sub_g, seed=42)
        nx.draw_networkx(
            sub_g,
            pos=pos,
            with_labels=False,
            node_size=40,
            width=0.5,
            edge_color="#AAB2BD",
            node_color="#2E86C1",
            ax=ax,
        )
        ax.set_title(f"Community {selected_cid} Subgraph ({sub_g.number_of_nodes()} nodes)")
        ax.axis("off")
        st.pyplot(fig)

with tab_details:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Community ID", int(selected_cid))
    d2.metric("Size", int(community_stats["size"]))
    d3.metric("Internal Density", f"{community_stats['internal_density']:.4f}")
    d4.metric("Max Degree", int(community_stats["max_degree"]))
    st.metric("Average Degree", f"{community_stats['avg_degree']:.2f}")

    st.markdown("**Representative Developers (by degree)**")
    reps = community_representatives.get(str(int(selected_cid)), [])
    if reps:
        st.dataframe(reps, use_container_width=True, hide_index=True)
    else:
        st.info("No representative users available.")

st.divider()

st.subheader("Developer Lookup")
if submit_button:
    user_row = lookup_user(user_query, node_df)
    if user_row is None:
        st.warning("User not found. Try a numeric ID (0-37699) or exact username.")
    else:
        c_info = community_summary[community_summary["community_id"] == int(user_row["community_id"])].iloc[0]
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**User Profile**")
            username = user_row["name"] if str(user_row["name"]).strip() else "(unknown)"
            st.metric("Username", username)
            st.metric("User ID", int(user_row["id"]))
            st.metric("Degree", int(user_row["degree"]))

        with col2:
            st.markdown("**Community Details**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Community ID", int(user_row["community_id"]))
            m2.metric("Community Size", int(c_info["size"]))
            m3.metric("Density", f"{c_info['internal_density']:.4f}")

        st.markdown(f"**Top {k_value} Similar Developers (Toy Baseline)**")
        sim_df = toy_similar_users(user_row, node_df, k_value)
        st.dataframe(
            sim_df,
            column_config={
                "similarity_score": st.column_config.ProgressColumn(
                    "Similarity", min_value=0, max_value=1, format="%.2f"
                )
            },
            use_container_width=True,
            hide_index=True,
        )
