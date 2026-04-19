import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_services import (
    ALGORITHMS,
    ALGO_DISPLAY,
    algo_artifacts_exist,
    build_algorithm,
    build_pyvis_graph,
    community_subgraph,
    feature_similar_users,
    load_algorithm_data,
    load_comparison_stats,
    load_edges,
    load_features_matrix,
    lookup_user,
    sampled_global_graph,
    toy_similar_users,
)

st.set_page_config(
    page_title="GitHub Dev Communities",
    layout="wide",
    page_icon="🔗",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080D14 !important;
    border-right: 1px solid #21262D;
    padding-top: 8px;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    color: #C9D1D9 !important;
    padding: 4px 0;
}

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #161B22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 20px;
    color: #8B949E;
    font-weight: 500;
    font-size: 14px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Divider ──────────────────────────────────────────── */
hr { border-color: #21262D !important; margin: 24px 0 !important; }

/* ── Dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #30363D;
}

/* ── Selectbox ────────────────────────────────────────── */
[data-testid="stSelectbox"] > div {
    border-radius: 8px !important;
}

/* ── Alerts ───────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Primary button ───────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1F6FEB 0%, #388BFD 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88;
}

/* ── Caption text ─────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    color: #8B949E !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_algo_data(algo: str) -> dict:
    return load_algorithm_data(algo)


@st.cache_data(show_spinner=False)
def get_edges() -> pd.DataFrame:
    return load_edges()


@st.cache_resource(show_spinner=False)
def get_features() -> object:
    return load_features_matrix(37700)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<p style='color:#58A6FF; font-weight:700; font-size:18px; margin:0 0 16px;'>"
        "🔗 DevGraph"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='color:#8B949E; font-size:11px; font-weight:600; "
        "text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px;'>"
        "Search Developer</p>",
        unsafe_allow_html=True,
    )
    user_query = st.text_input(
        "search",
        placeholder="Username or User ID…",
        label_visibility="collapsed",
    )
    analyze_btn = st.button("Find Developer", use_container_width=True, type="primary")
    k_value = st.slider("Recommendations to show", min_value=1, max_value=20, value=8)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8B949E; font-size:11px; font-weight:600; "
        "text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px;'>"
        "Algorithm</p>",
        unsafe_allow_html=True,
    )
    algo = st.radio(
        "algo",
        options=ALGORITHMS,
        format_func=lambda x: ALGO_DISPLAY[x],
        label_visibility="collapsed",
    )


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner(f"Loading {ALGO_DISPLAY[algo]}…"):
    data = get_algo_data(algo)

node_df = data["node_df"]
community_summary = data["community_summary"]
community_representatives = data["community_representatives"]
global_stats = data["global_stats"]
edges_df = get_edges()

# ── Hero header ───────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1C2333 0%, #0D1117 100%);
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 28px 32px 24px;
    margin-bottom: 20px;
">
    <h1 style="margin:0 0 6px; color:#E6EDF3; font-size:26px; font-weight:700; line-height:1.2;">
        GitHub Developer Communities
    </h1>
    <p style="margin:0; color:#8B949E; font-size:14px; line-height:1.6;">
        37,700 developers &nbsp;·&nbsp; 289,003 connections &nbsp;·&nbsp;
        Active algorithm:&nbsp;
        <span style="
            background:#1F6FEB22;
            color:#58A6FF;
            border:1px solid #1F6FEB55;
            border-radius:20px;
            padding:2px 12px;
            font-weight:600;
            font-size:13px;
        ">{ALGO_DISPLAY[algo]}</span>
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Developers", f"{global_stats['num_nodes']:,}")
c2.metric("Connections", f"{global_stats['num_edges']:,}")
c3.metric("Communities", f"{global_stats['num_communities']:,}")
c4.metric("Modularity", f"{global_stats['modularity']:.4f}")
c5.metric("Graph Density", f"{global_stats['density']:.6f}")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Developer Lookup ──────────────────────────────────────────────────────────

if analyze_btn and user_query.strip():
    features_matrix = get_features()
    user_row = lookup_user(user_query, node_df)

    if user_row is None:
        st.warning("User not found. Try a numeric ID (0–37699) or an exact username.")
        st.session_state.pop("last_user_id", None)
        st.session_state.pop("last_community_id", None)
    else:
        st.session_state["last_user_id"] = int(user_row["id"])
        st.session_state["last_community_id"] = int(user_row["community_id"])
        c_info = community_summary[
            community_summary["community_id"] == int(user_row["community_id"])
        ].iloc[0]
        username = str(user_row["name"]).strip() or "(unknown)"
        uid = int(user_row["id"])
        degree = int(user_row["degree"])
        comm_id = int(user_row["community_id"])

        # ── Profile card ──
        st.markdown(f"""
<div style="
    background:#161B22;
    border:1px solid #30363D;
    border-radius:14px;
    padding:24px 28px;
    margin-bottom:16px;
">
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
        <div style="
            width:52px; height:52px; border-radius:50%;
            background:linear-gradient(135deg,#1F6FEB,#388BFD);
            display:flex; align-items:center; justify-content:center;
            font-size:22px; font-weight:700; color:white; flex-shrink:0;
        ">{username[0].upper()}</div>
        <div>
            <div style="font-size:20px; font-weight:700; color:#E6EDF3;">{username}</div>
            <a href="https://github.com/{username}" target="_blank"
               style="color:#58A6FF; font-size:13px; text-decoration:none;">
               github.com/{username} ↗
            </a>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="color:#8B949E; font-size:11px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em;">Community</div>
            <div style="
                font-size:18px; font-weight:700; color:#58A6FF;
                background:#1F6FEB15; border:1px solid #1F6FEB40;
                border-radius:8px; padding:2px 14px; display:inline-block; margin-top:2px;
            ">#{comm_id}</div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:12px;">
        <div style="background:#0D1117; border-radius:8px; padding:12px 14px; border:1px solid #21262D;">
            <div style="color:#8B949E; font-size:10px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;">User ID</div>
            <div style="color:#E6EDF3; font-size:16px; font-weight:700;">{uid}</div>
        </div>
        <div style="background:#0D1117; border-radius:8px; padding:12px 14px; border:1px solid #21262D;">
            <div style="color:#8B949E; font-size:10px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;">Connections</div>
            <div style="color:#E6EDF3; font-size:16px; font-weight:700;">{degree}</div>
        </div>
        <div style="background:#0D1117; border-radius:8px; padding:12px 14px; border:1px solid #21262D;">
            <div style="color:#8B949E; font-size:10px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;">Comm. Size</div>
            <div style="color:#E6EDF3; font-size:16px; font-weight:700;">{int(c_info["size"]):,}</div>
        </div>
        <div style="background:#0D1117; border-radius:8px; padding:12px 14px; border:1px solid #21262D;">
            <div style="color:#8B949E; font-size:10px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;">Density</div>
            <div style="color:#E6EDF3; font-size:16px; font-weight:700;">{c_info["internal_density"]:.4f}</div>
        </div>
        <div style="background:#0D1117; border-radius:8px; padding:12px 14px; border:1px solid #21262D;">
            <div style="color:#8B949E; font-size:10px; font-weight:600;
                        text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;">Avg Degree</div>
            <div style="color:#E6EDF3; font-size:16px; font-weight:700;">{c_info["avg_degree"]:.1f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Community top members + recommendations ──
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown(
                "<p style='color:#8B949E; font-size:12px; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px;'>"
                "Top Members in Community</p>",
                unsafe_allow_html=True,
            )
            reps = community_representatives.get(str(comm_id), [])
            if reps:
                st.dataframe(pd.DataFrame(reps), use_container_width=True, hide_index=True)

        with right_col:
            st.markdown(
                f"<p style='color:#8B949E; font-size:12px; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px;'>"
                f"Recommended Collaborators (top {k_value})</p>",
                unsafe_allow_html=True,
            )
            rec_feat, rec_deg = st.tabs(["Feature Similarity", "Degree Baseline"])
            with rec_feat:
                sim_df = feature_similar_users(user_row, node_df, get_features(), k_value)
                if sim_df.empty:
                    st.info("No candidates found.")
                else:
                    st.dataframe(
                        sim_df,
                        column_config={
                            "similarity_score": st.column_config.ProgressColumn(
                                "Similarity", min_value=0, max_value=1, format="%.3f"
                            )
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
            with rec_deg:
                toy_df = toy_similar_users(user_row, node_df, k_value)
                if toy_df.empty:
                    st.info("No candidates found.")
                else:
                    st.dataframe(
                        toy_df,
                        column_config={
                            "similarity_score": st.column_config.ProgressColumn(
                                "Similarity", min_value=0, max_value=1, format="%.3f"
                            )
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_overview, tab_communities, tab_compare = st.tabs(
    ["📊  Overview", "🔍  Community Explorer", "⚖️  Compare Algorithms"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────────────────

with tab_overview:
    v1, v2 = st.columns(2)

    with v1:
        st.markdown(
            "<p style='color:#8B949E; font-size:12px; font-weight:600; "
            "text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;'>"
            "Top 15 Communities by Size</p>",
            unsafe_allow_html=True,
        )
        top_size = (
            community_summary.head(15)
            .copy()
            .set_index(community_summary.head(15)["community_id"].astype(str))[["size"]]
        )
        st.bar_chart(top_size, color="#1F6FEB")

    with v2:
        st.markdown(
            "<p style='color:#8B949E; font-size:12px; font-weight:600; "
            "text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;'>"
            "Community Size vs Internal Density</p>",
            unsafe_allow_html=True,
        )
        st.scatter_chart(
            community_summary.rename(
                columns={"size": "Size", "internal_density": "Internal Density"}
            ),
            x="Size",
            y="Internal Density",
            color="#388BFD",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8B949E; font-size:12px; font-weight:600; "
        "text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px;'>"
        "Global Community Map</p>",
        unsafe_allow_html=True,
    )

    sg1, sg2 = st.columns(2)
    with sg1:
        n_comms = st.slider(
            "Communities to include",
            min_value=5,
            max_value=min(40, len(community_summary)),
            value=20,
        )
    with sg2:
        n_per_comm = st.slider("Nodes per community", min_value=2, max_value=12, value=4)

    sample_g, node_to_comm = sampled_global_graph(node_df, edges_df, n_per_comm, n_comms)

    if sample_g.number_of_nodes() > 0:
        highlight_global = st.session_state.get("last_user_id")
        st.caption(
            f"{sample_g.number_of_nodes()} nodes · {sample_g.number_of_edges()} edges "
            "· Hover to inspect · Click to open GitHub"
            + (" · Gold = searched developer" if highlight_global else "")
        )
        html = build_pyvis_graph(
            sample_g, node_df, node_to_comm, height="560px", highlight_id=highlight_global
        )
        components.html(html, height=580, scrolling=False)
    else:
        st.info("No nodes available for graph view.")

# ── Tab 2: Community Explorer ─────────────────────────────────────────────────

with tab_communities:
    cids = community_summary["community_id"].astype(int).tolist()

    def _comm_label(x: int) -> str:
        row = community_summary[community_summary["community_id"] == x]
        size = int(row["size"].values[0]) if not row.empty else 0
        return f"Community {x}  ({size:,} members)"

    _default_cid = st.session_state.get("last_community_id")
    _default_idx = cids.index(_default_cid) if _default_cid in cids else 0
    selected_cid = st.selectbox(
        "Select Community", cids, index=_default_idx, format_func=_comm_label
    )
    cs = community_summary[community_summary["community_id"] == selected_cid].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Members", f"{int(cs['size']):,}")
    m2.metric("Internal Density", f"{cs['internal_density']:.4f}")
    m3.metric("Avg Degree", f"{cs['avg_degree']:.1f}")
    m4.metric("Max Degree", int(cs["max_degree"]))

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    sub_graph_tab, sub_members_tab = st.tabs(["  Network View  ", "  Top Members  "])

    with sub_graph_tab:
        max_nodes = st.slider("Max nodes in view", 30, 200, 100, 10)
        highlight = st.session_state.get("last_user_id")
        pin_id = highlight if st.session_state.get("last_community_id") == selected_cid else None
        sub_g = community_subgraph(selected_cid, node_df, edges_df, max_nodes, ensure_node_id=pin_id)
        if sub_g.number_of_nodes() > 0:
            st.caption(
                f"{sub_g.number_of_nodes()} nodes · {sub_g.number_of_edges()} edges "
                "· Hover to inspect · Click to open GitHub"
                + (" · Gold = searched developer" if pin_id else "")
            )
            html2 = build_pyvis_graph(
                sub_g, node_df, node_to_comm=None, height="520px", highlight_id=pin_id
            )
            components.html(html2, height=540, scrolling=False)
        else:
            st.info("No nodes found for this community.")

    with sub_members_tab:
        reps = community_representatives.get(str(int(selected_cid)), [])
        if reps:
            st.dataframe(pd.DataFrame(reps), use_container_width=True, hide_index=True)
        else:
            st.info("No representative data available.")

# ── Tab 3: Algorithm Comparison ───────────────────────────────────────────────

with tab_compare:
    not_built = [a for a in ALGORITHMS if not algo_artifacts_exist(a)]

    if not_built:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(
                f"Not yet computed: **{', '.join(ALGO_DISPLAY[a] for a in not_built)}**. "
                "Results are cached to disk after the first run."
            )
        with col_btn:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("▶ Build Missing", type="primary", use_container_width=True):
                for a in not_built:
                    with st.spinner(f"Running {ALGO_DISPLAY[a]}…"):
                        build_algorithm(a)
                get_algo_data.clear()
                st.success("All algorithms computed.")
                st.rerun()

    comp_df = load_comparison_stats()

    if not comp_df.empty:
        st.markdown(
            "<p style='color:#8B949E; font-size:12px; font-weight:600; "
            "text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px;'>"
            "Metrics Summary</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            comp_df,
            column_config={
                "Modularity": st.column_config.ProgressColumn(
                    "Modularity", min_value=0, max_value=1, format="%.4f"
                ),
                "Communities": st.column_config.NumberColumn("# Communities"),
                "Avg Community Size": st.column_config.NumberColumn("Avg Size"),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown(
                "<p style='color:#8B949E; font-size:12px; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;'>"
                "Modularity — higher is better</p>",
                unsafe_allow_html=True,
            )
            st.bar_chart(comp_df.set_index("Algorithm")[["Modularity"]], color="#1F6FEB")

        with chart_col2:
            st.markdown(
                "<p style='color:#8B949E; font-size:12px; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;'>"
                "Communities Detected</p>",
                unsafe_allow_html=True,
            )
            st.bar_chart(comp_df.set_index("Algorithm")[["Communities"]], color="#388BFD")

        st.markdown(
            "<p style='color:#8B949E; font-size:12px; font-weight:600; "
            "text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px;'>"
            "Average Community Size</p>",
            unsafe_allow_html=True,
        )
        st.bar_chart(
            comp_df.set_index("Algorithm")[["Avg Community Size"]], color="#58A6FF"
        )
    else:
        st.info("Select an algorithm in the sidebar to run it, or build all above.")
