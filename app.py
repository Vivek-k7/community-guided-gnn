import matplotlib.pyplot as plt
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
    page_title="DevGraph",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Global ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

/* ── Algorithm pills (horizontal radio) ───────────── */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    gap: 6px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: #FDF9F4;
    border: 1px solid #D4B896;
    border-radius: 20px;
    padding: 5px 14px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #7A5C44;
    transition: all 0.15s;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: #FFF3DC;
    border-color: #A0522D;
    color: #7B3000;
    font-weight: 600;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* ── Search input ─────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #D4B896 !important;
    border-radius: 10px !important;
    color: #2D1B0E !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #C68642 !important;
    box-shadow: 0 0 0 3px rgba(198,134,66,0.18) !important;
}

/* ── Primary button ───────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #A0522D 0%, #C68642 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 0;
    transition: opacity 0.15s;
    color: #FFFFFF !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.85; }

/* ── Secondary (home) button ──────────────────────── */
[data-testid="stButton"] > button[kind="secondary"] {
    background: #FDF9F4;
    border: 1px solid #D4B896;
    border-radius: 10px;
    color: #7A5C44;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.15s;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #A0522D;
    color: #2D1B0E;
}

/* ── Tabs ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #EDE5D8;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #D4B896;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 8px 22px;
    color: #7A5C44;
    font-weight: 500;
    font-size: 14px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #2D1B0E !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Dataframe ────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #D4B896 !important;
}

/* ── Divider ──────────────────────────────────────── */
hr { border-color: #DDD0BE !important; margin: 20px 0 !important; }

/* ── Alerts ───────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Caption ──────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #7A5C44 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_algo_data(a: str) -> dict:
    return load_algorithm_data(a)

@st.cache_data(show_spinner=False)
def get_edges() -> pd.DataFrame:
    return load_edges()

@st.cache_resource(show_spinner=False)
def get_features():
    return load_features_matrix(37700)


# ── Top bar ───────────────────────────────────────────────────────────────────

title_col, algo_col = st.columns([1, 3])

with title_col:
    st.markdown(
        "<h2 style='margin:0; padding:6px 0; color:#2D1B0E; font-weight:700;'>"
        "DevGraph</h2>",
        unsafe_allow_html=True,
    )

with algo_col:
    algo = st.radio(
        "algo",
        options=ALGORITHMS,
        format_func=lambda x: ALGO_DISPLAY[x],
        label_visibility="collapsed",
        horizontal=True,
    )

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner(f"Loading {ALGO_DISPLAY[algo]}..."):
    data = get_algo_data(algo)

node_df           = data["node_df"]
community_summary = data["community_summary"]
comm_reps         = data["community_representatives"]
global_stats      = data["global_stats"]
edges_df          = get_edges()

# ── Search bar ────────────────────────────────────────────────────────────────

search_col, btn_col = st.columns([6, 1])
with search_col:
    user_query = st.text_input(
        "search",
        placeholder="Search by GitHub username or user ID...",
        label_visibility="collapsed",
    )
with btn_col:
    analyze_btn = st.button("Search", type="primary", use_container_width=True)

# Persist the last successful query so sliders don't clear results on rerun
if analyze_btn and user_query.strip():
    st.session_state["active_query"] = user_query.strip()

active_query = st.session_state.get("active_query", "")

# ── Developer view (search result) ───────────────────────────────────────────

if active_query:
    user_row = lookup_user(active_query, node_df)

    if user_row is None:
        st.warning("User not found. Try a numeric ID (0-37699) or an exact username.")
        st.session_state.pop("active_query", None)
        st.session_state.pop("last_user_id", None)
        st.session_state.pop("last_community_id", None)

    else:
        uid      = int(user_row["id"])
        degree   = int(user_row["degree"])
        comm_id  = int(user_row["community_id"])
        username = str(user_row["name"]).strip() or f"ID {uid}"
        c_info   = community_summary[community_summary["community_id"] == comm_id].iloc[0]

        st.session_state["last_user_id"]      = uid
        st.session_state["last_community_id"] = comm_id

        # ── Profile banner + Home button ──────────────────────────────────────
        banner_col, home_col = st.columns([6, 1])

        with banner_col:
            st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#FDF0DC 0%,#F7F3ED 100%);
    border:1px solid #D4B896;
    border-radius:14px;
    padding:22px 28px;
    margin:16px 0 20px;
    display:flex;
    align-items:center;
    gap:20px;
">
    <div style="
        width:56px;height:56px;border-radius:50%;flex-shrink:0;
        background:linear-gradient(135deg,#A0522D,#C68642);
        display:flex;align-items:center;justify-content:center;
        font-size:24px;font-weight:700;color:#fff;
    ">{username[0].upper()}</div>
    <div style="flex:1;">
        <div style="font-size:20px;font-weight:700;color:#2D1B0E;">{username}</div>
        <a href="https://github.com/{username}" target="_blank"
           style="color:#A0522D;font-size:13px;text-decoration:none;">
           github.com/{username} &uarr;
        </a>
    </div>
    <div style="display:flex;gap:28px;">
        <div style="text-align:center;">
            <div style="color:#7A5C44;font-size:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;">User ID</div>
            <div style="color:#2D1B0E;font-size:18px;font-weight:700;margin-top:2px;">{uid}</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#7A5C44;font-size:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;">Connections</div>
            <div style="color:#2D1B0E;font-size:18px;font-weight:700;margin-top:2px;">{degree}</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#7A5C44;font-size:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;">Community</div>
            <div style="color:#A0522D;font-size:18px;font-weight:700;margin-top:2px;">#{comm_id}</div>
        </div>
        <div style="text-align:center;">
            <div style="color:#7A5C44;font-size:10px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;">Comm. Size</div>
            <div style="color:#2D1B0E;font-size:18px;font-weight:700;margin-top:2px;">{int(c_info["size"]):,}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        with home_col:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            if st.button("Home", use_container_width=True):
                st.session_state.pop("active_query", None)
                st.session_state.pop("last_user_id", None)
                st.session_state.pop("last_community_id", None)
                st.rerun()

        # ── Similar developers + community explorer ───────────────────────────
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown(
                "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;'>"
                "Recommended Collaborators</p>",
                unsafe_allow_html=True,
            )
            k_val = st.slider("Top-K", 1, 20, 8, key="k_result", label_visibility="collapsed")
            rec_feat, rec_deg = st.tabs(["Feature Similarity", "Degree Baseline"])

            with rec_feat:
                sim_df = feature_similar_users(user_row, node_df, get_features(), k_val)
                if sim_df.empty:
                    st.info("No candidates found in this community.")
                else:
                    sim_display = sim_df.copy()
                    sim_display["name"] = sim_display["name"].apply(
                        lambda n: f"https://github.com/{n}"
                    )
                    st.dataframe(
                        sim_display,
                        column_config={
                            "name": st.column_config.LinkColumn(
                                "Username",
                                display_text=r"https://github\.com/(.+)",
                            ),
                            "similarity_score": st.column_config.ProgressColumn(
                                "Similarity", min_value=0, max_value=1, format="%.3f"
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

            with rec_deg:
                toy_df = toy_similar_users(user_row, node_df, k_val)
                if toy_df.empty:
                    st.info("No candidates found in this community.")
                else:
                    toy_display = toy_df.copy()
                    toy_display["name"] = toy_display["name"].apply(
                        lambda n: f"https://github.com/{n}"
                    )
                    st.dataframe(
                        toy_display,
                        column_config={
                            "name": st.column_config.LinkColumn(
                                "Username",
                                display_text=r"https://github\.com/(.+)",
                            ),
                            "similarity_score": st.column_config.ProgressColumn(
                                "Similarity", min_value=0, max_value=1, format="%.3f"
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

        with right_col:
            st.markdown(
                f"<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;'>"
                f"Community #{comm_id} Explorer</p>",
                unsafe_allow_html=True,
            )
            max_n = st.slider("Nodes in view", 10, 80, 50, 5, key="comm_nodes")
            sub_g = community_subgraph(
                comm_id, node_df, edges_df, max_n, ensure_node_id=uid
            )
            if sub_g.number_of_nodes() > 0:
                st.caption(
                    f"{sub_g.number_of_nodes()} nodes · {sub_g.number_of_edges()} edges "
                    "· Hover to inspect · Click to open GitHub · Gold = you"
                )
                html = build_pyvis_graph(
                    sub_g, node_df, node_to_comm=None, height="460px", highlight_id=uid
                )
                components.html(html, height=480, scrolling=False)

# ── Default view (no active search) ──────────────────────────────────────────

else:
    tab_overview, tab_compare = st.tabs(["Overview", "Compare Algorithms"])

    # ── Overview tab ─────────────────────────────────────────────────────────
    with tab_overview:
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, val in [
            (m1, "Developers",    f"{global_stats['num_nodes']:,}"),
            (m2, "Connections",   f"{global_stats['num_edges']:,}"),
            (m3, "Communities",   f"{global_stats['num_communities']:,}"),
            (m4, "Modularity",    f"{global_stats['modularity']:.4f}"),
            (m5, "Graph Density", f"{global_stats['density']:.6f}"),
        ]:
            col.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #D4B896;border-radius:10px;
            padding:16px 20px;margin-bottom:4px;">
    <div style="color:#7A5C44;font-size:10px;font-weight:600;text-transform:uppercase;
                letter-spacing:.06em;">{label}</div>
    <div style="color:#2D1B0E;font-size:22px;font-weight:700;margin-top:4px;">{val}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.06em;'>Top 15 Communities by Size</p>",
                unsafe_allow_html=True,
            )
            top15 = community_summary.head(15).copy()
            fig1, ax1 = plt.subplots(figsize=(6, 3.4))
            fig1.patch.set_facecolor("#FFFFFF")
            ax1.set_facecolor("#FDF9F4")
            ax1.bar(
                top15["community_id"].astype(str),
                top15["size"],
                color="#C68642",
                width=0.6,
                zorder=2,
            )
            ax1.set_xlabel("Community ID", fontsize=9, color="#7A5C44")
            ax1.set_ylabel("Size", fontsize=9, color="#7A5C44")
            ax1.tick_params(colors="#7A5C44", labelsize=8)
            ax1.spines[["top", "right"]].set_visible(False)
            ax1.spines[["left", "bottom"]].set_color("#D4B896")
            ax1.yaxis.grid(True, color="#EDE5D8", linewidth=0.7, zorder=1)
            ax1.set_axisbelow(True)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(pad=0.5)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with c2:
            st.markdown(
                "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.06em;'>Size vs Internal Density</p>",
                unsafe_allow_html=True,
            )
            fig2, ax2 = plt.subplots(figsize=(6, 3.4))
            fig2.patch.set_facecolor("#FFFFFF")
            ax2.set_facecolor("#FDF9F4")
            ax2.scatter(
                community_summary["size"],
                community_summary["internal_density"],
                color="#A0522D",
                alpha=0.55,
                s=22,
                edgecolors="none",
                zorder=2,
            )
            ax2.set_xlabel("Size", fontsize=9, color="#7A5C44")
            ax2.set_ylabel("Internal Density", fontsize=9, color="#7A5C44")
            ax2.tick_params(colors="#7A5C44", labelsize=8)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.spines[["left", "bottom"]].set_color("#D4B896")
            ax2.yaxis.grid(True, color="#EDE5D8", linewidth=0.7, zorder=1)
            ax2.xaxis.grid(True, color="#EDE5D8", linewidth=0.7, zorder=1)
            ax2.set_axisbelow(True)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.markdown(
            "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
            "text-transform:uppercase;letter-spacing:.06em;margin-top:8px;'>"
            "Global Community Map</p>",
            unsafe_allow_html=True,
        )
        sg1, sg2 = st.columns(2)
        with sg1:
            n_comms = st.slider("Communities", 5, min(40, len(community_summary)), 20)
        with sg2:
            n_per = st.slider("Nodes per community", 2, 12, 4)

        sample_g, node_to_comm = sampled_global_graph(node_df, edges_df, n_per, n_comms)
        if sample_g.number_of_nodes() > 0:
            st.caption(
                f"{sample_g.number_of_nodes()} nodes · {sample_g.number_of_edges()} edges "
                "· Hover to inspect · Click to open GitHub"
            )
            components.html(
                build_pyvis_graph(sample_g, node_df, node_to_comm, height="560px"),
                height=580, scrolling=False,
            )

    # ── Compare tab ──────────────────────────────────────────────────────────
    with tab_compare:
        not_built = [a for a in ALGORITHMS if not algo_artifacts_exist(a)]
        if not_built:
            info_col, btn_col2 = st.columns([3, 1])
            with info_col:
                st.info(
                    f"Not yet computed: **{', '.join(ALGO_DISPLAY[a] for a in not_built)}**. "
                    "Results cache to disk after the first run."
                )
            with btn_col2:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if st.button("Build Missing", type="primary", use_container_width=True):
                    for a in not_built:
                        with st.spinner(f"Running {ALGO_DISPLAY[a]}..."):
                            build_algorithm(a)
                    get_algo_data.clear()
                    st.success("Done.")
                    st.rerun()

        comp_df = load_comparison_stats()
        if not comp_df.empty:
            st.markdown(
                "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>"
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
            cc1, cc2 = st.columns(2)

            with cc1:
                st.markdown(
                    "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:.06em;'>Modularity</p>",
                    unsafe_allow_html=True,
                )
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                fig3.patch.set_facecolor("#FFFFFF")
                ax3.set_facecolor("#FDF9F4")
                ax3.bar(comp_df["Algorithm"], comp_df["Modularity"], color="#C68642", width=0.5, zorder=2)
                ax3.tick_params(colors="#7A5C44", labelsize=8)
                ax3.spines[["top", "right"]].set_visible(False)
                ax3.spines[["left", "bottom"]].set_color("#D4B896")
                ax3.yaxis.grid(True, color="#EDE5D8", linewidth=0.7, zorder=1)
                ax3.set_axisbelow(True)
                plt.xticks(rotation=20, ha="right", fontsize=8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

            with cc2:
                st.markdown(
                    "<p style='color:#7A5C44;font-size:11px;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:.06em;'>Communities Detected</p>",
                    unsafe_allow_html=True,
                )
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                fig4.patch.set_facecolor("#FFFFFF")
                ax4.set_facecolor("#FDF9F4")
                ax4.bar(comp_df["Algorithm"], comp_df["Communities"], color="#A0522D", width=0.5, zorder=2)
                ax4.tick_params(colors="#7A5C44", labelsize=8)
                ax4.spines[["top", "right"]].set_visible(False)
                ax4.spines[["left", "bottom"]].set_color("#D4B896")
                ax4.yaxis.grid(True, color="#EDE5D8", linewidth=0.7, zorder=1)
                ax4.set_axisbelow(True)
                plt.xticks(rotation=20, ha="right", fontsize=8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4)
