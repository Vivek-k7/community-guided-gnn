import streamlit as st
import requests
import pandas as pd

# 1. Configuration & Backend URL
BACKEND_URL = "http://localhost:8000/analyze"  # Update this when backend is live

st.set_page_config(page_title="GitHub Developer Insights", layout="wide")


# 2. Mock Backend Function (For development/testing)
def fetch_user_data(username):
    """
    Simulates the backend response defined in the Frontend Integration Brief.
    In production, replace the 'return' with:
    response = requests.get(f"{BACKEND_URL}/{username}")
    return response.json()
    """
    # Example structure based on Page 2 of your document
    return {
        "user": {
            "username": username,
            "user_id": 1432,
            "degree": 56
        },
        "community": {
            "community_id": 12,
            "size": 438,
            "density": 0.21,
            "top_features": [1574, 3773, 2478, 3098],
            "representative_users": ["alice", "bob", "charlie"]
        },
        "similar_users": [
            {"username": "dev_x", "similarity_score": 0.91, "community_id": 12, "degree": 45},
            {"username": "dev_y", "similarity_score": 0.88, "community_id": 12, "degree": 30}
        ]
    }


# 3. Frontend Layout
st.title("Community-Aware Developer Insights")
st.markdown("Analyze GitHub developers based on follower relationships and repository interests.")

# User Input Section
with st.sidebar:
    st.header("Search Parameters")
    username_input = st.text_input("Enter GitHub Username", placeholder="e.g., john_doe")
    k_value = st.slider("Top-K Similar Developers", min_value=1, max_value=20, value=5)
    submit_button = st.button("Analyze Developer")

if submit_button and username_input:
    with st.spinner(f"Fetching data for {username_input}..."):
        # Integration Point: Calling the backend
        data = fetch_user_data(username_input)

        # Display Results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("User Profile")
            st.metric("Username", data["user"]["username"])
            st.metric("User ID", data["user"]["user_id"])
            st.metric("Connections (Degree)", data["user"]["degree"])

        with col2:
            st.subheader("Community Details")
            c_info = data["community"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Community ID", c_info["community_id"])
            m2.metric("Community Size", c_info["size"])
            m3.metric("Density", c_info["density"])

            st.write("**Top Features (Repos/Topics):**")
            st.info(", ".join(map(str, c_info["top_features"])))

            st.write("**Representative Users:**")
            st.write(", ".join(c_info["representative_users"]))

        st.divider()

        # Output Top 'K' Similar Usernames
        st.subheader(f"Top {k_value} Similar Developers")
        similar_df = pd.DataFrame(data["similar_users"]).head(k_value)

        # Display as a clean table
        st.dataframe(
            similar_df,
            column_config={
                "similarity_score": st.column_config.ProgressColumn(
                    "Similarity", min_value=0, max_value=1, format="%.2f"
                ),
            },
            use_container_width=True,
            hide_index=True
        )
elif not username_input and submit_button:
    st.warning("Please enter a valid GitHub username.")