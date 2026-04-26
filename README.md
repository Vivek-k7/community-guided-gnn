# DevGraph — GitHub Developer Community Explorer

A Streamlit web application that detects communities in the GitHub developer network using five unsupervised machine learning algorithms, and recommends collaborators based on repository activity similarity.

---

## Dataset

This project uses the **MUSAE GitHub Social Network** dataset.

| File | Description |
|------|-------------|
| `git_web_ml/musae_git_edges.csv` | 289,003 follow relationships between developers |
| `git_web_ml/musae_git_target.csv` | 37,700 developer nodes with GitHub usernames |
| `git_web_ml/musae_git_features.json` | 4005-dim sparse binary feature vectors (repository memberships) |

**Source:** [MUSAE GitHub Dataset — Stanford SNAP / Benedek Rozemberczki](https://snap.stanford.edu/data/github-social.html)

The dataset files are included in the repository under `git_web_ml/`.

---

## Requirements

- **Python >= 3.10** (required for `int | None` union syntax used in `data_services.py`)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Key packages

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.56.0 | Web application framework |
| `pandas` | 2.2.3 | Data loading and manipulation |
| `numpy` | 1.26.4 | Numerical operations (pinned to 1.x for compatibility with python-louvain) |
| `matplotlib` | 3.10.8 | Static chart rendering |
| `networkx` | 3.6.1 | Graph construction and community detection |
| `scipy` | 1.17.1 | Sparse matrix operations |
| `scikit-learn` | 1.8.0 | TruncatedSVD, KMeans, cosine similarity |
| `python-louvain` | 0.16 | Louvain community detection (imported as `community`) |
| `pyvis` | 0.3.2 | Interactive graph visualization |

---

## Project Structure

```
community-guided-gnn/
├── app.py                          # Streamlit frontend
├── data_services.py                # Data loading, algorithm builders, graph utilities
├── requirements.txt
├── gnn_vgae.ipynb                  # Notebook used to train the VGAE and generate embeddings
├── git_web_ml/
│   ├── musae_git_edges.csv
│   ├── musae_git_target.csv
│   └── musae_git_features.json
├── artifacts/
│   ├── vgae_embeddings.npy         # Pre-trained VGAE node embeddings (37700 × 64)
│   └── frontend/                   # Cached algorithm artifacts (auto-generated on first run)
│       ├── louvain/
│       ├── label_propagation/
│       ├── spectral/
│       ├── kmeans/
│       └── gnn/
└── .streamlit/
    └── config.toml                 # UI theme configuration
```

---

## Running the App

### 1. Clone the repository

```bash
git clone https://github.com/Vivek-k7/community-guided-gnn.git
cd community-guided-gnn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## First Run Notes

- On first launch, the app will automatically build and cache the community detection artifacts for the selected algorithm. This takes 1–3 minutes per algorithm depending on your machine.
- Subsequent runs load from the cache instantly.
- The **VGAE + K-Means** algorithm requires `artifacts/vgae_embeddings.npy` to already exist (included in the repo). If you want to retrain the VGAE from scratch, run `gnn_vgae.ipynb` in Jupyter — it will overwrite the embeddings file.

---

## Algorithms

| Algorithm | Approach | Uses Graph | Uses Features |
|-----------|----------|-----------|--------------|
| Louvain | Modularity optimisation | Yes | No |
| Label Propagation | Neighbour voting (asynchronous) | Yes | No |
| Spectral Clustering | Eigendecomposition + K-Means | Yes | No |
| K-Means (Features) | TruncatedSVD + K-Means on feature vectors | No | Yes |
| VGAE + K-Means | Graph Convolutional encoder + K-Means on learned embeddings | Yes | Yes |
