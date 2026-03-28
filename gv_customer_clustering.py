"""
Golden Village — Customer Segmentation & Clustering
=====================================================
Purpose : Cluster GV customers into distinct behavioural personas
          using RFM metrics + entertainment preference signals,
          to power targeted marketing campaigns.

Cluster outputs (example personas — actual labels determined by data):
  • "Blockbuster Weekend Warriors"  — high frequency, weekend evenings
  • "Family Matinee Regulars"       — moderate frequency, daytime, animation
  • "Casual Occasionals"            — low frequency, price-sensitive
  • "Premium Experience Seekers"    — low frequency but high spend per visit
  • "Lapsed Loyalists"              — formerly active, now churned

Algorithm : K-Means with elbow method + silhouette scoring
            (extendable to DBSCAN or Gaussian Mixture Models)

Dependencies : pandas, numpy, matplotlib, seaborn, scikit-learn
Install      : pip install pandas numpy matplotlib seaborn scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
SEED     = 42
rng      = np.random.default_rng(SEED)
N_USERS  = 8_000
SNAPSHOT_DATE = pd.Timestamp("2025-01-01")   # "today" reference for recency

GENRES   = ["Action", "Romance", "Horror", "Animation",
            "Thriller", "Comedy", "Drama", "Sci-Fi"]
LOCATIONS = [
    "Vivocity", "JEM", "Jurong Point", "Tampines",
    "Yishun", "Paya Lebar", "Bishan", "Kallang Leisure Park"
]

# Persona labels — assign after inspecting cluster centroids
PERSONA_LABELS = {
    0: "Blockbuster Weekend Warriors",
    1: "Family Matinee Regulars",
    2: "Casual Occasionals",
    3: "Premium Experience Seekers",
    4: "Lapsed Loyalists"
}

PERSONA_COLOURS = {
    "Blockbuster Weekend Warriors":  "#E63946",
    "Family Matinee Regulars":       "#2A9D8F",
    "Casual Occasionals":            "#F4A261",
    "Premium Experience Seekers":    "#6A0572",
    "Lapsed Loyalists":              "#457B9D"
}

## import CSV Data
df_raw = pd.read_csv("data/gv_customers_v3.csv")



# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the numeric feature matrix for clustering.
    All categorical variables are one-hot encoded.
    """
    feat = df[[
        "customer_id",
        "recency_days",        # R — how recently they visited
        "total_visits",        # F — how often they come
        "total_spend",         # M — total monetary value
        "avg_ticket_price",    # proxy for premium preference
        "pct_weekend_visits",  # lifestyle signal
        "pct_evening_visits",  # time-of-day preference
        "gv_plus_member"       # loyalty signal
    ]].copy()

    # One-hot encode genre preference
    genre_dummies = pd.get_dummies(
        df["favourite_genre"], prefix="genre", drop_first=False
    )
    feat = pd.concat([feat, genre_dummies], axis=1)

    return feat


# ─────────────────────────────────────────────
# 3.  ELBOW METHOD + SILHOUETTE SCORING
# ─────────────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray,
                   k_range: range = range(2, 11),
                   save_path: str = None) -> int:
    """
    Run K-Means for k in k_range, plot inertia (elbow) and
    silhouette scores, and return the recommended K.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(list(k_range), inertias, marker="o", color="#E63946", linewidth=2)
    axes[0].set_title("Elbow Method — Inertia", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].plot(list(k_range), silhouettes, marker="s", color="#2A9D8F", linewidth=2)
    axes[1].set_title("Silhouette Score (higher = better)", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    optimal_k = list(k_range)[int(np.argmax(silhouettes))]
    axes[1].axvline(x=optimal_k, color="orange", linestyle="--",
                    label=f"Recommended K={optimal_k}")
    axes[1].legend()

    plt.suptitle("Optimal Cluster Count Selection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    print(f"    ✔ Recommended K (max silhouette) : {optimal_k}")
    return optimal_k


# ─────────────────────────────────────────────
# 4.  CLUSTERING
# ─────────────────────────────────────────────
def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=15, max_iter=500)
    km.fit(X_scaled)
    return km


def assign_persona_labels(df_clustered: pd.DataFrame,
                           feature_cols: list) -> pd.DataFrame:
    """
    Auto-label clusters by their centroid characteristics.

    Strategy:
      - Sort clusters by avg recency (ascending) and avg spend (descending)
      - Map rank order to PERSONA_LABELS dict
    Note: In production, review centroids manually and update PERSONA_LABELS.
    """
    summary = (
        df_clustered.groupby("cluster")[feature_cols]
                    .mean()
                    .reset_index()
    )
    # Sort: low recency (recent), high spend → "best" clusters first
    summary = summary.sort_values(
        ["recency_days", "total_spend"], ascending=[True, False]
    ).reset_index(drop=True)

    label_map = {
        int(row["cluster"]): PERSONA_LABELS.get(i, f"Segment {i}")
        for i, row in summary.iterrows()
    }
    df_clustered["persona"] = df_clustered["cluster"].map(label_map)
    return df_clustered, label_map


# ─────────────────────────────────────────────
# 5.  CLUSTER PROFILING
# ─────────────────────────────────────────────
def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each cluster along key marketing-relevant dimensions.
    """
    numeric_cols = [
        "recency_days", "total_visits", "total_spend",
        "avg_ticket_price", "pct_weekend_visits",
        "pct_evening_visits", "gv_plus_member"
    ]
    profile = (
        df.groupby("persona")[numeric_cols]
          .mean()
          .round(2)
    )
    profile["cluster_size"] = df.groupby("persona")["customer_id"].count()
    profile["cluster_pct"]  = (profile["cluster_size"] / len(df) * 100).round(1)
    profile["top_genre"] = (
        df.groupby("persona")["favourite_genre"]
          .agg(lambda x: x.value_counts().idxmax())
    )
    return profile


def marketing_playbook(profile: pd.DataFrame) -> None:
    """Print a plain-language marketing recommendation per persona."""
    playbook = {
        "Blockbuster Weekend Warriors": (
            "Push weekend blockbuster pre-sale alerts via app. "
            "Offer combo deals (movie + F&B). Reward streaks with bonus GV+ points."
        ),
        "Family Matinee Regulars": (
            "Target Saturday/Sunday matinee packages. Family bundles (4-pack). "
            "Promote Animation and Comedy releases 2 weeks in advance."
        ),
        "Casual Occasionals": (
            "Re-engagement emails with discount vouchers. Mid-week promotions. "
            "Lower-friction GV+ trial offers to upgrade loyalty."
        ),
        "Premium Experience Seekers": (
            "Exclusive early access to premium halls (Gemini, IMAX). "
            "Personalised genre recommendations. Partner dining/hotel bundles."
        ),
        "Lapsed Loyalists": (
            "Win-back campaign: 'We miss you' voucher. Highlight what's new (halls, "
            "titles). Remind of unused GV+ benefits before expiry."
        )
    }
    print("\n  🎯  MARKETING PLAYBOOK\n  " + "─" * 56)
    for persona, tactic in playbook.items():
        size_row = profile.loc[persona] if persona in profile.index else None
        pct = f"{size_row['cluster_pct']:.1f}%" if size_row is not None else "N/A"
        print(f"\n  [{persona}]  ({pct} of base)")
        print(f"  → {tactic}")


# ─────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────
def plot_pca_clusters(df: pd.DataFrame,
                      X_scaled: np.ndarray,
                      save_path: str = None):
    """2-D PCA scatter to visualise cluster separation."""
    pca   = PCA(n_components=2, random_state=SEED)
    comps = pca.fit_transform(X_scaled)
    var   = pca.explained_variance_ratio_

    plot_df = pd.DataFrame({
        "PC1":    comps[:, 0],
        "PC2":    comps[:, 1],
        "persona": df["persona"].values
    })

    fig, ax = plt.subplots(figsize=(11, 7))
    for persona, grp in plot_df.groupby("persona"):
        colour = PERSONA_COLOURS.get(persona, "#888888")
        ax.scatter(grp["PC1"], grp["PC2"], label=persona,
                   color=colour, alpha=0.55, s=18, edgecolors="none")

    ax.set_xlabel(f"PC 1 ({var[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var[1]:.1%} variance)", fontsize=11)
    ax.set_title("Customer Segments — PCA Projection",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Persona", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_radar(profile: pd.DataFrame, save_path: str = None):
    """Radar / spider chart comparing clusters on 5 key dimensions."""
    dims = ["total_visits", "total_spend", "pct_weekend_visits",
            "pct_evening_visits", "gv_plus_member"]
    dim_labels = ["Visits", "Spend", "% Weekend", "% Evening", "GV+ Member"]

    # Normalise to 0-1 for radar
    scaler = MinMaxScaler()
    normed  = pd.DataFrame(
        scaler.fit_transform(profile[dims]),
        index=profile.index, columns=dims
    )

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={"polar": True})

    for persona in normed.index:
        values = normed.loc[persona, dims].tolist() + \
                 [normed.loc[persona, dims[0]]]
        colour = PERSONA_COLOURS.get(persona, "#888")
        ax.plot(angles, values, "o-", linewidth=2, color=colour, label=persona)
        ax.fill(angles, values, alpha=0.12, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=11)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8)
    ax.set_title("Cluster Persona Radar",
                 fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_segment_distribution(df: pd.DataFrame, save_path: str = None):
    counts = df["persona"].value_counts()
    colours = [PERSONA_COLOURS.get(p, "#888") for p in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=colours, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({val/len(df)*100:.1f}%)",
                va="center", fontsize=10)
    ax.set_title("Customer Segment Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Customers")
    ax.set_xlim(0, counts.max() * 1.3)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 7.  EXPORT HELPER
# ─────────────────────────────────────────────
def export_segments(df: pd.DataFrame, path: str = "data/gv_customer_segments.csv"):
    """
    Export customer_id + persona + key features to CSV for
    upload into CRM / marketing automation platform.
    """
    export_cols = [
        "customer_id", "persona", "recency_days",
        "total_visits", "total_spend", "avg_ticket_price",
        "pct_weekend_visits", "pct_evening_visits",
        "gv_plus_member", "favourite_genre", "preferred_location"
    ]
    df[export_cols].to_csv(path, index=False)
    print(f"\n  💾  Segment export saved → {path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────
# 8.  MAIN RUNNER
# ─────────────────────────────────────────────
def run_customer_clustering(optimal_k: int = None):
    print("\n" + "=" * 60)
    print("  GOLDEN VILLAGE — CUSTOMER SEGMENTATION")
    print("=" * 60)

    # ── 1. Load customers ──────────────────────────────────────
    print("\n[1] Loading customer data …")
    # df_raw = generate_customer_data(N_USERS)
    print(f"    Customers loaded : {len(df_raw):,}")

    # ── 2. Feature engineering ────────────────────────────────
    print("[2] Building feature matrix …")
    feat_df   = build_feature_matrix(df_raw)
    feature_cols = [c for c in feat_df.columns if c != "customer_id"]
    X = feat_df[feature_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    Features used : {len(feature_cols)}")

    # ── 3. Optimal K ──────────────────────────────────────────
    print("[3] Finding optimal number of clusters …")
    if optimal_k is None:
        optimal_k = find_optimal_k(X_scaled, k_range=range(2, 10))
    else:
        print(f"    Using supplied K = {optimal_k}")

    # ── 4. Fit K-Means ────────────────────────────────────────
    print(f"[4] Fitting K-Means with K={optimal_k} …")
    km = fit_kmeans(X_scaled, optimal_k)
    df_raw["cluster"] = km.labels_

    sil = silhouette_score(X_scaled, km.labels_)
    print(f"    Silhouette score : {sil:.4f}")

    # ── 5. Label personas ─────────────────────────────────────
    print("[5] Assigning persona labels …")
    df_clustered, label_map = assign_persona_labels(
        df_raw, ["recency_days", "total_visits", "total_spend"]
    )
    print("    Cluster → Persona mapping:")
    for cluster_id, persona in label_map.items():
        n = (df_clustered["cluster"] == cluster_id).sum()
        print(f"      Cluster {cluster_id} → {persona}  ({n:,} customers)")

    # ── 6. Profile clusters ───────────────────────────────────
    print("\n[6] Cluster profiles:")
    profile = profile_clusters(df_clustered)
    print(profile.to_string())

    # ── 7. Visualisations ─────────────────────────────────────
    print("\n[7] Generating visualisations …")
    plot_segment_distribution(df_clustered)
    plot_pca_clusters(df_clustered, X_scaled)
    plot_cluster_radar(profile)

    # ── 8. Marketing playbook ─────────────────────────────────
    marketing_playbook(profile)

    # ── 9. Export ─────────────────────────────────────────────
    export_segments(df_clustered, "data/gv_customer_segments.csv")

    print("\n✅  Clustering complete.")
    return df_clustered, profile


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df_segments, cluster_profile = run_customer_clustering()
