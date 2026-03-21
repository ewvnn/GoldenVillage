"""
Golden Village — Analytics Master Runner
=========================================
Runs both the occupancy analytics and customer clustering
pipelines in sequence.

Usage:
    python gv_analytics_main.py

Modules:
    gv_occupancy_analytics.py  — screening scheduling optimisation
    gv_customer_clustering.py  — customer segmentation for marketing
"""

from gv_occupancy_analytics  import run_occupancy_analysis
from gv_customer_clustering  import run_customer_clustering

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   GOLDEN VILLAGE — DIGITAL TRANSFORMATION ANALYTICS     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Module 1: Occupancy Analytics ──────────────────────────────
    booking_df = run_occupancy_analysis()

    # ── Module 2: Customer Clustering ──────────────────────────────
    # Pass optimal_k=5 to skip elbow calculation (recommended for prod),
    # or leave as None to auto-detect via silhouette scoring.
    segments_df, cluster_profile = run_customer_clustering(optimal_k=5)

    print("\n\n" + "═" * 60)
    print("  ALL PIPELINES COMPLETE")
    print("  Outputs:")
    print("    • gv_customer_segments.csv  (CRM upload-ready)")
    print("    • Charts rendered inline")
    print("═" * 60)
