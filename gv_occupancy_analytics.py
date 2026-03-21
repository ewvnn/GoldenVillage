"""
Golden Village — Cinema Occupancy Analytics
============================================
Purpose : Analyse historical booking data to compute occupancy rates
          across dimensions (day, time, location, movie genre, etc.)
          and produce scheduling recommendations for future screenings.

Outputs :
  - Occupancy summary DataFrames
  - Heatmaps (day × time, location × day)
  - Top/bottom performing slots
  - Recommended prime-time windows per location

Dependencies : pandas, numpy, matplotlib, seaborn, scikit-learn
Install      : pip install pandas numpy matplotlib seaborn scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

LOCATIONS = [
    "Vivocity", "JEM", "Jurong Point", "Tampines",
    "Yishun", "Paya Lebar", "Bishan", "Kallang Leisure Park"
]

GENRES = ["Action", "Romance", "Horror", "Animation",
          "Thriller", "Comedy", "Drama", "Sci-Fi"]

TIME_SLOTS = [
    "10:00", "12:00", "14:00", "16:00",
    "18:00", "20:00", "22:00"
]

DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

CAPACITY_MAP = {          # seats per hall (location-level average)
    "Vivocity": 280, "JEM": 230, "Jurong Point": 210,
    "Tampines": 260, "Yishun": 200, "Paya Lebar": 220,
    "Bishan": 190,  "Kallang Leisure Park": 175
}

# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATOR
#     Replace generate_booking_data() with a
#     function that reads your actual database.
# ─────────────────────────────────────────────
## import booking data from CSV
df = pd.read_csv("data/gv_bookings.csv", parse_dates=["screening_date"])




# def generate_booking_data(n_records: int = 50_000) -> pd.DataFrame:
#     """
#     Simulate 12 months of screening-level booking records.

#     Real usage:
#         df = pd.read_csv("gv_bookings.csv", parse_dates=["screening_date"])
#         # OR connect to your DB / data warehouse here

#     Expected schema (minimum required columns):
#         screening_date  : datetime
#         location        : str
#         time_slot       : str   (e.g. "20:00")
#         day_of_week     : str   (e.g. "Fri")
#         genre           : str
#         seats_sold      : int
#         capacity        : int
#         ticket_price    : float
#     """
#     dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

#     records = []
#     for _ in range(n_records):
#         loc   = rng.choice(LOCATIONS)
#         date  = pd.Timestamp(rng.choice(dates))
#         dow   = DAYS_OF_WEEK[date.dayofweek]
#         slot  = rng.choice(TIME_SLOTS)
#         genre = rng.choice(GENRES)
#         cap   = CAPACITY_MAP[loc]

#         # Realistic occupancy patterns
#         base = 0.45
#         if dow in ["Fri", "Sat", "Sun"]:   base += 0.22   # weekend boost
#         if slot in ["20:00", "22:00"]:     base += 0.18   # prime evening
#         if slot in ["10:00", "12:00"]:     base -= 0.15   # morning dip
#         if genre in ["Action", "Animation"]: base += 0.08
#         if genre == "Drama":               base -= 0.06
#         base += rng.normal(0, 0.08)          # noise

#         occ_rate  = float(np.clip(base, 0.05, 1.0))
#         seats_sold = int(cap * occ_rate)
#         price      = rng.choice([13.50, 15.00, 16.50, 19.00, 22.00])

#         records.append({
#             "screening_date": date,
#             "location":       loc,
#             "time_slot":      slot,
#             "day_of_week":    dow,
#             "month":          date.month,
#             "genre":          genre,
#             "seats_sold":     seats_sold,
#             "capacity":       cap,
#             "ticket_price":   price,
#             "revenue":        seats_sold * price
#         })

#     df = pd.DataFrame(records)
#     df["occupancy_rate"] = df["seats_sold"] / df["capacity"]
#     return df


# ─────────────────────────────────────────────
# 2.  CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def occupancy_by_day_time(df: pd.DataFrame) -> pd.DataFrame:
    """Mean occupancy rate for each (day_of_week × time_slot) cell."""
    pivot = (
        df.groupby(["day_of_week", "time_slot"])["occupancy_rate"]
          .mean()
          .unstack("time_slot")
          .reindex(DAYS_OF_WEEK)          # canonical Mon→Sun order
          .reindex(columns=TIME_SLOTS)
    )
    return pivot


def occupancy_by_location_day(df: pd.DataFrame) -> pd.DataFrame:
    """Mean occupancy rate for each (location × day_of_week) cell."""
    pivot = (
        df.groupby(["location", "day_of_week"])["occupancy_rate"]
          .mean()
          .unstack("day_of_week")
          .reindex(columns=DAYS_OF_WEEK)
    )
    return pivot


def occupancy_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate occupancy and revenue statistics per genre."""
    return (
        df.groupby("genre")
          .agg(
              avg_occupancy =("occupancy_rate", "mean"),
              total_revenue =("revenue",        "sum"),
              screenings    =("genre",          "count")
          )
          .sort_values("avg_occupancy", ascending=False)
          .round(3)
    )


def occupancy_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Month-level occupancy trend (seasonality view)."""
    return (
        df.groupby("month")["occupancy_rate"]
          .mean()
          .rename("avg_occupancy")
          .reset_index()
    )


def top_bottom_slots(df: pd.DataFrame, n: int = 5) -> dict:
    """Return top-N and bottom-N (day, time, location) combinations."""
    agg = (
        df.groupby(["location", "day_of_week", "time_slot"])
          ["occupancy_rate"].mean()
          .reset_index()
          .sort_values("occupancy_rate", ascending=False)
    )
    return {
        "top":    agg.head(n).reset_index(drop=True),
        "bottom": agg.tail(n).reset_index(drop=True)
    }


# ─────────────────────────────────────────────
# 3.  SCHEDULING RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend_screening_slots(
    df: pd.DataFrame,
    location: str,
    top_n: int = 5,
    min_occupancy_threshold: float = 0.70
) -> pd.DataFrame:
    """
    For a given location, rank (day × time_slot) pairs by historical
    occupancy and flag those above the viability threshold.

    Parameters
    ----------
    location              : cinema branch name
    top_n                 : number of recommended slots to return
    min_occupancy_threshold: minimum avg occupancy to be "recommended"

    Returns
    -------
    DataFrame with columns: day_of_week, time_slot, avg_occupancy,
                             recommended, rank
    """
    loc_df = df[df["location"] == location]
    slot_perf = (
        loc_df.groupby(["day_of_week", "time_slot"])
              .agg(
                  avg_occupancy=("occupancy_rate", "mean"),
                  screenings   =("occupancy_rate", "count")
              )
              .reset_index()
              .sort_values("avg_occupancy", ascending=False)
    )
    slot_perf["recommended"] = slot_perf["avg_occupancy"] >= min_occupancy_threshold
    slot_perf["rank"] = range(1, len(slot_perf) + 1)
    return slot_perf.head(top_n).reset_index(drop=True)


def generate_all_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Compile scheduling recommendations for every location."""
    frames = []
    for loc in LOCATIONS:
        rec = recommend_screening_slots(df, loc, top_n=5)
        rec.insert(0, "location", loc)
        frames.append(rec)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 4.  VISUALISATIONS
# ─────────────────────────────────────────────
def plot_day_time_heatmap(pivot: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".0%", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Average Occupancy Rate"}
    )
    ax.set_title("Occupancy Rate — Day of Week × Time Slot",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time Slot", fontsize=11)
    ax.set_ylabel("Day of Week", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_location_day_heatmap(pivot: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".0%", cmap="Blues",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Average Occupancy Rate"}
    )
    ax.set_title("Occupancy Rate — Location × Day of Week",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Location", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_genre_performance(genre_df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Occupancy bar
    axes[0].barh(genre_df.index, genre_df["avg_occupancy"],
                 color=sns.color_palette("RdYlGn", len(genre_df)))
    axes[0].set_xlabel("Average Occupancy Rate")
    axes[0].set_title("Occupancy by Genre", fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Revenue bar
    axes[1].barh(genre_df.index, genre_df["total_revenue"] / 1e6,
                 color=sns.color_palette("Blues_r", len(genre_df)))
    axes[1].set_xlabel("Total Revenue (SGD Millions)")
    axes[1].set_title("Revenue by Genre", fontweight="bold")

    plt.suptitle("Genre Performance Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_monthly_trend(month_df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(month_df["month"], month_df["avg_occupancy"],
            marker="o", linewidth=2.5, color="#E63946")
    ax.fill_between(month_df["month"], month_df["avg_occupancy"],
                    alpha=0.15, color="#E63946")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Monthly Occupancy Trend", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Occupancy Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 5.  MAIN RUNNER
# ─────────────────────────────────────────────
def run_occupancy_analysis():
    print("=" * 60)
    print("  GOLDEN VILLAGE — OCCUPANCY ANALYTICS")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1] Loading booking data …")
    # df = generate_booking_data(n_records=50_000)
    print(f"    Records loaded : {len(df):,}")
    print(f"    Date range     : {df['screening_date'].min().date()} "
          f"→ {df['screening_date'].max().date()}")
    print(f"    Locations      : {df['location'].nunique()}")

    # ── Day × Time heatmap ────────────────────────────────────
    print("\n[2] Generating Day × Time occupancy heatmap …")
    dt_pivot = occupancy_by_day_time(df)
    plot_day_time_heatmap(dt_pivot)

    # ── Location × Day heatmap ────────────────────────────────
    print("[3] Generating Location × Day occupancy heatmap …")
    ld_pivot = occupancy_by_location_day(df)
    plot_location_day_heatmap(ld_pivot)

    # ── Genre performance ─────────────────────────────────────
    print("[4] Genre performance breakdown …")
    genre_df = occupancy_by_genre(df)
    print(genre_df.to_string())
    plot_genre_performance(genre_df)

    # ── Monthly trend ─────────────────────────────────────────
    print("\n[5] Monthly occupancy trend …")
    month_df = occupancy_by_month(df)
    plot_monthly_trend(month_df)

    # ── Top / Bottom slots ────────────────────────────────────
    print("\n[6] Top 5 and Bottom 5 slots across all locations:")
    tb = top_bottom_slots(df, n=5)
    print("\n  TOP 5 highest-occupancy slots:")
    print(tb["top"].to_string(index=False))
    print("\n  BOTTOM 5 lowest-occupancy slots (consider reducing screenings):")
    print(tb["bottom"].to_string(index=False))

    # ── Scheduling recommendations ────────────────────────────
    print("\n[7] Scheduling recommendations per location:")
    recs = generate_all_recommendations(df)
    for loc in LOCATIONS:
        loc_recs = recs[recs["location"] == loc]
        print(f"\n  📍 {loc}")
        print(loc_recs[["rank","day_of_week","time_slot",
                         "avg_occupancy","recommended"]].to_string(index=False))

    print("\n✅  Occupancy analysis complete.")
    return df   # return df so it can be piped into clustering module


if __name__ == "__main__":
    df = run_occupancy_analysis()
