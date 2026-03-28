"""
Golden Village — Cinema Occupancy Analytics
============================================
Purpose : Analyse historical booking data to compute occupancy rates
          across dimensions (day, time, location, movie genre, etc.)
          and produce scheduling recommendations for future screenings.
          Includes a Linear Regression model to predict occupancy rate
          for any future screening based on independent factors.

Outputs :
  - Occupancy summary DataFrames
  - Heatmaps (day × time, location × day)
  - Top/bottom performing slots
  - Recommended prime-time windows per location
  - Linear regression model with feature coefficients + diagnostics
  - predict_occupancy() function for forward planning

Dependencies : pandas, numpy, matplotlib, seaborn, scikit-learn
Install      : pip install pandas numpy matplotlib seaborn scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

CAPACITY_MAP = {
    "Vivocity": 280, "JEM": 230, "Jurong Point": 210,
    "Tampines": 260, "Yishun": 200, "Paya Lebar": 220,
    "Bishan": 190,  "Kallang Leisure Park": 175
}

# ─────────────────────────────────────────────
# 1.  DATA LOADER
#     Reads from gv_bookings.csv (same folder as script).
# ─────────────────────────────────────────────
def generate_booking_data(n_records: int = 50_000) -> pd.DataFrame:
    """
    Load booking data from gv_bookings.csv (same folder as this script).
    Falls back to synthetic generation if the CSV is not found.

    To always use the CSV, ensure gv_bookings.csv is in:
        /Applications/MAMP/htdocs/GoldenVillage/
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data/gv_bookings.csv")

    if os.path.exists(csv_path):
        print(f"    Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["screening_date"])
        df["occupancy_rate"] = df["seats_sold"] / df["capacity"]
        if "month" not in df.columns:
            df["month"] = df["screening_date"].dt.month
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["screening_date"].dt.strftime("%a")
        return df
    
# ─────────────────────────────────────────────
# 2.  CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def occupancy_by_day_time(df: pd.DataFrame) -> pd.DataFrame:
    """Mean occupancy rate for each (day_of_week × time_slot) cell."""
    pivot = (
        df.groupby(["day_of_week", "time_slot"])["occupancy_rate"]
          .mean()
          .unstack("time_slot")
          .reindex(DAYS_OF_WEEK)
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
              avg_occupancy=("occupancy_rate", "mean"),
              total_revenue=("revenue",        "sum"),
              screenings   =("genre",          "count")
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
# 4.  LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────
def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer numeric features for the regression model.

    Independent variables (X):
        time_slot_hour    — hour of screening (numeric, 10–22)
        is_weekend        — 1 if Fri/Sat/Sun, else 0
        is_evening        — 1 if slot is 18:00, 20:00 or 22:00
        is_morning        — 1 if slot is 10:00 or 12:00
        month             — calendar month 1–12 (seasonality signal)
        is_school_holiday — 1 if month is Jun or Dec (SG school holidays)
        ticket_price      — ticket price in SGD
        genre_*           — one-hot encoded genre (drop_first to avoid multicollinearity)
        location_*        — one-hot encoded location (drop_first)

    Target variable (y):
        occupancy_rate    — seats_sold / capacity  (0.0 to 1.0)
    """
    feat = df.copy()

    feat["time_slot_hour"]    = feat["time_slot"].str.split(":").str[0].astype(int)
    feat["is_weekend"]        = feat["day_of_week"].isin(["Fri", "Sat", "Sun"]).astype(int)
    feat["is_evening"]        = feat["time_slot"].isin(["18:00", "20:00", "22:00"]).astype(int)
    feat["is_morning"]        = feat["time_slot"].isin(["10:00", "12:00"]).astype(int)
    feat["is_school_holiday"] = feat["month"].isin([6, 12]).astype(int)

    genre_dummies    = pd.get_dummies(feat["genre"],    prefix="genre",    drop_first=True)
    location_dummies = pd.get_dummies(feat["location"], prefix="location", drop_first=True)

    feat = pd.concat([feat, genre_dummies, location_dummies], axis=1)
    return feat


def train_occupancy_model(df: pd.DataFrame) -> dict:
    """
    Train a Ridge Regression model to predict occupancy_rate.

    Ridge is used over plain LinearRegression to add mild regularisation
    (alpha=0.5), which stabilises coefficients when one-hot dummy columns
    are correlated with each other.

    Returns
    -------
    dict with keys:
        model        — fitted sklearn Pipeline (StandardScaler + Ridge)
        feature_cols — list of feature names in training order
        metrics      — MAE, RMSE, R², cross-validated R²
        coef_df      — DataFrame of coefficients sorted by absolute impact
        X_test       — held-out feature matrix
        y_test       — held-out true values
        y_pred       — model predictions on X_test
    """
    feat = build_regression_features(df)

    base_features = [
        "time_slot_hour", "is_weekend", "is_evening", "is_morning",
        "month", "is_school_holiday", "ticket_price"
    ]
    genre_cols    = [c for c in feat.columns if c.startswith("genre_")]
    location_cols = [c for c in feat.columns if c.startswith("location_")]
    feature_cols  = base_features + genre_cols + location_cols

    X = feat[feature_cols].values
    y = feat["occupancy_rate"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     Ridge(alpha=0.5))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    metrics = {
        "MAE":          round(mae,   4),
        "RMSE":         round(rmse,  4),
        "R²":           round(r2,    4),
        "Cross-val R²": round(cv_r2, 4),
        "Train size":   len(X_train),
        "Test size":    len(X_test),
    }

    coefficients = model.named_steps["lr"].coef_
    coef_df = (
        pd.DataFrame({
            "feature":     feature_cols,
            "coefficient": coefficients
        })
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    return {
        "model":        model,
        "feature_cols": feature_cols,
        "metrics":      metrics,
        "coef_df":      coef_df,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
    }


def predict_occupancy(
    model_bundle: dict,
    df_historical: pd.DataFrame,
    screenings: list
) -> pd.DataFrame:
    """
    Predict occupancy rate for one or more future screenings.

    Parameters
    ----------
    model_bundle    : dict returned by train_occupancy_model()
    df_historical   : full historical DataFrame (used for column alignment)
    screenings      : list of dicts, each containing:
                        location, time_slot, day_of_week,
                        month, genre, ticket_price

    Returns
    -------
    DataFrame with input fields + predicted_occupancy_pct + predicted_seats_sold

    Example
    -------
    future = [
        {"location": "Vivocity", "time_slot": "20:00", "day_of_week": "Sat",
         "month": 6, "genre": "Action", "ticket_price": 22.0},
    ]
    results = predict_occupancy(model_bundle, df, future)
    print(results)
    """
    model        = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    rows = []
    for s in screenings:
        row = s.copy()
        row["time_slot_hour"]    = int(s["time_slot"].split(":")[0])
        row["is_weekend"]        = int(s["day_of_week"] in ["Fri", "Sat", "Sun"])
        row["is_evening"]        = int(s["time_slot"] in ["18:00", "20:00", "22:00"])
        row["is_morning"]        = int(s["time_slot"] in ["10:00", "12:00"])
        row["is_school_holiday"] = int(s["month"] in [6, 12])
        rows.append(row)

    pred_df = pd.DataFrame(rows)

    genre_dummies    = pd.get_dummies(pred_df["genre"],    prefix="genre")
    location_dummies = pd.get_dummies(pred_df["location"], prefix="location")
    pred_df = pd.concat([pred_df, genre_dummies, location_dummies], axis=1)

    # Fill any dummy columns absent from this small batch with 0
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0

    X_pred    = pred_df[feature_cols].values
    predicted = np.clip(model.predict(X_pred), 0.05, 1.0)

    output = pd.DataFrame(screenings)
    output["predicted_occupancy"]     = predicted.round(4)
    output["predicted_occupancy_pct"] = (predicted * 100).round(1)
    output["predicted_seats_sold"]    = (
        predicted * output["location"].map(CAPACITY_MAP)
    ).astype(int)

    return output


# ─────────────────────────────────────────────
# 5.  VISUALISATIONS — DESCRIPTIVE
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

    axes[0].barh(genre_df.index, genre_df["avg_occupancy"],
                 color=sns.color_palette("RdYlGn", len(genre_df)))
    axes[0].set_xlabel("Average Occupancy Rate")
    axes[0].set_title("Occupancy by Genre", fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

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
# 6.  VISUALISATIONS — REGRESSION MODEL
# ─────────────────────────────────────────────
def plot_feature_coefficients(coef_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: str = None):
    """
    Horizontal bar chart of the top-N most impactful features by
    absolute coefficient value.
    Green = increases occupancy, Red = decreases occupancy.
    """
    plot_df = coef_df.head(top_n).copy()
    colours = ["#2A9D8F" if v >= 0 else "#E63946"
               for v in plot_df["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"][::-1],
            plot_df["coefficient"][::-1],
            color=colours[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Linear Regression — Top {top_n} Feature Coefficients\n"
        "(Positive = increases occupancy   |   Negative = decreases occupancy)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Coefficient value (effect on occupancy rate)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend = [Patch(color="#2A9D8F", label="Increases occupancy"),
              Patch(color="#E63946", label="Decreases occupancy")]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_actual_vs_predicted(y_test: np.ndarray,
                              y_pred: np.ndarray,
                              save_path: str = None):
    """Scatter of actual vs predicted + residuals histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.25, s=8, color="#457B9D")
    lims = [max(0, min(y_test.min(), y_pred.min()) - 0.02),
            min(1, max(y_test.max(), y_pred.max()) + 0.02)]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Occupancy Rate")
    axes[0].set_ylabel("Predicted Occupancy Rate")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].legend(); axes[0].grid(linestyle="--", alpha=0.35)

    # Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=50, color="#6A0572",
                 edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution\n(centred at 0 = unbiased model)",
                      fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle("Linear Regression — Model Diagnostics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_prediction_examples(predictions_df: pd.DataFrame,
                              save_path: str = None):
    """Bar chart of predicted occupancy for a set of future screenings."""
    labels = (
        predictions_df["location"] + "\n" +
        predictions_df["day_of_week"] + " " +
        predictions_df["time_slot"]
    )
    colours = [
        "#2A9D8F" if v >= 0.70 else
        "#F4A261" if v >= 0.50 else "#E63946"
        for v in predictions_df["predicted_occupancy"]
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(predictions_df) * 1.6), 5))
    bars = ax.bar(labels, predictions_df["predicted_occupancy"],
                  color=colours, edgecolor="white", width=0.6)
    for bar, val in zip(bars, predictions_df["predicted_occupancy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.0%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0.70, color="#2A9D8F", linestyle="--",
               linewidth=1.5, label="70% recommended threshold")
    ax.axhline(0.50, color="#F4A261", linestyle=":",
               linewidth=1.5, label="50% viability floor")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Predicted Occupancy — Future Screenings",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted Occupancy Rate")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_regression_line(model_bundle: dict,
                          df: pd.DataFrame,
                          feature: str,
                          save_path: str = None):
    """
    Plot the linear regression relationship for a single feature.
    Shows actual data points and the model's predicted line.
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    
    # Get the feature index
    if feature not in feature_cols:
        print(f"Feature '{feature}' not in model features.")
        return
    
    feat_idx = feature_cols.index(feature)
    
    # Prepare data for plotting
    plot_df = build_regression_features(df)
    X = plot_df[feature_cols].values
    y_actual = plot_df["occupancy_rate"].values
    
    # Sort by the feature for smooth line
    sort_idx = np.argsort(X[:, feat_idx])
    X_sorted = X[sort_idx]
    y_actual_sorted = y_actual[sort_idx]
    
    # Predict for the sorted data
    y_pred_sorted = model.predict(X_sorted)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of actual data
    ax.scatter(X_sorted[:, feat_idx], y_actual_sorted, alpha=0.3, s=10, color="#457B9D", label="Actual data")
    
    # Regression line
    ax.plot(X_sorted[:, feat_idx], y_pred_sorted, color="#E63946", linewidth=2, label="Regression prediction")
    
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Occupancy Rate")
    ax.set_title(f"Linear Regression: Occupancy Rate vs {feature.replace('_', ' ').title()}", 
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_actual_vs_predicted(y_test: np.ndarray,
                              y_pred: np.ndarray,
                              save_path: str = None):
    """Scatter of actual vs predicted + residuals histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.25, s=8, color="#457B9D")
    lims = [max(0, min(y_test.min(), y_pred.min()) - 0.02),
            min(1, max(y_test.max(), y_pred.max()) + 0.02)]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Occupancy Rate")
    axes[0].set_ylabel("Predicted Occupancy Rate")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].legend(); axes[0].grid(linestyle="--", alpha=0.35)

    # Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=50, color="#6A0572",
                 edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution\n(centred at 0 = unbiased model)",
                      fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    plt.suptitle("Linear Regression — Model Diagnostics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_prediction_examples(predictions_df: pd.DataFrame,
                              save_path: str = None):
    """Bar chart of predicted occupancy for a set of future screenings."""
    labels = (
        predictions_df["location"] + "\n" +
        predictions_df["day_of_week"] + " " +
        predictions_df["time_slot"]
    )
    colours = [
        "#2A9D8F" if v >= 0.70 else
        "#F4A261" if v >= 0.50 else "#E63946"
        for v in predictions_df["predicted_occupancy"]
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(predictions_df) * 1.6), 5))
    bars = ax.bar(labels, predictions_df["predicted_occupancy"],
                  color=colours, edgecolor="white", width=0.6)
    for bar, val in zip(bars, predictions_df["predicted_occupancy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.0%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0.70, color="#2A9D8F", linestyle="--",
               linewidth=1.5, label="70% recommended threshold")
    ax.axhline(0.50, color="#F4A261", linestyle=":",
               linewidth=1.5, label="50% viability floor")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Predicted Occupancy — Future Screenings",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted Occupancy Rate")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 7.  MAIN RUNNER
# ─────────────────────────────────────────────
def run_occupancy_analysis():
    print("=" * 60)
    print("  GOLDEN VILLAGE — OCCUPANCY ANALYTICS")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1] Loading booking data …")
    df = generate_booking_data(n_records=50_000)
    print(f"    Records loaded : {len(df):,}")
    print(f"    Date range     : {pd.to_datetime(df['screening_date']).min().date()} "
          f"→ {pd.to_datetime(df['screening_date']).max().date()}")
    print(f"    Locations      : {df['location'].nunique()}")
    print(f"    Avg occupancy  : {df['occupancy_rate'].mean():.1%}")

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

    # ── Linear Regression Model ───────────────────────────────
    print("\n" + "=" * 60)
    print("[8] Training Linear Regression (Ridge) model …")
    model_bundle = train_occupancy_model(df)

    print("\n  Model Performance Metrics:")
    for k, v in model_bundle["metrics"].items():
        print(f"    {k:<20}: {v}")

    print("\n  Top 10 most impactful features:")
    print(model_bundle["coef_df"].head(10).to_string(index=False))

    print("\n  Plotting feature coefficients …")
    plot_feature_coefficients(model_bundle["coef_df"], top_n=20)

    print("  Plotting actual vs predicted diagnostics …")
    plot_actual_vs_predicted(model_bundle["y_test"], model_bundle["y_pred"])

    print("  Plotting regression line for time_slot_hour …")
    plot_regression_line(model_bundle, df, "time_slot_hour")

    # ── Example predictions for future screenings ─────────────
    print("\n[9] Predicting occupancy for example future screenings …")
    future_screenings = [
        {"location": "Vivocity",             "time_slot": "20:00", "day_of_week": "Sat",
         "month": 6,  "genre": "Action",     "ticket_price": 22.00},
        {"location": "Vivocity",             "time_slot": "10:00", "day_of_week": "Tue",
         "month": 3,  "genre": "Drama",      "ticket_price": 13.50},
        {"location": "Tampines",             "time_slot": "20:00", "day_of_week": "Fri",
         "month": 12, "genre": "Animation",  "ticket_price": 19.00},
        {"location": "Jurong Point",         "time_slot": "22:00", "day_of_week": "Sat",
         "month": 8,  "genre": "Thriller",   "ticket_price": 16.50},
        {"location": "Yishun",               "time_slot": "12:00", "day_of_week": "Mon",
         "month": 2,  "genre": "Romance",    "ticket_price": 13.50},
        {"location": "Kallang Leisure Park", "time_slot": "18:00", "day_of_week": "Sun",
         "month": 6,  "genre": "Horror",     "ticket_price": 15.00},
    ]
    predictions = predict_occupancy(model_bundle, df, future_screenings)
    print(predictions[[
        "location", "day_of_week", "time_slot", "genre",
        "predicted_occupancy_pct", "predicted_seats_sold"
    ]].to_string(index=False))
    plot_prediction_examples(predictions)

    print("\n✅  Occupancy analysis complete.")
    return df


if __name__ == "__main__":
    df = run_occupancy_analysis()
