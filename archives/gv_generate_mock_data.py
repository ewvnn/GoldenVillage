"""
Golden Village — Mock Data Generator
======================================
Produces three CSV files that serve as realistic mock datasets
for the GV occupancy and clustering analytics pipelines.

Output files:
  gv_bookings.csv          — 50,000 screening-level booking records
  gv_customers.csv         — 8,000 customer-level CRM profiles
  gv_movies.csv            — 120 movie reference records

Usage:
  python gv_generate_mock_data.py

To load into the analytics pipelines, replace the generate_*() calls:
  # occupancy:
  df = pd.read_csv("gv_bookings.csv", parse_dates=["screening_date"])
  # clustering:
  df = pd.read_csv("gv_customers.csv", parse_dates=["last_visit_date"])
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────
LOCATIONS = [
    "Vivocity", "JEM", "Jurong Point", "Tampines",
    "Yishun", "Paya Lebar", "Bishan", "Kallang Leisure Park"
]

CAPACITY_MAP = {
    "Vivocity": 280, "JEM": 230, "Jurong Point": 210,
    "Tampines": 260, "Yishun": 200, "Paya Lebar": 220,
    "Bishan": 190,  "Kallang Leisure Park": 175
}

HALL_MAP = {           # number of halls per location
    "Vivocity": 8, "JEM": 6, "Jurong Point": 6,
    "Tampines": 7, "Yishun": 5, "Paya Lebar": 6,
    "Bishan": 5,   "Kallang Leisure Park": 4
}

GENRES = ["Action", "Romance", "Horror", "Animation",
          "Thriller", "Comedy", "Drama", "Sci-Fi"]

TIME_SLOTS = ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]
DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TICKET_PRICES = [13.50, 15.00, 16.50, 19.00, 22.00]   # standard → premium

# Hall type affects price tier
HALL_TYPES = ["Standard", "Standard", "Standard", "Premium", "IMAX", "Dolby"]
HALL_PRICE_MULTIPLIER = {
    "Standard": 1.0, "Premium": 1.3, "IMAX": 1.6, "Dolby": 1.5
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "GV App Wallet",
                   "PayNow", "GrabPay", "Cash"]

FIRST_NAMES_SG = [
    "Wei Jie", "Xiu Ying", "Rajan", "Priya", "Nadia", "Hafiz",
    "Jun Hao", "Li Ting", "Arjun", "Mei Ling", "Siti", "Karthik",
    "Zhen Wei", "Amelia", "Darren", "Jasmine", "Ryan", "Hui Fen",
    "Amirah", "Chen Hao", "Surya", "Ling Ling", "Faiz", "Tan Wei",
    "Nicole", "Marcus", "Hui Min", "Farhan", "Jia Qi", "Shreya"
]

LAST_NAMES_SG = [
    "Tan", "Lim", "Lee", "Ng", "Wong", "Chen", "Goh", "Chua",
    "Ong", "Koh", "Nair", "Pillai", "Ramachandran", "Abdullah",
    "Hussain", "Singh", "Kumar", "Patel", "Teo", "Yeo"
]

# ─────────────────────────────────────────────
# MOVIE TITLES  (120 realistic fictional titles)
# ─────────────────────────────────────────────
MOVIE_TEMPLATES = {
    "Action":    ["Steel Storm", "Iron Vanguard", "Last Operative", "Rogue Protocol",
                  "Titanfall Rising", "Red Horizon", "Zero Hour Strike", "Shadow Breach",
                  "Dark Sentinel", "Final Countdown", "Operation Blackout", "Overdrive",
                  "Code Nemesis", "Thunder Legion", "Alpha Strike", "Night Pursuit",
                  "Classified", "Ghost Protocol 2", "Killswitch", "Delta Force Rising"],
    "Romance":   ["Always You", "Second Chances", "Rainy Season", "Love in Transit",
                  "The Promise", "Between Us", "Chasing Sunsets", "Letters to Her",
                  "One More Chance", "A Perfect Lie", "Before Sunrise Again",
                  "Our Little Secret", "Still", "Unspoken", "Forever After"],
    "Horror":    ["The Hollow", "Nightwatch", "Beneath the Floor", "Sleepless",
                  "The Entity Returns", "Dusk", "Haunted Frequency", "The Last Rite",
                  "Something in the Dark", "Withered", "The Visitor", "Cold Sweat",
                  "Possession", "Unseen", "The Basement"],
    "Animation": ["Cosmic Pals", "Little Dragon Academy", "The Great Reef",
                  "Skybound", "Panda Troopers", "Robot Puppy", "Magic Garden 2",
                  "Ocean Friends", "Cloud Kingdom", "The Tiny Giants",
                  "Star Critters", "Jumbo and Friends", "Zap & Zoom"],
    "Thriller":  ["Blackout", "The Setup", "False Identity", "Missing Persons",
                  "Conspiracy", "The Informant", "Trust No One", "Loose End",
                  "Evidence", "Double Cross", "The Witness", "Deception"],
    "Comedy":    ["Weekend Disaster", "Totally Normal Family", "My Boss is Crazy",
                  "Road Trippin'", "Five Weddings", "The Accident", "Oops!",
                  "New Neighbours", "Date Night Gone Wrong", "Chaos & Order",
                  "The Promotion", "Backfire"],
    "Drama":     ["The Bridge", "Inheritance", "Broken Seasons", "Parallel Lives",
                  "The Decision", "Quiet Storm", "Echoes", "Home Again",
                  "When It Rains", "The Reckoning", "Aftermath", "Forgiven"],
    "Sci-Fi":    ["Exo-Planet", "Singularity", "Void Walker", "Parallel Earth",
                  "Genesis Protocol", "The Last Transmission", "Omega Station",
                  "Neural Link", "Beyond the Belt", "Exodus Colony", "AI Rising",
                  "Quantum Breach"]
}

RATINGS = ["G", "PG", "PG13", "NC16", "M18"]

# ─────────────────────────────────────────────
# 1.  MOVIES TABLE
# ─────────────────────────────────────────────
def generate_movies() -> pd.DataFrame:
    """
    120 movie records spanning Jan 2024 – Dec 2024.
    Schema: movie_id, title, genre, rating, duration_mins,
            release_date, language, distributor
    """
    languages    = ["English", "English", "English", "Mandarin",
                    "Tamil", "Malay", "Korean", "Japanese"]
    distributors = ["GV Films", "Shaw Distribution", "Cathay Asia",
                    "Disney SEA", "Universal SEA", "Sony Pictures SEA",
                    "Warner Bros SEA", "Indie SG"]

    rows = []
    movie_id = 1
    for genre, titles in MOVIE_TEMPLATES.items():
        rating_weights = {
            "Action":    [0,  5, 35, 40, 20],
            "Romance":   [5, 20, 45, 25,  5],
            "Horror":    [0,  0, 10, 40, 50],
            "Animation": [30,40, 25,  5,  0],
            "Thriller":  [0,  5, 30, 40, 25],
            "Comedy":    [5, 25, 50, 15,  5],
            "Drama":     [0, 10, 40, 35, 15],
            "Sci-Fi":    [0,  5, 35, 45, 15],
        }
        w = np.array(rating_weights[genre]) / 100

        for title in titles:
            release = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 365))
            )
            rows.append({
                "movie_id":      f"MOV{str(movie_id).zfill(4)}",
                "title":         title,
                "genre":         genre,
                "rating":        rng.choice(RATINGS, p=w),
                "duration_mins": int(rng.integers(85, 175)),
                "release_date":  release.date(),
                "language":      rng.choice(languages,
                                             p=[0.55,0,0,0.15,0.07,0.05,0.10,0.08]),
                "distributor":   rng.choice(distributors),
                "budget_sgd_m":  round(float(rng.uniform(2, 80)), 1),
            })
            movie_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2.  BOOKINGS TABLE  (50 000 rows)
# ─────────────────────────────────────────────
def generate_bookings(movies_df: pd.DataFrame,
                      n_records: int = 50_000) -> pd.DataFrame:
    """
    Screening-level booking records.

    Schema:
        booking_id, screening_date, day_of_week, time_slot,
        location, hall_number, hall_type,
        movie_id, title, genre,
        capacity, seats_sold, occupancy_rate,
        ticket_price, revenue,
        payment_method, advance_booking_days
    """
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    # Pre-build a movie pool with release-date awareness
    movie_pool = movies_df[["movie_id", "title", "genre", "release_date"]].copy()
    movie_pool["release_date"] = pd.to_datetime(movie_pool["release_date"])

    records = []
    for _ in range(n_records):
        loc      = rng.choice(LOCATIONS)
        date     = pd.Timestamp(rng.choice(dates))
        dow      = DAYS_OF_WEEK[date.dayofweek]
        slot     = rng.choice(TIME_SLOTS)
        cap      = CAPACITY_MAP[loc]
        n_halls  = HALL_MAP[loc]
        hall_no  = int(rng.integers(1, n_halls + 1))
        hall_type = rng.choice(HALL_TYPES,
                                p=[0.55, 0.15, 0.10, 0.10, 0.05, 0.05])

        # Pick a movie available on that date (released before screening)
        available = movie_pool[movie_pool["release_date"] <= date]
        if len(available) == 0:
            available = movie_pool   # fallback
        movie = available.sample(1, random_state=None).iloc[0]
        genre = movie["genre"]

        # ── Occupancy model ──────────────────────────────────
        base = 0.45
        if dow in ["Fri", "Sat", "Sun"]:       base += 0.22
        if slot in ["20:00", "22:00"]:         base += 0.18
        if slot in ["10:00", "12:00"]:         base -= 0.15
        if genre in ["Action", "Animation"]:   base += 0.08
        if genre == "Drama":                   base -= 0.06
        if hall_type in ["IMAX", "Dolby"]:     base += 0.05   # premium draw
        # School holiday boost (Jun & Dec)
        if date.month in [6, 12]:              base += 0.10
        base += rng.normal(0, 0.08)
        occ_rate   = float(np.clip(base, 0.05, 1.0))
        seats_sold = int(cap * occ_rate)

        # ── Pricing ───────────────────────────────────────────
        base_price = rng.choice(TICKET_PRICES)
        price = round(base_price * HALL_PRICE_MULTIPLIER[hall_type], 2)
        revenue = round(seats_sold * price, 2)

        # ── Other fields ──────────────────────────────────────
        advance_days = int(rng.integers(0, 14))   # days booked ahead

        records.append({
            "booking_id":           f"BK{str(len(records)+1).zfill(8)}",
            "screening_date":       date.date(),
            "month":                date.month,
            "day_of_week":          dow,
            "time_slot":            slot,
            "location":             loc,
            "hall_number":          hall_no,
            "hall_type":            hall_type,
            "movie_id":             movie["movie_id"],
            "title":                movie["title"],
            "genre":                genre,
            "capacity":             cap,
            "seats_sold":           seats_sold,
            "occupancy_rate":       round(occ_rate, 4),
            "ticket_price":         price,
            "revenue":              revenue,
            "payment_method":       rng.choice(
                PAYMENT_METHODS,
                p=[0.30, 0.20, 0.25, 0.10, 0.10, 0.05]
            ),
            "advance_booking_days": advance_days,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 3.  CUSTOMERS TABLE  (8 000 rows)
# ─────────────────────────────────────────────
def generate_customers(n_users: int = 8_000) -> pd.DataFrame:
    """
    Customer-level CRM profiles derived from aggregated booking history.

    Schema:
        customer_id, name, email, age, gender,
        gv_plus_member, gv_plus_tier,
        last_visit_date, recency_days,
        total_visits, total_spend, avg_ticket_price,
        pct_weekend_visits, pct_evening_visits,
        favourite_genre, preferred_location,
        acquisition_channel, registration_date
    """
    SNAPSHOT = pd.Timestamp("2025-01-01")

    segments = rng.choice(5, size=n_users,
                          p=[0.25, 0.20, 0.30, 0.10, 0.15])
    channels = ["GV App", "Walk-in", "Website", "Partner Referral",
                "Social Media", "Corporate"]

    rows = []
    for i, seg in enumerate(segments):
        uid = f"GV{str(i+1).zfill(6)}"

        # Name
        fname = rng.choice(FIRST_NAMES_SG)
        lname = rng.choice(LAST_NAMES_SG)
        name  = f"{fname} {lname}"
        email = (fname.lower().replace(" ", ".") + "." +
                 lname.lower() + str(rng.integers(10, 99)) + "@email.com")

        age    = int(rng.integers(16, 70))
        gender = rng.choice(["M", "F", "Prefer not to say"], p=[0.46, 0.48, 0.06])

        reg_date = SNAPSHOT - pd.Timedelta(days=int(rng.integers(30, 1200)))

        # ── Segment-specific behaviour ────────────────────────
        if seg == 0:   # Blockbuster Weekend Warriors
            visits   = int(rng.integers(18, 40))
            spend    = visits * rng.uniform(22, 35)
            recency  = int(rng.integers(1, 30))
            pct_wknd = rng.uniform(0.70, 1.0)
            pct_eve  = rng.uniform(0.65, 1.0)
            fav      = rng.choice(["Action", "Thriller", "Sci-Fi"])
            gv_plus  = rng.random() < 0.75
            tier     = rng.choice(["Gold", "Platinum"], p=[0.5, 0.5])

        elif seg == 1: # Family Matinee Regulars
            visits   = int(rng.integers(8, 20))
            spend    = visits * rng.uniform(55, 90)
            recency  = int(rng.integers(10, 60))
            pct_wknd = rng.uniform(0.60, 0.90)
            pct_eve  = rng.uniform(0.10, 0.40)
            fav      = rng.choice(["Animation", "Comedy", "Romance"])
            gv_plus  = rng.random() < 0.55
            tier     = rng.choice(["Silver", "Gold"], p=[0.6, 0.4])

        elif seg == 2: # Casual Occasionals
            visits   = int(rng.integers(1, 7))
            spend    = visits * rng.uniform(12, 20)
            recency  = int(rng.integers(60, 200))
            pct_wknd = rng.uniform(0.30, 0.70)
            pct_eve  = rng.uniform(0.30, 0.70)
            fav      = rng.choice(GENRES)
            gv_plus  = rng.random() < 0.15
            tier     = "Standard"

        elif seg == 3: # Premium Experience Seekers
            visits   = int(rng.integers(4, 12))
            spend    = visits * rng.uniform(45, 80)
            recency  = int(rng.integers(5, 45))
            pct_wknd = rng.uniform(0.40, 0.70)
            pct_eve  = rng.uniform(0.55, 0.90)
            fav      = rng.choice(["Drama", "Thriller", "Action"])
            gv_plus  = rng.random() < 0.90
            tier     = rng.choice(["Gold", "Platinum"], p=[0.3, 0.7])

        else:          # Lapsed Loyalists
            visits   = int(rng.integers(5, 15))
            spend    = visits * rng.uniform(15, 30)
            recency  = int(rng.integers(150, 365))
            pct_wknd = rng.uniform(0.40, 0.80)
            pct_eve  = rng.uniform(0.40, 0.80)
            fav      = rng.choice(GENRES)
            gv_plus  = rng.random() < 0.45
            tier     = rng.choice(["Standard", "Silver"], p=[0.5, 0.5])

        if not gv_plus:
            tier = "Non-member"

        avg_price = round(spend / visits, 2)
        last_visit = SNAPSHOT - pd.Timedelta(days=recency)

        rows.append({
            "customer_id":          uid,
            "name":                 name,
            "email":                email,
            "age":                  age,
            "gender":               gender,
            "gv_plus_member":       int(gv_plus),
            "gv_plus_tier":         tier,
            "registration_date":    reg_date.date(),
            "last_visit_date":      last_visit.date(),
            "recency_days":         recency,
            "total_visits":         visits,
            "total_spend":          round(float(spend), 2),
            "avg_ticket_price":     avg_price,
            "pct_weekend_visits":   round(float(pct_wknd), 3),
            "pct_evening_visits":   round(float(pct_eve), 3),
            "favourite_genre":      fav,
            "preferred_location":   rng.choice(LOCATIONS),
            "acquisition_channel":  rng.choice(
                channels, p=[0.35, 0.20, 0.20, 0.10, 0.10, 0.05]
            ),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    out = Path(".")
    print("=" * 60)
    print("  GOLDEN VILLAGE — MOCK DATA GENERATOR")
    print("=" * 60)

    print("\n[1] Generating movies table (120 records) …")
    movies_df = generate_movies()
    movies_path = out / "gv_movies.csv"
    movies_df.to_csv(movies_path, index=False)
    print(f"    ✔ Saved → {movies_path}")
    print(f"    Columns : {list(movies_df.columns)}")
    print(movies_df.groupby("genre").size().to_string())

    print("\n[2] Generating bookings table (50,000 records) …")
    bookings_df = generate_bookings(movies_df, n_records=50_000)
    bookings_path = out / "gv_bookings.csv"
    bookings_df.to_csv(bookings_path, index=False)
    print(f"    ✔ Saved → {bookings_path}")
    print(f"    Columns : {list(bookings_df.columns)}")
    print(f"    Date range : {bookings_df['screening_date'].min()} "
          f"→ {bookings_df['screening_date'].max()}")
    print(f"    Avg occupancy : {bookings_df['occupancy_rate'].mean():.1%}")
    print(f"    Total revenue : SGD {bookings_df['revenue'].sum():,.0f}")

    print("\n[3] Generating customers table (8,000 records) …")
    customers_df = generate_customers(n_users=8_000)
    customers_path = out / "gv_customers.csv"
    customers_df.to_csv(customers_path, index=False)
    print(f"    ✔ Saved → {customers_path}")
    print(f"    Columns : {list(customers_df.columns)}")
    print(f"    GV+ members : "
          f"{customers_df['gv_plus_member'].sum():,} "
          f"({customers_df['gv_plus_member'].mean():.1%})")
    print(f"    Tier breakdown:")
    print(customers_df["gv_plus_tier"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("  FILES GENERATED:")
    print(f"    gv_movies.csv     — {len(movies_df):,} rows")
    print(f"    gv_bookings.csv   — {len(bookings_df):,} rows")
    print(f"    gv_customers.csv  — {len(customers_df):,} rows")
    print("=" * 60)
    print("""
HOW TO LOAD INTO ANALYTICS PIPELINES:
──────────────────────────────────────
# In gv_occupancy_analytics.py, replace generate_booking_data():
df = pd.read_csv("gv_bookings.csv", parse_dates=["screening_date"])
df["occupancy_rate"] = df["seats_sold"] / df["capacity"]

# In gv_customer_clustering.py, replace generate_customer_data():
df = pd.read_csv("gv_customers.csv", parse_dates=["last_visit_date"])
df["recency_days"] = (pd.Timestamp("2025-01-01") - df["last_visit_date"]).dt.days
""")


if __name__ == "__main__":
    main()
