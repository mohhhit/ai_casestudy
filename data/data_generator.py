"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: data/data_generator.py
 Purpose: Generate realistic synthetic datasets for three AI use-cases:
          1. AR Lens / Filter Recommendation
          2. Programmatic Ad Targeting
          3. User Churn Prediction
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Ensure output directory exists
DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1.  AR Lens Recommendation Dataset
# ---------------------------------------------------------------------------

def generate_ar_lens_data(n_users: int = 5000) -> pd.DataFrame:
    """
    Simulates Snapchat user interactions with AR lenses/filters.

    Features
    --------
    user_id          : Unique user identifier
    age_group        : Teen / Young Adult / Adult
    gender           : M / F / NB
    country          : Top Snap markets
    daily_snaps_sent : Average snaps sent per day
    story_views      : Average story views per day
    session_minutes  : Average daily session length (minutes)
    lens_category    : Category of lens last used
    time_of_day      : Morning / Afternoon / Evening / Night
    device_type      : iOS / Android
    lens_used_cnt    : # unique lenses used last 30 days
    friend_count     : Number of friends on platform
    clicked_discover : Did user click Discover content? (bool)

    Target
    ------
    engaged_with_ar  : 1 = opened/used recommended lens, 0 = ignored
    """
    age_groups     = np.random.choice(["Teen", "Young Adult", "Adult"], n_users, p=[0.35, 0.45, 0.20])
    gender         = np.random.choice(["M", "F", "NB"], n_users, p=[0.40, 0.55, 0.05])
    country        = np.random.choice(["US", "UK", "IN", "AU", "BR"], n_users, p=[0.40, 0.15, 0.25, 0.10, 0.10])
    daily_snaps    = np.random.poisson(lam=12, size=n_users).clip(0, 80)
    story_views    = np.random.poisson(lam=8,  size=n_users).clip(0, 50)
    session_min    = np.random.gamma(shape=3, scale=10, size=n_users).clip(1, 120).round(1)
    lens_category  = np.random.choice(["Face", "World", "Selfie", "Fun", "Brand"], n_users)
    time_of_day    = np.random.choice(["Morning", "Afternoon", "Evening", "Night"], n_users, p=[0.20, 0.25, 0.35, 0.20])
    device_type    = np.random.choice(["iOS", "Android"], n_users, p=[0.55, 0.45])
    lens_used_cnt  = np.random.poisson(lam=7, size=n_users).clip(0, 50)
    friend_count   = np.random.lognormal(mean=3.5, sigma=0.8, size=n_users).astype(int).clip(0, 500)
    clicked_disc   = np.random.binomial(1, 0.45, size=n_users)

    # Engagement probability driven by features (realistic signal)
    base_prob = (
        0.15
        + 0.10 * (age_groups == "Teen")
        + 0.08 * (age_groups == "Young Adult")
        + 0.05 * (daily_snaps / 80)
        + 0.07 * (session_min / 120)
        + 0.05 * (lens_used_cnt / 50)
        + 0.06 * clicked_disc
        + np.random.normal(0, 0.05, n_users)
    ).clip(0.05, 0.95)

    engaged = np.random.binomial(1, base_prob, n_users)

    df = pd.DataFrame({
        "user_id":         [f"U{i:06d}" for i in range(n_users)],
        "age_group":       age_groups,
        "gender":          gender,
        "country":         country,
        "daily_snaps_sent": daily_snaps,
        "story_views":     story_views,
        "session_minutes": session_min,
        "lens_category":   lens_category,
        "time_of_day":     time_of_day,
        "device_type":     device_type,
        "lens_used_cnt":   lens_used_cnt,
        "friend_count":    friend_count,
        "clicked_discover": clicked_disc,
        "engaged_with_ar": engaged,
    })
    return df


# ---------------------------------------------------------------------------
# 2.  Ad Targeting Dataset
# ---------------------------------------------------------------------------

def generate_ad_targeting_data(n_samples: int = 8000) -> pd.DataFrame:
    """
    Simulates Snapchat ad impression data for programmatic targeting.

    Features
    --------
    impression_id    : Unique impression identifier
    user_segment     : Snap audience segment
    ad_format        : Snap Ad / Story Ad / Collection Ad / AR Ad
    ad_category      : Fashion / Gaming / Food / Travel / Beauty
    bid_price_usd    : Advertiser bid price per impression
    hour_of_day      : Hour auction occurred (0-23)
    day_of_week      : 0=Mon … 6=Sun
    device_type      : iOS / Android
    placement        : Between Friends / Discover / Spotlight
    user_age_bucket  : 13-17 / 18-24 / 25-34 / 35+
    frequency        : How many times user saw this ad before
    relevance_score  : Platform relevance estimate (0-10)

    Targets
    -------
    clicked          : 1 = ad click, 0 = impression only
    converted        : 1 = downstream conversion (purchase/signup)
    """
    segments      = np.random.choice(["Lifestyle", "Gamer", "Fashionista", "Foodie", "Traveller"], n_samples)
    ad_format     = np.random.choice(["Snap Ad", "Story Ad", "Collection Ad", "AR Ad"], n_samples, p=[0.35, 0.30, 0.20, 0.15])
    ad_category   = np.random.choice(["Fashion", "Gaming", "Food", "Travel", "Beauty"], n_samples)
    bid_price     = np.random.uniform(0.01, 2.50, n_samples).round(4)
    hour_of_day   = np.random.randint(0, 24, n_samples)
    day_of_week   = np.random.randint(0, 7, n_samples)
    device_type   = np.random.choice(["iOS", "Android"], n_samples, p=[0.55, 0.45])
    placement     = np.random.choice(["Between Friends", "Discover", "Spotlight"], n_samples, p=[0.50, 0.30, 0.20])
    age_bucket    = np.random.choice(["13-17", "18-24", "25-34", "35+"], n_samples, p=[0.18, 0.42, 0.27, 0.13])
    frequency     = np.random.poisson(lam=2, size=n_samples).clip(0, 10)
    relevance     = np.random.uniform(0, 10, n_samples).round(2)

    ctr_base = (
        0.02
        + 0.04 * (relevance / 10)
        + 0.02 * (ad_format == "AR Ad")
        + 0.01 * (placement == "Between Friends")
        - 0.005 * frequency
        + np.random.normal(0, 0.01, n_samples)
    ).clip(0.005, 0.30)

    clicked   = np.random.binomial(1, ctr_base, n_samples)
    cvr_base  = (0.10 + 0.15 * (relevance / 10) + np.random.normal(0, 0.03, n_samples)).clip(0.02, 0.50)
    converted = clicked * np.random.binomial(1, cvr_base, n_samples)

    df = pd.DataFrame({
        "impression_id":   [f"IMP{i:07d}" for i in range(n_samples)],
        "user_segment":    segments,
        "ad_format":       ad_format,
        "ad_category":     ad_category,
        "bid_price_usd":   bid_price,
        "hour_of_day":     hour_of_day,
        "day_of_week":     day_of_week,
        "device_type":     device_type,
        "placement":       placement,
        "user_age_bucket": age_bucket,
        "frequency":       frequency,
        "relevance_score": relevance,
        "clicked":         clicked,
        "converted":       converted,
    })
    return df


# ---------------------------------------------------------------------------
# 3.  User Churn Prediction Dataset
# ---------------------------------------------------------------------------

def generate_churn_data(n_users: int = 6000) -> pd.DataFrame:
    """
    Simulates 30-day Snapchat user behaviour for churn prediction.

    Features
    --------
    user_id           : Unique user identifier
    tenure_days       : Days since account creation
    age_group         : Teen / Young Adult / Adult
    country           : Top Snap markets
    avg_daily_opens   : Average app opens per day (last 30 days)
    avg_snaps_sent    : Average snaps sent per day
    avg_stories_posted: Average stories posted per week
    streak_max        : Longest Snap streak (days)
    streak_current    : Current streak length
    friend_count      : Friends on platform
    notifications_on  : Push notifications enabled (bool)
    used_spotlight    : Used Spotlight feature (bool)
    used_bitmoji      : Uses Bitmoji avatar (bool)
    premium_features  : Paid for any premium feature (bool)
    days_since_last_open: Days since last app open

    Target
    ------
    churned           : 1 = inactive 30+ days, 0 = still active
    """
    tenure        = np.random.exponential(scale=400, size=n_users).clip(1, 3000).astype(int)
    age_group     = np.random.choice(["Teen", "Young Adult", "Adult"], n_users, p=[0.35, 0.45, 0.20])
    country       = np.random.choice(["US", "UK", "IN", "AU", "BR"], n_users, p=[0.40, 0.15, 0.25, 0.10, 0.10])
    daily_opens   = np.random.gamma(3, 5, n_users).clip(0, 50).round(1)
    snaps_sent    = np.random.poisson(10, n_users).clip(0, 80)
    stories_pw    = np.random.poisson(2, n_users).clip(0, 20)
    streak_max    = np.random.geometric(p=0.05, size=n_users).clip(0, 365)
    streak_cur    = np.minimum(streak_max, np.random.geometric(p=0.10, size=n_users).clip(0, 365))
    friend_count  = np.random.lognormal(3.5, 0.8, n_users).astype(int).clip(0, 500)
    notif_on      = np.random.binomial(1, 0.65, n_users)
    spotlight     = np.random.binomial(1, 0.30, n_users)
    bitmoji       = np.random.binomial(1, 0.70, n_users)
    premium       = np.random.binomial(1, 0.08, n_users)
    days_since    = np.random.exponential(scale=8, size=n_users).clip(0, 90).astype(int)

    churn_prob = (
        0.10
        + 0.20 * (days_since > 14)
        + 0.15 * (daily_opens < 2)
        + 0.10 * (streak_cur == 0)
        - 0.08 * notif_on
        - 0.06 * bitmoji
        - 0.05 * premium
        - 0.07 * (snaps_sent > 5)
        + np.random.normal(0, 0.05, n_users)
    ).clip(0.02, 0.95)

    churned = np.random.binomial(1, churn_prob, n_users)

    df = pd.DataFrame({
        "user_id":              [f"U{i:06d}" for i in range(n_users)],
        "tenure_days":          tenure,
        "age_group":            age_group,
        "country":              country,
        "avg_daily_opens":      daily_opens,
        "avg_snaps_sent":       snaps_sent,
        "avg_stories_posted":   stories_pw,
        "streak_max":           streak_max,
        "streak_current":       streak_cur,
        "friend_count":         friend_count,
        "notifications_on":     notif_on,
        "used_spotlight":       spotlight,
        "used_bitmoji":         bitmoji,
        "premium_features":     premium,
        "days_since_last_open": days_since,
        "churned":              churned,
    })
    return df


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def generate_all(save: bool = True) -> dict:
    """Generate all three datasets and optionally save as CSV."""
    print("=" * 60)
    print("  Snap Inc.  |  Synthetic Data Generation")
    print("=" * 60)

    datasets = {
        "ar_lens":      generate_ar_lens_data(),
        "ad_targeting": generate_ad_targeting_data(),
        "churn":        generate_churn_data(),
    }

    for name, df in datasets.items():
        print(f"\n[{name}]  shape={df.shape}")
        print(df.head(3).to_string(index=False))
        if save:
            path = DATA_DIR / f"{name}_data.csv"
            df.to_csv(path, index=False)
            print(f"  --> Saved: {path}")

    return datasets


if __name__ == "__main__":
    generate_all()

