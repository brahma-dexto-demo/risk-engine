"""Stable feature ordering shared by training and inference."""

import numpy as np

FEATURE_NAMES = ("days_since_last_login", "open_tickets", "monthly_spend_usd")


def features(accounts):
    return np.array([[row[name] for name in FEATURE_NAMES] for row in accounts], dtype=float)
