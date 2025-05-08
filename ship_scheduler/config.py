# ship_scheduler/config.py
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ROUTES_CSV      = os.path.join(DATA_DIR, "routes.csv")
SHIPS_CSV       = os.path.join(DATA_DIR, "ships.csv")
SHIPMENTS_CSV   = os.path.join(DATA_DIR, "shipments.csv")
DISTANCES_CSV   = os.path.join(DATA_DIR, "distances.csv")
STORMS_CSV      = os.path.join(DATA_DIR, "storms.csv")
BASELINE = os.path.join(RESULTS_DIR, "baseline.csv")
RESULTS = os.path.join(RESULTS_DIR, "trained.csv")
LEARNING_CURVE = os.path.join(RESULTS_DIR, "cumulative_rewards.csv")
LOG_CSV      = os.path.join(RESULTS_DIR, "event_log.csv")
PARAMS = os.path.join(RESULTS_DIR, "tuned_params.csv")

SEED = 2003

# CHANGE ACCORDINGLY
YEARS = 5



