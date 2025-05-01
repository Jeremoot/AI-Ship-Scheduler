import os
import pandas as pd

from .config import (
    ROUTES_CSV,
    DISTANCES_CSV,
    STORMS_CSV,
    SHIPS_CSV,
    SHIPMENTS_CSV
)
from .utils import (
    generate_routes_df,
    generate_distances_df,
    generate_storm_logs_df,
    initialize_ships,
    generate_shipment_data
)

def main():
    data_dir = os.path.dirname(ROUTES_CSV)
    os.makedirs(data_dir, exist_ok=True)  # make sure data directory exists

    routes_df = generate_routes_df()
    routes_df.to_csv(ROUTES_CSV, index=False)
    print(f"Wrote {len(routes_df)} routes → {ROUTES_CSV}")  # save routes

    dist_df = generate_distances_df()
    dist_df.to_csv(DISTANCES_CSV, index=False)
    print(f"Wrote {len(dist_df)} distances → {DISTANCES_CSV}")  # save distances

    storms_df = generate_storm_logs_df(start_year=2022, num_years=11, seed=123)
    storms_df.to_csv(STORMS_CSV, index=False)
    print(f"Wrote {len(storms_df)} storm events → {STORMS_CSV}")  # save storm logs

    routes_list = [r.split(";") for r in routes_df["route"]]
    ships = initialize_ships(routes_list)
    ships_df = pd.DataFrame([vars(s) for s in ships])
    ships_df.to_csv(SHIPS_CSV, index=False)
    print(f"Generated {len(ships)} ships → {SHIPS_CSV}")  # save ships

    shipments_df = generate_shipment_data(ships, num_rows=1000)
    shipments_df.to_csv(SHIPMENTS_CSV, index=False)
    print(f"Generated shipments data ({len(shipments_df)} rows) → {SHIPMENTS_CSV}")  # save shipments

if __name__ == "__main__":
    main()
