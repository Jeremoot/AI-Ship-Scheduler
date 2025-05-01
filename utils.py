# utils.py
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from .entities import Ship, PathSegment, StormZone


def initialize_ships(routes: List[List[str]]) -> List[Ship]:
    ships = []
    ship_id = 1
    for route in routes:
        big = len(route) >= 4
        looped_route = len(route) in [2, 5]
        max_capacity = 12000 if big else int(12000 * 0.6)
        containers = random.randint(int(0.5 * max_capacity), max_capacity)
        ships.append(Ship(
            id=ship_id,
            route=route,
            big=big,
            looped_route=looped_route,
            containers=containers
        ))
        ship_id += 1
    return ships


def generate_shipment_data(ships: List[Ship], num_rows: int = 200) -> pd.DataFrame:
    records = []
    for _ in range(num_rows):
        row = {}
        for ship in ships:
            max_capacity = 12000 if ship.big else int(12000 * 0.6)
            unload = random.randint(0, ship.containers)
            remaining = ship.containers - unload
            free_space = max_capacity - remaining
            load = random.randint(0, free_space)
            row[f"Ship_{ship.id}_start"]   = ship.containers
            row[f"Ship_{ship.id}_unload"]  = unload
            row[f"Ship_{ship.id}_load"]    = load
            ship.containers = remaining + load
        records.append(row)
    return pd.DataFrame(records)


# static list of predefined routes
ROUTES_LIST = [
    ["Antwerp", "Suez"],
    ["Suez", "Singapore"],
    ["Singapore", "Shanghai"],
    ["Singapore", "Panama"],
    ["Shanghai", "Panama"],
    ["Panama", "Antwerp"],

    ["Antwerp","Suez","Singapore"],
    ["Suez","Singapore","Shanghai"],
    ["Suez","Singapore","Panama"],
    ["Singapore","Shanghai","Panama"],
    ["Singapore","Panama","Antwerp"],
    ["Shanghai","Panama","Antwerp"],
    ["Panama","Antwerp","Suez"],

    ["Antwerp","Suez","Singapore","Shanghai"],
    ["Suez","Singapore","Shanghai","Panama"],
    ["Suez","Singapore","Panama","Antwerp"],
    ["Singapore","Panama","Antwerp","Suez"],
    ["Singapore","Shanghai","Panama","Antwerp"],
    ["Shanghai","Panama","Antwerp","Suez"],
    ["Panama","Antwerp","Suez","Singapore"],

    ["Antwerp","Suez","Singapore","Shanghai","Panama"],
    ["Panama","Shanghai","Singapore","Suez","Antwerp"]
]

def generate_routes_df() -> pd.DataFrame:
    return pd.DataFrame({"route": [";".join(r) for r in ROUTES_LIST]})


DIST_RAW = {
    ("Antwerp","Suez"): [
        PathSegment("Antwerp","AWCP", 425.00749114401043, None),
        PathSegment("AWCP","Suez", 6566.051775985024, None)
    ],
    ("Suez","Singapore"): [
        PathSegment("Suez","Singapore", 7353.9178699623535, None)
    ],
    ("Singapore","Shanghai"): [
        PathSegment("Singapore","SGWNP",  768.0609283805868, None),
        PathSegment("SGWNP","WNPCP",    1887.583311742735, "WNP"),
        PathSegment("WNPCP","CP",            530.3300858899106, None),
        PathSegment("CP","Shanghai",           628.3218763304216, None)
    ],
    ("Singapore","Panama"): [
        PathSegment("Singapore","SGWNP",  768.0609283805868, None),
        PathSegment("SGWNP","WNPCP",    1887.583311742735, "WNP"),
        PathSegment("WNPCP","CP",            530.3300858899106, None),
        PathSegment("CP","CPENP",        11453.454605845673, None),
        PathSegment("CPENP","ENPPM",     2291.3575625136305, "ENP"),
        PathSegment("ENPPM","Panama",         665.5289175033902, None),
    ],
    ("Shanghai","Panama"): [
        PathSegment("Shanghai","CP",           628.3218763304216, None),
        PathSegment("CP","CPENP",        11453.454605845673, None),
        PathSegment("CPENP","ENPPM",     2291.3575625136305, "ENP"),
        PathSegment("ENPPM","Panama",         665.5289175033902, None),
    ],
    ("Panama","Antwerp"): [
        PathSegment("Panama","NAR",      2148.0692247245806, "NA"),
        PathSegment("NAR","E",      5632.47758365195,    None),
        PathSegment("E","AW",      1987.772473344937,   None),
        PathSegment("AW","Antwerp",       425.00749114401043, None)
    ],
}

def generate_distances_df() -> pd.DataFrame:
    local = dict(DIST_RAW)
    for (a, b), segs in list(local.items()):
        rev = [PathSegment(s.to, s.frm, s.distance, s.storm) for s in reversed(segs)]
        local[(b, a)] = rev

    aggregated = []
    for (route_from, route_to), segs in local.items():
        # sum total distance
        total_distance = sum(s.distance for s in segs)

        storms = sorted({s.storm for s in segs if s.storm is not None})

        storm_distance = sum(s.distance for s in segs if s.storm is not None)

        storm_ratio = storm_distance / total_distance if total_distance > 0 else 0.0

        aggregated.append({
            "route_from":     route_from,
            "route_to":       route_to,
            "total_distance": total_distance,
            "storms":         storms,
            "storm_ratio":    storm_ratio
        })

    return pd.DataFrame(aggregated)


zones = {
    "WNP": StormZone("WNP", months=(6, 12)),  # June–Dec
    "ENP": StormZone("ENP", months=(6, 10)),  # June–Oct
    "NA":  StormZone("NA",  months=(8, 10)),  # Aug–Oct
}

def generate_storm_logs_df(start_year: int, num_years: int = 4, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    recs = []
    for year in range(start_year, start_year + num_years):
        for zone in zones.values():
            start_season = datetime(year, zone.months[0], 1)
            if zone.months[1] == 12:
                end_season = datetime(year, 12, 31)
            else:
                end_season = datetime(year, zone.months[1] + 1, 1) - timedelta(days=1)
            num_storms = random.randint(1,5)
            for _ in range(num_storms):
                days = (end_season - start_season).days
                off = random.randint(0, days)
                sdt = start_season + timedelta(days=off)
                dur = random.randint(1,14)
                edt = sdt + timedelta(days=dur)
                recs.append({"zone":zone.name, "start":sdt, "end":edt})
    return pd.DataFrame(recs)