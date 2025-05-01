import pandas as pd
from .env import ShipSchedulingEnv
from .config import BASELINE, YEARS

def run_baseline():
    env = ShipSchedulingEnv(years=YEARS)

    state = env.reset()
    while any(len(ship.best_time) == 0 for ship in env.ships.values()):
        action = {'delay': 0.0} if state['decision_type'] == 'departure' else {'berth_choice': 0}
        state, _, done, _ = env.step(action)
        if done:
            break

    initial_bests = {sid: sum(ship.best_time) for sid, ship in env.ships.items()}

    while True:
        action = {'delay': 0.0} if state['decision_type'] == 'departure' else {'berth_choice': 0}
        state, _, done, _ = env.step(action)
        if done:
            break

    final_bests = {sid: sum(ship.best_time) for sid, ship in env.ships.items()}

    rows = []
    for sid in sorted(env.ships):
        ship = env.ships[sid]
        ib = initial_bests[sid]
        fb = final_bests.get(sid)
        imp = ib - fb if fb is not None else None
        loops = len(list(ship.all_time))
        ratio = (imp / ib * 100) if (ib and imp is not None) else None

        rows.append({
            "ship_id": sid,
            "improvement_h": imp,
            "improvement_perc": ratio,
            "time_in_queue": env.time_in_queue[sid],
            "time_in_service": env.time_in_service[sid],
            "time_in_storm": env.time_in_storm[sid],
            "loops": loops,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(BASELINE, index=False)
    print("\nSaved baseline")
    print(df_out)

if __name__ == '__main__':
    run_baseline()
