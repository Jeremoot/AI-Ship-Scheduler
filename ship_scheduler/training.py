# ship_scheduler/training.py

import pandas as pd
from .env import ShipSchedulingEnv
from .agent import A2CAgent
from .config import PARAMS, RESULTS, LEARNING_CURVE, YEARS, LOG_CSV
from .config import RESULTS

def main():
    df = pd.read_csv(PARAMS)
    best_row = df.loc[df["mean_norm_improvement"].idxmax()]
    alpha, beta, gamma = best_row["alpha"], best_row["beta"], best_row["gamma"]
    print(f"Using tuned params: ALPHA={alpha:.4f}, BETA={beta:.4f}, GAMMA={gamma:.4f}\n")

    env_warm = ShipSchedulingEnv()
    env_warm.ALPHA = alpha
    env_warm.BETA  = beta
    env_warm.GAMMA = gamma

    warm_agent = A2CAgent(env_warm)
    initial_bests = {}
    state = env_warm.reset()

    while len(initial_bests) < len(env_warm.ships):
        action = warm_agent.select_action(state)

        state, _, done, _ = env_warm.step(action)

        # as soon as a ship finishes its first loop, record it
        for sid, ship in env_warm.ships.items():
            if sid not in initial_bests and ship.best_time:
                initial_bests[sid] = sum(ship.best_time)

        if done:
            break

    assert len(initial_bests) == len(env_warm.ships), "Warm-up failed: not every ship completed one loop"

    # train on fresh env
    env_train = ShipSchedulingEnv(years=YEARS)
    env_train.ALPHA = alpha
    env_train.BETA  = beta
    env_train.GAMMA = gamma

    agent = A2CAgent(env_train)
    agent.train(env_train)

    final_bests = {
        sid: sum(ship.best_time)
        for sid, ship in env_train.ships.items()
        if ship.best_time
    }

    rows = []
    for sid in sorted(env_train.ships):
        ship = env_train.ships[sid]   
        ib = initial_bests[sid]
        fb = final_bests.get(sid, None)
        imp = ib - fb if fb is not None else None
        loop_no = len(list(ship.all_time))
        
        ratio = (imp / ib * 100) if (ib and imp is not None) else None

        rows.append({
            "ship_id":           sid,
            "improvement_h":     imp,
            "improvement_perc": ratio,
            "time_in_queue":     env_train.time_in_queue[sid],
            "time_in_service":   env_train.time_in_service[sid],
            "time_in_storm":     env_train.time_in_storm[sid],
            "loops":             loop_no,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS, index=False)
    pd.DataFrame(agent.learning_curve).to_csv(LEARNING_CURVE, index=False)
    print(f"\nSaved results")
    print(df_out)

    log_df = env_train.get_log_df()
    log_df['berth_id'] = log_df['berth_id'].astype('Int64')

    print(log_df)
    log_df.to_csv(LOG_CSV, index=False)
    print(f"Event log written to {LOG_CSV}")

if __name__ == "__main__":
    main()
