# tune_hyperparams.py
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import pandas as pd

from .env import ShipSchedulingEnv
from .agent import A2CAgent

from .config import PARAMS, YEARS


def objective(params):
    α, β, γ = params

    env_warm = ShipSchedulingEnv()
    env_warm.ALPHA = α
    env_warm.BETA  = β
    env_warm.GAMMA = γ

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


    # training
    env_train = ShipSchedulingEnv()
    env_train.ALPHA = α
    env_train.BETA  = β
    env_train.GAMMA = γ
    env_warm.years = YEARS

    agent = A2CAgent(env_train)
    agent.train(env_train)

    final_bests = {
        sid: sum(ship.best_time)
        for sid, ship in env_train.ships.items()
        if ship.best_time
    }

    # compute normalised per-ship improvement
    improvements = []
    for sid, ib in initial_bests.items():
        fb = final_bests.get(sid, ib)
        if np.isfinite(ib) and ib > 0:
            improvements.append((ib - fb) / ib)

    mean_norm = float(np.mean(improvements)) if improvements else 0.0

    return -mean_norm


if __name__ == "__main__":
    space = [
        Real(0.0, 0.5, name="alpha"), # queue penalty
        Real(0.0, 2.0, name="beta"), # storm penalty
        Real(0.0, 1.0, name="gamma"), # leg-speed reward
    ]

    res = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",
        n_calls=30,
        n_random_starts=5,
        random_state=42,
    )

    df = pd.DataFrame(res.x_iters, columns=['alpha','beta','gamma'])
    df['mean_norm_improvement'] = [-v for v in res.func_vals]
    df.to_csv(PARAMS, index=False)    

    print("BEST (alpha, beta, gamma)           =", res.x)
    print("BEST mean normalized improvement =", -res.fun)
