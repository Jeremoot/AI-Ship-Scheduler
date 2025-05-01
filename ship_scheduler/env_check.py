# env_check.py
from .env    import ShipSchedulingEnv
from .config import LOG_CSV

# ship
SHIP_ID = 1

env   = ShipSchedulingEnv()
state = env.reset()
done  = False
while not done:
    #FCFS
    state, _, done, _ = env.step({'departure_choice': 0, 'berth_choice': 0})

log_df = env.get_log_df()

print(log_df)
log_df.to_csv(LOG_CSV, index=False)
print(f"Event log written to {LOG_CSV}")
