# 🚢 AI-Ship-Scheduler

This study presents a reinforcement learning (RL) framework for real-time maritime scheduling across port rotations. Traditional approaches rely on static timetables or heuristics and often fail to respond effectively to congestion and weather disruptions. In contrast, the proposed method employs a self-comparative reward mechanism, where each ship is incentivised to improve its own historical best route completion time (RCT) rather than adhere to external schedules. An Advantage Actor-Critic (A2C) agent is trained online within an event-driven simulation environment that models storm-affected segments and berth contention across global ports. The agent continuously refines its queued ship selection and departure timing strategies throughout a single, multi-year simulation without episodic resets. Results show that the trained policy outperforms first-come-first-served (FCFS) baselines by 23%. These findings suggest that self-referenced RL offers a resilient and adaptive scheduling solution for dynamic maritime operations.

For full methodology, experiments, and results, see `report.pdf`

## Setup

**Clone the repo**

  ```bash
  git clone https://github.com/Jeremoot/AI-Ship-Scheduler.git

  cd AI-Ship-Scheduler
```
<br/>

**Create and activate a virtual environment**<br/>
```bash
python -m venv shipscheduler

source shipscheduler/bin/activate 
# or shipscheduler\Scripts\activate on Windows

pip install -r requirements.txt
```


## Running the Code
Edit the YEARS value in ship_scheduler/config.py to control how long the simulation runs (max 24 years).
<br/>

- To train the A2C agent:<br/>
```bash
python -m ship_scheduler.training
```

- To run the FCFS baseline:<br/>
```bash
python -m ship_scheduler.baseline
```

Output files will appear in ship_scheduler/results/.
<br/>
## Visualisation
Go to ship_scheduler/visualise.ipynb for plotting and analysis

## Contributors
Farah Qistina binti Alnizam<br/>
Jeremy Klement Jim
