# AI-Ship-Scheduler
## Setup
**Clone the repo**

git clone https://github.com/Jeremoot/AI-Ship-Scheduler.git

cd AI-Ship-Scheduler
<br/>
<br/>
<br/>
**Create and activate a virtual environment**

python -m venv shipscheduler

source shipscheduler/bin/activate (or shipscheduler\Scripts\activate on Windows)

pip install -r requirements.txt
<br/>
<br/>

## Running the Code
Edit the YEARS value in ship_scheduler/config.py to control how long the simulation runs (max 10 years).
<br/>
<br/>
‼️Make sure you are in the directory above ship_scheduler.
<br/>
<br/>

To train the A2C agent:

python -m ship_scheduler.training

To run the FCFS baseline:

python -m ship_scheduler.baseline
<br/>
<br/>
<br/>
Output files will appear in ship_scheduler/results/.
<br/>
<br/>
## Visualisation
Open visualise.ipynb for plotting and analysis


## Contributors
Farah Qistina binti Alnizam
Jeremy Klement Jim
