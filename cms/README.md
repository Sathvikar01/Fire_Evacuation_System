# Fire Evacuation System (CMS-DACO)

This project simulates building evacuation during an emergency using Agent-based methods and Ant Colony Optimization (ACO). The user interface is built with PyQt5 so you can configure the grid, tune ACO settings, and watch the evacuation progress in real-time.

Key features
- ACO-based movement mode (adaptive pheromone trails)
- Distance-based and Random baseline movement modes
- Fire and smoke spread model with configurable wind and spread rates
- Detailed performance metrics (completion, casualties, congestion, evacuation time)
- Precompute ants for faster initial pheromone seeding

Quick start
1. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv venv
; .\venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run application from repository root:
```powershell
python main.py
```

Configuration
- Most configuration lives in `config.py` — change defaults there to tune movement, ACO, fire, or visualization.
- `GridSpec` in `core/grid.py` sets grid size, initial crowd, number of exits and wall density.

Developer notes
- Run `main.py` or import `Simulation` from `core/simulation.py` to run programmatically.
- To change movement mode call `Simulation(spec, movement_mode=...)` with `aco`, `distance`, or `random`.
- The UI lives under `ui/` — modifies Qt widgets, controls and graphing of results.

Tests
- There are no automated tests yet; please consider adding unit tests in `core/` for core algorithms.

Contribution
- Create issues, forks and PRs in https://github.com/Sathvikar01/Fire_Evacuation_System

License
- Please add or update `LICENSE` to the repo as desired.
