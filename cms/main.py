import sys
from PyQt5.QtWidgets import QApplication

from core.grid import GridSpec
from core.simulation import Simulation
from ui.ui_main import UIMain
from config import (
    GRID_DEFAULT, CROWD_DEFAULT, EXITS_DEFAULT, WALL_DENSITY_DEFAULT, 
    PRECOMPUTE_SECONDS_DEFAULT, PRECOMPUTE_ANTS,
    MOVEMENT_MODE_DEFAULT, RANDOM_SEED
)

def main():
    app = QApplication(sys.argv)

    spec = GridSpec(
        rows=GRID_DEFAULT,
        cols=GRID_DEFAULT,
        crowd=CROWD_DEFAULT,
        exits=EXITS_DEFAULT,
        wall_density=WALL_DENSITY_DEFAULT,
    )

    sim = Simulation(spec, seed=RANDOM_SEED, movement_mode=MOVEMENT_MODE_DEFAULT)

    # quick precompute (seed + small ACO refine) for fast startup
    sim.start_precompute(total_seconds=PRECOMPUTE_SECONDS_DEFAULT, iterations_total=PRECOMPUTE_ANTS)

    def on_regen(new_spec):
        was_running = sim.running
        sim.pause()
        sim.regenerate(new_spec)
        # optional quick precompute after regen
        sim.start_precompute(total_seconds=PRECOMPUTE_SECONDS_DEFAULT, iterations_total=PRECOMPUTE_ANTS)
        if was_running:
            sim.start()

    ui = UIMain(sim, on_regen)
    ui.resize(1220, 780)
    ui.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
