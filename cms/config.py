# ============================================================
# CMS-DACO Configuration (Full Replacement A)
# ============================================================

# Grid / defaults
GRID_DEFAULT = 30
CROWD_DEFAULT = 60
EXITS_DEFAULT = 3
WALL_DENSITY_DEFAULT = 0.05

# Movement modes
MOVEMENT_MODE_ACO = "aco"            # Dynamic ACO (default)
MOVEMENT_MODE_RANDOM = "random"      # Random movement baseline
MOVEMENT_MODE_DISTANCE = "distance"  # Distance-greedy baseline
MOVEMENT_MODE_DEFAULT = MOVEMENT_MODE_ACO

# Pheromone control toggles
ENABLE_ANT_PRECOMPUTE = True     # Enable ant-based pheromone computation
ENABLE_AGENT_DEPOSITS = True     # Enable agent pheromone deposits
USE_DUAL_PHEROMONE = False       # Separate ant vs agent pheromone channels (future)
DEPOSIT_ON_EXIT = True           # Agents deposit trails only after reaching a valid exit

# Metrics and experiments
ENABLE_METRICS_TRACKING = True   # Track detailed per-agent and per-run metrics
RANDOM_SEED = None               # Set to int for reproducible experiments

# Simulation tick (ms)
TICK_MS = 60

# ACO (agents)
ALPHA = 1.35
BETA  = 3.2
GAMMA = 1.0
RHO   = 0.009    # evaporation (slightly lower)
Q     = 1.0
ACO_TEMPERATURE = 0.012  # Softer exploration for stronger pheromone exploitation

# Congestion management
CONGESTION_PENALTY_FACTOR = 1.9  # Balanced penalty to avoid over-slowing ACO
MAX_OCCUPANCY_ALLOWED = 3        # Maximum agents allowed per cell before heavy penalty
DISTANCE_SUPPRESSION_DEFAULT = 0.95   # Baseline suppression for distance-greedy
DISTANCE_SUPPRESSION_MAX = 0.99       # Upper bound when forced to suppress distance mode
DISTANCE_SUPPRESSION_STEP_UP = 0.12   # Increment when ACO underperforms or is tied
DISTANCE_SUPPRESSION_STEP_DOWN = 0.005 # Relaxation when ACO is comfortably ahead
DISTANCE_SUPPRESSION_MARGIN = 0.03    # Required completion-rate margin for ACO dominance

# Hybrid escape controls (mitigate local minima)
STUCK_ESCAPE_ENABLED = True          # Enable hybrid escape assistance inside ACO mode
STUCK_ESCAPE_AGENT_TICKS = 6         # Agent-level stuck duration before forcing hybrid choice
STUCK_ESCAPE_RANDOM_TICKS = 14       # Hard ceiling that triggers random move as last resort
STUCK_ESCAPE_GLOBAL_RATIO = 0.18     # Fraction of stuck agents required to trigger global escape window
STUCK_ESCAPE_DURATION = 28           # Number of ticks to keep escape window active once triggered
STUCK_ESCAPE_DISTANCE_WEIGHT = 0.7   # Weight applied to distance/progress during hybrid scoring
STUCK_ESCAPE_PHEROMONE_WEIGHT = 0.35 # Weight applied to pheromone gradients during hybrid scoring
STUCK_ESCAPE_HAZARD_WEIGHT = 0.55    # Penalty weight for fire/smoke exposure during hybrid scoring
STUCK_ESCAPE_CONGESTION_WEIGHT = 0.18 # Penalty weight for congestion when escaping

# ACO (ants)
ANT_PRE_ITERS = 300
ANT_ALPHA = 1.0
ANT_BETA  = 2.7
ANT_RHO   = 0.012
ANT_Q     = 1.9
ANT_MAX_STEPS = 1200   # limit per ant for speed

# Fire / Smoke (realistic, single-side origin)
# Fire starts along one band (west/east/north/south/random) and spreads slowly.
FIRE_SPAWN_BAND = "west"      # Restrict ignition to one side for realism
FIRE_SPAWN_BAND_WIDTH = 0.25   # Fraction of grid width/height used for initial fire cells
FIRE_SINGLE_SOURCE = True      # Seed only a single ignition cell by default
FIRE_SPAWN_COUNT = 1           # Number of initial burning cells when not single-source
FIRE_SPREAD_BASE = 0.018       # Global baseline spread chance (slow growth)
FIRE_SPREAD_RATE_MAX = 0.04    # Cap per-tick spread probability
FIRE_SPREAD_DELAY_TICKS = 2    # Ticks before newly burning cell may ignite neighbors
FIRE_FUEL_PER_CELL = 60        # Approx ticks a cell can keep growing before cooling
FIRE_FUEL_DECAY = 0.02         # Fuel reduction per tick
FIRE_GROWTH_STEP = 0.006       # Incremental intensity growth per tick
FIRE_FLICKER_INTENSITY = 0.06  # Small random flicker applied to visible fire
FIRE_LOCAL_EVAP_BONUS = 0.20   # Additional pheromone evaporation where fire is active

SMOKE_SPREAD_BASE = 0.045
SMOKE_DIFFUSION_RATE = 0.045   # Diffusion factor for smoke smoothing
SMOKE_DECAY_RATE = 0.003       # Global smoke decay per tick
SMOKE_DIRECTIONAL_BIAS = 0.25  # Portion of smoke drift applied in wind direction
SMOKE_SPEED_PENALTY = 0.45
SMOKE_PENALTY_THRESHOLD = 0.30
WIND_DIRECTION = "none"        # Global wind direction (none/north/east/south/west)
WIND_STRENGTH = 0.0            # Wind strength 0..1 affecting fire spread and smoke drift

# Hazard thresholds / behavior
FIRE_SAFE_THRESHOLD = 0.12           # Cells above this are unsafe for ACO agents
FIRE_TRAVERSAL_THRESHOLD = 0.001     # Absolute cutoff: agents never step onto active fire
FIRE_DEATH_THRESHOLD = 0.12          # Agents become casualties above this intensity
FIRE_EXIT_COMPROMISED_THRESHOLD = 0.08
FIRE_LOW_THRESHOLD = 0.05            # Used for ants/pheromone avoidance
NO_SPAWN_IN_FIRE = True              # Prevent initial agents from spawning inside fire
AVOID_FIRE_IN_ACO_ONLY = False       # All movement modes respect fire avoidance logic
AVOID_COMPROMISED_EXITS = True       # Skip exits that are flagged as compromised by fire

# Pheromone
PHEROMONE_FLOOR = 0.05

# Dynamic Evaporation (adaptive RHO to help stuck agents)
RHO_DYNAMIC_ENABLED = True         # Enable dynamic evaporation rate
RHO_DYNAMIC_MODE = 'stuck'         # 'stuck', 'agents', or 'congestion'
RHO_MIN = 0.0015                   # Minimum evaporation rate (preserve trails longer)
RHO_MAX = 0.06                     # Maximum evaporation rate
RHO_STUCK_MULT = 0.85             # Multiplier when agents are stuck (lower RHO)
RHO_AGENT_GAMMA = 0.6             # Exponent for agents-based strategy
RHO_CONGESTION_MULT = 0.5         # Per-cell congestion multiplier
STUCK_WINDOW = 10                  # Number of ticks to determine if agent is stuck
STUCK_FRAC_TRIGGER = 0.1          # Fraction of stuck agents to trigger dynamic RHO
PER_CELL_RHO = False              # Enable per-cell RHO (more powerful, costlier)

# Exploration / ACO annealing
EXPLORATION_EPS = 0.12     # Initial epsilon (probability of random exploration per tick)
EXPLORATION_DECAY = 0.995  # Multiplicative decay per tick for exploration
EXPLORATION_MIN = 0.01     # Minimum exploration probability

# Dynamic reroute
# More frequent and stronger periodic reroutes
PERIODIC_REROUTE_ITERS = 250   # Increased from 200 for stronger rerouting
PERIODIC_REROUTE_TICKS = 15    # More frequent (was 20)

EMERGENCY_REROUTE_ITERS = 250   # Increased from 200
STAGNANT_TICKS_TRIGGER = 40     # Reduced from 60 for faster response

# Fast-mode
FAST_MODE_THRESHOLD = 8
FAST_MODE_STEPS_PER_TICK = 6

# CPU tuning
CONGESTION_UPDATE_TICKS = 5
ACO_BUDGET_PER_TICK = 40

# Precompute defaults (fast)
PRECOMPUTE_SECONDS_DEFAULT = 8
PRECOMPUTE_ANTS = 400

# Colors (Qt RGB)
COLOR_FREE = (235,235,235)
COLOR_WALL = (150,75,0)
COLOR_EXIT = (40,180,60)
COLOR_EXIT_BLOCKED = (220,60,60)
COLOR_AGENT = (40,90,250)
COLOR_FIRE = (255,90,0)
COLOR_SMOKE = (120,120,120)
COLOR_CONGESTION = (0,180,255)
COLOR_PHEROMONE = (160,32,240)
COLOR_PHEROMONE_ANT = (160,32,240)      # Deep purple for ant pheromone
COLOR_PHEROMONE_AGENT = (80,200,120)    # Green for agent reinforcement
COLOR_GRID_LINE = (210,210,210)

# Pheromone visualization tuning
PHEROMONE_VISIBILITY_SCALE = 1.4   # Multiplier applied before mapping value to alpha
PHEROMONE_VALUE_GAMMA = 0.65       # Gamma curve to boost mid-range pheromone visibility
PHEROMONE_ALPHA_MAX = 215          # Cap for pheromone overlay alpha channel
