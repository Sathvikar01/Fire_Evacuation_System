# ============================================================
# CMS-DACO Configuration (Full Replacement A)
# ============================================================

# Grid / defaults
GRID_DEFAULT = 30
CROWD_DEFAULT = 60
EXITS_DEFAULT = 3
WALL_DENSITY_DEFAULT = 0.05

# Movement modes
MOVEMENT_MODE_ACO = "aco"            # Dynamic ACO (default, full CMS-DACO)
MOVEMENT_MODE_RANDOM = "random"      # Random movement baseline
MOVEMENT_MODE_DISTANCE = "distance"  # Distance-greedy baseline (BFS)
MOVEMENT_MODE_ASTAR = "astar"        # Hazard-aware A* baseline (dynamic costs)
MOVEMENT_MODE_STANDARD_ACO = "standard_aco"  # ACO without DACO enhancements
MOVEMENT_MODE_DEFAULT = MOVEMENT_MODE_ACO

# Pheromone control toggles
ENABLE_ANT_PRECOMPUTE = True     # Enable ant-based pheromone computation
ENABLE_AGENT_DEPOSITS = True     # Enable agent pheromone deposits
# DEPOSIT_ON_EXIT removed — deposit is always on successful exit (research: fair comparison)
# USE_DUAL_PHEROMONE alias removed — use DUAL_PHEROMONE_ENABLED only

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
# Research reproducibility: 0.012 was near-deterministic (1/T=83, exp(400) overflow).
# 0.45 gives genuine softmax exploration while preserving pheromone exploitation.
ACO_TEMPERATURE = 0.45  # Softmax temperature — higher = more exploration (was 0.012)

# Congestion management
CONGESTION_PENALTY_FACTOR = 1.9  # Balanced penalty to avoid over-slowing ACO
DISTANCE_SUPPRESSION_DEFAULT = 0.95   # Baseline suppression for distance-greedy (used for session tracking)
# DISTANCE_SUPPRESSION_MAX/STEP_UP/DOWN/MARGIN removed — rigging logic was deleted (session_tracker no-op)

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
# Research decision: FIRE_TRAVERSAL was 0.001 (120x gap vs death 0.12) → agents froze
# while survivable low-fire (0.02) blocked them; they still died when fire grew underfoot.
# For reproducibility set traversal to 0.08 (just below SAFE/DEATH 0.12) — allows low-fire
# traversal with 0.04 safety buffer before casualty. Documented for paper comparison.
FIRE_SAFE_THRESHOLD = 0.12           # Cells above this are unsafe for ACO agents
FIRE_TRAVERSAL_THRESHOLD = 0.08      # Traversal cutoff — was 0.001, now 0.08 (buffer before 0.12 death)
FIRE_DEATH_THRESHOLD = 0.12          # Agents become casualties above this intensity
FIRE_EXIT_COMPROMISED_THRESHOLD = 0.08
FIRE_LOW_THRESHOLD = 0.05            # Used for ants/pheromone avoidance
NO_SPAWN_IN_FIRE = True              # Prevent initial agents from spawning inside fire
AVOID_COMPROMISED_EXITS = True       # Skip exits that are flagged as compromised by fire

# Ablation toggles (for systematic evaluation)
BFS_SEED_ENABLED = True              # If False, disable BFS distance seeding (ablation)
HAZARD_AWARE_ROUTING_ENABLED = True  # If False, agents ignore fire/smoke (ablation)

# Pheromone
PHEROMONE_FLOOR = 0.05

# BFS seed annealing (BUG-10): The distance-seed dominates ant deposits by
# ~300x. To let the ACO actually shape the pheromone field, the seed weight
# decays linearly over SEED_ANNEAL_ITERS iterations during precomputation.
SEED_ANNEAL_ENABLED = True
SEED_ANNEAL_ITERS = 150    # Iterations over which the BFS seed decays to zero
SEED_ANNEAL_FLOOR = 0.15   # Residual seed weight preserved after annealing

# Hazard forecasting (R4): ants use predicted fire state for their heuristic.
HAZARD_FORECAST_ENABLED = True
HAZARD_FORECAST_HORIZON = 3   # Ticks ahead to forecast fire intensity
HAZARD_FORECAST_GAMMA = 1.2   # Exponent for hazard penalty in ant heuristic

# Dual-channel multi-objective ACO (R5)
DUAL_PHEROMONE_ENABLED = False   # Separate speed vs safety pheromone channels
DUAL_PHEROMONE_BLEND = 0.5       # Weight for speed channel (1-blend for safety)

# Predictive congestion pheromone (R6)
PREDICTIVE_CONGESTION_ENABLED = True
PREDICTIVE_CONGESTION_DIFFUSION = 0.3   # Diffusion rate for negative pheromone
PREDICTIVE_CONGESTION_DECAY = 0.05      # Decay rate for negative pheromone

# Dynamic Evaporation (adaptive RHO to help stuck agents)
RHO_DYNAMIC_ENABLED = True         # Enable dynamic evaporation rate
RHO_DYNAMIC_MODE = 'stuck'         # 'stuck', 'agents', or 'congestion'
RHO_MIN = 0.0015                   # Minimum evaporation rate (preserve trails longer)
RHO_MAX = 0.06                     # Maximum evaporation rate
RHO_STUCK_MULT = 0.70             # Multiplier when agents are stuck (was 0.85, too weak — only 15% drop, now 30%)ier when agents are stuck (lower RHO)
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
COLOR_GRID_LINE = (210,210,210)

# Pheromone visualization tuning
PHEROMONE_VISIBILITY_SCALE = 1.4   # Multiplier applied before mapping value to alpha
PHEROMONE_VALUE_GAMMA = 0.65       # Gamma curve to boost mid-range pheromone visibility
PHEROMONE_ALPHA_MAX = 215          # Cap for pheromone overlay alpha channel
