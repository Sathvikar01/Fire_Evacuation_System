import numpy as np
from config import (
    FIRE_SPREAD_BASE,
    FIRE_SPREAD_RATE_MAX,
    FIRE_SPREAD_DELAY_TICKS,
    FIRE_GROWTH_STEP,
    FIRE_FUEL_PER_CELL,
    FIRE_FUEL_DECAY,
    FIRE_SAFE_THRESHOLD,
    FIRE_LOW_THRESHOLD,
    FIRE_FLICKER_INTENSITY,
    SMOKE_SPREAD_BASE,
    SMOKE_DIFFUSION_RATE,
    SMOKE_DECAY_RATE,
    SMOKE_DIRECTIONAL_BIAS,
    WIND_DIRECTION,
    WIND_STRENGTH,
)
from .grid import WALL


def _wind_vector(direction: str) -> tuple[int, int]:
    mapping = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }
    return mapping.get(direction, (0, 0))


def _apply_roll_mask(arr: np.ndarray, dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift array without wrap-around and return validity mask."""
    shifted = np.roll(arr, shift=(dr, dc), axis=(0, 1))
    mask = np.ones_like(arr, dtype=bool)
    if dr > 0:
        mask[:dr, :] = False
    elif dr < 0:
        mask[dr:, :] = False
    if dc > 0:
        mask[:, :dc] = False
    elif dc < 0:
        mask[:, dc:] = False
    shifted[~mask] = 0.0
    return shifted, mask


def step_fire_and_smoke(
    types,
    fire,
    smoke,
    rng,
    fire_params: dict | None = None,
    smoke_params: dict | None = None,
    wind_params: dict | None = None,
):
    """Advance fire/smoke with configurable spread, wind bias, and diffusion."""
    fire_cfg = {
        "spread_base": FIRE_SPREAD_BASE,
        "spread_rate_max": FIRE_SPREAD_RATE_MAX,
        "spread_delay": FIRE_SPREAD_DELAY_TICKS,
        "growth_step": FIRE_GROWTH_STEP,
        "fuel_per_cell": FIRE_FUEL_PER_CELL,
        "fuel_decay": FIRE_FUEL_DECAY,
        "safe_threshold": FIRE_SAFE_THRESHOLD,
        "low_threshold": FIRE_LOW_THRESHOLD,
        "flicker": FIRE_FLICKER_INTENSITY,
    }
    smoke_cfg = {
        "base_spread": SMOKE_SPREAD_BASE,
        "diffusion_rate": SMOKE_DIFFUSION_RATE,
        "decay_rate": SMOKE_DECAY_RATE,
        "directional_bias": SMOKE_DIRECTIONAL_BIAS,
    }
    wind_cfg = {
        "direction": WIND_DIRECTION,
        "strength": WIND_STRENGTH,
    }
    if fire_params:
        fire_cfg.update({k: v for k, v in fire_params.items() if k in fire_cfg})
    if smoke_params:
        smoke_cfg.update({k: v for k, v in smoke_params.items() if k in smoke_cfg})
    if wind_params:
        wind_cfg.update({k: v for k, v in wind_params.items() if k in wind_cfg})

    R, C = fire.shape
    new_fire = fire.copy()
    new_smoke = smoke.copy()

    spread_ready = fire_cfg["safe_threshold"] + fire_cfg["spread_delay"] * fire_cfg["growth_step"]
    max_new = max(1, int((R * C) / max(1, fire_cfg["fuel_per_cell"])))
    new_ignitions = 0

    wind_dir = str(wind_cfg.get("direction", "none")).lower()
    wind_strength = float(wind_cfg.get("strength", 0.0))
    wind_vec = _wind_vector(wind_dir)

    cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(R):
        for c in range(C):
            if types[r, c] == WALL:
                continue

            intensity = fire[r, c]
            if intensity <= 0.01:
                continue

            growth = fire_cfg["growth_step"] * (1.0 - intensity)
            new_fire[r, c] = min(1.0, intensity + growth)
            if new_fire[r, c] >= 0.98:
                new_fire[r, c] = max(0.92, new_fire[r, c] - fire_cfg["fuel_decay"])

            if intensity < spread_ready:
                continue

            base_prob = min(
                fire_cfg["spread_base"] + intensity * 0.05,
                fire_cfg["spread_rate_max"],
            )
            for dr, dc in cardinal_dirs:
                if new_ignitions >= max_new:
                    break
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and types[nr, nc] != WALL:
                    if new_fire[nr, nc] > fire_cfg["low_threshold"]:
                        continue
                    wind_bias = 1.0
                    if wind_strength > 0.0 and wind_vec != (0, 0):
                        dot = dr * wind_vec[0] + dc * wind_vec[1]
                        if dot > 0:
                            wind_bias += wind_strength * 0.9 * dot
                        elif dot < 0:
                            wind_bias -= wind_strength * 0.45 * (-dot)
                    prob = base_prob * wind_bias
                    prob = max(0.0, min(fire_cfg["spread_rate_max"], prob))
                    if rng.random() < prob:
                        upper = min(0.35, fire_cfg["safe_threshold"] + 0.15)
                        new_fire[nr, nc] = float(rng.uniform(fire_cfg["safe_threshold"], upper))
                        new_smoke[nr, nc] = max(new_smoke[nr, nc], 0.45)
                        new_ignitions += 1

    fires = np.argwhere(new_fire > fire_cfg["low_threshold"])
    for r, c in fires:
        new_smoke[r, c] = max(new_smoke[r, c], 0.7)
        for dr, dc in cardinal_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and types[nr, nc] != WALL:
                if new_smoke[nr, nc] < 0.3 and rng.random() < smoke_cfg["base_spread"]:
                    bias = 1.0
                    if wind_strength > 0.0 and wind_vec != (0, 0):
                        dot = dr * wind_vec[0] + dc * wind_vec[1]
                        if dot > 0:
                            bias += 0.6 * wind_strength * dot
                    new_smoke[nr, nc] = max(new_smoke[nr, nc], 0.3 * bias)

    diffusion_rate = max(0.0, float(smoke_cfg.get("diffusion_rate", 0.0)))
    if diffusion_rate > 0.0:
        neighbor_sum = np.zeros_like(new_smoke)
        neighbor_count = np.zeros_like(new_smoke, dtype=np.int16)
        for dr, dc in cardinal_dirs:
            shifted, mask = _apply_roll_mask(new_smoke, dr, dc)
            neighbor_sum += shifted
            neighbor_count += mask.astype(np.int16)
        valid_mask = neighbor_count > 0
        avg = np.zeros_like(new_smoke)
        avg[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask]
        new_smoke += diffusion_rate * (avg - new_smoke)

    directional_bias = max(0.0, float(smoke_cfg.get("directional_bias", 0.0)))
    if directional_bias > 0.0 and wind_strength > 0.0 and wind_vec != (0, 0):
        drift = min(0.95, directional_bias * wind_strength)
        drifted, _ = _apply_roll_mask(new_smoke, wind_vec[0], wind_vec[1])
        new_smoke = (1.0 - drift) * new_smoke + drift * drifted

    decay = min(0.95, max(0.0, float(smoke_cfg.get("decay_rate", 0.0))))
    if decay > 0.0:
        new_smoke *= (1.0 - decay)

    flicker = max(0.0, float(fire_cfg.get("flicker", 0.0)))
    if flicker > 0.0:
        flicker_mask = new_fire > fire_cfg["low_threshold"]
        if np.any(flicker_mask):
            noise = (rng.random(size=new_fire.shape) - 0.5) * flicker
            new_fire = np.clip(new_fire + noise * flicker_mask, 0.0, 1.0)

    new_smoke = np.clip(new_smoke, 0.0, 1.0)
    return new_fire, new_smoke
