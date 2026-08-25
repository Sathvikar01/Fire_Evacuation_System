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

    wind_dir = str(wind_cfg.get("direction", "none")).lower()
    wind_strength = float(wind_cfg.get("strength", 0.0))
    wind_vec = _wind_vector(wind_dir)

    cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # --- Vectorized fire growth ---
    not_wall = (types != WALL)
    intensity = fire
    burning_mask = (intensity > 0.01) & not_wall
    if burning_mask.any():
        growth = fire_cfg["growth_step"] * (1.0 - intensity)
        new_fire = np.where(
            burning_mask,
            np.minimum(1.0, intensity + growth),
            new_fire,
        )
        # fuel decay at near-saturation
        sat = new_fire >= 0.98
        if sat.any():
            new_fire = np.where(sat, np.maximum(0.92, new_fire - fire_cfg["fuel_decay"]), new_fire)

    # --- Vectorized fire spread ---
    spread_source = (intensity >= spread_ready) & not_wall
    if spread_source.any() and max_new > 0:
        base_prob = np.minimum(
            fire_cfg["spread_base"] + intensity * 0.05,
            fire_cfg["spread_rate_max"],
        )
        upper = min(0.35, fire_cfg["safe_threshold"] + 0.15)

        ignition_mask_total = np.zeros_like(new_fire, dtype=bool)
        ignition_values = np.zeros_like(new_fire, dtype=new_fire.dtype)

        for dr, dc in cardinal_dirs:
            if not spread_source.any():
                break
            if ignition_mask_total.sum() >= max_new:
                break
            # Shift source intensity and base_prob toward neighbor direction.
            src_intensity, src_mask = _apply_roll_mask(intensity, dr, dc)
            src_base, _ = _apply_roll_mask(base_prob, dr, dc)
            src_ready, _ = _apply_roll_mask(spread_source, dr, dc)

            # Neighbor cell is a candidate to ignite
            valid_neighbor = src_mask & not_wall
            # Only ignite cells not already burning above low threshold
            target_low = new_fire <= fire_cfg["low_threshold"]
            candidate = valid_neighbor & src_ready & target_low
            if not candidate.any():
                continue

            wind_bias = np.ones_like(new_fire)
            if wind_strength > 0.0 and wind_vec != (0, 0):
                dot = dr * wind_vec[0] + dc * wind_vec[1]
                if dot > 0:
                    wind_bias += wind_strength * 0.9 * dot
                elif dot < 0:
                    wind_bias -= wind_strength * 0.45 * (-dot)

            # FIX: use source base_prob (src_base) not target base_prob — was ~2.2x too slow
            prob = np.clip(src_base * wind_bias, 0.0, fire_cfg["spread_rate_max"])
            # Per-direction independent rolls (previously shared across directions → correlated)
            rolls = rng.random(size=new_fire.shape)
            ignites = candidate & (rolls < prob)

            # Apply ignitions not already ignited this tick
            new_ignitions = ignites & ~ignition_mask_total
            if new_ignitions.any():
                ignition_mask_total |= new_ignitions
                ignition_values = np.where(
                    new_ignitions,
                    rng.uniform(fire_cfg["safe_threshold"], upper, size=new_fire.shape).astype(new_fire.dtype),
                    ignition_values,
                )
                if ignition_mask_total.sum() >= max_new:
                    # Trim excess ignitions (keep first max_new by random subset)
                    total_idx = np.argwhere(ignition_mask_total)
                    if len(total_idx) > max_new:
                        keep = total_idx[rng.choice(len(total_idx), size=max_new, replace=False)]
                        trimmed = np.zeros_like(ignition_mask_total)
                        for r, c in keep:
                            trimmed[int(r), int(c)] = True
                        ignition_mask_total = trimmed
                        ignition_values = np.where(ignition_mask_total, ignition_values, 0.0)
                    break

        if ignition_mask_total.any():
            # new_fire[ignition_mask_total] = ignition_values[ignition_mask_total]
            new_fire = np.where(ignition_mask_total, ignition_values, new_fire)
            new_smoke = np.where(
                ignition_mask_total,
                np.maximum(new_smoke, 0.45),
                new_smoke,
            )

    # --- Smoke from active fire cells (vectorized) ---
    fires = new_fire > fire_cfg["low_threshold"]
    if fires.any():
        new_smoke = np.where(fires, np.maximum(new_smoke, 0.7), new_smoke)
        # Smoke spread to neighbors of fire cells — per-direction rolls for independence
        for dr, dc in cardinal_dirs:
            src_fire, mask = _apply_roll_mask(fires, dr, dc)
            target = src_fire & mask & (new_smoke < 0.3) & not_wall
            if not target.any():
                continue
            bias = 1.0
            if wind_strength > 0.0 and wind_vec != (0, 0):
                dot = dr * wind_vec[0] + dc * wind_vec[1]
                if dot > 0:
                    bias += 0.6 * wind_strength * dot
                elif dot < 0:
                    # Symmetric headwind penalty (previously only tailwind) for consistency with fire wind
                    bias -= 0.3 * wind_strength * (-dot)
                    bias = max(0.2, bias)
            smoke_rolls = rng.random(size=new_smoke.shape)
            spread_mask = target & (smoke_rolls < smoke_cfg["base_spread"])
            new_smoke = np.where(
                spread_mask,
                np.maximum(new_smoke, np.minimum(0.3 * bias, 1.0)),
                new_smoke,
            )

    diffusion_rate = max(0.0, float(smoke_cfg.get("diffusion_rate", 0.0)))
    if diffusion_rate > 0.0:
        neighbor_sum = np.zeros_like(new_smoke)
        neighbor_count = np.zeros_like(new_smoke, dtype=np.int16)
        for dr, dc in cardinal_dirs:
            shifted, mask = _apply_roll_mask(new_smoke, dr, dc)
            # Exclude WALL cells from diffusion average (previously diluted smoke near walls)
            wall_shifted, _ = _apply_roll_mask(not_wall.astype(np.float32), dr, dc)
            # wall_shifted is 1 where neighbor not wall, 0 otherwise
            valid = mask & (wall_shifted > 0.5)
            neighbor_sum += np.where(valid, shifted, 0.0)
            neighbor_count += valid.astype(np.int16)
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
