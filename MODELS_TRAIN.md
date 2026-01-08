# RLMAPF Training Configs - Model Differences and Guidance

This document summarizes the key differences between training configs, maps to train on, and how major parameters shape behavior and outcomes. It also suggests a next model idea.

## Baseline vs CNN (general)
- **Baseline**: MLP policy on flattened observations (`use_cnn_observation: false`). Typically larger fully connected layers (e.g., `[1024, 512, 512]`). Simpler observation, no spatial conv filters.
- **CNN**: Convolutional encoder on grid observations (`use_cnn_observation: true`, `conv_filters` defined). Typically smaller FC head (e.g., `[512, 256]` or `[512, 512, 256]`) after conv stack. Better at spatial patterns (obstacles, other agents) but heavier compute.
- Common PPO/RLlib stack; differences are in the model and observation space.

## CNN Variants (key differences)
- **cnn** (vanilla CNN): Conv filters (e.g., 32/64/64), FC head 512/256, `use_d_star_lite: false`, standard PPO reward shaping.
- **cnn_iterative_dstar / cnn_iterative_dstar_different**: CNN + D* Lite enabled, iterative replanning (`use_d_star_lite: true`, `d_star_iterations` > 1), often with congestion weighting/path progress. Better for dynamic rerouting.
- **cnn_congestion_dstar**: CNN + D* Lite with congestion cost emphasis (`d_star_congestion_weight` > 1), often with higher collision penalties. Targets crowded layouts.
- **cnn_dispersal**: CNN + D* Lite with path progress reward (`d_star_path_progress_weight: 0.1`), multiple maps emphasizing spread-out routes; collision penalty higher (0.5), step_cost moderate. Aims to reduce clustering and encourage path shortening.
- **cnn_periphery**: CNN tuned for edge/perimeter spawning/goals; typically uses maps that favor border routes. May keep D* on or off; key idea is bias toward periphery navigation (config differs mainly by map set and potentially collision/step costs).

## Suggested training maps (3–4 each)
- **baseline**: `big_empty_1-100a-20x20`, `basic2_1-100a-35x35`, `pprai_1-20a-42x39`.
- **cnn** (vanilla): `pprai_1-20a-42x39`, `basic3_1-100a-45x35`, `maze1_1-100a-30x30`.
- **cnn_iterative_dstar / cnn_iterative_dstar_different**: `pprai_1-20a-42x39`, `maze1_1-100a-30x30`, `corridors_10-10a-32x32` (or `basic2_1-100a-35x35`).
- **cnn_congestion_dstar**: `pprai_1-20a-42x39` (aisles), `slots_10-10a-39x29` (parking/slots), `corridors_10-10a-32x32` (chokepoints).
- **cnn_dispersal**: `basic2_1-100a-35x35`, `maze1_1-100a-30x30`, `grid_1-100a-20x23` (or `pprai_1-20a-42x39`).
- **cnn_periphery**: `big_empty_1-100a-20x20`, `basic3_1-100a-45x35`, `pprai_1-20a-42x39` (edges vs aisles).

## Parameter cheatsheet (influence on training/behavior)
- **use_cnn_observation**: Enables spatial grid input; needed for conv filters. Improves spatial awareness; more compute.
- **conv_filters**: Depth/size of conv stack. More/larger filters capture richer patterns but slow training.
- **fcnet_hiddens**: Capacity of FC head. Larger = more expressive but slower and risk of overfit.
- **use_d_star_lite**: Enable D* Lite path planning features and extra channels (when CNN). Helps rerouting around congestion/obstacles; small overhead.
- **d_star_iterations**: More iterations = more congestion-aware replanning; higher compute per step.
- **d_star_congestion_weight**: Higher weight penalizes crowded cells; encourages detours; too high can over-avoid and lengthen paths.
- **d_star_path_progress_weight**: Positive weight rewards shortening D* path each step; promotes progress and dispersal. Too high can cause jitter if paths fluctuate.
- **collision_penalty**: Cost per collision. Higher discourages collisions but can make agents overly cautious; too low yields reckless behavior.
- **step_cost**: Cost per step. Higher pushes faster goal seeking but can encourage risky shortcuts; too low can lead to wandering.
- **wait_cost_multiplier**: Multiplier on step cost for waits. >1 discourages waiting (can increase collisions); <1 encourages yielding; =1 neutral.
- **reward_closer_to_goal_final / reward_final_d_star**: Provide positive rewards when reaching goal or reducing D* path at the end. Good for sparse rewards; balance to avoid myopic behavior.
- **penalize_collision / penalize_waiting / penalize_steps**: Gates for applying penalties; typically on for collisions and step costs; toggle as needed for shaping.
- **use_collision_priority_multiplier**: Adjusts collision penalties by direction; can bias avoidance patterns; use carefully to avoid artifacts.
- **agents_num** (train): Sets density. Training at lower counts generalizes poorly to high-density; consider curricula or ranges.
- **maps_names_with_variants**: Map curriculum/diversity. More varied maps improve robustness but slow convergence.
- **eval/training seeds**: Fix seeds for reproducibility; varied seeds improve robustness but increase variance in reported metrics.

## Next model idea (to differentiate)
**CNN + D* with adaptive waiting and density reward**:
- Base on cnn_congestion_dstar, but:
  - Lower `wait_cost_multiplier` to 1.0 (or <1.0) and increase `collision_penalty` slightly (e.g., 0.6–0.8) to encourage yielding over collisions.
  - Add a mild per-step negative reward scaled by local agent density (requires env change), or use `reward_low_density: true` if available, to encourage dispersal.
  - Train on a mix of high-density corridor/slots maps and a mid-size maze (e.g., `slots_10-10a-39x29`, `corridors_10-10a-32x32`, `maze1_1-100a-30x30`, plus one open map).
- Goal: improve congestion handling by rewarding deconfliction rather than pure progress, differentiating it from current models that heavily penalize waits or over-avoid congestion.
