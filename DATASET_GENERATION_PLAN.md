# FREA Dataset Generation Plan

## Goal

Build a CARLA-based dangerous-driving dataset inspired by DeepAccident-style organization, using FREA where it fits best and script-driven logic where FREA is not the right tool.

The dataset should support:

- multiple CARLA towns
- near-miss and collision variants
- synchronized multi-sensor recordings
- later use for downstream perception / prediction / safety experiments

## Current Technical Baseline

The current working baseline is:

- CARLA `0.9.13`
- Python `3.8`
- FREA running successfully in `render` mode
- pretrained `fppo_adv` and `HJR` models loading correctly
- custom support added for saving front-camera image sequences from:
  - `ego`
  - current nearest `CBV`

Important decision:

- Do **not** switch to CARLA `0.9.14` just to use more towns.
- The main bottleneck is not map availability in CARLA itself.
- The real bottleneck is FREA's scenario assets, routes, pretrained resources, and compatibility.
- Keep the stack on `CARLA 0.9.13` for stability.

## Town Selection

Use only:

- `Town01`
- `Town02`
- `Town03`
- `Town04`
- `Town05`
- `Town06`

Reason:

- This avoids the extra engineering burden of forcing `Town07` and `Town10` into the current FREA workflow too early.
- It keeps the total dataset size close to the original target while staying realistic for a pilot and later expansion.

## Target Dataset Size

Original target:

- 696 scenes total

Revised target with 6 towns:

- `696 / 6 = 116` scenes per town

Total remains:

- `6 * 116 = 696`

## Train / Val / Test Split

Per town:

- train: `82`
- val: `17`
- test: `17`

Total per town:

- `82 + 17 + 17 = 116`

Recommendation:

- split by **route / scene instance**, not by individual clip only
- avoid train/test leakage from nearly identical scenes with only different weather or seeds

## Weather and Time Distribution

### Weather

Target ratio:

- clear : rainy : cloudy : wet = `4 : 4 : 1 : 1`

Do **not** use pure roulette sampling.

Use quota-constrained sampling instead.

Per town, for 116 scenes, a practical allocation is:

- clear: `46`
- rainy: `46`
- cloudy: `12`
- wet: `12`

### Time of Day

Target ratio:

- noon : sunset : night = `3 : 1 : 1`

Per town, for 116 scenes:

- noon: `70`
- sunset: `23`
- night: `23`

### Risk Type

Target ratio:

- collision : near-miss = `7 : 3`

Per town, for 116 scenes:

- collision: `81`
- near-miss: `35`

Note:

- These values can be adjusted slightly per town if exact scene generation is difficult.
- The key is to preserve approximate global balance, not force every town to match perfectly.

## Scenario Types

Three scenario types are planned.

### Type 1

Description:

- ego vehicle drives normally
- ego suddenly steers left or right
- ego collides with curb / street edge / adjacent lane / other lane users
- other car drives normally in the adjacent rear lane

Assessment:

- This is **not a natural FREA-first scenario**
- The main dangerous behavior comes from **ego**
- Better generated with:
  - scripted ego perturbation
  - fault injection
  - rule-based trigger logic

Conclusion:

- Type 1 should be handled mainly by **scripted ego disturbance**
- FREA can optionally be used as a secondary traffic interaction tool, but should not be the main generator

### Type 2

Description:

- other car drives normally
- ego starts from a roadside parked state

Assessment:

- This can be done as a hybrid scenario
- The base scene is best built by script
- FREA can then be used to make the moving other vehicle more adversarial if needed

Conclusion:

- Type 2 should be handled by **script + optional FREA assistance**

### Type 3

Description:

- ego drives normally behind a leading vehicle
- another car drives in the adjacent rear lane
- the leading vehicle suddenly slows down or brakes to stop

Assessment:

- This is the best fit for the current FREA workflow
- It naturally supports:
  - near-miss
  - collision
  - lane interaction
  - timing-based conflict generation

Conclusion:

- Type 3 should be the **primary FREA scenario family**

## Per-Town Scenario Count by Type

To scale the original `22 : 15 : 50` proportion to 116 scenes per town, use:

- Type 1: `29`
- Type 2: `20`
- Type 3: `67`

Check:

- `29 + 20 + 67 = 116`

## Sensor Plan

Both core vehicles should be preserved for downstream experiments:

- `ego car`
- `other car`

Planned full sensor suite per vehicle:

- `6 cameras`
- `1 lidar`

This is accepted as a design requirement, but it has major storage and synchronization implications.

### Engineering Implications

For one 20-second scene at 10 Hz:

- 200 frames per sensor
- 2 vehicles
- 6 cameras each

This means:

- `2 * 6 * 200 = 2400` images per scene

Across 696 scenes:

- more than 1.6 million images, before lidar is counted

Therefore:

- storage planning is required
- timestamp alignment must be explicit
- sensor calibration files must be saved

## Recommended Data Organization

Each generated scene should have a dedicated folder.

Recommended structure:

```text
scene_xxxx/
  meta.json
  ego/
    cam_front/
    cam_front_left/
    cam_front_right/
    cam_back/
    cam_back_left/
    cam_back_right/
    lidar/
  other/
    cam_front/
    cam_front_left/
    cam_front_right/
    cam_back/
    cam_back_left/
    cam_back_right/
    lidar/
  annotations/
    trajectory.json
    actors.json
    events.json
```

### `meta.json` should contain

- map name
- weather
- time of day
- scenario type
- near-miss / collision label
- route id
- seed
- frame count
- ego actor id
- other actor id
- sensor intrinsics / extrinsics
- start / stop timestamp

## Generation Strategy

Do not rely on purely random generation.

Recommended strategy:

1. define a small set of scenario templates per town
2. assign quotas for:
   - weather
   - time
   - risk type
   - scenario type
3. randomize only **within** the remaining quota
4. save all outputs
5. filter or score them afterward

This is better than unconstrained roulette sampling because:

- global balance is preserved
- town-level imbalance is reduced
- reproducibility improves

## Pilot Strategy

Before building the full 696-scene dataset, run a pilot.

### Pilot Phase A: Town Feasibility

Purpose:

- verify that every selected town can run stably
- verify route creation
- verify sensor attachment
- verify output saving

Scope:

- towns: `Town01` to `Town06`
- only **Type 3**
- each town: `2 to 3` scene instances
- each instance: generate both:
  - near-miss
  - collision

Estimated pilot scale:

- about `36` scenes total

This phase validates:

- town compatibility
- route/scenario construction
- multi-sensor synchronization
- feasibility of FREA for the main scenario family

### Pilot Phase B: Scenario Expansion

Only after Phase A succeeds:

- keep Type 3 as the main FREA-generated family
- add Type 2 as script + optional FREA-assisted generation
- add Type 1 as script-driven ego disturbance generation

## Near-Miss vs Collision Strategy

Do not expect a single template to produce both cleanly.

Recommended approach:

- create separate template variants for:
  - `near-miss`
  - `collision`

Practical template differences:

- `near-miss`
  - larger initial gap
  - later conflict trigger
  - shallower cut-in / conflict angle

- `collision`
  - smaller initial gap
  - earlier trigger
  - more direct conflict geometry

## Role of FREA

FREA should not be treated as a universal scene generator.

Use it where it is strongest:

- adversarial behavior generation for the **other vehicle**
- especially in scenarios where ego is nominal and danger comes from surrounding actors

Therefore:

- Type 3: FREA-first
- Type 2: hybrid
- Type 1: script-first

## Immediate Next Steps

1. Keep the project on `CARLA 0.9.13`
2. Limit the town set to `Town01` to `Town06`
3. Build a pilot around **Type 3 only**
4. Define exact per-town pilot templates
5. Expand the current image-export logic from:
   - ego front camera
   - CBV front camera
   to:
   - full 6-camera + lidar for ego
   - full 6-camera + lidar for other car
6. Finalize output folder structure and metadata schema
7. Run the pilot and measure:
   - runtime
   - disk usage
   - trajectory quality
   - failure rate

## Summary

The current agreed plan is:

- use `Town01` to `Town06`
- keep total size near `696`
- use `116` scenes per town
- use quota-controlled weather / time / risk balancing
- treat FREA as one generator inside a broader mixed pipeline
- begin with a pilot that tests all six towns using **Type 3**
- expand to Types 2 and 1 only after the pilot is stable

This is the most realistic path to a controllable, reproducible, and research-useful dataset.
