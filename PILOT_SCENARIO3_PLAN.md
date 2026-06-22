# Pilot Plan: Scenario Type 3

## Scope

This pilot only targets **Scenario Type 3**:

- ego drives normally behind a leading vehicle
- another car is present in the adjacent rear lane
- the leading vehicle suddenly slows down or brakes to a stop

This is the best fit for the current FREA workflow and should be validated before expanding to Scenario Types 1 and 2.

## Immediate Feasibility Finding

The repository already contains a matching scenario template:

- `scenario_03.json`
- scenario name: `OtherLeadingVehicle`

This is the correct starting point for the pilot.

However, the repository does **not** currently provide route coverage for all planned pilot towns.

### Existing `scenario_03` route coverage

Available route files currently map to:

- `Town03`
- `Town04`
- `Town05`
- plus `Town_Safebench_Light`

Missing for immediate pilot use:

- `Town01`
- `Town02`
- `Town06`

## Recommended Pilot Strategy

Split the pilot into two steps.

### Step A: Immediate pilot on existing assets

Use only:

- `Town03`
- `Town04`
- `Town05`

Purpose:

- validate that Scenario Type 3 can run end-to-end
- validate route/scenario parsing
- validate current rendering/export pipeline
- validate future multi-sensor expansion logic

This is the fastest way to get a meaningful pilot started.

### Step B: Complete six-town pilot

Create missing Scenario 3 route assets for:

- `Town01`
- `Town02`
- `Town06`

Then expand the pilot to:

- `Town01`
- `Town02`
- `Town03`
- `Town04`
- `Town05`
- `Town06`

## Suggested Minimal Existing Pilot Set

Start with one route per currently supported town:

- `Town03` -> `scenario_03_route_05.xml`
- `Town04` -> `scenario_03_route_09.xml`
- `Town05` -> `scenario_03_route_04.xml`

This is not yet a near-miss / collision split. It is only the first structural pilot.

## Why start with standard, not FREA, for the first pilot

The current pretrained FREA resources in this repository are centered around other scenario families, especially the existing Scenario 9 workflow.

For Scenario 3, the first pilot should verify:

- scenario geometry works
- route logic works
- actors spawn correctly
- output export works

Only after that should the pipeline move toward:

- FREA-based adversarialization
- near-miss / collision balancing
- quota-controlled batch generation

## Pilot Output Goals

For the first pilot, success means:

- all selected towns run stably
- the route is valid
- the leading-vehicle interaction is visible
- ego and other vehicle can both be recorded
- outputs are saved without crashes

The first pilot does **not** need to hit full dataset balance yet.

## Next Engineering Tasks

1. Create a Scenario 3 pilot manifest for the already-supported towns.
2. Add a dedicated Scenario 3 eval config.
3. Run render-mode pilot clips for:
   - Town03
   - Town04
   - Town05
4. Inspect the quality of:
   - route geometry
   - braking interaction
   - camera outputs
5. Build missing Scenario 3 route assets for:
   - Town01
   - Town02
   - Town06

## After This Pilot

Once the structural pilot works, the next step is:

- convert Scenario 3 into a full six-town pilot
- then separate:
  - near-miss variants
  - collision variants
- then scale to batch generation
