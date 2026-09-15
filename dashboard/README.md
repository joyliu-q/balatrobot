# Balatro dashboard component

`TrajectoryViewer.svelte` is the Balatro-specific trajectory viewer shipped
with this repository. `batro_modal_rl.py` resolves it relative to its own
file and attaches it to each launched run through Training Gym's
`training-gym-dashboard-overlay` volume.

Training Gym also ships a framework-default viewer at
`training-gym/dashboards/frontend/src/components/TrajectoryViewer.svelte`.
That file is the fallback used by the dashboard when no Balatro component is
attached (and can be replaced globally with `training-gym setup`).

The Balatro viewer follows the same Svelte props contract as the default:
`sample`, `samples`, `trajectory`, `rewardEvents`, `rollout`, and `run`.

The board shows chips scored, the current blind's target, and chips still
needed. Targets are per blind, not per ante; selection starts at zero and
shop/pack screens never reuse the previous blind's score. The initial
selection is available as the `S` frame. Blind selection lists each target,
skip tag, and its recorded effect; skip outcomes distinguish the tag taken
from the next blind's offer. Missing recorded effects are explicitly marked.

Run compiler and rendered-output regression tests with
`node --test dashboard/TrajectoryViewer.test.mjs`. This uses Svelte installed
in `../training-gym/dashboards/frontend`; set `TRAINING_GYM_FRONTEND` to
override that location.
