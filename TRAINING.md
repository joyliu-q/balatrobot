# Training launch safety

Run `batro_modal_rl.py` with the Training Gym dependencies in **Python 3.12**,
matching the remote image. Keep `colocate=True` in the recipe.

Validate a one-rollout checkpoint continuation without allocating GPUs:

```sh
python batro_modal_rl.py --resume-run nullary-bezier-022caf585c3e \
  --resume-iteration 199 --num-rollout 201 --save-interval 1
```

Slime resumes at checkpoint iteration + 1; `--num-rollout` is an exclusive
upper bound, not the number of additional rollouts. From checkpoint 199,
target 200 does no work, 201 performs one rollout, and 400 performs 200.

To submit, add `--launch --launch-record /absolute/path/to/new-launch.json`.
Keep the local command alive until it prints `Training Gym run id` and the
receipt says `submitted`, with both a Modal app ID and function-call ID.
After that, Training Gym's detached app outlives the local process.

Do not use an auto-restarting supervisor for the submission command. Reuse
the same receipt path if an existing invocation is restarted: its exclusive
creation prevents duplicate submissions. A receipt left at `submitting`
means the outcome is uncertain, not that retrying is safe. Check the launch
log and Training Gym run before deliberately choosing a new receipt path.

A dashboard cancellation reason of `stale_pre_active_no_modal_app` means
the run record never acquired a persisted Modal app ID before the stale-run
cleanup timeout. It is not evidence of a GPU initialization failure. Look
at the **local submission log**; remote training may never have been spawned.

CPU-only regression checks:

```sh
uv run pytest tests/test_training_launch.py -q
```
