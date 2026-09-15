"""CPU-only launcher regression tests, without importing the GPU dependencies."""

import ast
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def launcher():
    source = Path(__file__).resolve().parents[1] / "batro_modal_rl.py"
    tree = ast.parse(source.read_text())
    selected = {
        "_remaining_rollouts", "_launch_training", "_select_resume_checkpoint",
        "_checkpoint_iteration",
    }
    module = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ], type_ignores=[])
    pin = Mock()
    manifests = Mock(return_value={"shard": 100})
    checkpoints = Mock()
    namespace = {
        "TrainConfig": object, "TrainingRun": object, "Checkpoint": object,
        "Path": Path, "json": json, "os": os, "re": re,
        "_pin_resume_checkpoint": pin,
        "_checkpoint_manifest": manifests,
        "_checkpoint_manifest_health": Mock(return_value=(True, "complete")),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(
        remaining=namespace["_remaining_rollouts"],
        launch=namespace["_launch_training"], pin=pin,
        select=namespace["_select_resume_checkpoint"],
        checkpoints=checkpoints, manifests=manifests,
    )


@pytest.mark.parametrize("target", [1, 199, 200])
def test_empty_resume_rejected(launcher, target):
    with pytest.raises(ValueError, match="at least --num-rollout 201"):
        launcher.remaining(199, target)


@pytest.mark.parametrize("target,expected", [(201, 1), (400, 200)])
def test_resume_count(launcher, target, expected):
    assert launcher.remaining(199, target) == expected


def test_checkpoint_zero_is_also_exclusive(launcher):
    with pytest.raises(ValueError):
        launcher.remaining(0, 1)
    assert launcher.remaining(0, 2) == 1


def test_receipt_prevents_duplicate_submission(launcher, tmp_path):
    receipt = tmp_path / "launch.json"
    run = SimpleNamespace(training_run_id="run", modal_app_id="app", function_call_id="fc")
    config = SimpleNamespace(checkpoint="checkpoint", launch=Mock(return_value=run))
    assert launcher.launch(config, receipt) is run
    assert json.loads(receipt.read_text()) == {
        "state": "submitted", "training_run_id": "run",
        "modal_app_id": "app", "function_call_id": "fc",
    }
    with pytest.raises(RuntimeError, match="No run submitted"):
        launcher.launch(config, receipt)
    config.launch.assert_called_once()
    launcher.pin.assert_called_once_with("checkpoint")


@pytest.mark.parametrize("error", [RuntimeError("upload failed"), KeyboardInterrupt()])
def test_interrupted_submission_is_not_retried(launcher, tmp_path, error):
    receipt = tmp_path / "launch.json"
    config = SimpleNamespace(checkpoint=None, launch=Mock(side_effect=error))
    with pytest.raises(type(error)):
        launcher.launch(config, receipt)
    assert json.loads(receipt.read_text())["state"] == "submitting"
    with pytest.raises(RuntimeError, match="No run submitted"):
        launcher.launch(config, receipt)
    config.launch.assert_called_once()
    launcher.pin.assert_not_called()


def test_partial_receipt_is_safe(launcher, tmp_path):
    receipt = tmp_path / "launch.json"
    receipt.touch()
    config = SimpleNamespace(checkpoint=None, launch=Mock())
    with pytest.raises(RuntimeError, match="No run submitted"):
        launcher.launch(config, receipt)
    config.launch.assert_not_called()


@pytest.mark.parametrize("requested", [None, 199])
def test_checkpoint_selection_only_reads_two_manifests(launcher, requested):
    checkpoints = [SimpleNamespace(name=f"iter_{i:07d}") for i in range(200)]
    launcher.checkpoints.return_value = checkpoints
    result = SimpleNamespace(training_run_id="run", checkpoints=launcher.checkpoints)
    assert launcher.select(result, requested) is checkpoints[-1]
    assert launcher.manifests.call_count == 2


def test_missing_checkpoint_does_not_read_manifests(launcher):
    launcher.checkpoints.return_value = [SimpleNamespace(name="iter_0000199")]
    with pytest.raises(RuntimeError, match="was not found"):
        launcher.select(
            SimpleNamespace(training_run_id="run", checkpoints=launcher.checkpoints),
            200,
        )
    launcher.manifests.assert_not_called()


@pytest.fixture
def reward_helpers():
    source = Path(__file__).resolve().parents[1] / "batro_modal_rl.py"
    tree = ast.parse(source.read_text())
    selected = {"_progress_value", "_transition_reward_parts", "_ante_clear_reward", "_episode_reward_parts"}
    module = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ], type_ignores=[])
    namespace = {"Any": object}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"ANTE_CHECKPOINT_BONUS", "FINAL_ANTE", "DECISION_CAP_PENALTY"}:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
    exec(compile(module, str(source), "exec"), namespace)
    return SimpleNamespace(
        progress=namespace["_progress_value"],
        transition=namespace["_transition_reward_parts"],
        ante_reward=namespace["_ante_clear_reward"],
        episode=namespace["_episode_reward_parts"],
    )


def test_skipped_blind_is_not_rewardable_progress(reward_helpers):
    before = {
        "ante_num": 1,
        "state": "BLIND_SELECT",
        "blinds": {"small": {"status": "SELECT"}},
        "round": {"chips": 0},
    }
    skipped = {
        "ante_num": 1,
        "state": "BLIND_SELECT",
        "blinds": {
            "small": {"status": "SKIPPED"},
            "big": {"status": "SELECT"},
        },
        "round": {"chips": 0},
    }
    defeated = {
        "ante_num": 1,
        "state": "SHOP",
        "blinds": {
            "small": {"status": "DEFEATED"},
            "big": {"status": "SELECT"},
        },
        "round": {"chips": 0},
    }
    assert reward_helpers.progress(skipped) == (0, 0.0)
    assert reward_helpers.transition(before, skipped, valid_action=True)["total"] == 0.0
    assert reward_helpers.transition(before, defeated, valid_action=True)["total"] == 0.0

    playing = {**before, "state": "SELECTING_HAND", "round": {"chips": 120}}
    assert reward_helpers.transition(playing, defeated, valid_action=True)["total"] == 4.0

    # Advancing after a boss clear must not reconstruct two previously skipped
    # blinds from the new ante number.
    after_boss = {**defeated, "ante_num": 2}
    parts = reward_helpers.transition(playing, after_boss, valid_action=True)
    assert parts["progress"] == 1.0
    assert parts["blind_clear_bonus"] == 3.0
    assert parts["ante_checkpoint_bonus"] == 8.0


@pytest.mark.parametrize("ante", range(1, 9))
def test_ante_reward_scales_by_cleared_ante(reward_helpers, ante):
    parts = reward_helpers.transition(
        {"ante_num": ante}, {"ante_num": ante + 1},
        valid_action=True, highest_rewarded_ante=ante - 1,
    )
    assert parts["ante_checkpoint_bonus"] == 8.0 * ante
    assert parts["ante_checkpoints"] == 1
    assert parts["highest_rewarded_ante"] == ante


def test_vouchers_cannot_repeat_ante_rewards(reward_helpers):
    highest = 0
    rewards, checkpoints = [], 0
    antes = [1, 2, 3, 2, 3, 2, 3, 4]
    for before, after in zip(antes, antes[1:]):
        parts = reward_helpers.transition(
            {"ante_num": before}, {"ante_num": after},
            valid_action=True, highest_rewarded_ante=highest,
        )
        highest = parts["highest_rewarded_ante"]
        rewards.append(parts["ante_checkpoint_bonus"])
        checkpoints += parts["ante_checkpoints"]
    assert rewards == [8, 16, 0, 0, 0, 0, 24]
    assert highest == checkpoints == 3


def test_win_top_up_uses_weighted_missing_antes_only(reward_helpers):
    # Before the ante-8 boss: seven unique clears. A win may or may not
    # advance the selector to ante 9; either path must total 8*(1+...+8).
    assert reward_helpers.ante_reward(0, 7) == 224
    assert reward_helpers.ante_reward(7, 8) == 64
    assert reward_helpers.ante_reward(8, 8) == 0
    for ante in (8, 9):
        parts = reward_helpers.episode(
            {"ante_num": ante, "won": True}, invalid_actions=0, turns=1, capped=False,
        )
        assert parts["ante_checkpoint_bonus"] == 288


def test_multiple_new_antes_sum_their_individual_rewards(reward_helpers):
    parts = reward_helpers.transition(
        {"ante_num": 2}, {"ante_num": 5}, valid_action=True, highest_rewarded_ante=1,
    )
    assert parts["ante_checkpoint_bonus"] == 8 * (2 + 3 + 4)
    assert parts["ante_checkpoints"] == 3
