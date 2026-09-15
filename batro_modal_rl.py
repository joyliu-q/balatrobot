"""Train a Balatro policy with Training Gym and one Modal Sandbox per game.

The macOS ``Balatro.love`` archive is platform independent, but the executable
is not. This file uploads the private game assets to a Modal Volume and runs
them with the official Linux LOVE 11.5 build. Each rollout gets its own
Sandbox, so concurrent games never share a Balatro process or save directory.

Safe workflow (from the Training Gym Python 3.12 environment)::

    python batro_modal_rl.py --prepare-assets
    python batro_modal_rl.py --smoke-env
    python batro_modal_rl.py --launch

The first two commands use CPU containers only. Training is never launched
unless ``--launch`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from modal_training_gym import (
    DatasetConfig,
    Qwen3_8_27B,
    Qwen3_8_27B_Recipe,
    TrainConfig,
)
try:
    from modal_training_gym import DashboardComponent
except ImportError:
    # Dashboard components were added after the core Training Gym launch API.
    # Keep the training entrypoint usable with compatible older clients.
    DashboardComponent = None

from modal_training_gym.common.checkpoint import Checkpoint, _list_checkpoints
from modal_training_gym.common.environments.base import (
    Observation,
    SandboxEnvironment,
    SandboxEnvironmentPool,
    StepResult,
)
from modal_training_gym.common.models.base import ToolCall
from modal_training_gym.common.run import TrainingRun, set_checkpoint_location

# ---------------------------------------------------------------------------
# Private assets: local Mac -> private Modal Volume
# ---------------------------------------------------------------------------

LOVE_VERSION = "11.5"
LOVELY_VERSION = "0.9.0"
SMODS_VERSION = "1.0.0-beta-1814a"

ASSET_VOLUME_NAME = "balatro-rl-assets"
ASSET_MOUNT = "/opt/balatro-assets"
ASSET_APP_NAME = "balatro-rl-assets"
SANDBOX_APP_NAME = "balatro-rl-sandboxes"

REPO = Path(__file__).resolve().parent
# The Balatro-specific viewer is shipped with this repository.  Keep this
# path relative to the module rather than the caller's working directory so
# ``modal run`` and ``python`` resolve the same local source file.
BALATRO_TRAJECTORY_VIEWER_PATH = REPO / "dashboard" / "TrajectoryViewer.svelte"

LOVE_ARCHIVE_CANDIDATES = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common/Balatro"
    / "Balatro.app/Contents/Resources/Balatro.love",
    Path.home() / ".local/share/Steam/steamapps/common/Balatro/Balatro.love",
    Path.home() / ".steam/steam/steamapps/common/Balatro/Balatro.love",
)
SAVE_DIR_CANDIDATES = (
    Path.home() / "Library/Application Support/Balatro",
    Path.home() / ".local/share/love/Balatro",
)


def _find_balatro_love() -> Path:
    if override := os.environ.get("BALATRO_LOVE_ARCHIVE"):
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"BALATRO_LOVE_ARCHIVE does not exist: {path}")
    for path in LOVE_ARCHIVE_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find Balatro.love. Install Balatro with Steam or set "
        "BALATRO_LOVE_ARCHIVE."
    )


def _find_balatro_save() -> Path:
    if override := os.environ.get("BALATRO_SAVE_DIR"):
        path = Path(override).expanduser()
    else:
        path = next(
            (p for p in SAVE_DIR_CANDIDATES if (p / "settings.jkr").is_file()),
            None,
        )
    if path is None or not (path / "settings.jkr").is_file():
        raise FileNotFoundError(
            "No Balatro settings.jkr found. Launch Balatro once or set "
            "BALATRO_SAVE_DIR."
        )
    for relative in ("1/profile.jkr", "1/meta.jkr"):
        if not (path / relative).is_file():
            raise FileNotFoundError(f"Balatro save is missing {relative}: {path}")
    return path


assets_app = modal.App(ASSET_APP_NAME)
asset_volume = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
asset_seed_image = modal.Image.debian_slim(python_version="3.12")


@assets_app.function(
    image=asset_seed_image,
    volumes={"/assets": asset_volume},
    timeout=15 * 60,
    serialized=True,
    include_source=False,
)
def prepare_assets(release: str, upload_id: str) -> dict[str, Any]:
    """Publish a content-addressed Balatro release to the private Volume."""
    import shutil
    import uuid
    import zipfile

    releases = Path("/assets/releases")
    destination = releases / release
    releases.mkdir(parents=True, exist_ok=True)
    reused = destination.exists()

    if not reused:
        seed = Path("/assets/uploads") / upload_id
        if not upload_id or not seed.is_dir():
            raise FileNotFoundError(f"Asset upload is missing: {seed}")
        staging = Path(f"/assets/.staging-{release}-{uuid.uuid4().hex[:8]}")
        game = staging / "game"
        game.mkdir(parents=True)
        with zipfile.ZipFile(seed / "Balatro.love") as archive:
            archive.extractall(game)

        # A packaged Balatro executable is fused and implicitly uses the save
        # identity "Balatro". The portable .love directory needs it explicitly.
        conf_path = game / "conf.lua"
        conf = conf_path.read_text()
        if "t.identity" not in conf:
            conf = conf.replace(
                "t.title = 'Balatro'",
                "t.title = 'Balatro'\n\tt.identity = 'Balatro'",
                1,
            )
            conf_path.write_text(conf)

        shutil.copytree(seed / "mod", staging / "mods/balatrobot")
        shutil.copytree(seed / "python", staging / "python")
        shutil.copytree(seed / "save", staging / "save")
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "release": release,
                    "love": LOVE_VERSION,
                    "lovely": LOVELY_VERSION,
                    "smods": SMODS_VERSION,
                },
                indent=2,
            )
        )
        staging.rename(destination)

    if upload_id:
        shutil.rmtree(Path("/assets/uploads") / upload_id, ignore_errors=True)
    current_tmp = Path(f"/assets/.CURRENT-{os.getpid()}")
    current_tmp.write_text(release + "\n")
    current_tmp.replace("/assets/CURRENT")
    asset_volume.commit()
    return {
        "release": release,
        "volume": ASSET_VOLUME_NAME,
        "path": str(destination),
        "reused": reused,
    }


# ---------------------------------------------------------------------------
# Linux game image and one-sandbox-per-episode environment
# ---------------------------------------------------------------------------

_LOVE_WRAPPER = """#!/bin/sh
set -eu
LOVELY_PRELOAD="${LD_PRELOAD:-}"
unset LD_PRELOAD
DISPLAY_NUM="${BALATRO_DISPLAY:-99}"
if ! xdpyinfo -display ":$DISPLAY_NUM" >/dev/null 2>&1; then
    Xvfb ":$DISPLAY_NUM" -screen 0 1280x720x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    i=0
    while [ "$i" -lt 100 ]; do
        xdpyinfo -display ":$DISPLAY_NUM" >/dev/null 2>&1 && break
        i=$((i + 1))
        sleep 0.1
    done
fi
export DISPLAY=":$DISPLAY_NUM"
export LD_LIBRARY_PATH="/opt/love/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$LOVELY_PRELOAD"
exec /opt/balatro/bin/love "$@"
"""

_SANDBOX_ENTRYPOINT = f"""#!/bin/sh
set -eu
if [ ! -s {ASSET_MOUNT}/CURRENT ]; then
    echo "Balatro assets are missing. Run: python batro_modal_rl.py --prepare-assets" >&2
    exit 2
fi
RELEASE="$(tr -d '\\r\\n' < {ASSET_MOUNT}/CURRENT)"
ROOT="{ASSET_MOUNT}/releases/$RELEASE"
test -d "$ROOT/game"
test -d "$ROOT/mods/balatrobot"
test -d "$ROOT/python/balatrobot"

ln -sfn "$ROOT/game" /opt/balatro/game
ln -sfn "$ROOT/mods/balatrobot" /opt/balatro/Mods/balatrobot

# Saves are per-sandbox writable copies. Concurrent episodes must not mutate a
# shared profile in the Volume.
rm -rf /root/.local/share/love/Balatro
mkdir -p /root/.local/share/love/Balatro
cp -a "$ROOT/save/." /root/.local/share/love/Balatro/

export PYTHONPATH="$ROOT/python"
exec python -m balatrobot serve
"""


def _balatro_runtime_image() -> modal.Image:
    """Public Linux runtime; private game bytes live only in the Volume."""
    return (
        modal.Image.from_registry("debian:trixie-slim", add_python="3.12")
        .run_commands(
            "sed -i '/path-exclude.*man/d' /etc/dpkg/dpkg.cfg.d/* 2>/dev/null || true",
            "rm -f /etc/dpkg/dpkg.cfg.d/docker",
        )
        .apt_install(
            "love",  # pulls the AppImage's shared runtime dependencies
            "xvfb",
            "x11-utils",
            "libgl1-mesa-dri",
            "ca-certificates",
            "curl",
            "git",
            "unzip",
            "procps",
        )
        .run_commands(
            "curl -fsSL -o /tmp/love.AppImage "
            "https://github.com/love2d/love/releases/download/"
            f"{LOVE_VERSION}/love-{LOVE_VERSION}-x86_64.AppImage",
            "chmod +x /tmp/love.AppImage",
            "cd /tmp && ./love.AppImage --appimage-extract >/dev/null",
            "mv /tmp/squashfs-root /opt/love",
            "rm -f /tmp/love.AppImage",
            "curl -fsSL -o /tmp/lovely.tar.gz "
            "https://github.com/ethangreen-dev/lovely-injector/releases/download/"
            f"v{LOVELY_VERSION}/lovely-x86_64-unknown-linux-gnu.tar.gz",
            "tar xzf /tmp/lovely.tar.gz -C /tmp",
            "install -m 755 /tmp/liblovely.so /usr/local/lib/liblovely.so",
            "rm -f /tmp/lovely.tar.gz /tmp/liblovely.so",
            "mkdir -p /opt/balatro/bin /opt/balatro/Mods "
            "/root/.local/share/love /root/.config/love",
            "ln -sfn /opt/love/bin/love /opt/balatro/bin/love",
            "git clone --depth 1 --branch "
            f"{SMODS_VERSION} https://github.com/Steamodded/smods.git "
            "/opt/balatro/Mods/smods",
            "ln -sfn /opt/balatro/Mods /root/.local/share/love/Mods",
            "ln -sfn /opt/balatro/Mods /root/.config/love/Mods",
            "cat > /usr/local/bin/balatro-love <<'LOVE_EOF'\n"
            + _LOVE_WRAPPER
            + "LOVE_EOF",
            "chmod +x /usr/local/bin/balatro-love",
            "cat > /usr/local/bin/balatro-sandbox-entrypoint <<'ENTRY_EOF'\n"
            + _SANDBOX_ENTRYPOINT
            + "ENTRY_EOF",
            "chmod +x /usr/local/bin/balatro-sandbox-entrypoint",
        )
        .pip_install("httpx", "typer")
        .env(
            {
                "BALATROBOT_PLATFORM": "native",
                "BALATROBOT_BALATRO_PATH": "/opt/balatro/game",
                "BALATROBOT_LOVE_PATH": "/usr/local/bin/balatro-love",
                "BALATROBOT_LOVELY_PATH": "/usr/local/lib/liblovely.so",
                "BALATROBOT_LOGS_PATH": "/tmp/balatrobot-logs",
                "BALATROBOT_HEADLESS": "1",
                # Render a single Xvfb frame on RPC calls. BalatroBot disables
                # headless internally when this is set, allowing screenshots
                # without continuously rendering the game.
                "BALATROBOT_RENDER_ON_API": "1",
                "BALATROBOT_FAST": "1",
                "BALATROBOT_AUDIO": "0",
                "BALATROBOT_NO_SHADERS": "1",
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "SDL_AUDIODRIVER": "dummy",
                "ALSOFT_DRIVERS": "null",
            }
        )
    )


_RPC_SNIPPET = r"""
import json, sys, urllib.request
method, params = sys.argv[1], json.loads(sys.argv[2])
payload = json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:12346",
    data=payload,
    headers={"Content-Type":"application/json"},
)
with urllib.request.urlopen(request, timeout=8) as response:
    data = json.loads(response.read())
print("__BALATRO_RPC__" + json.dumps(data, separators=(",", ":")))
""".strip()

DECISION_STATES = frozenset(
    {"BLIND_SELECT", "SELECTING_HAND", "SHOP", "SMODS_BOOSTER_OPENED"}
)
TERMINAL_STATES = frozenset({"GAME_OVER"})


class BalatroSandboxEnvironment(SandboxEnvironment):
    """One isolated Balatro process and JSON-RPC server."""

    def __init__(self, sandbox: Any, state: dict[str, Any]) -> None:
        super().__init__(sandbox)
        self.state = state

    def rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        process = self.sandbox.exec(
            "python",
            "-c",
            _RPC_SNIPPET,
            method,
            json.dumps(params or {}, separators=(",", ":")),
            timeout=15,
        )
        process.wait()
        output = process.stdout.read()
        for line in reversed(output.splitlines()):
            if line.startswith("__BALATRO_RPC__"):
                payload = json.loads(line.removeprefix("__BALATRO_RPC__"))
                if "error" in payload:
                    raise RuntimeError(f"{method} failed: {payload['error']}")
                return payload["result"]
        error = process.stderr.read()
        raise RuntimeError(
            f"{method} RPC failed (rc={process.returncode}): {(error or output)[-1200:]}"
        )

    @staticmethod
    def _terminal(state: dict[str, Any]) -> bool:
        return bool(state.get("won")) or state.get("state") in TERMINAL_STATES

    @staticmethod
    def _stable(state: dict[str, Any]) -> bool:
        """Return whether a raw gamestate is safe to use for an action."""
        # The defaults preserve compatibility with an older BalatroBot asset
        # release that predates the readiness fields.
        return bool(state.get("state_complete", True)) and not bool(
            state.get("controller_locked", False)
        )

    def wait_for_stable_state(self, expected: str) -> dict[str, Any]:
        """Wait for a particular non-decision state to finish transitioning."""
        current = self.rpc("gamestate")
        for _ in range(120):
            if current.get("state") == expected and self._stable(current):
                self.state = current
                return current
            time.sleep(0.1)
            current = self.rpc("gamestate")
        raise TimeoutError(
            f"Balatro did not reach stable state {expected!r}: "
            f"{current.get('state')!r}"
        )

    def settle(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Advance automatic phases until the next model decision or terminal."""
        current = state or self.rpc("gamestate")
        for _ in range(120):
            name = current.get("state", "")
            # A gamestate response can briefly expose the next decision state
            # while Balatro is still finishing its UI/event transition.  Do
            # not hand that frame to the model or use it to issue an action.
            # The readiness fields are ignored when talking to an older mod so
            # local smoke tests and mixed deployments remain compatible.
            stable = self._stable(current)
            if self._terminal(current) or (
                name in DECISION_STATES and stable
            ):
                self.state = current
                return current
            if name == "ROUND_EVAL":
                current = self.rpc("cash_out")
            else:
                time.sleep(0.1)
                current = self.rpc("gamestate")
        raise TimeoutError(
            f"Balatro did not reach a decision state: {current.get('state')}"
        )

    def start(self, seed: str) -> dict[str, Any]:
        self.rpc("menu")
        state = self.rpc("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
        return self.settle(state)

    def screenshot_data_uri(self) -> str:
        """Capture the current game frame for Training Gym's rollout viewer."""
        path = f"/tmp/balatro-rollout-{time.time_ns()}.png"
        self.rpc("screenshot", {"path": path})
        process = self.sandbox.exec("base64", "-w", "0", path, timeout=15)
        process.wait()
        encoded = process.stdout.read().strip()
        if process.returncode != 0 or not encoded:
            error = process.stderr.read().strip()
            raise RuntimeError(f"screenshot encoding failed: {error}")
        return f"data:image/png;base64,{encoded}"

    def step(self, action: ToolCall) -> StepResult:
        try:
            # Refresh immediately before every action.  This closes the gap
            # between the model's previous observation and the live game, and
            # also waits out automatic transitions after a blind/round action.
            current = self.settle()
            action_response = self.rpc(action.name, action.arguments)
            # A successful play can return through ROUND_EVAL, after which
            # settle() immediately calls cash_out and receives SHOP. Preserve
            # its commit record across that automatic transition.
            committed_play = action_response.get("play_commit")
            state = self.settle(action_response)
            if isinstance(committed_play, dict) and "play_commit" not in state:
                state["play_commit"] = committed_play
            return StepResult(
                observation=Observation(
                    text=_compact_state(state), metadata={"state": state}
                ),
                done=self._terminal(state),
                info={"state_name": state.get("state", "")},
            )
        except Exception as exc:  # action errors are learning signal, not harness death
            # An action can race a state transition even after the preflight
            # refresh (for example when the server observes a new frame just
            # after accepting the request).  Return the freshest recoverable
            # state so the episode loop can choose a fallback for the actual
            # phase rather than blindly retrying a stale select/skip action.
            error_text = str(exc)
            recovered = self.state
            try:
                recovered = self.settle()
            except Exception as refresh_exc:
                error_text += f"; state refresh failed: {refresh_exc}"
            return StepResult(
                observation=Observation(
                    text=error_text,
                    is_error=True,
                    metadata={"state": recovered, "error": error_text},
                ),
                done=False,
                info={
                    "exception": repr(exc),
                    "state_refreshed": recovered is not self.state,
                },
            )


class BalatroEnvironmentPool(SandboxEnvironmentPool):
    @property
    def app_name(self) -> str:
        return SANDBOX_APP_NAME

    def build_image(self) -> modal.Image:
        return _balatro_runtime_image()

    def _diagnostics(self, sandbox: Any) -> str:
        parts: list[str] = []
        try:
            process = sandbox.exec(
                "sh",
                "-lc",
                "find /tmp/balatrobot-logs -type f -name '*.log' -print "
                "-exec tail -80 {} \\; 2>/dev/null",
                timeout=15,
            )
            process.wait()
            parts.append(process.stdout.read() + process.stderr.read())
        except Exception as exc:
            parts.append(f"diagnostic exec unavailable: {exc!r}")
        for name in ("stdout", "stderr"):
            try:
                parts.append(f"--- sandbox {name} ---\n{getattr(sandbox, name).read()}")
            except Exception as exc:
                parts.append(f"{name} unavailable: {exc!r}")
        return "\n".join(parts)[-8000:]

    def acquire(self, seed: str) -> BalatroSandboxEnvironment:
        volume = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=False)
        sandbox = self.create_sandbox(
            image=self.build_image(),
            command=("/usr/local/bin/balatro-sandbox-entrypoint",),
            volumes={ASSET_MOUNT: volume},
            timeout=60 * 60,
            idle_timeout=10 * 60,
            cpu=2.0,
            memory=4096,
        )
        env = BalatroSandboxEnvironment(sandbox, {})
        deadline = time.monotonic() + 90
        last_error = "not ready"
        while time.monotonic() < deadline:
            try:
                env.rpc("health")
                return env
            except Exception as exc:
                last_error = repr(exc)
                if sandbox.poll() is not None:
                    break
                time.sleep(0.5)
        diagnostics = self._diagnostics(sandbox)
        env.close()
        raise RuntimeError(
            f"Balatro sandbox did not become healthy: {last_error}\n{diagnostics}"
        )


_ENV_POOL: BalatroEnvironmentPool | None = None


def _get_env_pool() -> BalatroEnvironmentPool:
    global _ENV_POOL
    if _ENV_POOL is None:
        _ENV_POOL = BalatroEnvironmentPool()
    return _ENV_POOL


# ---------------------------------------------------------------------------
# Agent prompt, action parsing, and episode reward
# ---------------------------------------------------------------------------

# A completed ante is a useful curriculum checkpoint, but winning ante 8 must
# remain substantially more valuable than any collection of partial rewards.
ANTE_CHECKPOINT_BONUS = 8.0
FINAL_ANTE = 8
# A complete action budget is evidence that the policy failed to finish the
# game, not an environment fault. Keep the cost large enough that a loop of
# otherwise legal no-op actions cannot outrank actual blind progress.
DECISION_CAP_PENALTY = -10.0

_ALLOWED_METHODS = {
    "SELECTING_HAND": {
        "play",
        "discard",
        "sell",
        "use",
    },
    "SHOP": {"buy", "reroll", "next_round", "sell", "use"},
    "SMODS_BOOSTER_OPENED": {"pack"},
    "BLIND_SELECT": {"select", "skip"},
}
BALATRO_SYSTEM_PROMPT = (
    "Play Balatro through the provided JSON action interface. Choose exactly "
    "one legal API action and return only minified JSON: "
    '{"method":"name","params":{...}}. Card and item indices are 0-based; '
    "every index in a list must be distinct. State uses "
    "hand_stats=[level,total_played,played_this_round]. After the first "
    "turn, balatro_state contains only fields changed since the previous "
    "turn; absent fields are unchanged and null means removed. Poker-hand "
    "scoring: the state field hand_values maps each hand to "
    "[level, base_chips, base_mult]. A played hand's rough score is "
    "(base_chips + scored card chips + bonuses) × (base_mult + bonuses); "
    "jokers, enhancements, debuffs, and boss effects can change it. "
    "At level 1 the common bases are High Card 5/1, Pair 10/2, Two Pair "
    "20/2, Three of a Kind 30/3, Straight 30/4, Flush 35/4, Full House "
    "40/4, Four of a Kind 60/7, and Straight Flush 100/8; Five of a Kind, "
    "Flush House, and Flush Five are stronger special hands. The exact "
    "current values in hand_values are authoritative. "
    "Choose hands that clear the blind target, not just the highest raw "
    "rank. At BLIND_SELECT, choose select to enter the blind or skip to take "
    "the displayed tag (Boss blinds cannot be skipped). After a successful "
    "skip, the next state includes action_result with skipped, tag, tag_effect, "
    "and next blind details; action_result is one-turn transition metadata, "
    "not a replacement for the persistent game state."
)
_INDEX = r"[0-9]+"
_INDEX_LIST = rf"\[{_INDEX}(,{_INDEX})*\]"
_INDEX_LIST_5 = rf"\[{_INDEX}(,{_INDEX}){{0,4}}\]"
_ACTION_REGEX_BY_STATE = {
    "BLIND_SELECT": r'\{"method":"(select|skip)","params":\{\}\}',
    "SELECTING_HAND": (
        rf'\{{"method":"(play|discard)","params":\{{"cards":{_INDEX_LIST_5}\}}\}}'
        rf'|\{{"method":"sell","params":\{{"(joker|consumable)":{_INDEX}\}}\}}'
        rf'|\{{"method":"use","params":\{{"consumable":{_INDEX}'
        rf'(,"cards":{_INDEX_LIST_5})?\}}\}}'
    ),
    "SHOP": (
        rf'\{{"method":"buy","params":\{{"(card|voucher|pack)":{_INDEX}\}}\}}'
        r'|\{"method":"(reroll|next_round)","params":\{\}\}'
        rf'|\{{"method":"sell","params":\{{"(joker|consumable)":{_INDEX}\}}\}}'
        rf'|\{{"method":"use","params":\{{"consumable":{_INDEX}'
        rf'(,"cards":{_INDEX_LIST_5})?\}}\}}'
    ),
    "SMODS_BOOSTER_OPENED": (
        r'\{"method":"pack","params":\{"skip":true\}\}'
        rf'|\{{"method":"pack","params":\{{"card":{_INDEX}'
        rf'(,"targets":{_INDEX_LIST_5})?\}}\}}'
    ),
}


def _action_regex(state_name: str) -> str:
    return _ACTION_REGEX_BY_STATE.get(
        state_name,
        r'\{"method":"(play|discard|buy|reroll|next_round|sell|use|pack)",'
        r'"params":\{[^{}\n]*\}\}',
    )


def _compact_state(state: dict[str, Any], limit: int = 6_000) -> str:
    """Return a phase-specific state small enough for an entire ante-8 game."""

    def card_view(card: Any, *, playing_card: bool = False, owned: bool = False) -> Any:
        if not isinstance(card, dict):
            return card
        value = card.get("value") if isinstance(card.get("value"), dict) else {}
        modifier = card.get("modifier")
        if isinstance(modifier, dict):
            modifier = {key: value for key, value in modifier.items() if value}
        else:
            modifier = None
        card_state = card.get("state")
        debuffed = bool(
            card.get("debuff")
            or (isinstance(card_state, dict) and card_state.get("debuff"))
        )
        key = card.get("key") or card.get("label")
        if playing_card:
            # Most playing cards become a single token-like key such as S_A.
            if not modifier and not debuffed:
                return key
            view = {"id": key, "mod": modifier or None}
            if debuffed:
                view["debuff"] = True
            return {name: item for name, item in view.items() if item is not None}

        view = {"id": key, "effect": card.get("effect") or value.get("effect")}
        if modifier:
            view["mod"] = modifier
        cost = card.get("cost")
        buy = cost.get("buy") if isinstance(cost, dict) else card.get("buy")
        sell = cost.get("sell") if isinstance(cost, dict) else card.get("sell")
        if owned and sell is not None:
            view["sell"] = sell
        elif not owned and buy is not None:
            view["cost"] = buy
        return {name: item for name, item in view.items() if item is not None}

    def area_cards(
        area: Any, *, playing_cards: bool = False, owned: bool = False
    ) -> list[Any]:
        if not isinstance(area, dict):
            return []
        cards = area.get("cards") if isinstance(area.get("cards"), list) else []
        return [
            card_view(card, playing_card=playing_cards, owned=owned) for card in cards
        ]

    def slots(area: Any, default_limit: int) -> str:
        if not isinstance(area, dict):
            return f"0/{default_limit}"
        cards = area.get("cards") if isinstance(area.get("cards"), list) else []
        count = int(area.get("count", len(cards)) or 0)
        area_limit = int(area.get("limit", default_limit) or default_limit)
        return f"{count}/{area_limit}"

    state_name = str(state.get("state", ""))
    phase = {
        "SELECTING_HAND": "hand",
        "SHOP": "shop",
        "SMODS_BOOSTER_OPENED": "pack",
    }.get(state_name, state_name)
    compact: dict[str, Any] = {
        "phase": phase,
        "ante": state.get("ante_num"),
        "round": state.get("round_num"),
        "money": state.get("money"),
    }

    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    current_blind = next(
        (
            blind
            for blind in blinds.values()
            if isinstance(blind, dict) and blind.get("status") == "CURRENT"
        ),
        None,
    )
    if current_blind:
        compact["blind"] = {
            "type": str(current_blind.get("type", "")).lower(),
            "score": [round_info.get("chips"), current_blind.get("score")],
            "hands": round_info.get("hands_left"),
            "discards": round_info.get("discards_left"),
            "effect": current_blind.get("effect") or None,
        }

    # Blind selection is a model decision. Preserve every blind's status,
    # score, and tag so the agent can choose between selecting and skipping.
    # This is also useful during hand/shop phases when planning around the next
    # blind's effect or tag.
    blind_views = {
        name: {
            "status": blind.get("status"),
            "type": str(blind.get("type", "")).lower(),
            "name": blind.get("name"),
            "score": blind.get("score"),
            "effect": blind.get("effect") or None,
            "tag": blind.get("tag_name") or None,
            "tag_effect": blind.get("tag_effect") or None,
        }
        for name, blind in blinds.items()
        if isinstance(blind, dict)
    }
    if blind_views:
        compact["blinds"] = blind_views

    boss = blinds.get("boss")
    if isinstance(boss, dict) and boss.get("status") != "DEFEATED":
        compact["boss"] = [
            item
            for item in (boss.get("name"), boss.get("score"), boss.get("effect"))
            if item not in (None, "")
        ]

    jokers = state.get("jokers") if isinstance(state.get("jokers"), dict) else {}
    consumables = (
        state.get("consumables") if isinstance(state.get("consumables"), dict) else {}
    )
    compact["jokers"] = {
        "slots": slots(jokers, 5),
        "items": area_cards(jokers, owned=True),
    }
    compact["consumables"] = {
        "slots": slots(consumables, 2),
        "items": area_cards(consumables, owned=True),
    }

    # Vouchers move out of the shop once purchased. Keep their owned state in
    # the compact trajectory so the agent and debugger can distinguish an
    # active voucher from the current shop offers.
    used_vouchers = state.get("used_vouchers")
    if isinstance(used_vouchers, dict) and used_vouchers:
        compact["owned_vouchers"] = [
            {"id": key, "effect": value}
            for key, value in sorted(used_vouchers.items())
            if value not in (None, "")
        ]

    # Expose the known deck composition, not the engine's draw order.  The
    # composition is what a player can use to plan a build; preserving order
    # would leak future draws and would also make every delta unnecessarily
    # large.  Sort the compact views so draws do not reshuffle this field.
    deck_area = state.get("cards") if isinstance(state.get("cards"), dict) else {}
    deck = area_cards(deck_area, playing_cards=True)
    if deck:
        compact["deck"] = sorted(
            deck,
            key=lambda card: json.dumps(
                card, separators=(",", ":"), ensure_ascii=False
            ),
        )
    discard = area_cards(state.get("discard", {}), playing_cards=True)
    if discard:
        compact["discard"] = sorted(
            discard,
            key=lambda card: json.dumps(
                card, separators=(",", ":"), ensure_ascii=False
            ),
        )

    if state_name == "SELECTING_HAND":
        deck_cards = deck_area or state.get("deck_cards", {})
        compact["deck_left"] = (
            deck_cards.get("count") if isinstance(deck_cards, dict) else None
        )
        compact["hand"] = area_cards(state.get("hand", {}), playing_cards=True)
        hands = state.get("hands")
        if not isinstance(hands, dict):
            hands = (
                state.get("poker_hand_changes")
                if isinstance(state.get("poker_hand_changes"), dict)
                else {}
            )
        hand_stats = {
            name: [
                int(info.get("level") or 1),
                int(info.get("played") or 0),
                int(info.get("played_this_round") or 0),
            ]
            for name, info in hands.items()
            if isinstance(info, dict)
            and (
                int(info.get("level") or 1) != 1
                or int(info.get("played") or 0) > 0
                or int(info.get("played_this_round") or 0) > 0
            )
        }
        if hand_stats:
            compact["hand_stats"] = hand_stats
        hand_values = {
            name: [
                int(info.get("level") or 1),
                int(info.get("chips") or 0),
                int(info.get("mult") or 0),
            ]
            for name, info in hands.items()
            if isinstance(info, dict)
        }
        if hand_values:
            compact["hand_values"] = hand_values
    elif state_name == "SHOP":
        compact["reroll_cost"] = round_info.get("reroll_cost")
        for key in ("shop", "vouchers", "packs"):
            cards = area_cards(state.get(key, {}))
            if cards:
                compact[key] = cards
    elif state_name == "SMODS_BOOSTER_OPENED":
        compact["pack"] = area_cards(state.get("pack", {}))
        hand = area_cards(state.get("hand", {}), playing_cards=True)
        if hand:
            compact["hand"] = hand

    if state.get("won"):
        compact["won"] = True
    for key in ("play_commit",):
        if isinstance(state.get(key), dict):
            compact[key] = state[key]

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, item in value.items()
                if (cleaned := clean(item)) not in (None, {}, [])
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    compact = clean(compact)

    text = json.dumps(compact, separators=(",", ":"), ensure_ascii=False)
    if len(text) > limit:
        # Deck/discard composition is useful planning context, but the fields
        # below are required to choose a legal action. Do not fall back to a
        # summary that silently removes the hand, shop, pack, or hand values.
        # The compact card views are already bounded to the information needed
        # for indexed actions, so this keeps the important state available even
        # in unusually modifier-heavy runs.
        essential = (
            "phase",
            "ante",
            "round",
            "money",
            "blind",
            "blinds",
            "reroll_cost",
            "deck_left",
            "hand",
            "hand_stats",
            "hand_values",
            "jokers",
            "consumables",
            "owned_vouchers",
            "shop",
            "vouchers",
            "packs",
            "pack",
        )
        reduced = {key: compact[key] for key in essential if key in compact}
        reduced["deck_omitted"] = "deck" in compact
        reduced["discard_omitted"] = "discard" in compact
        reduced["truncated"] = True
        text = json.dumps(
            reduced,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return text


def _area_items(compact: dict[str, Any], name: str) -> list[Any]:
    area = compact.get(name)
    if isinstance(area, dict) and isinstance(area.get("items"), list):
        return area["items"]
    value = compact.get(name)
    return value if isinstance(value, list) else []


def _slot_usage(compact: dict[str, Any], name: str) -> tuple[int, int]:
    area = compact.get(name)
    slots = area.get("slots", "0/0") if isinstance(area, dict) else "0/0"
    try:
        used, limit = str(slots).split("/", 1)
        return int(used), int(limit)
    except (TypeError, ValueError):
        return 0, 0


def _purchasable_indices(
    compact: dict[str, Any], kind: str, *, enforce_slots: bool = False
) -> list[int]:
    money = float(compact.get("money") or 0)
    jokers_used, jokers_limit = _slot_usage(compact, "jokers")
    consumables_used, consumables_limit = _slot_usage(compact, "consumables")
    allowed: list[int] = []
    for index, item in enumerate(_area_items(compact, kind)):
        if not isinstance(item, dict) or float(item.get("cost") or 0) > money:
            continue
        card_id = str(item.get("id") or "")
        if enforce_slots and card_id.startswith("j_") and jokers_used >= jokers_limit:
            continue
        if (
            enforce_slots
            and card_id.startswith("c_")
            and consumables_used >= consumables_limit
        ):
            continue
        allowed.append(index)
    return allowed


_CONSUMABLE_TARGET_BOUNDS: dict[str, tuple[int, int]] = {
    # Tarot cards. Balatro permits choosing fewer than the displayed maximum
    # for multi-card enhancements/conversions, except Death (exactly two).
    "c_magician": (1, 2),
    "c_empress": (1, 2),
    "c_heirophant": (1, 2),
    "c_lovers": (1, 1),
    "c_chariot": (1, 1),
    "c_justice": (1, 1),
    "c_strength": (1, 2),
    "c_hanged_man": (1, 2),
    "c_death": (2, 2),
    "c_devil": (1, 1),
    "c_tower": (1, 1),
    "c_star": (1, 3),
    "c_moon": (1, 3),
    "c_sun": (1, 3),
    "c_world": (1, 3),
    # Spectral cards that highlight a playing card.
    "c_talisman": (1, 1),
    "c_aura": (1, 1),
    "c_deja_vu": (1, 1),
    "c_trance": (1, 1),
    "c_medium": (1, 1),
    "c_cryptid": (1, 1),
}
_CONSUMABLE_REQUIRES_JOKER = frozenset({"c_ankh"})


def _consumable_target_bounds(item: Any) -> tuple[int, int] | None:
    if not isinstance(item, dict):
        return None
    return _CONSUMABLE_TARGET_BOUNDS.get(str(item.get("id") or ""))


def _consumable_requires_joker(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and str(item.get("id") or "") in _CONSUMABLE_REQUIRES_JOKER
    )


def _target_action_help(
    *, method: str, item_name: str, index: int, item: Any, hand_size: int
) -> str | None:
    bounds = _consumable_target_bounds(item)
    if bounds is None:
        return f"{method} {item_name}={index} (no targets)"
    if hand_size <= 0:
        return None
    low, high = bounds
    count = str(low) if low == high else f"{low}-{high}"
    key = "cards" if method == "use" else "targets"
    return (
        f"{method} {item_name}={index} with {key}={count} distinct "
        f"hand indices in 0..{hand_size - 1}"
    )


def _action_help(state: dict[str, Any]) -> str:
    """Describe only actions that are legal under the current visible state."""
    compact = json.loads(_compact_state(state))
    state_name = str(state.get("state", ""))
    jokers = _area_items(compact, "jokers")
    consumables = _area_items(compact, "consumables")

    if state_name == "BLIND_SELECT":
        blinds = (
            compact.get("blinds") if isinstance(compact.get("blinds"), dict) else {}
        )
        selected = next(
            (
                (name, blind)
                for name, blind in blinds.items()
                if isinstance(blind, dict)
                and str(blind.get("status") or "").upper() == "SELECT"
            ),
            None,
        )
        if selected is None:
            return "select"
        name, blind = selected
        label = blind.get("name") or name
        score = blind.get("score")
        actions = [f"select (currently selected: {label}, score {score})"]
        if str(blind.get("type") or "").upper() != "BOSS":
            tag = blind.get("tag") or "no tag"
            tag_effect = blind.get("tag_effect")
            tag_text = f"{tag} ({tag_effect})" if tag_effect else tag
            actions.append(
                f"skip (currently selected: {label}, tag: {tag_text})"
            )
        return "; ".join(actions)

    if state_name == "SELECTING_HAND":
        hand = compact.get("hand") if isinstance(compact.get("hand"), list) else []
        high = len(hand) - 1
        blind = compact.get("blind") if isinstance(compact.get("blind"), dict) else {}
        play_count = (
            "exactly 5"
            if "must play 5 cards" in str(blind.get("effect") or "").lower()
            else "1-5"
        )
        actions = [
            f"play cards={play_count} distinct indices in 0..{high}",
        ]
        discards = int(blind.get("discards") or 0)
        if discards > 0:
            actions.append(
                f"discard cards=1-5 distinct indices in 0..{high} ({discards} left)"
            )
        if jokers:
            actions.append(f"sell joker=0..{len(jokers) - 1}")
        if consumables:
            actions.append(f"sell consumable=0..{len(consumables) - 1}")
            actions.extend(
                help_text
                for index, item in enumerate(consumables)
                if (not _consumable_requires_joker(item) or jokers)
                and (
                    help_text := _target_action_help(
                        method="use",
                        item_name="consumable",
                        index=index,
                        item=item,
                        hand_size=len(hand),
                    )
                )
            )
        return "; ".join(actions)

    if state_name == "SHOP":
        actions = ["next_round"]
        for kind, indices in (
            ("card", _purchasable_indices(compact, "shop", enforce_slots=True)),
            ("voucher", _purchasable_indices(compact, "vouchers")),
            ("pack", _purchasable_indices(compact, "packs")),
        ):
            if indices:
                actions.append(f"buy {kind}=one of {indices}")
        reroll_cost = float(compact.get("reroll_cost") or 0)
        if reroll_cost > 0 and float(compact.get("money") or 0) >= reroll_cost:
            actions.append(f"reroll (cost {reroll_cost:g})")
        if jokers:
            actions.append(f"sell joker=0..{len(jokers) - 1}")
        if consumables:
            actions.append(f"sell consumable=0..{len(consumables) - 1}")
            actions.extend(
                help_text
                for index, item in enumerate(consumables)
                if (not _consumable_requires_joker(item) or jokers)
                and _consumable_target_bounds(item) is None
                and (
                    help_text := _target_action_help(
                        method="use",
                        item_name="consumable",
                        index=index,
                        item=item,
                        hand_size=0,
                    )
                )
            )
        return "; ".join(actions)

    if state_name == "SMODS_BOOSTER_OPENED":
        pack = compact.get("pack") if isinstance(compact.get("pack"), list) else []
        jokers_used, jokers_limit = _slot_usage(compact, "jokers")
        selectable = [
            index
            for index, item in enumerate(pack)
            if not (
                isinstance(item, dict)
                and str(item.get("id") or "").startswith("j_")
                and jokers_used >= jokers_limit
            )
            and not (
                isinstance(item, dict)
                and str(item.get("id") or "") == "c_ankh"
                and jokers_used == 0
            )
        ]
        actions = ["pack skip=true"]
        hand = compact.get("hand") if isinstance(compact.get("hand"), list) else []
        actions.extend(
            help_text
            for index in selectable
            if (
                help_text := _target_action_help(
                    method="pack",
                    item_name="card",
                    index=index,
                    item=pack[index],
                    hand_size=len(hand),
                )
            )
        )
        return "; ".join(actions)

    return "none"


def _turn_prompt(state: dict[str, Any]) -> str:
    return (
        "You are playing Balatro. Choose exactly one legal API action. Card and "
        "item indices are 0-based. Optimize for winning ante 8. Return only "
        'minified JSON: {"method":"name","params":{...}} with no markdown or '
        "reasoning. Every index in a list must be distinct. Obey the exact "
        "Legal line; omitted actions are illegal. State uses slots=used/limit and "
        "hand_stats=[level,total_played,played_this_round] and hand_values="
        "[level,base_chips,base_mult]. After this first "
        "turn, <balatro_state> carries only the fields that CHANGED since the "
        "previous turn; anything absent is unchanged, null means removed.\n"
        "At BLIND_SELECT, blinds contains the selected blind and its tag; use "
        "select to enter it or skip to take the tag (Boss blinds cannot be "
        "skipped). After skip, action_result explicitly reports the skipped "
        "blind, tag/tag_effect, and next blind; action_result is one-turn "
        "transition metadata.\n"
        "The deck field is the sorted composition of cards remaining in the "
        "draw pile (not draw order); discard is the known discarded pile. Use "
        "these to plan hand selection and future deck/joker synergies.\n"
        f"Legal actions now: {_action_help(state)}\n"
        f"Gamestate:\n{_compact_state(state)}"
    )


def _action_is_locally_legal(action: ToolCall, state: dict[str, Any]) -> bool:
    compact = json.loads(_compact_state(state))
    method, params = action.name, action.arguments
    state_name = str(state.get("state", ""))

    def valid_indices(value: Any, size: int, *, complete: bool = False) -> bool:
        if not isinstance(value, list) or not value or len(value) != len(set(value)):
            return False
        if any(not isinstance(index, int) or not 0 <= index < size for index in value):
            return False
        return not complete or sorted(value) == list(range(size))

    hand = compact.get("hand") if isinstance(compact.get("hand"), list) else []
    jokers = _area_items(compact, "jokers")
    consumables = _area_items(compact, "consumables")
    blind = compact.get("blind") if isinstance(compact.get("blind"), dict) else {}
    compact_blinds = (
        compact.get("blinds") if isinstance(compact.get("blinds"), dict) else {}
    )

    if method in {"select", "skip"}:
        if state_name != "BLIND_SELECT" or params:
            return False
        selected = next(
            (
                blind
                for blind in compact_blinds.values()
                if isinstance(blind, dict)
                and str(blind.get("status") or "").upper() == "SELECT"
            ),
            None,
        )
        if selected is None:
            return False
        return method == "select" or str(selected.get("type") or "").upper() != "BOSS"

    def valid_targets(key: str, item: Any) -> bool:
        if _consumable_requires_joker(item) and not jokers:
            return False
        bounds = _consumable_target_bounds(item)
        if bounds is None:
            return key not in params
        values = params.get(key)
        return (
            valid_indices(values, len(hand)) and bounds[0] <= len(values) <= bounds[1]
        )

    if method in {"play", "discard"}:
        if set(params) != {"cards"}:
            return False
        cards = params.get("cards")
        if not valid_indices(cards, len(hand)) or len(cards) > 5:
            return False
        counter = "hands" if method == "play" else "discards"
        if int(blind.get(counter) or 0) <= 0:
            return False
        effect = str(blind.get("effect") or "").lower()
        return (
            not (method == "play" and "must play 5 cards" in effect) or len(cards) == 5
        )
    if method == "sell":
        if set(params) not in ({"joker"}, {"consumable"}):
            return False
        if "joker" in params:
            return isinstance(params["joker"], int) and 0 <= params["joker"] < len(
                jokers
            )
        if "consumable" in params:
            return isinstance(params["consumable"], int) and 0 <= params[
                "consumable"
            ] < len(consumables)
        return False
    if method == "use":
        if set(params) - {"consumable", "cards"}:
            return False
        index = params.get("consumable")
        if not isinstance(index, int) or not 0 <= index < len(consumables):
            return False
        if _consumable_target_bounds(consumables[index]) is not None:
            if str(state.get("state", "")) != "SELECTING_HAND":
                return False
        return valid_targets("cards", consumables[index])
    if method == "buy":
        if len(params) != 1:
            return False
        for kind, area, enforce_slots in (
            ("card", "shop", True),
            ("voucher", "vouchers", False),
            ("pack", "packs", False),
        ):
            if kind in params:
                return (
                    isinstance(params[kind], int)
                    and not isinstance(params[kind], bool)
                    and params[kind]
                    in _purchasable_indices(compact, area, enforce_slots=enforce_slots)
                )
        return False
    if method == "reroll":
        if params:
            return False
        cost = float(compact.get("reroll_cost") or 0)
        return cost > 0 and float(compact.get("money") or 0) >= cost
    if method == "next_round":
        return state_name == "SHOP" and not params
    if method == "pack":
        if params.get("skip") is True:
            return set(params) == {"skip"}
        if set(params) - {"card", "targets"}:
            return False
        index = params.get("card")
        pack = compact.get("pack") if isinstance(compact.get("pack"), list) else []
        if not isinstance(index, int) or not 0 <= index < len(pack):
            return False
        item = pack[index]
        jokers_used, jokers_limit = _slot_usage(compact, "jokers")
        if (
            isinstance(item, dict)
            and str(item.get("id") or "").startswith("j_")
            and jokers_used >= jokers_limit
        ):
            return False
        if (
            isinstance(item, dict)
            and str(item.get("id") or "") == "c_ankh"
            and jokers_used == 0
        ):
            return False
        return valid_targets("targets", item)
    return False


def _decode_action(text: str | None, state: dict[str, Any]) -> ToolCall | None:
    """Decode the first model JSON action allowed in the current phase.

    This deliberately does not enforce index legality; callers can distinguish
    malformed-but-repairable model output from output that needs the emergency
    fallback action.
    """
    if not text:
        return None
    state_name = str(state.get("state", ""))
    candidate = text.strip().removeprefix("```json").removeprefix("```")
    candidate = candidate.removesuffix("```").strip()
    for match in re.finditer(r"\{", candidate):
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        method, params = value.get("method"), value.get("params", {})
        if method not in _ALLOWED_METHODS.get(state_name, set()):
            continue
        if not isinstance(params, dict):
            continue
        params = dict(params)
        params.pop("reasoning", None)
        return ToolCall(name=method, arguments=params)
    return None


def _parse_action(text: str | None, state: dict[str, Any]) -> ToolCall | None:
    action = _decode_action(text, state)
    if action is None:
        return None
    params = action.arguments
    for key in ("cards", "targets"):
        indices = params.get(key)
        if indices is not None and (
            not isinstance(indices, list)
            or not 1 <= len(indices) <= 5
            or any(not isinstance(index, int) for index in indices)
        ):
            return None
    if _action_is_locally_legal(action, state):
        return action
    return None


def _repair_action(text: str | None, state: dict[str, Any]) -> ToolCall | None:
    """Repair common index mistakes without hiding genuinely unknown output.

    Qwen occasionally emits duplicate card indices or an index just outside the
    visible hand. Preserve the requested method, deduplicate/filter indices,
    and only use the emergency fallback when no meaningful action can be
    recovered. The episode still marks the repaired turn invalid for reward.
    """
    action = _decode_action(text, state)
    if action is None or action.name not in {"play", "discard"}:
        return None
    compact = json.loads(_compact_state(state))
    hand = compact.get("hand") if isinstance(compact.get("hand"), list) else []
    if not hand:
        return None
    values = action.arguments.get("cards")
    if not isinstance(values, list):
        return None
    cleaned: list[int] = []
    for value in values:
        if isinstance(value, int) and 0 <= value < len(hand) and value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        cleaned = [0]
    cleaned = cleaned[:5]
    blind = compact.get("blind") if isinstance(compact.get("blind"), dict) else {}
    effect = str(blind.get("effect") or "").lower()
    if action.name == "play" and "must play 5 cards" in effect:
        cleaned.extend(index for index in range(len(hand)) if index not in cleaned)
        cleaned = cleaned[:5]
    repaired = ToolCall(action.name, {**action.arguments, "cards": cleaned})
    return repaired if _action_is_locally_legal(repaired, state) else None


def _fallback_action(state: dict[str, Any]) -> ToolCall:
    name = state.get("state")
    if name == "BLIND_SELECT":
        return ToolCall("select", {})
    if name == "SELECTING_HAND":
        return ToolCall("play", {"cards": [0]})
    if name == "SHOP":
        return ToolCall("next_round", {})
    if name == "SMODS_BOOSTER_OPENED":
        return ToolCall("pack", {"skip": True})
    raise RuntimeError(f"No fallback action for state {name!r}")


def _episode_reward_parts(
    state: dict[str, Any],
    *,
    invalid_actions: int,
    turns: int,
    capped: bool,
    terminal_reason: str = "",
    partial_rewards: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return the complete-game reward and its components.

    ``partial_rewards`` contains transition-level shaping accumulated while the
    game was played.  The trainer still consumes one scalar per full-game
    sample, so this is accumulated rather than emitted as separate samples.
    Keeping the fallback calculation preserves the helper's useful standalone
    behavior for callers that only have a terminal state.
    """
    del capped

    if partial_rewards is not None:
        parts = {
            "progress": float(partial_rewards.get("progress", 0.0)),
            "blind_clear_bonus": float(partial_rewards.get("blind_clear_bonus", 0.0)),
            "ante_checkpoint_bonus": float(
                partial_rewards.get("ante_checkpoint_bonus", 0.0)
            ),
            "legal_action_bonus": float(partial_rewards.get("legal_action_bonus", 0.0)),
            "invalid_penalty": float(partial_rewards.get("invalid_penalty", 0.0)),
        }
        parts["win_bonus"] = 100.0 if state.get("won") else 0.0
        parts["cap_penalty"] = (
            DECISION_CAP_PENALTY if terminal_reason == "decision_cap" else 0.0
        )
        parts["total"] = sum(parts.values())
        parts["partial_total"] = parts["total"] - parts["win_bonus"]
        return parts

    # Terminal-only fallback for tests and older callers.
    blinds_cleared, fraction = _progress_value(state)
    progress = float(blinds_cleared + fraction)
    blind_clear_bonus = float(3.0 * blinds_cleared)
    ante = max(1, int(state.get("ante_num") or 1))
    ante_checkpoints = max(0, ante - 1)
    if state.get("won"):
        ante_checkpoints = max(ante_checkpoints, FINAL_ANTE)
    ante_checkpoint_bonus = float(ANTE_CHECKPOINT_BONUS * ante_checkpoints)
    # Formatting a legal JSON action is a prerequisite, not game progress.
    # Rewarding it made repeatable no-op actions (notably rearrange) a policy
    # objective. Keep the metric field for dashboard compatibility at zero.
    legal_action_bonus = 0.0
    win_bonus = 100.0 if state.get("won") else 0.0
    invalid_penalty = float(-2.0 * invalid_actions)
    # Context exhaustion may be a harness/token-budget limitation. Exhausting
    # the explicit decision budget is a policy failure and must not pay.
    cap_penalty = (
        DECISION_CAP_PENALTY if terminal_reason == "decision_cap" else 0.0
    )
    return {
        "progress": progress,
        "blind_clear_bonus": blind_clear_bonus,
        "ante_checkpoint_bonus": ante_checkpoint_bonus,
        "legal_action_bonus": legal_action_bonus,
        "win_bonus": win_bonus,
        "invalid_penalty": invalid_penalty,
        "cap_penalty": cap_penalty,
        "partial_total": (
            progress
            + blind_clear_bonus
            + ante_checkpoint_bonus
            + legal_action_bonus
            + invalid_penalty
        ),
        "total": (
            progress
            + blind_clear_bonus
            + ante_checkpoint_bonus
            + legal_action_bonus
            + win_bonus
            + invalid_penalty
            + cap_penalty
        ),
    }


def _progress_value(state: dict[str, Any]) -> tuple[int, float]:
    """Return rewardable blind progress and current-blind score fraction.

    ``SKIPPED`` does advance the game's blind selector, but it must not count
    as progress for reward.  Treating it as a clear paid the same +4 shaping
    as defeating a blind and trained the policy to skip Small and Big blinds
    before immediately losing to the Boss.  ``DEFEATED`` is the only status
    that represents score-earned game progress.
    """
    ante = max(1, int(state.get("ante_num") or 1))
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    chips = float(round_info.get("chips") or 0)
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    status_values = [
        blind.get("status")
        for name in ("small", "big", "boss")
        for blind in [blinds.get(name)]
        if isinstance(blind, dict)
    ]
    completed_statuses = {"DEFEATED"}
    if status_values:
        completed = sum(status in completed_statuses for status in status_values)
    else:
        # Preserve compatibility with older/partial game-state fixtures.
        completed = max(0, int(state.get("round_num") or 0) - 1)
    blinds_completed = (ante - 1) * 3 + completed
    current_blind = next(
        (
            float(blind.get("score") or 0)
            for blind in blinds.values()
            if isinstance(blind, dict) and blind.get("status") == "CURRENT"
        ),
        0.0,
    )
    fraction = min(1.0, chips / current_blind) if current_blind > 0 else 0.0
    return blinds_completed, fraction


def _transition_reward_parts(
    previous_state: dict[str, Any],
    state: dict[str, Any],
    *,
    valid_action: bool,
    invalid_events: int = 0,
) -> dict[str, float]:
    """Shape reward from one environment transition.

    Progress is potential-based within the currently played blind. A clear is
    recognized from the state transition itself rather than reconstructed from
    ``ante_num``: the latter cannot distinguish previously skipped blinds from
    defeated ones after the game advances to a new ante.
    """
    _, previous_fraction = _progress_value(previous_state)
    _, fraction = _progress_value(state)
    previous_phase = str(previous_state.get("state") or "")
    phase = str(state.get("state") or "")
    blind_defeated = previous_phase == "SELECTING_HAND" and phase == "SHOP"
    if blind_defeated:
        # The completed blind has potential 1.0; do not double-pay score
        # fraction that was already credited on earlier plays.
        progress_delta = float(max(0.0, 1.0 - previous_fraction))
    elif previous_phase == "SELECTING_HAND" and phase == "SELECTING_HAND":
        progress_delta = float(max(0.0, fraction - previous_fraction))
    else:
        # BLIND_SELECT transitions include skips and tags. They are real game
        # actions but create no score-earned progress.
        progress_delta = 0.0
    blind_clear_delta = 3.0 if blind_defeated else 0.0
    previous_ante = max(1, int(previous_state.get("ante_num") or 1))
    ante = max(1, int(state.get("ante_num") or 1))
    ante_checkpoints = max(0, ante - previous_ante)
    ante_checkpoint_bonus = float(ANTE_CHECKPOINT_BONUS * ante_checkpoints)
    # Legal syntax alone is not a rewardable transition.
    legal_bonus = 0.0
    invalid_penalty = -2.0 * max(0, int(invalid_events))
    return {
        "progress": progress_delta,
        "blind_clear_bonus": blind_clear_delta,
        "ante_checkpoint_bonus": ante_checkpoint_bonus,
        "ante_checkpoints": ante_checkpoints,
        "blind_clear_count": int(blind_defeated),
        "legal_action_bonus": legal_bonus,
        "invalid_penalty": invalid_penalty,
        "total": (
            progress_delta
            + blind_clear_delta
            + ante_checkpoint_bonus
            + legal_bonus
            + invalid_penalty
        ),
    }


def _episode_reward(
    state: dict[str, Any],
    *,
    invalid_actions: int,
    turns: int,
    capped: bool,
    terminal_reason: str = "",
) -> float:
    return _episode_reward_parts(
        state,
        invalid_actions=invalid_actions,
        turns=turns,
        capped=capped,
        terminal_reason=terminal_reason,
    )["total"]


_TOKENIZERS: dict[str, Any] = {}


def _tokenizer(checkpoint: str) -> Any:
    if checkpoint not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[checkpoint] = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
    return _TOKENIZERS[checkpoint]


def _chat_prompt(tokenizer: Any, user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": BALATRO_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_message},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


async def _generate_action(
    args: Any,
    sampling_params: dict[str, Any],
    prompt_ids: list[int],
    state_name: str,
) -> tuple[list[int], str | None, str]:
    from slime.utils.http_utils import post

    params = deepcopy(sampling_params)
    context_limit = int(getattr(args, "balatro_context_len", 32768))
    params["max_new_tokens"] = min(
        int(params.get("max_new_tokens", 96)), 96, context_limit - len(prompt_ids)
    )
    if params["max_new_tokens"] <= 0:
        return [], None, "context"
    # SGLang structured-output constraints belong inside sampling_params.
    params["regex"] = _action_regex(state_name)

    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "input_ids": prompt_ids,
            "sampling_params": params,
            "return_logprob": True,
        },
    )
    finish = output.get("meta_info", {}).get("finish_reason", {}).get("type", "")
    if finish == "abort":
        return [], None, finish

    response_ids = [
        item[1] for item in output.get("meta_info", {}).get("output_token_logprobs", [])
    ]
    return response_ids, output.get("text", ""), finish


def _action_payload(action: ToolCall) -> dict[str, Any]:
    return {"method": action.name, "params": action.arguments}


# Sentinel: a subtree that is identical to the previous turn's.
_UNCHANGED = object()


def _state_delta(previous: Any, current: Any) -> Any:
    """Recursive diff of two compact states.

    The transcript is one ever-growing sequence, so re-sending the whole state
    every turn means turn N carries N copies of largely identical JSON - the
    dominant cost of a long episode. Only what changed is sent instead; the
    model reconstructs the rest from earlier turns.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return current if current != previous else _UNCHANGED

    delta: dict[str, Any] = {}
    for key, value in current.items():
        if key not in previous:
            delta[key] = value
            continue
        nested = _state_delta(previous[key], value)
        if nested is not _UNCHANGED:
            delta[key] = nested
    # Keys the game dropped this turn (a joker sold, a pack closed).
    for key in previous:
        if key not in current:
            delta[key] = None
    return delta or _UNCHANGED


# Cheap scalars always echoed, so the model never has to scan back to orient.
_DELTA_ANCHORS = ("phase", "ante", "round", "money")


def _delta_payload(previous: dict[str, Any] | None, current: dict[str, Any]) -> Any:
    """Delta against the previous turn, or the full state when that is smaller."""
    if previous is None:
        return current

    delta = _state_delta(previous, current)
    delta = {} if delta is _UNCHANGED else delta
    for anchor in _DELTA_ANCHORS:
        if anchor in current:
            delta.setdefault(anchor, current[anchor])

    # A near-total change (a new round, say) can serialize larger than the
    # state itself; send whichever is smaller.
    def encode(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    return delta if len(encode(delta)) < len(encode(current)) else current


def _selected_blind(compact: dict[str, Any] | None) -> tuple[str, dict[str, Any]] | None:
    """Return the blind currently offered at BLIND_SELECT.

    Compact states keep every blind in ``blinds`` so a skipped blind and the
    newly offered blind can be explained without consulting the raw engine
    state.  Keep this helper deliberately tolerant of partial state deltas.
    """
    if not isinstance(compact, dict):
        return None
    blinds = compact.get("blinds")
    if not isinstance(blinds, dict):
        return None
    for name, blind in blinds.items():
        if isinstance(blind, dict) and str(blind.get("status") or "").upper() == "SELECT":
            return str(name), blind
    return None


def _skip_transition(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> dict[str, Any] | None:
    """Summarize what a valid ``skip`` action changed for the model/viewer."""
    before = _selected_blind(previous)
    after = _selected_blind(current)
    if before is None:
        return None
    before_name, before_blind = before
    next_name, next_blind = after if after is not None else (None, None)
    result: dict[str, Any] = {
        "method": "skip",
        "skipped": {
            "id": before_name,
            "name": before_blind.get("name") or before_name,
            "type": before_blind.get("type"),
        },
    }
    tag = before_blind.get("tag")
    tag_effect = before_blind.get("tag_effect")
    if tag:
        result["tag"] = tag
    if tag_effect:
        result["tag_effect"] = tag_effect
    if next_blind is not None:
        result["next"] = {
            "id": next_name,
            "name": next_blind.get("name") or next_name,
            "type": next_blind.get("type"),
            "score": next_blind.get("score"),
            "effect": next_blind.get("effect"),
        }
    return result


def _environment_feedback(
    state: dict[str, Any],
    *,
    previous_state_name: str,
    previous_compact: dict[str, Any] | None,
    executed: ToolCall,
    valid: bool,
    error: str = "",
) -> tuple[str, dict[str, Any]]:
    compact_state = json.loads(_compact_state(state))
    view = _delta_payload(previous_compact, compact_state)
    if valid and not error:
        payload = view
        # A skip is otherwise easy to miss in a sparse delta: the old blind
        # becomes SKIPPED and the next one becomes SELECT.  Include a compact,
        # explicit explanation so both the model and trajectory viewers know
        # which tag was earned and what blind is now available.
        if executed.name == "skip" and isinstance(payload, dict):
            if transition := _skip_transition(previous_compact, compact_state):
                payload = {**payload, "action_result": transition}
    else:
        payload = {
            "fallback": _action_payload(executed),
            "error": error or "invalid_model_action",
            "state": view,
        }
    feedback = (
        "\n<balatro_state>"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        + "</balatro_state>"
    )
    feedback += f"\nLegal: {_action_help(state)}"
    return feedback + "\nReturn one JSON action.\n", compact_state


def _log_balatro_event(kind: str, payload: dict[str, Any]) -> None:
    print(
        f"{kind} {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}",
        flush=True,
    )


async def generate_balatro_episode(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Play one full game and return it as exactly one multi-turn sample."""
    from slime.utils.types import Sample

    del evaluation
    source_metadata = dict(getattr(sample, "metadata", None) or {})
    seed = str(source_metadata.get("balatro_seed") or sample.prompt)
    # This budget covers only state-changing game decisions.
    max_decisions = int(getattr(args, "balatro_max_decisions", 250))
    context_limit = int(getattr(args, "balatro_context_len", 32768))
    capture_screenshot = bool(getattr(args, "balatro_capture_screenshot", True))
    rollout_id = time.time_ns() & ((1 << 63) - 1)
    env: BalatroSandboxEnvironment | None = None
    episode_started = time.monotonic()
    episode_trace: list[dict[str, Any]] = []

    def span(name: str, started: float, **attributes: Any) -> None:
        episode_trace.append(
            {
                "name": name,
                "start": started - episode_started,
                "end": time.monotonic() - episode_started,
                "attributes": attributes,
            }
        )

    # A transient sandbox startup failure should not kill a whole GPU job.
    for attempt in range(3):
        acquire_started = time.monotonic()
        try:
            env = await asyncio.to_thread(_get_env_pool().acquire, seed)
            span(
                "balatro.sandbox.acquire", acquire_started, attempt=attempt + 1, ok=True
            )
            break
        except Exception as exc:
            span(
                "balatro.sandbox.acquire",
                acquire_started,
                attempt=attempt + 1,
                ok=False,
            )
            _log_balatro_event(
                "BALATRO_SANDBOX_ERROR",
                {"seed": seed, "attempt": attempt + 1, "error": str(exc)[-1200:]},
            )
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
    if env is None:
        sample.status = Sample.Status.ABORTED
        return sample

    tokenizer = _tokenizer(args.hf_checkpoint)
    initial_prompt = ""
    initial_prompt_ids: list[int] = []
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    trajectory_text = ""
    trajectory_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": BALATRO_SYSTEM_PROMPT,
        }
    ]
    invalid_actions = 0
    invalid_turns = 0
    fallback_actions = 0
    repaired_actions = 0
    api_errors = 0
    turns_taken = 0
    state: dict[str, Any] = {}
    # Previous turn's compact state, so feedback can be sent as a delta.
    previous_compact: dict[str, Any] | None = None

    def append_tool_feedback(feedback: str) -> bool:
        """Append an environment response, including terminal error frames."""
        nonlocal trajectory_text, response_token_ids, loss_mask, capped, terminal_reason
        trajectory_messages.append({"role": "tool", "content": feedback.strip()})
        trajectory_text += feedback
        feedback_ids = tokenizer(feedback, add_special_tokens=False)["input_ids"]
        room = context_limit - 128 - len(initial_prompt_ids) - len(response_token_ids)
        if len(feedback_ids) > max(0, room):
            capped = True
            terminal_reason = "context_cap"
            return False
        response_token_ids.extend(feedback_ids)
        loss_mask.extend([0] * len(feedback_ids))
        return True

    capped = False
    aborted = False
    episode_error = ""
    terminal_reason = "game_over"
    screenshot = ""
    partial_reward_parts = {
        "progress": 0.0,
        "blind_clear_bonus": 0.0,
        "ante_checkpoint_bonus": 0.0,
        "legal_action_bonus": 0.0,
        "invalid_penalty": 0.0,
    }
    ante_checkpoints = 0
    partial_reward_steps = 0
    # Keep checkpoint-window rewards separate from the episode accumulators.
    # A blind-clear event credits everything earned since the previous blind
    # clear (all card moves included); an ante-clear event credits everything
    # earned across that ante, including skipped blinds.  These windows are
    # observability/credit-assignment records; ``sample.reward`` remains the
    # single additive game-level training signal.
    partial_reward_trace: list[dict[str, Any]] = []
    blind_reward_pending = dict(partial_reward_parts)
    ante_reward_pending = dict(partial_reward_parts)
    blind_window_turn: int | None = None
    ante_window_turn: int | None = None
    blind_window_token: int | None = None
    ante_window_token: int | None = None

    def _add_pending(target: dict[str, float], values: dict[str, float]) -> None:
        for key in target:
            target[key] += float(values.get(key, 0.0))

    def _pending_total(values: dict[str, float]) -> float:
        return float(sum(values.values()))

    def _reset_pending(values: dict[str, float]) -> None:
        for key in values:
            values[key] = 0.0

    def _append_checkpoint_event(
        *,
        label: str,
        state_name: str,
        turn: int,
        token_start: int | None,
        token_end: int,
        values: dict[str, float],
    ) -> None:
        if not any(abs(value) > 1e-12 for value in values.values()):
            return
        event: dict[str, Any] = {
            "turn": turn,
            "state": state_name,
            "label": label,
            "reward": _pending_total(values),
            # This is the actual game-level cumulative total. It is preferable
            # to summing event rewards because an ante window intentionally
            # overlaps its blind-clear windows.
            "cumulative_reward": sum(partial_reward_parts.values()),
            "progress": values["progress"],
            "blind_clear_bonus": values["blind_clear_bonus"],
            "ante_checkpoint_bonus": values["ante_checkpoint_bonus"],
            "legal_action_bonus": values["legal_action_bonus"],
            "invalid_penalty": values["invalid_penalty"],
            "total": _pending_total(values),
        }
        if token_start is not None:
            event["token_start"] = token_start
        event["token_end"] = token_end
        partial_reward_trace.append(event)

    previous_reward_state: dict[str, Any] | None = None
    try:
        game_start = time.monotonic()
        state = await asyncio.to_thread(env.start, seed)
        previous_reward_state = state
        span("balatro.game.start", game_start, state=state.get("state", ""))

        first_turn = _turn_prompt(state)
        trajectory_messages.append({"role": "user", "content": first_turn})
        initial_prompt = _chat_prompt(tokenizer, first_turn)
        initial_prompt_ids = tokenizer(initial_prompt, add_special_tokens=False)[
            "input_ids"
        ]

        for turn in range(max_decisions):
            if env._terminal(state):
                terminal_reason = "won" if state.get("won") else "game_over"
                break
            if len(initial_prompt_ids) + len(response_token_ids) >= context_limit - 128:
                capped = True
                terminal_reason = "context_cap"
                break

            turns_taken = turn + 1
            state_name = str(state.get("state", ""))
            # Reward events are aligned to the response/loss-mask sequence,
            # which excludes the initial prompt. The transition reward is
            # emitted after this action and can therefore be attached to the
            # action tokens that caused it.
            action_token_start = len(response_token_ids)
            generate_started = time.monotonic()
            generated_ids, response, finish = await _generate_action(
                args,
                sampling_params,
                initial_prompt_ids + response_token_ids,
                state_name,
            )
            span(
                "balatro.model.generate",
                generate_started,
                turn=turn,
                state=state_name,
                tokens=len(generated_ids),
                finish=finish,
            )
            if response is None:
                aborted = finish == "abort"
                capped = finish == "context"
                terminal_reason = "model_abort" if aborted else "context_cap"
                break

            response_token_ids.extend(generated_ids)
            action_token_end = len(response_token_ids)
            loss_mask.extend([1] * len(generated_ids))
            trajectory_text += response
            trajectory_messages.append({"role": "assistant", "content": response})

            parse_started = time.monotonic()
            action = _parse_action(response, state)
            invalid = action is None
            repaired_turn = False
            if action is None:
                repaired = _repair_action(response, state)
                invalid_actions += 1
                invalid_turns += 1
                if repaired is not None:
                    repaired_actions += 1
                    repaired_turn = True
                    action = repaired
                    decoded = _decode_action(response, state)
                    _log_balatro_event(
                        "BALATRO_ACTION_REPAIRED",
                        {
                            "seed": seed,
                            "episode_id": rollout_id,
                            "turn": turn,
                            "state": state_name,
                            "response": response[:500],
                            "requested": _action_payload(decoded) if decoded else None,
                            "repaired": _action_payload(action),
                        },
                    )
                else:
                    fallback_actions += 1
                    action = _fallback_action(state)
                    _log_balatro_event(
                        "BALATRO_ACTION_ERROR",
                        {
                            "seed": seed,
                            "episode_id": rollout_id,
                            "turn": turn,
                            "state": state_name,
                            "error": "invalid_model_output",
                            "response": response[:500],
                            "fallback": _action_payload(action),
                        },
                    )
            span("balatro.action.parse", parse_started, turn=turn, valid=not invalid)

            executed = action
            rpc_started = time.monotonic()
            result = await asyncio.to_thread(env.step, action)
            api_error = ""
            if result.observation.is_error:
                invalid_actions += 1
                api_errors += 1
                api_error = result.observation.text[:500]
                # ``step`` refreshes the environment after an RPC error. Use
                # that state for recovery; ``state`` is the model's previous
                # observation and may already be stale.
                refreshed_state = result.observation.metadata.get("state")
                if isinstance(refreshed_state, dict) and refreshed_state:
                    state = refreshed_state
                try:
                    fallback = _fallback_action(state)
                except Exception as fallback_exc:
                    episode_error = f"{api_error}; no fallback: {fallback_exc}"
                    terminal_reason = (
                        "game_over" if env._terminal(state) else "rpc_error"
                    )
                    partial_reward_parts["invalid_penalty"] -= 2.0 * (
                        int(invalid) + 1
                    )
                    feedback, previous_compact = _environment_feedback(
                        state,
                        previous_state_name=state_name,
                        previous_compact=previous_compact,
                        executed=action,
                        valid=False,
                        error=episode_error,
                    )
                    append_tool_feedback(feedback)
                    span(
                        f"balatro.rpc.{action.name}",
                        rpc_started,
                        turn=turn,
                        valid=False,
                        fallback=False,
                        refreshed_state=state.get("state", ""),
                    )
                    break
                fallback_actions += 1
                executed = fallback
                _log_balatro_event(
                    "BALATRO_ACTION_ERROR",
                    {
                        "seed": seed,
                        "episode_id": rollout_id,
                        "turn": turn,
                        "state": state_name,
                        "state_after_error": state.get("state", ""),
                        "error": api_error,
                        "requested": _action_payload(action),
                        "fallback": _action_payload(fallback),
                    },
                )
                result = await asyncio.to_thread(env.step, fallback)
                if result.observation.is_error:
                    api_errors += 1
                    episode_error = result.observation.text[:500]
                    terminal_reason = "rpc_error"
                    refreshed_state = result.observation.metadata.get("state")
                    if isinstance(refreshed_state, dict) and refreshed_state:
                        state = refreshed_state
                    # Preserve the failed turn in the trajectory. Previously
                    # this branch broke before adding a tool message, leaving
                    # viewers with an unpaired final action that looked like a
                    # spontaneous game over.
                    feedback, previous_compact = _environment_feedback(
                        state,
                        previous_state_name=state_name,
                        previous_compact=previous_compact,
                        executed=executed,
                        valid=False,
                        error=episode_error,
                    )
                    append_tool_feedback(feedback)
                    partial_reward_parts["invalid_penalty"] -= 2.0 * (int(invalid) + 1)
                    span(
                        f"balatro.rpc.{action.name}",
                        rpc_started,
                        turn=turn,
                        valid=False,
                        fallback=True,
                        refreshed_state=state.get("state", ""),
                    )
                    break
            span(
                f"balatro.rpc.{executed.name}",
                rpc_started,
                turn=turn,
                valid=not result.observation.is_error,
                fallback=bool(api_error),
                repaired=repaired_turn,
            )
            state = result.observation.metadata["state"]

            # Accumulate shaping at the transition where it occurs.  The
            # trainer still receives one reward for this complete game, but
            # the breakdown now reflects the sequence of partial outcomes.
            if previous_reward_state is not None:
                transition_rewards = _transition_reward_parts(
                    previous_reward_state,
                    state,
                    valid_action=not invalid and not api_error,
                    invalid_events=int(invalid) + int(bool(api_error)),
                )
                for key in partial_reward_parts:
                    partial_reward_parts[key] += transition_rewards[key]
                ante_checkpoints += int(transition_rewards["ante_checkpoints"])
                partial_reward_steps += 1
                _add_pending(blind_reward_pending, transition_rewards)
                _add_pending(ante_reward_pending, transition_rewards)
                if blind_window_turn is None:
                    blind_window_turn = turn + 1
                    blind_window_token = action_token_start
                if ante_window_turn is None:
                    ante_window_turn = turn + 1
                    ante_window_token = action_token_start

                blind_clear_count = int(transition_rewards["blind_clear_count"])
                ante_clear_count = int(transition_rewards["ante_checkpoints"])
                if ante_clear_count:
                    # An ante event intentionally spans all of its blinds,
                    # including any that were skipped in one transition.
                    _append_checkpoint_event(
                        label="ANTE_CLEAR",
                        state_name=state_name,
                        turn=turn + 1,
                        token_start=ante_window_token,
                        token_end=action_token_end,
                        values=ante_reward_pending,
                    )
                    _reset_pending(ante_reward_pending)
                    _reset_pending(blind_reward_pending)
                    ante_window_turn = None
                    blind_window_turn = None
                    ante_window_token = None
                    blind_window_token = None
                elif blind_clear_count:
                    _append_checkpoint_event(
                        label="BLIND_CLEAR",
                        state_name=state_name,
                        turn=turn + 1,
                        token_start=blind_window_token,
                        token_end=action_token_end,
                        values=blind_reward_pending,
                    )
                    _reset_pending(blind_reward_pending)
                    blind_window_turn = None
                    blind_window_token = None
            previous_reward_state = state

            feedback, previous_compact = _environment_feedback(
                state,
                previous_state_name=state_name,
                previous_compact=previous_compact,
                executed=executed,
                valid=not invalid and not api_error,
                error=api_error,
            )
            if not append_tool_feedback(feedback):
                break

            if result.done:
                terminal_reason = "won" if state.get("won") else "game_over"
                break
        else:
            capped = True
            terminal_reason = "decision_cap"
    except Exception as exc:
        episode_error = str(exc)[-1200:]
        terminal_reason = "episode_error"
        invalid_actions += 3
        partial_reward_parts["invalid_penalty"] -= 6.0
        _log_balatro_event(
            "BALATRO_EPISODE_ERROR",
            {"seed": seed, "episode_id": rollout_id, "error": episode_error},
        )
    finally:
        if capture_screenshot and env is not None:
            screenshot_started = time.monotonic()
            try:
                screenshot = await asyncio.to_thread(env.screenshot_data_uri)
                span("balatro.screenshot", screenshot_started, ok=True)
            except Exception as exc:
                span("balatro.screenshot", screenshot_started, ok=False)
                _log_balatro_event(
                    "BALATRO_SCREENSHOT_ERROR",
                    {"seed": seed, "episode_id": rollout_id, "error": str(exc)},
                )
        await asyncio.to_thread(env.close)

    if not initial_prompt_ids or not any(loss_mask):
        sample.status = Sample.Status.ABORTED
        return sample

    # A win can occur on the ante-8 boss before the game increments
    # ``ante_num``. Credit that final checkpoint exactly once, while avoiding
    # double-counting if the environment has already advanced the ante.
    final_ante = max(1, int(state.get("ante_num") or 1))
    expected_checkpoints = min(FINAL_ANTE, final_ante)
    if state.get("won") and ante_checkpoints < expected_checkpoints:
        missing_checkpoints = expected_checkpoints - ante_checkpoints
        checkpoint_reward = ANTE_CHECKPOINT_BONUS * missing_checkpoints
        partial_reward_parts["ante_checkpoint_bonus"] += checkpoint_reward
        ante_checkpoints += missing_checkpoints
        ante_reward_pending["ante_checkpoint_bonus"] += checkpoint_reward
        _append_checkpoint_event(
            label="WON",
            state_name="WON",
            turn=turns_taken,
            token_start=ante_window_token,
            token_end=len(response_token_ids),
            values=ante_reward_pending,
        )
        _reset_pending(ante_reward_pending)
        _reset_pending(blind_reward_pending)
    elif any(abs(value) > 1e-12 for value in blind_reward_pending.values()):
        # Preserve the final blind's card-move credit when the game ends or
        # hits a context/decision cap before a clear event.
        _append_checkpoint_event(
            label="GAME_END",
            state_name=str(state.get("state", "")),
            turn=turns_taken,
            token_start=blind_window_token,
            token_end=len(response_token_ids),
            values=blind_reward_pending,
        )

    reward_parts = _episode_reward_parts(
        state,
        invalid_actions=invalid_actions,
        turns=turns_taken,
        capped=capped,
        terminal_reason=terminal_reason,
        partial_rewards=partial_reward_parts,
    )
    reward = reward_parts["total"]
    won = bool(state.get("won"))
    final_ante = int(state.get("ante_num") or 0)
    final_round = int(state.get("round_num") or 0)
    context_tokens = len(initial_prompt_ids) + len(response_token_ids)
    context_utilization = context_tokens / max(1, context_limit)
    tokens_per_turn = len(response_token_ids) / max(1, turns_taken)
    duration = time.monotonic() - episode_started
    span(
        "balatro.episode",
        episode_started,
        turns=turns_taken,
        won=won,
        capped=capped,
    )

    metadata = dict(getattr(sample, "metadata", None) or {})
    metadata["balatro"] = {
        "seed_group": seed,
        "episode_id": rollout_id,
        "turns": turns_taken,
        "terminal_reason": terminal_reason,
        "episode_reward": reward,
        "partial_reward": reward_parts["partial_total"],
        "partial_reward_steps": partial_reward_steps,
        "reward_checkpoint_events": len(partial_reward_trace),
        "ante_checkpoints": ante_checkpoints,
        "partial_reward_trace": partial_reward_trace,
        "final_ante": final_ante,
        "final_round": final_round,
        "won": won,
        "capped": capped,
        "invalid_turns": invalid_turns,
        "invalid_actions": invalid_actions,
        "fallback_actions": fallback_actions,
        "repaired_actions": repaired_actions,
        "api_errors": api_errors,
        "context_tokens": context_tokens,
        "context_limit": context_limit,
        "context_utilization": context_utilization,
        "tokens_per_turn": tokens_per_turn,
        "duration_s": round(duration, 3),
        "error": episode_error,
    }
    episode_metrics = {
        "balatro/reward_total": reward,
        "balatro/reward_progress": reward_parts["progress"],
        "balatro/reward_blind_clear": reward_parts["blind_clear_bonus"],
        "balatro/reward_ante_checkpoint": reward_parts["ante_checkpoint_bonus"],
        "balatro/reward_legal_action": reward_parts["legal_action_bonus"],
        "balatro/reward_win_bonus": reward_parts["win_bonus"],
        "balatro/reward_invalid_penalty": reward_parts["invalid_penalty"],
        "balatro/reward_cap_penalty": reward_parts["cap_penalty"],
        "balatro/reward_partial": reward_parts["partial_total"],
        "balatro/partial_reward_steps": partial_reward_steps,
        "balatro/reward_checkpoint_events": len(partial_reward_trace),
        "balatro/ante_checkpoints": ante_checkpoints,
        "balatro/final_ante": final_ante,
        "balatro/final_round": final_round,
        "balatro/turns": turns_taken,
        "balatro/invalid_actions": invalid_actions,
        "balatro/fallback_actions": fallback_actions,
        "balatro/repaired_actions": repaired_actions,
        "balatro/api_errors": api_errors,
        "balatro/context_tokens": context_tokens,
        "balatro/context_utilization": context_utilization,
        "balatro/tokens_per_turn": tokens_per_turn,
        "balatro/won": int(won),
    }
    # Training Gym renders numeric top-level sample metadata as per-rollout
    # time-series charts. Keep the nested metrics object for the sample detail
    # view, and promote the same values so every Balatro metric is chartable.
    metadata["metrics"] = episode_metrics
    metadata.update(episode_metrics)
    metadata["exit_status"] = terminal_reason
    metadata["trajectory_messages"] = trajectory_messages
    if screenshot:
        metadata["image"] = screenshot
    # Keep the source compatible with framework Sample implementations that do
    # not allow arbitrary attributes. Training Gym promotes these reserved
    # metadata keys into its first-class rollout fields during extraction.
    metadata["reward_granularity"] = "checkpoint"
    metadata["reward_events"] = partial_reward_trace
    # Training Gym exposes checkpoint-window rewards as first-class Sample
    # fields.
    # Slime's sample type is version-dependent, so retain the metadata fallback
    # above while promoting the fields whenever the runtime object supports it.
    try:
        sample.reward_granularity = "checkpoint"
        sample.reward_events = partial_reward_trace
    except (AttributeError, TypeError, ValueError):
        pass

    sample.prompt = initial_prompt
    sample.tokens = initial_prompt_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = trajectory_text
    sample.loss_mask = loss_mask
    sample.status = Sample.Status.ABORTED if aborted else Sample.Status.COMPLETED
    sample.reward = reward
    sample.metadata = metadata
    sample.rollout_id = rollout_id
    existing_trace = getattr(sample, "trace", None)
    sample.trace = (
        list(existing_trace) if isinstance(existing_trace, list) else []
    ) + episode_trace

    _log_balatro_event(
        "BALATRO_EPISODE",
        {
            "seed": seed,
            "episode_id": rollout_id,
            "reward": reward,
            "ante": final_ante,
            "round": final_round,
            "turns": turns_taken,
            "won": won,
            "capped": capped,
            "invalid_actions": invalid_actions,
            "terminal_reason": terminal_reason,
            "duration_s": round(duration, 3),
        },
    )
    return sample


def _sample_reward(sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    return float(reward) if reward is not None else 0.0


async def passthrough_reward(args: Any, samples: Any, **kwargs: Any) -> Any:
    del args, kwargs
    if isinstance(samples, (list, tuple)):
        return [_sample_reward(sample) for sample in samples]
    return _sample_reward(samples)


def normalize_episode_rewards(
    args: Any, samples: list[Any]
) -> tuple[list[float], list[float]]:
    """Normalize complete-game rewards within each same-seed GRPO group."""
    from collections import defaultdict

    del args
    raw = [_sample_reward(sample) for sample in samples]
    seed_groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None) or {}
        seed = metadata.get("balatro", {}).get(
            "seed_group", str(getattr(sample, "group_index", 0))
        )
        seed_groups[str(seed)].append(index)

    advantages = [0.0] * len(samples)
    for indices in seed_groups.values():
        games: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            games[int(getattr(samples[index], "rollout_id", index))].append(index)
        game_ids = list(games)
        scores = [sum(raw[i] for i in games[g]) / len(games[g]) for g in game_ids]
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
        game_advantage = {
            game_id: (score - mean) / (std + 1e-6)
            for game_id, score in zip(game_ids, scores)
        }
        for game_id, game_indices in games.items():
            for index in game_indices:
                advantages[index] = game_advantage[game_id]
    return raw, advantages


# ---------------------------------------------------------------------------
# Training Gym dataset and active run configuration
# ---------------------------------------------------------------------------


class BalatroSeedsDataset(DatasetConfig):
    num_seeds = 256

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return True

    def _rows(self, count: int) -> list[dict[str, Any]]:
        rows = []
        for index in range(count):
            seed = f"BALATRO-RL-{index:08d}"
            rows.append(
                {
                    "messages": [{"role": "user", "content": seed}],
                    "label": "",
                    "metadata": {"balatro_seed": seed},
                }
            )
        return rows

    @staticmethod
    def _write(rows: list[dict[str, Any]], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")

    def rows(self):
        return self._rows(int(self.num_seeds))


def build_train_config(
    num_rollout: int | None = None,
    load: str | None = None,
    no_load_optim: bool = False,
) -> TrainConfig:
    if num_rollout is not None and num_rollout < 1:
        raise ValueError(f"num_rollout must be >= 1, got {num_rollout}")
    return TrainConfig(
        model=Qwen3_8_27B(),
        dataset=BalatroSeedsDataset(),
        recipe=Qwen3_8_27B_Recipe(
            num_rollout=200 if num_rollout is None else num_rollout,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
            global_batch_size=32,
            # Rollout engines share the requested 8×8 actor cluster. A colocated recipe
            # must not set ``rollout_num_gpus``: that field describes a
            # separate rollout pool and only applies to disaggregated runs.
            colocate=True,
            actor_num_nodes=8,
            actor_num_gpus_per_node=8,
            rollout_num_gpus_per_engine=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            pipeline_model_parallel_size=2,
            context_parallel_size=4,
            rollout_max_response_len=96,
            rollout_temperature=1.0,
            max_tokens_per_gpu=12288,
            sglang_speculative_algorithm=None,
            sglang_speculative_num_steps=None,
            sglang_speculative_eagle_topk=None,
            sglang_speculative_num_draft_tokens=None,
            save_interval=50,
            load=load or "",
            no_load_optim=no_load_optim,
            trace_sample_limit=8,
            # Long-horizon tool use is especially prone to collapsing into a
            # single brittle action line. Keep a light anchor to ref_load and
            # a small exploration incentive rather than relying on optimizer
            # weight decay as the only regularizer.
            use_kl_loss=True,
            kl_loss_type="low_var_kl",
            kl_loss_coef=0.001,
            kl_coef=0.0,
            entropy_coef=0.001,
            weight_decay=0.01,
            custom_generate_function=generate_balatro_episode,
            custom_rm_function=passthrough_reward,
            custom_reward_post_process_function=normalize_episode_rewards,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            environment={
                "PYTHONPATH": "/root/Megatron-LM/",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "TRAINING_GYM_TRAJECTORY_SAMPLE_LIMIT": "64",
                "TRAINING_GYM_IMAGE_SAMPLE_LIMIT": "0",
            },
            extra_config={
                "balatro_max_decisions": 250,
                "balatro_context_len": 49152,
                "balatro_capture_screenshot": False,
            },
            train_function_kwargs={"ephemeral_disk": 1_048_576},
        ),
    )


def _local_asset_release() -> tuple[
    str, list[tuple[Path, str]], list[tuple[Path, str]]
]:
    """Return the content hash plus file/directory upload manifest."""
    import hashlib

    save = _find_balatro_save()
    files = [
        (_find_balatro_love(), "Balatro.love"),
        (REPO / "balatrobot.json", "mod/balatrobot.json"),
        (REPO / "balatrobot.lua", "mod/balatrobot.lua"),
        (save / "settings.jkr", "save/settings.jkr"),
        (save / "1/profile.jkr", "save/1/profile.jkr"),
        (save / "1/meta.jkr", "save/1/meta.jkr"),
    ]
    directories = [
        (REPO / "src/lua", "mod/src/lua"),
        (REPO / "src/balatrobot", "python/balatrobot"),
    ]

    digest = hashlib.sha256()

    def update(path: Path, logical_path: str) -> None:
        digest.update(logical_path.encode())
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)

    for path, remote in files:
        update(path, remote)
    for directory, remote in directories:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            update(path, f"{remote}/{path.relative_to(directory)}")
    return digest.hexdigest()[:16], files, directories


def _prepare_assets_from_cli() -> dict[str, Any]:
    import uuid

    release, files, directories = _local_asset_release()
    try:
        reused = bool(asset_volume.listdir(f"/releases/{release}"))
    except Exception:
        reused = False

    upload_id = ""
    with modal.enable_output():
        if not reused:
            upload_id = f"{release}-{uuid.uuid4().hex[:10]}"
            root = f"/uploads/{upload_id}"
            with asset_volume.batch_upload() as batch:
                for path, remote in files:
                    batch.put_file(str(path), f"{root}/{remote}")
                for path, remote in directories:
                    batch.put_directory(str(path), f"{root}/{remote}")
        with assets_app.run():
            return prepare_assets.remote(release, upload_id)


def _smoke_environment(seed: str = "BALATRO-RL-SMOKE") -> dict[str, Any]:
    pool = BalatroEnvironmentPool()
    env = pool.acquire(seed)
    try:
        state = env.start(seed)
        initial = {
            "state": state.get("state"),
            "ante": state.get("ante_num"),
            "round": state.get("round_num"),
            "hand_size": len(state.get("hand") or []),
        }
        skip_result = env.step(ToolCall("skip", {}))
        skip_state = skip_result.observation.metadata.get("state", {})
        skip_big_result = env.step(ToolCall("skip", {}))
        skip_big_state = skip_big_result.observation.metadata.get("state", {})
        result = env.step(ToolCall("select", {}))
        selected_state = result.observation.metadata.get("state", {})
        commit_result = env.step(_fallback_action(selected_state))
        next_state = commit_result.observation.metadata.get("state", {})
        return {
            "initial": initial,
            "skip_action": {
                "error": skip_result.observation.is_error,
                "state": skip_state.get("state"),
                "selected_blinds": {
                    name: blind.get("status")
                    for name, blind in (skip_state.get("blinds") or {}).items()
                    if isinstance(blind, dict)
                },
            },
            "skip_big_action": {
                "error": skip_big_result.observation.is_error,
                "state": skip_big_state.get("state"),
                "selected_blinds": {
                    name: blind.get("status")
                    for name, blind in (skip_big_state.get("blinds") or {}).items()
                    if isinstance(blind, dict)
                },
            },
            "one_action": {
                "error": commit_result.observation.is_error,
                "state": next_state.get("state"),
                "ante": next_state.get("ante_num"),
                "round": next_state.get("round_num"),
            },
        }
    finally:
        pool.release(env)


def _checkpoint_iteration(checkpoint: Checkpoint) -> int:
    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    if match is None:
        raise RuntimeError(
            f"Cannot determine iteration from checkpoint {checkpoint.name!r}"
        )
    return int(match.group(1))


def _remaining_rollouts(checkpoint_iteration: int, target: int) -> int:
    # Slime resumes at the next rollout ID and uses an exclusive upper bound.
    start = checkpoint_iteration + 1
    if target <= start:
        raise ValueError(
            f"Checkpoint {checkpoint_iteration} resumes at rollout {start}, so "
            f"--num-rollout {target} would perform no training. "
            f"Use at least --num-rollout {start + 1} for one new rollout."
        )
    return target - start


def _launch_training(config: TrainConfig, record_path: Path | None = None):
    """Submit once; retain a receipt even if submission is interrupted.

    Reuse the same record path when a supervisor reruns this command. An
    existing receipt blocks submission, including an uncertain/partial one:
    automatically retrying those can create duplicate GPU jobs.
    """
    record = None
    if record_path is not None:
        try:
            record = record_path.open("x", encoding="utf-8")
        except FileExistsError:
            raise RuntimeError(
                f"Launch record already exists: {record_path}. No run submitted. "
                "Inspect its run ID/logs before using a new record for a new run."
            ) from None
    try:
        if record is not None:
            json.dump({"state": "submitting", "pid": os.getpid()}, record)
            record.flush()
            os.fsync(record.fileno())
        if config.checkpoint is not None:
            _pin_resume_checkpoint(config.checkpoint)
        # Training Gym's public launcher starts a detached Modal app. Keep this
        # process alive until it returns the persisted function-call handle.
        run = config.launch()
        if record is not None:
            record.seek(0)
            json.dump({
                "state": "submitted",
                "training_run_id": run.training_run_id,
                "modal_app_id": run.modal_app_id,
                "function_call_id": run.function_call_id,
            }, record)
            record.truncate()
            record.flush()
            os.fsync(record.fileno())
        return run
    finally:
        if record is not None:
            record.close()


def _checkpoint_manifest(checkpoint: Checkpoint) -> dict[str, int]:
    volume = modal.Volume.from_name(
        checkpoint.checkpoints_volume_name,
        create_if_missing=False,
    )
    manifest: dict[str, int] = {}
    for entry in volume.iterdir(checkpoint.path_relative_to_volume, recursive=False):
        path = str(getattr(entry, "path", ""))
        size = getattr(entry, "size", None)
        if path and isinstance(size, int):
            manifest[path.rsplit("/", 1)[-1]] = size
    return manifest


def _checkpoint_manifest_health(
    candidate: dict[str, int],
    reference: dict[str, int] | None,
) -> tuple[bool, str]:
    required = {".metadata", "metadata.json"}
    missing = sorted(required - candidate.keys())
    if missing:
        return False, f"missing {', '.join(missing)}"

    candidate_shards = {
        name: size for name, size in candidate.items() if name.endswith(".distcp")
    }
    if not candidate_shards:
        return False, "contains no .distcp shards"
    if any(size <= 0 for size in candidate_shards.values()):
        return False, "contains an empty .distcp shard"

    if reference is None:
        return True, "has checkpoint metadata and non-empty shards"

    reference_shards = {
        name: size for name, size in reference.items() if name.endswith(".distcp")
    }
    if candidate_shards.keys() != reference_shards.keys():
        return False, "distributed shard set differs from the previous checkpoint"

    truncated = [
        name
        for name, size in candidate_shards.items()
        if size < 0.9 * reference_shards[name]
    ]
    if truncated:
        return False, (
            f"{len(truncated)} distributed shards are more than 10% smaller "
            "than the previous checkpoint"
        )
    return True, "shard manifest matches the previous checkpoint"


def _pin_resume_checkpoint(checkpoint: Checkpoint) -> None:
    """Point Megatron's run-level tracker at the validated checkpoint.

    Training Gym mounts the parent run directory and passes that directory to
    ``--load``. Megatron then follows ``latest_checkpointed_iteration.txt``;
    selecting a Checkpoint object alone does not override that tracker.
    """
    iteration = _checkpoint_iteration(checkpoint)
    run_dir = PurePosixPath(checkpoint.path_relative_to_volume).parent
    marker_path = str(run_dir / "latest_checkpointed_iteration.txt")
    volume = modal.Volume.from_name(
        checkpoint.checkpoints_volume_name,
        create_if_missing=False,
    )
    marker = f"{iteration}\n".encode()
    with volume.batch_upload(force=True) as batch:
        batch.put_file(BytesIO(marker), marker_path)
    written = b"".join(volume.read_file(marker_path))
    if written.strip() != str(iteration).encode():
        raise RuntimeError(
            f"Failed to pin Megatron checkpoint tracker {marker_path!r} "
            f"to iteration {iteration}"
        )
    print(f"Pinned Megatron checkpoint tracker to iteration {iteration}.")


def _resume_result(config: TrainConfig, training_run_id: str) -> TrainingRun:
    """Load a run and supply legacy checkpoint metadata when it is absent.

    Older Training Gym launchers wrote the checkpoint directory to the legacy
    train-result store but not to ``TrainingRun.metadata``.  The current API
    discovers checkpoints exclusively from that metadata, so reconstruct the
    deterministic Slime location in memory for those older runs.  Do not save
    this compatibility value back to the record.
    """
    run = TrainingRun.from_id(training_run_id)
    if run.checkpoints():
        return run
    recipe_name = type(config.recipe).__name__.lstrip("_").lower()
    set_checkpoint_location(
        run,
        checkpoint_dir=f"/checkpoints/{training_run_id}",
        checkpoints_volume_name=f"slime-{recipe_name}-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    return run


def _select_resume_checkpoint(
    result: TrainingRun,
    requested_iteration: int | None,
) -> Checkpoint:
    # Use the persisted run's public helper in the ordinary case. A legacy run
    # may have had its location reconstructed in memory by ``_resume_result``;
    # calling ``TrainingRun.checkpoints()`` would reload and discard that local
    # compatibility metadata, so enumerate the same location directly there.
    location = getattr(result, "metadata", {}).get("checkpoint_location")
    if isinstance(location, dict):
        checkpoints = _list_checkpoints(
            str(location["checkpoint_dir"]),
            str(location["checkpoints_volume_name"]),
            str(location.get("checkpoints_mount_path") or "/checkpoints"),
            training_run_id=result.training_run_id,
            app_name=getattr(result, "app_name", ""),
        )
    else:
        checkpoints = result.checkpoints()
    checkpoints = sorted(checkpoints, key=_checkpoint_iteration)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found for run {result.training_run_id!r}")

    # Only inspect candidates we actually consider and their predecessor.
    # Eagerly listing every historical checkpoint can hit VolumeListFiles limits
    # before submission, particularly when validating and then launching.
    manifests: dict[str, dict[str, int]] = {}

    def manifest(checkpoint: Checkpoint) -> dict[str, int]:
        if checkpoint.name not in manifests:
            manifests[checkpoint.name] = _checkpoint_manifest(checkpoint)
        return manifests[checkpoint.name]

    candidates = checkpoints
    if requested_iteration is not None:
        candidates = [
            checkpoint
            for checkpoint in checkpoints
            if _checkpoint_iteration(checkpoint) == requested_iteration
        ]
        if not candidates:
            available = ", ".join(str(_checkpoint_iteration(c)) for c in checkpoints)
            raise RuntimeError(
                f"Checkpoint iteration {requested_iteration} was not found; "
                f"available iterations: {available}"
            )

    for checkpoint in reversed(candidates):
        checkpoint_index = checkpoints.index(checkpoint)
        reference = (
            manifest(checkpoints[checkpoint_index - 1])
            if checkpoint_index > 0
            else None
        )
        healthy, reason = _checkpoint_manifest_health(
            manifest(checkpoint), reference
        )
        if healthy:
            print(f"Checkpoint {checkpoint.name} accepted: {reason}.")
            return checkpoint
        print(f"Checkpoint {checkpoint.name} rejected: {reason}.")

    raise RuntimeError(
        f"No structurally complete checkpoint found for run {result.training_run_id!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--smoke-env", action="store_true")
    parser.add_argument(
        "--load",
        default=None,
        help=(
            "Checkpoint directory to resume from, e.g. /checkpoints/<run-id>. "
            "Omit to train from the base model."
        ),
    )
    parser.add_argument(
        "--no-load-optim",
        action="store_true",
        help=(
            "Skip optimizer state when resuming (--load). Required to resume "
            "with a different --num-rollout than the checkpoint was written with."
        ),
    )
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=None,
        help=(
            "Override the active TrainConfig's total target iteration count. When resuming, "
            "this exclusive upper bound must exceed checkpoint iteration + 1."
        ),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        metavar="TRAINING_RUN_ID",
        help="Resume from the latest structurally complete Megatron checkpoint.",
    )
    parser.add_argument(
        "--resume-iteration",
        type=int,
        default=None,
        help="Require this exact checkpoint iteration when resuming.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override checkpoint frequency; the final iteration is always saved.",
    )
    parser.add_argument(
        "--no-fault-tolerance",
        action="store_true",
        help="Fail once instead of repeatedly resuming from a newly damaged checkpoint.",
    )
    parser.add_argument(
        "--launch-record",
        type=Path,
        default=None,
        help="One-shot submission receipt; an existing file prevents duplicate launches.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually launch the H100 Training Gym run.",
    )
    args = parser.parse_args()
    if args.launch and sys.version_info[:2] != (3, 12):
        parser.error("Launch with Python 3.12 to match the Training Gym image.")
    if args.resume_iteration is not None and not args.resume_run:
        parser.error("--resume-iteration requires --resume-run")
    if args.resume_run and args.load:
        parser.error("Use either --resume-run or --load, not both")

    if args.prepare_assets:
        print(json.dumps(_prepare_assets_from_cli(), indent=2))
    if args.smoke_env:
        with modal.enable_output():
            smoke_result = _smoke_environment()
        print(json.dumps(smoke_result, indent=2))

    config = build_train_config(
        num_rollout=args.num_rollout,
        load=args.load,
        no_load_optim=args.no_load_optim,
    )
    if args.save_interval is not None:
        if args.save_interval < 1:
            raise ValueError("--save-interval must be at least 1")
        config.recipe.save_interval = args.save_interval
    if args.no_fault_tolerance:
        config.recipe.use_fault_tolerance = False
    if args.resume_run:
        result = _resume_result(config, args.resume_run)
        checkpoint = _select_resume_checkpoint(result, args.resume_iteration)
        checkpoint_iteration = _checkpoint_iteration(checkpoint)
        target_iteration = int(config.recipe.num_rollout)
        remaining = _remaining_rollouts(checkpoint_iteration, target_iteration)
        config.checkpoint = checkpoint
        # The checkpoint's scheduler was built for its original rollout
        # horizon. Continuing to a larger target must keep the new schedule
        # while still restoring model, optimizer, and RNG state.
        resume_extra_config = dict(config.recipe.extra_config or {})
        resume_extra_config["override_opt_param_scheduler"] = True
        config.recipe.extra_config = resume_extra_config
        print(
            f"Resuming {args.resume_run} from {checkpoint.name}; "
            f"target iterations={target_iteration}; new rollouts={remaining}."
        )
    if not args.launch:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "balatro-seeds.jsonl")
            config.dataset.write(path)
            config.dataset.validate_written(path)
        rollouts = int(config.recipe.num_rollout)
        print(
            "TrainConfig validated; no GPUs launched. "
            f"rollouts={rollouts}, max_decisions="
            f"{config.recipe.extra_config['balatro_max_decisions']}."
        )
        if not args.prepare_assets and not args.resume_run:
            print("First run: python batro_modal_rl.py --prepare-assets")
        launch_cmd = "python batro_modal_rl.py"
        if args.num_rollout is not None:
            launch_cmd += f" --num-rollout {args.num_rollout}"
        if args.resume_run:
            launch_cmd += f" --resume-run {args.resume_run}"
        if args.resume_iteration is not None:
            launch_cmd += f" --resume-iteration {args.resume_iteration}"
        if args.save_interval is not None:
            launch_cmd += f" --save-interval {args.save_interval}"
        if args.no_fault_tolerance:
            launch_cmd += " --no-fault-tolerance"
        print(f"Launch explicitly: {launch_cmd} --launch")
        return

    run = _launch_training(config, args.launch_record)
    viewer_path = BALATRO_TRAJECTORY_VIEWER_PATH

    if DashboardComponent is not None and hasattr(run, "add_dashboard_component"):
        run.add_dashboard_component(
            name="balatro",
            component_type=DashboardComponent.TRAJECTORY_VIEWER,
            from_path=viewer_path,
        )
    print(f"Training Gym run id: {run.training_run_id}")


if __name__ == "__main__":
    main()
