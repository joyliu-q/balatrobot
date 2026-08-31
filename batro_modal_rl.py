"""Train a Balatro policy with Training Gym and one Modal Sandbox per game.

The macOS ``Balatro.love`` archive is platform independent, but the executable
is not. This file uploads the private game assets to a Modal Volume and runs
them with the official Linux LOVE 11.5 build. Each rollout gets its own
Sandbox, so concurrent games never share a Balatro process or save directory.

Safe workflow (from the Training Gym Python 3.12 environment)::

    python batro_modal_rl.py --prepare-assets
    python batro_modal_rl.py --smoke-env
    python batro_modal_rl.py --profile proof --launch

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
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import modal
from modal_training_gym import (
    DatasetConfig,
    Qwen3_4B,
    Qwen3_8_27B,
    Qwen3_8_27b_Recipe,
    SlimeRecipe,
    TrainConfig,
)
from modal_training_gym.common.environments.base import (
    Observation,
    SandboxEnvironment,
    SandboxEnvironmentPool,
    StepResult,
)
from modal_training_gym.common.models.base import ToolCall
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

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

DECISION_STATES = frozenset({"SELECTING_HAND", "SHOP", "SMODS_BOOSTER_OPENED"})
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

    def settle(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Advance automatic phases until the next model decision or terminal."""
        current = state or self.rpc("gamestate")
        for _ in range(120):
            name = current.get("state", "")
            if self._terminal(current) or name in DECISION_STATES:
                self.state = current
                return current
            if name == "ROUND_EVAL":
                current = self.rpc("cash_out")
            elif name == "BLIND_SELECT":
                current = self.rpc("select")
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
            state = self.settle(self.rpc(action.name, action.arguments))
            return StepResult(
                observation=Observation(
                    text=_compact_state(state), metadata={"state": state}
                ),
                done=self._terminal(state),
                info={"state_name": state.get("state", "")},
            )
        except Exception as exc:  # action errors are learning signal, not harness death
            return StepResult(
                observation=Observation(
                    text=str(exc), is_error=True, metadata={"state": self.state}
                ),
                done=False,
                info={"exception": repr(exc)},
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

_ACTION_HELP = {
    "SELECTING_HAND": (
        "play|discard cards=[i,...] (1-5); sell joker|consumable=i; "
        "use consumable=i[,cards=[i,...]]; "
        "rearrange hand|jokers|consumables=[i,...]"
    ),
    "SHOP": (
        "buy card|voucher|pack=i; reroll; next_round; "
        "sell joker|consumable=i; use consumable=i[,cards=[i,...]]; "
        "rearrange jokers|consumables=[i,...]. Respect cost and slots; sell first if full"
    ),
    "SMODS_BOOSTER_OPENED": "pack card=i[,targets=[i,...]] OR pack skip=true",
}
_ALLOWED_METHODS = {
    "SELECTING_HAND": {"play", "discard", "sell", "use", "rearrange"},
    "SHOP": {"buy", "reroll", "next_round", "sell", "use", "rearrange"},
    "SMODS_BOOSTER_OPENED": {"pack"},
}
_INDEX = r"[0-9]+"
_INDEX_LIST = rf"\[{_INDEX}(,{_INDEX})*\]"
_INDEX_LIST_5 = rf"\[{_INDEX}(,{_INDEX}){{0,4}}\]"
_ACTION_REGEX_BY_STATE = {
    "SELECTING_HAND": (
        rf'\{{"method":"(play|discard)","params":\{{"cards":{_INDEX_LIST_5}\}}\}}'
        rf'|\{{"method":"sell","params":\{{"(joker|consumable)":{_INDEX}\}}\}}'
        rf'|\{{"method":"use","params":\{{"consumable":{_INDEX}'
        rf'(,"cards":{_INDEX_LIST_5})?\}}\}}'
        rf'|\{{"method":"rearrange","params":\{{"(hand|jokers|consumables)":'
        rf'{_INDEX_LIST}\}}\}}'
    ),
    "SHOP": (
        rf'\{{"method":"buy","params":\{{"(card|voucher|pack)":{_INDEX}\}}\}}'
        r'|\{"method":"(reroll|next_round)","params":\{\}\}'
        rf'|\{{"method":"sell","params":\{{"(joker|consumable)":{_INDEX}\}}\}}'
        rf'|\{{"method":"use","params":\{{"consumable":{_INDEX}'
        rf'(,"cards":{_INDEX_LIST_5})?\}}\}}'
        rf'|\{{"method":"rearrange","params":\{{"(jokers|consumables)":'
        rf'{_INDEX_LIST}\}}\}}'
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
        r'\{"method":"(play|discard|buy|reroll|next_round|sell|use|rearrange|pack)",'
        r'"params":\{[^{}\n]*\}\}',
    )


def _compact_state(state: dict[str, Any], limit: int = 6_000) -> str:
    """Return a phase-specific state small enough for an entire ante-8 game."""

    def card_view(
        card: Any, *, playing_card: bool = False, owned: bool = False
    ) -> Any:
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

    boss = blinds.get("boss")
    if isinstance(boss, dict) and boss.get("status") != "DEFEATED":
        compact["boss"] = [
            item
            for item in (boss.get("name"), boss.get("score"), boss.get("effect"))
            if item not in (None, "")
        ]

    jokers = state.get("jokers") if isinstance(state.get("jokers"), dict) else {}
    consumables = (
        state.get("consumables")
        if isinstance(state.get("consumables"), dict)
        else {}
    )
    compact["jokers"] = {
        "slots": slots(jokers, 5),
        "items": area_cards(jokers, owned=True),
    }
    compact["consumables"] = {
        "slots": slots(consumables, 2),
        "items": area_cards(consumables, owned=True),
    }

    if state_name == "SELECTING_HAND":
        deck_cards = (
            state.get("cards")
            if isinstance(state.get("cards"), dict)
            else state.get("deck_cards", {})
        )
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
    elif state_name == "SHOP":
        compact["reroll_cost"] = round_info.get("reroll_cost")
        for key in ("shop", "vouchers", "packs"):
            cards = area_cards(state.get(key, {}))
            if cards:
                compact[key] = cards
    elif state_name == "SMODS_BOOSTER_OPENED":
        compact["pack"] = area_cards(state.get("pack", {}))

    if state.get("won"):
        compact["won"] = True

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
        text = json.dumps(
            {
                "phase": compact.get("phase"),
                "money": compact.get("money"),
                "ante": compact.get("ante"),
                "round": compact.get("round"),
                "blind": compact.get("blind"),
                "hand": compact.get("hand"),
                "jokers": compact.get("jokers"),
                "consumables": compact.get("consumables"),
                "truncated": True,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return text


def _turn_prompt(state: dict[str, Any]) -> str:
    name = str(state.get("state", ""))
    return (
        "You are playing Balatro. Choose exactly one legal API action. Card and "
        "item indices are 0-based. Optimize for winning ante 8. Return only "
        'minified JSON: {"method":"name","params":{...}} with no markdown or '
        "reasoning. State uses slots=used/limit and "
        "hand_stats=[level,total_played,played_this_round]. After this first "
        "turn, <balatro_state> carries only the fields that CHANGED since the "
        "previous turn; anything absent is unchanged, null means removed.\n"
        f"Legal actions now: {_ACTION_HELP.get(name, 'none')}\n"
        f"Gamestate:\n{_compact_state(state)}"
    )


def _parse_action(text: str | None, state_name: str) -> ToolCall | None:
    if not text:
        return None
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
        params.pop("reasoning", None)
        for key in ("cards", "targets"):
            indices = params.get(key)
            if indices is not None and (
                not isinstance(indices, list)
                or not 1 <= len(indices) <= 5
                or any(not isinstance(index, int) for index in indices)
            ):
                break
        else:
            return ToolCall(name=method, arguments=params)
    return None


def _fallback_action(state: dict[str, Any]) -> ToolCall:
    name = state.get("state")
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
) -> dict[str, float]:
    """Dense progress reward; same-seed GRPO removes seed/luck baselines."""
    rounds = max(0, int(state.get("round_num") or 0))
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    chips = float(round_info.get("chips") or 0)
    current_blind = 0.0
    blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
    for blind in blinds.values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            current_blind = float(blind.get("score") or 0)
            break
    fraction = min(1.0, chips / current_blind) if current_blind > 0 else 0.0
    blinds_cleared = max(0, rounds - 1)
    progress = float(blinds_cleared + fraction)
    blind_clear_bonus = float(3.0 * blinds_cleared)
    legal_action_bonus = float(0.1 * max(0, turns - invalid_actions))
    win_bonus = 100.0 if state.get("won") else 0.0
    invalid_penalty = float(-2.0 * invalid_actions)
    # Running out of context is the harness's limit, not a policy mistake -
    # penalising it taxes exactly the long episodes worth rewarding. Spending
    # the whole decision budget without finishing is a real failure.
    cap_penalty = -1.0 if capped and terminal_reason != "context_cap" else 0.0
    return {
        "progress": progress,
        "blind_clear_bonus": blind_clear_bonus,
        "legal_action_bonus": legal_action_bonus,
        "win_bonus": win_bonus,
        "invalid_penalty": invalid_penalty,
        "cap_penalty": cap_penalty,
        "total": (
            progress
            + blind_clear_bonus
            + legal_action_bonus
            + win_bonus
            + invalid_penalty
            + cap_penalty
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
            "content": "Play Balatro through the provided JSON action interface.",
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
    encode = lambda obj: json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return delta if len(encode(delta)) < len(encode(current)) else current


def _environment_feedback(
    state: dict[str, Any],
    *,
    previous_state_name: str,
    previous_compact: dict[str, Any] | None,
    executed: ToolCall,
    valid: bool,
    error: str = "",
) -> tuple[str, dict[str, Any]]:
    state_name = str(state.get("state", ""))
    compact_state = json.loads(_compact_state(state))
    view = _delta_payload(previous_compact, compact_state)
    if valid and not error:
        payload = view
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
    if state_name != previous_state_name or not valid or error:
        feedback += f"\nLegal: {_ACTION_HELP.get(state_name, 'none')}"
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
    max_decisions = int(getattr(args, "balatro_max_decisions", 128))
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
            span("balatro.sandbox.acquire", acquire_started, attempt=attempt + 1, ok=True)
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
            "content": "Play Balatro through the provided JSON action interface.",
        }
    ]
    invalid_actions = 0
    invalid_turns = 0
    fallback_actions = 0
    api_errors = 0
    turns_taken = 0
    state: dict[str, Any] = {}
    # Previous turn's compact state, so feedback can be sent as a delta.
    previous_compact: dict[str, Any] | None = None
    capped = False
    aborted = False
    episode_error = ""
    terminal_reason = "game_over"
    screenshot = ""
    try:
        game_start = time.monotonic()
        state = await asyncio.to_thread(env.start, seed)
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
            loss_mask.extend([1] * len(generated_ids))
            trajectory_text += response
            trajectory_messages.append({"role": "assistant", "content": response})

            parse_started = time.monotonic()
            action = _parse_action(response, state_name)
            invalid = action is None
            if action is None:
                invalid_actions += 1
                invalid_turns += 1
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
                fallback = _fallback_action(state)
                fallback_actions += 1
                executed = fallback
                _log_balatro_event(
                    "BALATRO_ACTION_ERROR",
                    {
                        "seed": seed,
                        "episode_id": rollout_id,
                        "turn": turn,
                        "state": state_name,
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
                    span(
                        f"balatro.rpc.{action.name}",
                        rpc_started,
                        turn=turn,
                        valid=False,
                        fallback=True,
                    )
                    break
            span(
                f"balatro.rpc.{executed.name}",
                rpc_started,
                turn=turn,
                valid=not result.observation.is_error,
                fallback=invalid or bool(api_error),
            )
            state = result.observation.metadata["state"]

            feedback, previous_compact = _environment_feedback(
                state,
                previous_state_name=state_name,
                previous_compact=previous_compact,
                executed=executed,
                valid=not invalid and not api_error,
                error=api_error,
            )
            trajectory_messages.append({"role": "tool", "content": feedback.strip()})
            trajectory_text += feedback
            feedback_ids = tokenizer(feedback, add_special_tokens=False)["input_ids"]
            room = context_limit - 128 - len(initial_prompt_ids) - len(response_token_ids)
            if len(feedback_ids) > max(0, room):
                capped = True
                terminal_reason = "context_cap"
                break
            response_token_ids.extend(feedback_ids)
            loss_mask.extend([0] * len(feedback_ids))

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

    reward_parts = _episode_reward_parts(
        state,
        invalid_actions=invalid_actions,
        turns=turns_taken,
        capped=capped,
        terminal_reason=terminal_reason,
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
        "final_ante": final_ante,
        "final_round": final_round,
        "won": won,
        "capped": capped,
        "invalid_turns": invalid_turns,
        "invalid_actions": invalid_actions,
        "fallback_actions": fallback_actions,
        "api_errors": api_errors,
        "context_tokens": context_tokens,
        "context_limit": context_limit,
        "context_utilization": context_utilization,
        "tokens_per_turn": tokens_per_turn,
        "duration_s": round(duration, 3),
        "error": episode_error,
    }
    metadata["metrics"] = {
        "balatro/reward_total": reward,
        "balatro/reward_progress": reward_parts["progress"],
        "balatro/reward_blind_clear": reward_parts["blind_clear_bonus"],
        "balatro/reward_legal_action": reward_parts["legal_action_bonus"],
        "balatro/reward_win_bonus": reward_parts["win_bonus"],
        "balatro/reward_invalid_penalty": reward_parts["invalid_penalty"],
        "balatro/reward_cap_penalty": reward_parts["cap_penalty"],
        "balatro/final_ante": final_ante,
        "balatro/final_round": final_round,
        "balatro/turns": turns_taken,
        "balatro/invalid_actions": invalid_actions,
        "balatro/fallback_actions": fallback_actions,
        "balatro/api_errors": api_errors,
        "balatro/context_tokens": context_tokens,
        "balatro/context_utilization": context_utilization,
        "balatro/tokens_per_turn": tokens_per_turn,
        "balatro/won": int(won),
    }
    metadata["exit_status"] = terminal_reason
    metadata["trajectory_messages"] = trajectory_messages
    if screenshot:
        metadata["image"] = screenshot

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
# Training Gym dataset and profiles
# ---------------------------------------------------------------------------


class BalatroSeedsDataset(DatasetConfig):
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    output_format = "jsonl"
    always_prepare = True
    num_seeds = 256

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

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        self._write(self._rows(int(self.num_seeds)), path)
        for eval_path in (eval_paths or {}).values():
            self._write(self._rows(min(32, int(self.num_seeds))), eval_path)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class BalatroGRPORecipe(SlimeRecipe):
    gpu_type: str = "H100"
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.70

    num_rollout: int = 1
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 96
    rollout_temperature: float = 1.0
    rollout_shuffle: bool = True
    balance_data: bool = True

    advantage_estimator: str = "grpo"
    use_kl_loss: bool = True
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.001
    global_batch_size: int = 8
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 32768
    save_interval: int = 10
    eval_interval: int | None = None
    megatron_to_hf_mode: str = "bridge"
    capture_trace: bool = True
    trace_sample_limit: int = 16


PROFILES: dict[str, dict[str, int]] = {
    "27b-25x32": {
        "num_rollout": 25,
        "num_seeds": 256,
        "max_decisions": 96,
        "context_len": 49152,
        "trace_sample_limit": 32,
        "capture_screenshot": 0,
        "tensor_parallel": 4,
        "rollout_batch_size": 8,
        "n_samples_per_prompt": 4,
        "global_batch_size": 32,
        # Slime saves iteration 0 and every Nth iteration. Six also lands on
        # the final zero-indexed iteration (24), while avoiding 25 expensive
        # full-model checkpoints.
        "save_interval": 6,
    },
    "27b-proof": {
        "num_rollout": 1,
        "num_seeds": 32,
        "max_decisions": 96,
        "context_len": 32768,
        "trace_sample_limit": 8,
        "capture_screenshot": 0,
        "tensor_parallel": 4,
    },
    "proof": {
        "num_rollout": 1,
        "num_seeds": 32,
        "max_decisions": 32,
        "context_len": 16384,
        "trace_sample_limit": 8,
        "capture_screenshot": 0,
        "tensor_parallel": 1,
    },
    "observable": {
        "num_rollout": 3,
        "num_seeds": 128,
        "max_decisions": 64,
        "context_len": 16384,
        "trace_sample_limit": 16,
        "capture_screenshot": 0,
        "tensor_parallel": 2,
    },
    "smoke": {
        "num_rollout": 10,
        "num_seeds": 128,
        "max_decisions": 96,
        "context_len": 32768,
        "trace_sample_limit": 16,
        "capture_screenshot": 0,
        "tensor_parallel": 2,
    },
    "full": {
        "num_rollout": 250,
        "num_seeds": 5000,
        "max_decisions": 160,
        "context_len": 32768,
        "trace_sample_limit": 8,
        "capture_screenshot": 0,
        "tensor_parallel": 2,
    },
}


def build_train_config(
    profile: str = "proof", num_rollout: int | None = None
) -> TrainConfig:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    settings = dict(PROFILES[profile])
    if num_rollout is not None:
        if num_rollout < 1:
            raise ValueError(f"num_rollout must be >= 1, got {num_rollout}")
        # One training step per rollout batch; the dataset cycles once its
        # seeds are exhausted (num_seeds / rollout_batch_size steps per epoch).
        settings["num_rollout"] = num_rollout
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "TRAINING_GYM_TRAJECTORY_SAMPLE_LIMIT": "64",
        "TRAINING_GYM_IMAGE_SAMPLE_LIMIT": "0",
    }
    extra_config = {
        "balatro_max_decisions": settings["max_decisions"],
        "balatro_context_len": settings["context_len"],
        "balatro_capture_screenshot": bool(settings["capture_screenshot"]),
    }
    if profile.startswith("27b-"):
        return TrainConfig(
            model=Qwen3_8_27B(),
            dataset=BalatroSeedsDataset(num_seeds=settings["num_seeds"]),
            recipe=Qwen3_8_27b_Recipe(
                num_rollout=settings["num_rollout"],
                rollout_batch_size=settings.get("rollout_batch_size", 2),
                n_samples_per_prompt=settings.get("n_samples_per_prompt", 4),
                global_batch_size=settings.get("global_batch_size", 8),
                rollout_max_response_len=96,
                # Qwen3.8-27B uses context parallelism 4, so each GPU holds a
                # quarter of the full multi-turn trajectory.
                max_tokens_per_gpu=settings["context_len"] // 4,
                sglang_speculative_algorithm=None,
                sglang_speculative_num_steps=None,
                sglang_speculative_eagle_topk=None,
                sglang_speculative_num_draft_tokens=None,
                save_interval=settings.get("save_interval", 1),
                trace_sample_limit=settings["trace_sample_limit"],
                custom_generate_function=generate_balatro_episode,
                custom_rm_function=passthrough_reward,
                custom_reward_post_process_function=normalize_episode_rewards,
                apply_chat_template_kwargs='{"enable_thinking": false}',
                environment=environment,
                extra_config=extra_config,
            ),
        )
    return TrainConfig(
        model=Qwen3_4B(),
        dataset=BalatroSeedsDataset(num_seeds=settings["num_seeds"]),
        recipe=BalatroGRPORecipe(
            num_rollout=settings["num_rollout"],
            save_interval=1 if profile in {"proof", "observable"} else 10,
            tensor_model_parallel_size=settings["tensor_parallel"],
            sequence_parallel=settings["tensor_parallel"] > 1,
            max_tokens_per_gpu=settings["context_len"],
            trace_sample_limit=settings["trace_sample_limit"],
            custom_generate_function=generate_balatro_episode,
            custom_rm_function=passthrough_reward,
            custom_reward_post_process_function=normalize_episode_rewards,
            environment=environment,
            extra_config=extra_config,
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
        result = env.step(_fallback_action(state))
        next_state = result.observation.metadata.get("state", {})
        return {
            "initial": initial,
            "one_action": {
                "error": result.observation.is_error,
                "state": next_state.get("state"),
                "ante": next_state.get("ante_num"),
                "round": next_state.get("round_num"),
            },
        }
    finally:
        pool.release(env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--smoke-env", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="proof")
    parser.add_argument(
        "--num-rollout",
        type=int,
        default=None,
        help="Override the profile's training step count (slime rollouts).",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually launch the H100 Training Gym run.",
    )
    args = parser.parse_args()

    if args.prepare_assets:
        print(json.dumps(_prepare_assets_from_cli(), indent=2))
    if args.smoke_env:
        with modal.enable_output():
            smoke_result = _smoke_environment()
        print(json.dumps(smoke_result, indent=2))

    config = build_train_config(args.profile, num_rollout=args.num_rollout)
    if not args.launch:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "balatro-seeds.jsonl")
            config.dataset.prepare(path)
            config.dataset.validate_prepared(path)
        profile = PROFILES[args.profile]
        rollouts = args.num_rollout or profile["num_rollout"]
        print(
            "Training config validated; no GPUs launched. "
            f"profile={args.profile}, rollouts={rollouts}, "
            f"max_decisions={profile['max_decisions']}."
        )
        if not args.prepare_assets:
            print("First run: python batro_modal_rl.py --prepare-assets")
        launch_cmd = f"python batro_modal_rl.py --profile {args.profile}"
        if args.num_rollout is not None:
            launch_cmd += f" --num-rollout {args.num_rollout}"
        print(f"Launch explicitly: {launch_cmd} --launch")
        return

    run = config.launch()
    print(f"Training Gym run id: {run.training_run_id}")


if __name__ == "__main__":
    main()
