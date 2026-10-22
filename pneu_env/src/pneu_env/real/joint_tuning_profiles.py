"""Offline construction of the six-valve calibration input (no hardware I/O).

The 657 s profile order comes from the May 15 joint_tuning recording. This
is a new, reproducible 0.8--1.0 protocol, not the missing May 20 source code.
Only supply valves 1 and 2 have shortened low-command visits. The limit is
on commanded exposure, not a pressure limit or a hardware watchdog.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from pneu_env.real.flowrate_profiles import stair_value


@dataclass(frozen=True)
class JointTuningConfig:
    freq: float = 50.0
    command_min: float = 0.8
    command_max: float = 1.0
    supply_low_threshold: float = 0.85
    supply_low_max_sec: float = 0.2
    seed: int = 20261022
    idle_ramps: bool = True
    idle_ramp_period_sec: float = 20.0
    idle_ramp_fade_sec: float = 0.5
    solo_prefix_sec: float = 10.0

    def __post_init__(self):
        if not np.isfinite(self.freq) or self.freq <= 0:
            raise ValueError("freq must be finite and positive")
        if not 0 <= self.command_min < self.supply_low_threshold < self.command_max <= 1:
            raise ValueError("Require 0 <= min < low threshold < max <= 1")
        if not 1 / self.freq <= self.supply_low_max_sec < 2:
            raise ValueError("Low-command budget must be at least one sample and < 2 s")
        if (not np.isfinite(self.idle_ramp_period_sec)
                or self.idle_ramp_period_sec <= self.supply_low_max_sec):
            raise ValueError("Idle ramp period must exceed the low-command budget")
        if not np.isfinite(self.idle_ramp_fade_sec) or self.idle_ramp_fade_sec <= 0:
            raise ValueError("Idle ramp fade must be finite and positive")
        if not np.isfinite(self.solo_prefix_sec) or self.solo_prefix_sec < 0:
            raise ValueError("Solo prefix must be finite and nonnegative")


def joint_profiles() -> list[dict]:
    """Keep the approved 657 s windows; unused time becomes an open hold."""
    profiles = []

    def add(name, mode, duration, channel=None, **kwargs):
        profiles.append(dict(name=name, mode=mode, duration=float(duration),
                             channel=channel, **kwargs))

    def hold(name, duration=5.0):
        add(name, "hold", duration)

    def triangles(channel):
        for period, duration, suffix in ((10.0, 30.0, "0p1hz"), (2.0, 12.0, "0p5hz")):
            add(f"joint_v{channel}_triangle_{suffix}", "triangle_wave", duration,
                channel, period=period)

    def stairs(channel):
        add(f"joint_v{channel}_stairs", "random_staircase", 45.0, channel,
            hold_sec=2.0, transition_sec=0.15)

    hold("joint_all_open_start")
    for channel in (1, 2):
        triangles(channel)
        hold(f"joint_all_open_after_v{channel}")
    for channel in (3, 4, 5, 6):
        triangles(channel)
        stairs(channel)
    stairs(1)
    hold("joint_all_open_between_supply_stairs")
    stairs(2)
    add("joint_low_pulse_all_six", "hold", 5.0, low_all=True)
    hold("joint_all_open_after_low_pulse")
    for channel in (1, 2):
        add(f"joint_v{channel}_ramp", "ramp", 25.0, channel,
            ramp_up_sec=10.0, hold_peak_sec=2.5, ramp_down_sec=10.0)
        hold(f"joint_all_open_after_ramp_v{channel}")
    add("joint_random_all_six", "random", 40.0, hold_min=1, hold_max=3)
    hold("joint_all_open_end")
    return profiles


def _low_mask(values, threshold):
    return values < threshold - 1e-12


def _triangle_values(t, period, channel, config):
    phase = t % period
    low, high = config.command_min, config.command_max
    if channel in (1, 2):
        cross = config.supply_low_max_sec / 2
        return np.interp(phase,
            [0, cross, period / 2, period - cross, period],
            [low, config.supply_low_threshold, high, config.supply_low_threshold, low])
    return low + (high - low) * (1 - abs(2 * phase / period - 1))


def _shorten_low_visits(values, threshold, max_frames, open_value):
    """Compress each entire low run, including consecutive low stair levels."""
    low = _low_mask(values, threshold)
    boundaries = np.r_[0, np.flatnonzero(low[1:] != low[:-1]) + 1, len(values)]
    parts = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = values[start:end]
        if low[start] and len(segment) > max_frames:
            indices = np.rint(np.linspace(0, len(segment) - 1, max_frames)).astype(int)
            segment = segment[indices]
        parts.append(segment)
    compact = np.concatenate(parts)
    result = np.full_like(values, open_value)
    result[:len(compact)] = compact
    return result, len(compact)


def _limit_supply_runs(requested, threshold, max_frames, open_value):
    """Latch open for the rest of a requested low run, even across profiles."""
    commands = requested.copy()
    limited = np.zeros((len(commands), 2), dtype=bool)
    for channel in range(2):
        consecutive = 0
        for frame, is_low in enumerate(_low_mask(requested[:, channel], threshold)):
            consecutive = consecutive + 1 if is_low else 0
            if consecutive > max_frames:
                commands[frame, channel] = open_value
                limited[frame, channel] = True
    return commands, limited


class JointTuningSuite:
    def __init__(self, config: JointTuningConfig):
        self.config = config
        self.collection_type = "type2" if config.idle_ramps else "type1"
        self.saved_preset = None
        self.observation_timeout_sec = 0.5
        self.profiles = joint_profiles()
        self.max_low_frames = int(np.floor(config.supply_low_max_sec * config.freq + 1e-9))
        self.period = 1 / config.freq
        self.duration = sum(p["duration"] for p in self.profiles)
        commands, indices, local_times = [], [], []
        start = 0.0
        for index, profile in enumerate(self.profiles):
            n_float = profile["duration"] * config.freq
            if not np.isclose(n_float, round(n_float)):
                raise ValueError("Profile durations must be multiples of the command period")
            t = np.arange(int(round(n_float))) / config.freq
            profile["start_sec"] = start
            profile["end_sec"] = start + profile["duration"]
            profile["signal_duration_sec"] = profile["duration"]
            commands.append(self._profile_commands(profile, t, index))
            indices.append(np.full(len(t), index, dtype=int))
            local_times.append(t)
            start = profile["end_sec"]
        self.primary_requested = np.concatenate(commands)
        self.requested = self.primary_requested.copy()
        self.profile_indices = np.concatenate(indices)
        self.local_times = np.concatenate(local_times)
        self.times = np.arange(len(self.requested)) / config.freq
        self.idle_ramp_mask = np.zeros_like(self.requested, dtype=bool)
        if config.idle_ramps:
            self._add_idle_ramps()
        self.commands, self.limited = _limit_supply_runs(
            self.requested, config.supply_low_threshold, self.max_low_frames,
            config.command_max)

    @classmethod
    def from_preset(cls, collection_type):
        if collection_type not in ("type1", "type2"):
            raise ValueError("collection_type must be type1 or type2")
        path = (Path(__file__).resolve().parents[3] / "config" / "collection"
                / f"lib3_joint_tuning_{collection_type}.json")
        raw = path.read_bytes()
        preset = json.loads(raw)
        if preset["collection_type"] != collection_type:
            raise ValueError(f"Preset type does not match its filename: {path}")
        config = JointTuningConfig(**preset["config"])
        if config.idle_ramps is not (collection_type == "type2"):
            raise ValueError("type1 must disable idle ramps; type2 must enable them")
        timeout = float(preset["observation_timeout_sec"])
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Observation timeout must be finite and positive")
        suite = cls(config)
        suite.observation_timeout_sec = timeout
        suite.saved_preset = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                  contents=preset)
        return suite

    def _add_idle_ramps(self):
        """Excite non-target valves after an initial solo measurement window."""
        cfg = self.config
        for profile in self.profiles:
            if profile["mode"] not in ("triangle_wave", "random_staircase", "ramp"):
                continue  # Preserve all-open recovery, low pulse, and random blocks.
            prefix = (profile["period"] if profile["mode"] == "triangle_wave"
                      else cfg.solo_prefix_sec)
            prefix = min(prefix, profile["duration"])
            profile["solo_duration_sec"] = prefix
            start = int(round(profile["start_sec"] * cfg.freq))
            end = int(round(profile["end_sec"] * cfg.freq))
            local = self.local_times[start:end]
            fade = min(cfg.idle_ramp_fade_sec, (profile["duration"] - prefix) / 2)
            if fade <= 0:
                continue
            envelope = np.clip(np.minimum((local - prefix) / fade,
                                         (profile["duration"] - local) / fade), 0, 1)
            for channel in range(1, 7):
                if channel == profile["channel"]:
                    continue  # Keep the full target-valve waveform intact.
                phase_offset = (channel - 1) * cfg.idle_ramp_period_sec / 6
                ramp = _triangle_values(self.times[start:end] + phase_offset,
                                        cfg.idle_ramp_period_sec, channel, cfg)
                values = cfg.command_max + envelope * (ramp - cfg.command_max)
                self.requested[start:end, channel - 1] = values
                self.idle_ramp_mask[start:end, channel - 1] = envelope > 0

    def _profile_commands(self, profile, t, index):
        cfg = self.config
        low, high = cfg.command_min, cfg.command_max
        commands = np.full((len(t), 6), high, dtype=float)
        mode = profile["mode"]
        channel = profile["channel"]
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, index]))
        if mode == "hold":
            if profile.get("low_all"):
                commands[:] = low
            return commands
        if mode == "random":
            frame = 0
            while frame < len(t):
                hold = rng.integers(profile["hold_min"], profile["hold_max"] + 1)
                count = int(round(hold * cfg.freq))
                commands[frame:frame + count] = rng.uniform(low, high, size=6)
                frame += count
            return commands
        if mode == "triangle_wave":
            values = _triangle_values(t, profile["period"], channel, cfg)
        elif mode == "random_staircase":
            mids = np.sort(rng.uniform(low, high, size=9))
            rising = np.r_[low, mids, high]
            levels = np.r_[rising, rising[-2::-1]]
            values = np.array([stair_value(x, levels=levels,
                hold_s=profile["hold_sec"], transition_s=profile["transition_sec"])
                for x in t])
        elif mode == "ramp":
            up = profile["ramp_up_sec"]
            down_start = up + profile["hold_peak_sec"]
            down_end = down_start + profile["ramp_down_sec"]
            values = np.interp(t, [0, up, down_start, down_end], [low, high, high, low])
            # The former 2.5 s minimum hold is now an open recovery hold.
            values[t > down_end + 1e-12] = high
        else:
            raise ValueError(f"Unsupported joint profile: {mode}")
        if channel in (1, 2) and mode in ("random_staircase", "ramp"):
            budget = self.max_low_frames
            if mode == "ramp":
                budget = max(1, self.max_low_frames // 2)
            values, active_frames = _shorten_low_visits(
                values, cfg.supply_low_threshold, budget, high)
            profile["signal_duration_sec"] = active_frames / cfg.freq
        commands[:, channel - 1] = values
        return commands

    def sample(self, elapsed_s):
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        frame = int(np.floor(elapsed_s * self.config.freq + 1e-9))
        if frame >= len(self.commands):
            opened = np.full(6, self.config.command_max)
            return opened, dict(profile_idx=len(self.profiles), profile_mode="idle",
                profile_time=0.0, profile_period=0.0, requested=opened.copy(),
                low_limited=np.zeros(2, dtype=bool), idle_ramp_mask=np.zeros(6, dtype=bool))
        index = int(self.profile_indices[frame])
        profile = self.profiles[index]
        return self.commands[frame].copy(), dict(profile_idx=index,
            profile_mode=profile["mode"], profile_time=float(self.local_times[frame]),
            profile_period=float(profile.get("period", 0.0)),
            requested=self.requested[frame].copy(), low_limited=self.limited[frame].copy(),
            idle_ramp_mask=self.idle_ramp_mask[frame].copy())

    def max_low_seconds(self):
        maxima = []
        for channel in range(6):
            longest = count = 0
            for low in _low_mask(self.commands[:, channel], self.config.supply_low_threshold):
                count = count + 1 if low else 0
                longest = max(longest, count)
            maxima.append(longest / self.config.freq)
        return maxima

    def manifest(self):
        return dict(protocol="joint_tuning_657s_u080_supply_short_idle_ramps_v2"
                    if self.config.idle_ramps else "joint_tuning_657s_u080_supply_short_v1",
            basis="May 15 657 s profile order; not a reproduction of the May 20 CSV",
            collection_type=self.collection_type, saved_preset=self.saved_preset,
            config=asdict(self.config), duration_sec=self.duration,
            sample_count=len(self.commands), profiles=self.profiles,
            commands_sha256=hashlib.sha256(self.commands.astype("<f8").tobytes()).hexdigest(),
            max_contiguous_low_command_sec=self.max_low_seconds(),
            command_coverage=self.coverage(),
            supply_limit="u < threshold; continuous across profile/random boundaries",
            limitations=["Command dwell limit is not a pressure interlock or hardware watchdog.",
                "Real command duration depends on bridge/RT timing and communication.",
                "No pump control or relief model is added."])

    def coverage(self):
        cfg = self.config
        maxima = self.max_low_seconds()
        return [dict(valve=channel + 1,
            full_open_percent=float(100 * np.mean(self.commands[:, channel] >= cfg.command_max - 1e-9)),
            non_full_open_sec=float(np.count_nonzero(self.commands[:, channel] < cfg.command_max - 1e-9) / cfg.freq),
            idle_ramp_sec=float(np.count_nonzero(self.idle_ramp_mask[:, channel]) / cfg.freq),
            low_command_sec=float(np.count_nonzero(_low_mask(self.commands[:, channel], cfg.supply_low_threshold)) / cfg.freq),
            longest_low_command_sec=maxima[channel]) for channel in range(6)]

    def write_preview(self, output_dir, plot=True):
        """Export commands, profile windows, and provenance without bridge I/O."""
        import csv
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "commands.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "profile_idx"] + [f"ctrl{i}" for i in range(1, 7)]
                + ["supply_low_limited1", "supply_low_limited2"]
                + [f"idle_ramp{i}" for i in range(1, 7)])
            for t, index, ctrl, limited, ramp_mask in zip(self.times, self.profile_indices,
                    self.commands, self.limited, self.idle_ramp_mask):
                writer.writerow([t, index, *ctrl, *limited.astype(int), *ramp_mask.astype(int)])
        with (output_dir / "profiles.csv").open("w", newline="") as f:
            keys = ["name", "mode", "channel", "start_sec", "end_sec",
                    "duration", "signal_duration_sec", "solo_duration_sec"]
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.profiles)
        with (output_dir / "coverage.csv").open("w", newline="") as f:
            coverage = self.coverage()
            writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
            writer.writeheader()
            writer.writerows(coverage)
        (output_dir / "config.json").write_text(json.dumps(self.manifest(), indent=2) + "\n")
        if plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(6, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
            for i, ax in enumerate(axes):
                ax.plot(self.times, self.commands[:, i], linewidth=0.65, label="Command")
                ax.plot(self.times, np.where(self.idle_ramp_mask[:, i], self.commands[:, i], np.nan),
                        color="darkorange", linewidth=0.7, label="Added idle-channel ramp")
                ax.axhline(self.config.supply_low_threshold, color="gray", linestyle="--", linewidth=.6)
                ax.set_ylabel(f"Valve {i + 1}")
                ax.set_ylim(self.config.command_min - .02, self.config.command_max + .02)
                ax.grid(alpha=.2)
            cfg = self.config
            axes[0].set_title(f"{self.duration:g} s joint_tuning / "
                f"{cfg.command_min:g}--{cfg.command_max:g} / "
                f"supply low-command budget {cfg.supply_low_max_sec:g} s / "
                f"idle ramps {'ON' if cfg.idle_ramps else 'OFF'}")
            axes[0].legend(loc="lower right", fontsize=8)
            axes[-1].set_xlabel("Planned elapsed time (s)")
            fig.savefig(output_dir / "commands.png", dpi=150)
            plt.close(fig)
        return output_dir
