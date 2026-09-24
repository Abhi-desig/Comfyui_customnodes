"""Pydantic settings for the batch controller: YAML file + environment overrides.

Secrets (the Anthropic API key, Slack tokens) come from environment variables
ONLY -- never from the YAML file, and never logged. `AppConfig.load()` is the
single entry point every CLI command uses to get a validated config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .manifest import DEFAULT_MAX_FRAME_COUNT

# Keys that must never appear anywhere in the YAML file (checked recursively,
# case-insensitively) -- if a secret leaks into a committed config file, that
# defeats the whole point of keeping it in the environment.
_SECRET_YAML_KEYS = {
    "anthropic_api_key",
    "slack_bot_token",
    "slack_webhook_url",
    "api_key",
    "token",
    "bot_token",
    "webhook_url",
}

_ENV_PREFIX = "COMFYCTL_"


class ConfigError(RuntimeError):
    """Raised for a malformed config file, or a secret found where it must
    never be (the YAML file)."""


class BudgetConfig(BaseModel):
    max_wall_clock_s: float | None = None
    max_gpu_seconds: float | None = None
    max_judge_usd: float | None = None


class SupervisorConfig(BaseModel):
    stall_timeout: float = 600.0
    canary_timeout: float = 30.0
    hard_ceiling: float = 2700.0
    hysteresis_n: int = 2
    backoff_base: float = 10.0
    backoff_cap: float = 300.0
    intensity_max_restarts: int = 5
    intensity_window: float = 600.0
    recycle_every_n_jobs: int = 40
    launch_argv: list[str] | None = None


class BreakerConfig(BaseModel):
    failure_threshold: int = 5
    cooldown_s: float = 300.0


class GlobalBreakerConfig(BaseModel):
    crash_threshold: int = 3
    crash_window_s: float = 600.0


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comfy_base_url: str = "http://127.0.0.1:8188"
    db_path: str = "comfy_controller.db"
    model_version: str = "default"
    workflow_dir: str | None = None
    comfy_output_dir: str | None = None

    notifier: Literal["console", "slack"] = "console"
    slack_channel: str | None = None  # the channel name isn't secret; the token is

    judge_backend: Literal["claude", "fake"] = "claude"
    rubric_path: str = "config/rubric.yaml"

    # Cross-frame consistency checking for SEQUENCE assets, configured under
    # `frame_consistency:` in the rubric file. On by default -- it is pure
    # numpy/Pillow unless the optional torch/CLIP/LPIPS extra is installed.
    frame_consistency: bool = True
    # Ceiling on a manifest row's `frame_count`; one ComfyUI job per frame.
    max_frame_count: int = DEFAULT_MAX_FRAME_COUNT

    await_outcome_timeout_s: float = 1800.0
    concurrency: int = 4
    tick_interval_s: float = 30.0
    stall_alert_s: float = 900.0
    milestone_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    global_breaker: GlobalBreakerConfig = Field(default_factory=GlobalBreakerConfig)

    # Populated only in `load()`, only from the environment. Never set these
    # directly from a YAML mapping.
    anthropic_api_key: str | None = Field(default=None, exclude=True, repr=False)
    slack_bot_token: str | None = Field(default=None, exclude=True, repr=False)
    slack_webhook_url: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except OSError as exc:
            raise ConfigError(f"could not read config file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top-level YAML must be a mapping")

        found_secrets = _SECRET_YAML_KEYS & _all_keys(raw)
        if found_secrets:
            raise ConfigError(
                f"{path}: secret key(s) {sorted(found_secrets)} must never appear in the config "
                "file -- set them as environment variables instead (e.g. ANTHROPIC_API_KEY, "
                "SLACK_BOT_TOKEN, SLACK_WEBHOOK_URL)."
            )

        _apply_env_overrides(raw)

        try:
            cfg = cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"{path}: {exc}") from exc

        cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        cfg.slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
        cfg.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

        if cfg.judge_backend == "claude" and not cfg.anthropic_api_key:
            raise ConfigError("judge_backend=claude requires ANTHROPIC_API_KEY to be set in the environment")
        if cfg.notifier == "slack" and not (cfg.slack_bot_token or cfg.slack_webhook_url):
            raise ConfigError(
                "notifier=slack requires SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL to be set in the environment"
            )

        return cfg


def _all_keys(node: Any, acc: set[str] | None = None) -> set[str]:
    acc = acc if acc is not None else set()
    if isinstance(node, dict):
        for k, v in node.items():
            acc.add(str(k).lower())
            _all_keys(v, acc)
    elif isinstance(node, list):
        for item in node:
            _all_keys(item, acc)
    return acc


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """Non-secret overrides only, e.g. `COMFYCTL_DB_PATH=/tmp/x.db`. Only
    top-level scalar fields are overridable this way -- nested config
    (budget/supervisor/breaker/global_breaker) is YAML-only, which keeps this
    mechanism simple and keeps secrets from sneaking in through it too."""
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        field_name = key[len(_ENV_PREFIX) :].lower()
        if field_name in _SECRET_YAML_KEYS:
            continue
        raw[field_name] = value
