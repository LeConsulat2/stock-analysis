"""Edit settings here or in full-flow/.env. Demo imports no external SDKs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = ("fundamental", "news", "ecosystem", "skeptic", "decision")
DEFAULT_MODELS = {
    "openai": "gpt-6-astra",
    "claude": "claude-fable-5-1",
    "google": "gemini-3.8-flash",
}
KEYS = {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY"}


@dataclass(frozen=True)
class Route:
    provider: str
    model: str


@dataclass(frozen=True)
class Settings:
    display_timezone: str = "Pacific/Auckland"
    horizon_years: int = 3
    max_turns: int = 7
    max_calls_per_turn: int = 4
    max_requests: int = 300
    max_output_tokens: int = 6000
    max_input_chars: int = 100_000
    max_news: int = 6
    news_days: int = 7
    max_price_age_days: int = 5
    max_statement_age_days: int = 550
    buy_cushion: float = 0.15
    timeout_seconds: int = 90

    def __post_init__(self) -> None:
        for name in ("horizon_years", "max_turns", "max_calls_per_turn",
                     "max_requests", "max_output_tokens", "max_input_chars",
                     "max_news", "news_days", "max_price_age_days", "max_statement_age_days", "timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.buy_cushion <= 1:
            raise ValueError("buy_cushion must be between 0 and 1")


def load_routes(provider: str) -> dict[str, Route]:
    """Single provider by default; opt into role-specific providers with mixed."""
    if provider not in {*DEFAULT_MODELS, "mixed"}:
        raise ValueError("Provider must be openai, claude, google, or mixed")
    defaults = {"fundamental": "claude", "news": "google",
                "ecosystem": "google", "skeptic": "claude", "decision": "openai"}
    routes = {}
    for role in ROLES:
        chosen = (os.getenv(f"{role.upper()}_PROVIDER", defaults[role])
                  if provider == "mixed" else provider)
        if chosen not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider for {role}: {chosen}")
        model = os.getenv(f"{role.upper()}_MODEL") or os.getenv(
            f"{chosen.upper()}_MODEL", DEFAULT_MODELS[chosen])
        if not model.strip():
            raise ValueError(f"Empty model for {role}")
        routes[role] = Route(chosen, model)
    return routes


def check_keys(routes: dict[str, Route]) -> None:
    missing = sorted({KEYS[r.provider] for r in routes.values()
                      if not os.getenv(KEYS[r.provider])})
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
