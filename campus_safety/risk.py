"""The explainable risk engine — the project's core differentiator.

A transparent, rule-based score over named signals. Every factor's contribution
is returned individually so the dashboard can draw contribution bars and a judge
can add the numbers up by hand. We deliberately do NOT use a learned severity
model: we have no labelled incident data, and a "glass box" staff can audit and
override beats a black box trained on nothing.

Formula (Slide 6 / Round 3 section 3):

    priority = w_sos    * sos
             + w_video  * video_event
             + w_time   * time_of_day_risk
             + w_zone   * zone_history_risk
             + w_crowd  * crowd_level
             + w_corrob * (sos AND nearby video event)

Hard overrides then force a floor so the score can never under-rate an SOS or a
confirmed fall.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .models import RiskAssessment, Severity, Signal


@dataclass(slots=True)
class RiskInputs:
    """Everything the engine needs to score one incident.

    All continuous values are expected in [0, 1]. Flags are booleans.
    """

    sos: bool = False
    video_event: str | None = None  # "fall" | "crowd" | None
    held_seconds: float = 0.0       # how long the video event persisted
    hour: int = 12                  # local hour, for the time prior
    zone_history_risk: float = config.DEFAULT_ZONE_RISK  # in [0, 1]
    crowd_level: float = 0.0        # normalised crowd in [0, 1]
    corroborating: bool = False     # an SOS and a nearby video event in-window


def _time_risk(hour: int) -> float:
    return config.TIME_OF_DAY_RISK.get(hour % 24, 0.3)


def _band(priority: float) -> Severity:
    for threshold, name in config.SEVERITY_BANDS:
        if priority >= threshold:
            return name  # type: ignore[return-value]
    return "low"


def assess(inputs: RiskInputs) -> RiskAssessment:
    """Score one incident and return the score plus its full breakdown."""
    w = config.RISK_WEIGHTS
    signals: list[Signal] = []

    # --- SOS ---------------------------------------------------------------
    sos_value = 1.0 if inputs.sos else 0.0
    signals.append(
        Signal(
            signal="sos",
            label="Student SOS",
            weight=w["sos"],
            value=sos_value,
            contribution=w["sos"] * sos_value,
        )
    )

    # --- Video event (fall outweighs crowd) --------------------------------
    if inputs.video_event == "fall":
        video_weight, video_label = w["video_fall"], "Video: fall detected"
    elif inputs.video_event == "crowd":
        video_weight, video_label = w["video_crowd"], "Video: crowd detected"
    else:
        video_weight, video_label = w["video_fall"], "Video: no event"
    video_value = 1.0 if inputs.video_event else 0.0
    signals.append(
        Signal(
            signal=f"video_{inputs.video_event or 'none'}",
            label=video_label,
            weight=video_weight,
            value=video_value,
            contribution=video_weight * video_value,
        )
    )

    # --- Time of day -------------------------------------------------------
    t_risk = _time_risk(inputs.hour)
    signals.append(
        Signal(
            signal="time",
            label=f"Time of day ({inputs.hour:02d}:00)",
            weight=w["time"],
            value=t_risk,
            contribution=w["time"] * t_risk,
        )
    )

    # --- Zone history ------------------------------------------------------
    z_risk = max(0.0, min(1.0, inputs.zone_history_risk))
    signals.append(
        Signal(
            signal="zone",
            label="Zone incident history (30d)",
            weight=w["zone"],
            value=z_risk,
            contribution=w["zone"] * z_risk,
        )
    )

    # --- Crowd -------------------------------------------------------------
    c_level = max(0.0, min(1.0, inputs.crowd_level))
    signals.append(
        Signal(
            signal="crowd",
            label="Crowd level",
            weight=w["crowd"],
            value=c_level,
            contribution=w["crowd"] * c_level,
        )
    )

    # --- Corroboration (the verification bonus) ----------------------------
    corrob_value = 1.0 if inputs.corroborating else 0.0
    signals.append(
        Signal(
            signal="corroboration",
            label="SOS corroborated by video",
            weight=w["corroboration"],
            value=corrob_value,
            contribution=w["corroboration"] * corrob_value,
        )
    )

    priority = sum(s.contribution for s in signals)

    # --- Hard overrides ----------------------------------------------------
    overrides: list[str] = []
    if inputs.sos:
        floor = config.OVERRIDES["sos_min_priority"]
        if priority < floor:
            overrides.append(f"SOS floor -> {floor:g}")
            priority = floor
    if inputs.video_event == "fall" and inputs.held_seconds >= config.FALL_HOLD_S:
        floor = config.OVERRIDES["fall_min_priority"]
        if priority < floor:
            overrides.append(f"confirmed fall floor -> {floor:g}")
            priority = floor

    priority = min(priority, config.SCORE_MAX)

    return RiskAssessment(
        priority=priority,
        severity=_band(priority),
        signals=signals,
        overrides=overrides,
    )


def crowd_level_from_count(count: int, threshold: int | None = None) -> float:
    """Normalise a people-count into [0, 1] for the crowd signal."""
    threshold = threshold or config.CROWD_THRESHOLD
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, count / (threshold * 2.0)))


def published_weights() -> dict[str, object]:
    """Everything the admin UI and /api/weights expose.

    Transparency is the point: a judge can read this and reproduce any score.
    """
    return {
        "weights": dict(config.RISK_WEIGHTS),
        "overrides": dict(config.OVERRIDES),
        "severity_bands": [
            {"min": m, "severity": s} for m, s in config.SEVERITY_BANDS
        ],
        "time_of_day_risk": dict(config.TIME_OF_DAY_RISK),
        "default_zone_risk": config.DEFAULT_ZONE_RISK,
        "corroboration_window_s": config.CORROBORATION_WINDOW_S,
        "event_hold_s": {"fall": config.FALL_HOLD_S, "crowd": config.CROWD_HOLD_S},
        "note": (
            "Weights are hand-set for the MVP and published so scores are "
            "auditable. With real incident data we would calibrate them."
        ),
    }
