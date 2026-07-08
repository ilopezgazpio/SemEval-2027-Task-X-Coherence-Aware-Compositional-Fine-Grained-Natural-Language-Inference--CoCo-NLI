"""Track-level aggregation helpers for DiCo-NLI reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .errors import ValidationError
from .metrics import EvaluationReport


@dataclass(frozen=True, slots=True)
class ScoreProfile:
    """The three official scores for one track or aggregate."""

    weighted_f1: float
    soft_cons: float
    hard_cons: float

    def to_dict(self) -> dict[str, float]:
        return {
            "weighted_f1": self.weighted_f1,
            "soft_cons": self.soft_cons,
            "hard_cons": self.hard_cons,
        }


@dataclass(frozen=True, slots=True)
class TrackAggregateReport:
    """Macro-average profile over a selected set of tracks."""

    name: str
    tracks: tuple[str, ...]
    macro_average: ScoreProfile
    per_track: dict[str, ScoreProfile]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tracks": list(self.tracks),
            "macro_average": self.macro_average.to_dict(),
            "per_track": {
                track_name: profile.to_dict()
                for track_name, profile in self.per_track.items()
            },
        }


def macro_average_reports(
    reports_by_track: Mapping[str, EvaluationReport],
    *,
    tracks: tuple[str, ...] | None = None,
    name: str = "macro_average",
) -> TrackAggregateReport:
    """Macro-average the three official scores over selected tracks.

    This function intentionally gives every selected track equal weight. It does
    not micro-average by instance count, because DiCo-NLI's official reporting is
    track-specific and any cross-track summary should not let a larger track
    dominate a smaller language track.
    """

    selected_tracks = tracks if tracks is not None else tuple(reports_by_track.keys())
    if not selected_tracks:
        raise ValidationError("Cannot macro-average zero tracks.")

    missing = [track for track in selected_tracks if track not in reports_by_track]
    if missing:
        raise ValidationError(f"Missing report(s) for track(s): {', '.join(missing)}.")

    per_track = {
        track: ScoreProfile(
            weighted_f1=reports_by_track[track].weighted_f1,
            soft_cons=reports_by_track[track].soft_cons,
            hard_cons=reports_by_track[track].hard_cons,
        )
        for track in selected_tracks
    }
    denominator = len(selected_tracks)
    macro_average = ScoreProfile(
        weighted_f1=sum(profile.weighted_f1 for profile in per_track.values()) / denominator,
        soft_cons=sum(profile.soft_cons for profile in per_track.values()) / denominator,
        hard_cons=sum(profile.hard_cons for profile in per_track.values()) / denominator,
    )
    return TrackAggregateReport(
        name=name,
        tracks=selected_tracks,
        macro_average=macro_average,
        per_track=per_track,
    )
