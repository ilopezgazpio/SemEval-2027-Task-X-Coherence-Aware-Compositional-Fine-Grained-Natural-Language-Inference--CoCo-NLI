"""Exception types used by the DiCo-NLI scorer."""


class ScoringError(Exception):
    """Base exception for errors that should reject a submission."""


class ValidationError(ScoringError):
    """Raised when gold data or participant predictions are malformed."""

