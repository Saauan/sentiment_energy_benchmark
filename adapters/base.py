"""Contract for runner-owned construction and exclusively scheduled adapters."""

class Unavailable(RuntimeError):
    """A precondition is not met; the configured condition cannot be executed as specified."""
