"""Deterministic reliability boundary: retries, idempotency, breaker, and DLQ."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable


class FailureKind(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    AMBIGUOUS = "ambiguous"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class IdempotencyStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"


class OperationFailure(RuntimeError):
    def __init__(self, kind: FailureKind, message: str):
        super().__init__(message); self.kind = kind


@dataclass(frozen=True)
class AttemptResult:
    value: str
    deduplicated: bool = False


@dataclass
class IdempotencyRecord:
    key: str
    payload: str
    result: str | None = None
    status: IdempotencyStatus = IdempotencyStatus.STARTED


@dataclass(frozen=True)
class DeadLetter:
    key: str
    payload: str
    failure_kind: FailureKind
    attempts: int
    last_error: str


class IdempotentReceiver:
    def __init__(self):
        self.records: dict[str, IdempotencyRecord] = {}
        self.side_effect_count = 0
        self._lock = RLock()

    def refund(self, key: str, payload: str, outcome: str = "success") -> AttemptResult:
        with self._lock:
            return self._refund_locked(key, payload, outcome)

    def _refund_locked(self, key: str, payload: str, outcome: str) -> AttemptResult:
        if key in self.records:
            record = self.records[key]
            if record.payload != payload:
                raise ValueError("idempotency key reused with different payload")
            if record.status == IdempotencyStatus.STARTED:
                raise OperationFailure(FailureKind.TRANSIENT, "operation already in progress")
            assert record.result is not None
            return AttemptResult(record.result, deduplicated=True)
        if outcome == "transient":
            raise OperationFailure(FailureKind.TRANSIENT, "503 unavailable")
        if outcome == "permanent":
            raise OperationFailure(FailureKind.PERMANENT, "invalid account")
        record = IdempotencyRecord(key, payload)
        self.records[key] = record
        self.side_effect_count += 1
        result = f"refund-{self.side_effect_count}"
        record.result = result
        record.status = IdempotencyStatus.COMPLETED
        if outcome == "ambiguous":
            raise OperationFailure(FailureKind.AMBIGUOUS, "response lost after commit")
        return AttemptResult(result)


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: int = 30):
        self.threshold, self.cooldown = threshold, cooldown
        self.state, self.failures, self.opened_at = CircuitState.CLOSED, 0, None

    def allow(self, now: int) -> bool:
        if self.state == CircuitState.OPEN and now - int(self.opened_at) >= self.cooldown:
            self.state = CircuitState.HALF_OPEN
        return self.state != CircuitState.OPEN

    def success(self) -> None:
        self.state, self.failures, self.opened_at = CircuitState.CLOSED, 0, None

    def failure(self, now: int) -> None:
        self.failures += 1
        if self.state == CircuitState.HALF_OPEN or self.failures >= self.threshold:
            self.state, self.opened_at = CircuitState.OPEN, now


def full_jitter(base: float, attempt: int, random_value: Callable[[], float]) -> float:
    return random_value() * base * (2 ** (attempt - 1))


class ReliabilityBoundary:
    def __init__(
        self,
        max_attempts: int = 3,
        breaker: CircuitBreaker | None = None,
        random_value: Callable[[], float] = lambda: 0.5,
        sleeper: Callable[[float], None] = lambda _delay: None,
        attempt_timeout: float = 5.0,
        duration_source: Callable[[int], float] = lambda _attempt: 0.0,
    ):
        self.max_attempts = max_attempts
        self.breaker = breaker or CircuitBreaker()
        self.random_value = random_value
        self.sleeper = sleeper
        self.attempt_timeout = attempt_timeout
        self.duration_source = duration_source
        self.dead_letters: list[DeadLetter] = []
        self.telemetry: list[dict] = []

    def execute(self, key: str, payload: str, operation: Callable[[], AttemptResult], *, now: int = 0) -> AttemptResult | None:
        if not self.breaker.allow(now):
            self.dead_letters.append(DeadLetter(key, payload, FailureKind.TRANSIENT, 0, "circuit open"))
            return None
        for attempt in range(1, self.max_attempts + 1):
            try:
                duration = self.duration_source(attempt)
                if duration > self.attempt_timeout:
                    raise OperationFailure(FailureKind.TRANSIENT, "attempt deadline exceeded")
                result = operation()
                self.breaker.success()
                self.telemetry.append({"key": key, "attempt": attempt, "outcome": "success", "deduplicated": result.deduplicated, "duration": duration})
                return result
            except OperationFailure as error:
                self.telemetry.append({"key": key, "attempt": attempt, "outcome": error.kind.value, "error": str(error)})
                retryable = error.kind in {FailureKind.TRANSIENT, FailureKind.AMBIGUOUS}
                if not retryable or attempt == self.max_attempts:
                    self.breaker.failure(now)
                    self.dead_letters.append(DeadLetter(key, payload, error.kind, attempt, str(error)))
                    return None
                delay = full_jitter(1.0, attempt, self.random_value)
                self.telemetry[-1]["jitter_delay"] = delay
                self.sleeper(delay)
        raise AssertionError("unreachable")

    def replay(self, letter: DeadLetter, operation: Callable[[], AttemptResult], *, now: int) -> AttemptResult | None:
        return self.execute(letter.key, letter.payload, operation, now=now)


def classify_exception(error: Exception) -> FailureKind:
    if isinstance(error, OperationFailure): return error.kind
    if isinstance(error, (TimeoutError, ConnectionError)): return FailureKind.TRANSIENT
    return FailureKind.PERMANENT


def end_to_end_independent(step_probabilities: list[float]) -> float:
    """Multiply required independent step probabilities; correlation is not modeled."""
    result = 1.0
    for probability in step_probabilities: result *= probability
    return result


def validate_model_output(payload: dict) -> dict:
    if not isinstance(payload.get("amount"), (int, float)):
        raise ValueError("amount must be numeric")
    return payload


def repair_model_output(raw: dict, repairer: Callable[[dict, str], dict]) -> tuple[dict, str]:
    """Make one new model attempt with validation feedback, not a transport retry."""
    try:
        return validate_model_output(raw), ""
    except ValueError as error:
        feedback = str(error)
        return validate_model_output(repairer(raw, feedback)), feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", choices=("success", "transient", "permanent", "ambiguous", "malformed", "correlated"), default="ambiguous")
    args = parser.parse_args()
    if args.failure == "malformed":
        repaired, feedback = repair_model_output({"amount": "240"}, lambda raw, error: {"amount": float(raw["amount"]), "feedback": error})
        print(f"feedback={feedback}\nrepaired={repaired}")
        return
    if args.failure == "correlated":
        boundary = ReliabilityBoundary(max_attempts=1, breaker=CircuitBreaker(threshold=2))
        for number in range(3):
            boundary.execute(f"order-{number}:refund", "240.00", lambda: (_ for _ in ()).throw(OperationFailure(FailureKind.TRANSIENT, "shared dependency down")), now=number)
        print(f"breaker={boundary.breaker.state.value}\ndead_letters={boundary.dead_letters}")
        return
    receiver, boundary = IdempotentReceiver(), ReliabilityBoundary()
    outcomes = iter([args.failure, "success"])
    result = boundary.execute("order-1:refund", "240.00", lambda: receiver.refund("order-1:refund", "240.00", next(outcomes, "success")))
    print(f"result={result}\nside_effects={receiver.side_effect_count}\ntelemetry={boundary.telemetry}\ndead_letters={boundary.dead_letters}")


if __name__ == "__main__":
    main()
