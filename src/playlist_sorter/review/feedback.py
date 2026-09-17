"""Append-only, durable review feedback storage."""

import json
import os
from pathlib import Path

from playlist_sorter.core.contracts import ReviewAction


def validate_review_action(action: ReviewAction) -> None:
    """Validate an action-specific payload before it reaches durable feedback."""
    if not action.action_id or not action.candidate_id or not action.actor:
        raise ValueError("review action ID, candidate ID, and actor must be non-empty")
    if action.action in {"approve", "reject"}:
        valid = action.song_id is None and action.value is None
    elif action.action in {"rename", "split", "merge"}:
        valid = bool(action.value and action.value.strip()) and action.song_id is None
    else:
        valid = bool(action.song_id and action.song_id.strip()) and action.value is None
    if not valid:
        raise ValueError(f"invalid payload for review action: {action.action}")


class FeedbackStore:
    """Persist validated review actions as one fsync-safe JSON object per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(
        self,
        action: ReviewAction,
        *,
        candidate_ids: set[str] | frozenset[str] | None = None,
        song_ids: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Append exactly one action; optional known IDs make validation fail closed."""
        validate_review_action(action)
        if candidate_ids is not None and action.candidate_id not in candidate_ids:
            raise ValueError(f"unknown candidate ID: {action.candidate_id}")
        if action.song_id is not None and song_ids is not None and action.song_id not in song_ids:
            raise ValueError(f"unknown song ID: {action.song_id}")
        payload = json.dumps(action.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[ReviewAction, ...]:
        """Read existing feedback strictly; a corrupt line is never silently ignored."""
        if not self.path.exists():
            return ()
        actions: list[ReviewAction] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(f"blank feedback line {line_number}")
                    action = ReviewAction.model_validate_json(line)
                    validate_review_action(action)
                    actions.append(action)
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid feedback JSONL at {self.path}") from error
        return tuple(actions)
