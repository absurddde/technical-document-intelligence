"""Deterministic strict-hash incremental indexing decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from app.domain.models import Document, FileFingerprint, IndexAction, IndexPlanItem


class IncrementalIndexPlanner:
    """Plan inventory changes using SHA-256 as the final content identity."""

    def build_plan(
        self,
        discovered: Iterable[FileFingerprint],
        existing: Iterable[Document],
        *,
        include_missing: bool = True,
    ) -> tuple[IndexPlanItem, ...]:
        """Classify current and missing paths without parsing their contents."""

        current = {item.canonical_path: item for item in discovered}
        previous = {item.canonical_path: item for item in existing}
        hash_groups: dict[str, list[FileFingerprint]] = defaultdict(list)
        for fingerprint in current.values():
            hash_groups[fingerprint.sha256].append(fingerprint)

        existing_canonical_for_hash = {
            item.sha256: item.canonical_path
            for item in previous.values()
            if item.status.value not in {"duplicate", "missing"}
        }
        canonical_for_hash = {
            sha256: min(group, key=lambda item: item.canonical_path).canonical_path
            for sha256, group in hash_groups.items()
        }
        canonical_for_hash.update({
            sha256: path for sha256, path in existing_canonical_for_hash.items()
            if sha256 in hash_groups
        })

        plan: list[IndexPlanItem] = []
        for path in sorted(current):
            fingerprint = current[path]
            old = previous.get(path)
            duplicate_of = canonical_for_hash[fingerprint.sha256]
            if duplicate_of != path:
                action = IndexAction.DUPLICATE
            elif old is None:
                action = IndexAction.NEW
            elif old.sha256 == fingerprint.sha256:
                action = IndexAction.UNCHANGED
            else:
                action = IndexAction.MODIFIED

            plan.append(
                IndexPlanItem(
                    action=action,
                    canonical_path=path,
                    fingerprint=fingerprint,
                    existing_document_id=old.id if old else None,
                    duplicate_of_path=duplicate_of if duplicate_of != path else None,
                )
            )

        if include_missing:
            for path in sorted(set(previous) - set(current)):
                old = previous[path]
                plan.append(
                    IndexPlanItem(
                        action=IndexAction.MISSING,
                        canonical_path=path,
                        fingerprint=None,
                        existing_document_id=old.id,
                    )
                )
        return tuple(plan)
