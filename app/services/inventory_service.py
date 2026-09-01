"""Phase 1 folder inventory orchestration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.domain.models import DocumentStatus, IndexAction, IndexPlanItem, RunStatus
from app.indexing.planner import IncrementalIndexPlanner
from app.ingestion.scanner import FolderScanner, ScanResult
from app.infrastructure.paths import canonicalize_path
from app.persistence.connection import Database
from app.persistence.repositories import DocumentRepository, IndexingRunRepository
from app.persistence.schema import initialize_schema


class InventoryService:
    """Scan a folder and persist Phase 1 incremental inventory decisions."""

    def __init__(self, database: Database, scanner: FolderScanner) -> None:
        self._database = database
        self._scanner = scanner
        self._planner = IncrementalIndexPlanner()

    def inventory(self, root: Path) -> tuple[IndexPlanItem, ...]:
        """Run strict-hash discovery and atomically update inventory metadata."""

        scan = self._scanner.scan(root)
        return self._persist_scan(scan, canonicalize_path(root), include_missing=True)

    def inventory_files(self, paths: tuple[Path, ...]) -> tuple[IndexPlanItem, ...]:
        """Inventory explicit files selected by a desktop user."""

        scan = self._scanner.scan_paths(paths)
        root_label = canonicalize_path(paths[0].parent) if paths else "<explicit-files>"
        return self._persist_scan(scan, root_label, include_missing=False)

    def _persist_scan(self, scan: ScanResult, root_label: str, *,
                      include_missing: bool) -> tuple[IndexPlanItem, ...]:
        """Persist one scanner result through the shared incremental planner."""

        with self._database.transaction() as connection:
            initialize_schema(connection)
            documents = DocumentRepository(connection)
            runs = IndexingRunRepository(connection)
            run_id = runs.start(root_label)
            plan = self._planner.build_plan(
                scan.fingerprints, documents.list_all(), include_missing=include_missing
            )

            canonical_ids: dict[str, int] = {}
            for item in plan:
                if item.action in {IndexAction.DUPLICATE, IndexAction.MISSING}:
                    continue
                if item.fingerprint is None:
                    continue
                previous = documents.get_by_path(item.canonical_path)
                preserved_status = (
                    previous.status
                    if item.action is IndexAction.UNCHANGED and previous is not None
                    else DocumentStatus.DISCOVERED
                )
                saved = documents.save_fingerprint(
                    item.fingerprint, status=preserved_status
                )
                documents.ensure_index_state(
                    saved.id,
                    run_id,
                    reset_pipeline=item.action
                    in {IndexAction.NEW, IndexAction.MODIFIED},
                )
                canonical_ids[item.canonical_path] = saved.id

            for item in plan:
                if item.action is IndexAction.DUPLICATE and item.fingerprint is not None:
                    canonical_id = canonical_ids.get(item.duplicate_of_path or "")
                    if canonical_id is None and item.duplicate_of_path:
                        canonical = documents.get_by_path(item.duplicate_of_path)
                        canonical_id = canonical.id if canonical else None
                    if canonical_id is None:
                        raise RuntimeError("Duplicate canonical document was not persisted")
                    saved = documents.save_fingerprint(
                        item.fingerprint,
                        status=DocumentStatus.DUPLICATE,
                        duplicate_of_id=canonical_id,
                    )
                    documents.ensure_index_state(
                        saved.id,
                        run_id,
                        reset_pipeline=True,
                        inventory_status="skipped",
                    )
                elif item.action is IndexAction.MISSING:
                    if item.existing_document_id is not None:
                        documents.mark_missing(item.existing_document_id)

            action_counts = Counter(item.action.value for item in plan)
            counts = {
                "discovered_count": len(scan.fingerprints),
                "new_count": action_counts[IndexAction.NEW.value],
                "unchanged_count": action_counts[IndexAction.UNCHANGED.value],
                "modified_count": action_counts[IndexAction.MODIFIED.value],
                "duplicate_count": action_counts[IndexAction.DUPLICATE.value],
                "missing_count": action_counts[IndexAction.MISSING.value],
                "failed_count": len(scan.failures),
            }
            status = RunStatus.PARTIAL if scan.failures else RunStatus.COMPLETED
            runs.finish(run_id, status, counts)
            return plan
