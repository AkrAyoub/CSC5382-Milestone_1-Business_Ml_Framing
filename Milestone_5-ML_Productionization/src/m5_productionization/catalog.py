from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from m5_productionization.paths import DATA_RAW_DIR, UNCAPOPT_PATH

from m4_model_dev.data.benchmark import discover_raw_instances, parse_orlib_uncap, parse_uncapopt


@dataclass(frozen=True)
class CatalogEntry:
    instance_id: str
    file_name: str
    file_path: Path
    facility_count_m: int
    customer_count_n: int
    best_known: float | None
    file_size_bytes: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "file_name": self.file_name,
            "facility_count_m": self.facility_count_m,
            "customer_count_n": self.customer_count_n,
            "best_known": self.best_known,
            "file_size_bytes": self.file_size_bytes,
            "relative_path": str(self.file_path.relative_to(DATA_RAW_DIR.parent.parent)),
        }


@lru_cache(maxsize=1)
def load_catalog_entries() -> tuple[CatalogEntry, ...]:
    optima = parse_uncapopt(UNCAPOPT_PATH) if UNCAPOPT_PATH.exists() else {}
    entries: list[CatalogEntry] = []
    for path in discover_raw_instances(DATA_RAW_DIR):
        parsed = parse_orlib_uncap(path)
        entries.append(
            CatalogEntry(
                instance_id=parsed.instance_id,
                file_name=path.name,
                file_path=path,
                facility_count_m=parsed.facility_count_m,
                customer_count_n=parsed.customer_count_n,
                best_known=optima.get(parsed.instance_id),
                file_size_bytes=path.stat().st_size,
            )
        )
    return tuple(entries)


def list_catalog_entries() -> list[dict[str, object]]:
    return [entry.to_public_dict() for entry in load_catalog_entries()]


def resolve_catalog_entry(instance_id: str) -> CatalogEntry:
    normalized = instance_id.replace(".txt", "").strip()
    for entry in load_catalog_entries():
        if entry.instance_id == normalized:
            return entry
    raise KeyError(f"Unknown instance_id '{instance_id}'.")


def read_instance_text(instance_id: str) -> str:
    return resolve_catalog_entry(instance_id).file_path.read_text(encoding="utf-8")
