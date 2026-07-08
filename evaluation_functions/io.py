"""Safe loading for fixed DiCo-NLI reference and prediction CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .errors import ValidationError
from .labels import validate_label
from .schemas import (
    GoldInstance,
    PredictionRecord,
    REQUIRED_PREDICTION_FIELDS,
    REQUIRED_REFERENCE_FIELDS,
)

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 5_000_000
DEFAULT_MAX_FIELD_SIZE = 1_000_000


def read_prediction_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    allow_extra_fields: bool = False,
) -> list[PredictionRecord]:
    """Read a flat participant prediction CSV/TSV file."""

    rows = _read_rows(
        path,
        required_fields=REQUIRED_PREDICTION_FIELDS,
        allow_extra_fields=allow_extra_fields,
        max_bytes=max_bytes,
        max_rows=max_rows,
    )
    records: list[PredictionRecord] = []
    for row_number, row in rows:
        records.append(
            PredictionRecord(
                instance_id=_required_text(row, "instance_id", row_number),
                label=validate_label(row.get("label"), field_name=f"label at row {row_number}"),
            )
        )
    return records


def read_reference_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[GoldInstance]:
    """Read a fixed hidden reference CSV/TSV file.

    Reference files are produced by the organizer-side dataset production tools.
    The evaluator is intentionally agnostic to tracks: every row in this file is
    one final scorable instance.
    """

    rows = _read_rows(
        path,
        required_fields=REQUIRED_REFERENCE_FIELDS,
        allow_extra_fields=True,
        max_bytes=max_bytes,
        max_rows=max_rows,
    )
    records: list[GoldInstance] = []
    for row_number, row in rows:
        reverse_pair_id = row.get("reverse_pair_id", "").strip() or None
        records.append(
            GoldInstance(
                instance_id=_required_text(row, "instance_id", row_number),
                pair_id=_required_text(row, "pair_id", row_number),
                reverse_pair_id=reverse_pair_id,
                label=validate_label(row.get("label"), field_name=f"label at row {row_number}"),
            )
        )
    return records


def _read_rows(
    path: str | Path,
    *,
    required_fields: frozenset[str],
    allow_extra_fields: bool,
    max_bytes: int,
    max_rows: int,
) -> list[tuple[int, dict[str, str]]]:
    file_path = Path(path)
    _validate_file_path(file_path, max_bytes=max_bytes)

    csv.field_size_limit(DEFAULT_MAX_FIELD_SIZE)
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = _detect_dialect(sample)
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = _normalize_header(reader.fieldnames, file_path)
        reader.fieldnames = fieldnames
        _validate_header(
            fieldnames,
            required_fields=required_fields,
            allow_extra_fields=allow_extra_fields,
            file_path=file_path,
        )

        rows: list[tuple[int, dict[str, str]]] = []
        for row_index, row in enumerate(reader, start=2):
            if row_index - 1 > max_rows:
                raise ValidationError(f"{file_path} exceeds maximum row count {max_rows}.")
            _reject_parser_overflow(row, file_path=file_path, row_number=row_index)
            normalized = _normalize_row(row, file_path=file_path, row_number=row_index)
            if _is_blank_row(normalized):
                raise ValidationError(f"{file_path}: blank row at line {row_index}.")
            rows.append((row_index, normalized))

    if not rows:
        raise ValidationError(f"{file_path}: file contains no data rows.")
    return rows


def _validate_file_path(path: Path, *, max_bytes: int) -> None:
    if not path.exists():
        raise ValidationError(f"File does not exist: {path}.")
    if not path.is_file():
        raise ValidationError(f"Expected a regular file, got: {path}.")
    size = path.stat().st_size
    if size == 0:
        raise ValidationError(f"File is empty: {path}.")
    if size > max_bytes:
        raise ValidationError(f"{path} is too large: {size} bytes > {max_bytes} bytes.")


def _detect_dialect(sample: str) -> csv.Dialect:
    if "\x00" in sample:
        raise ValidationError("Input file contains NUL bytes.")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        class FallbackDialect(csv.excel):
            delimiter = ","

        return FallbackDialect


def _normalize_header(fieldnames: Iterable[str] | None, file_path: Path) -> list[str]:
    if fieldnames is None:
        raise ValidationError(f"{file_path}: missing header row.")
    normalized: list[str] = []
    seen: set[str] = set()
    for field in fieldnames:
        name = (field or "").strip()
        if not name:
            raise ValidationError(f"{file_path}: header contains an empty column name.")
        if "\x00" in name:
            raise ValidationError(f"{file_path}: header contains NUL bytes.")
        if name in seen:
            raise ValidationError(f"{file_path}: duplicate column {name!r}.")
        seen.add(name)
        normalized.append(name)
    return normalized


def _validate_header(
    fieldnames: list[str],
    *,
    required_fields: frozenset[str],
    allow_extra_fields: bool,
    file_path: Path,
) -> None:
    present = set(fieldnames)
    missing = sorted(required_fields - present)
    if missing:
        raise ValidationError(f"{file_path}: missing required column(s): {', '.join(missing)}.")
    if not allow_extra_fields:
        extra = sorted(present - required_fields)
        if extra:
            raise ValidationError(f"{file_path}: unexpected column(s): {', '.join(extra)}.")


def _reject_parser_overflow(row: dict[object, object], *, file_path: Path, row_number: int) -> None:
    if None in row:
        raise ValidationError(f"{file_path}: too many fields at line {row_number}.")


def _normalize_row(row: dict[object, object], *, file_path: Path, row_number: int) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        if not isinstance(key, str):
            raise ValidationError(f"{file_path}: non-string column name at line {row_number}.")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValidationError(f"{file_path}: non-string value at line {row_number}, column {key}.")
        if "\x00" in value:
            raise ValidationError(f"{file_path}: NUL byte at line {row_number}, column {key}.")
        normalized[key.strip()] = value.strip()
    return normalized


def _is_blank_row(row: dict[str, str]) -> bool:
    return all(value == "" for value in row.values())


def _required_text(row: dict[str, str], field_name: str, row_number: int) -> str:
    value = row.get(field_name, "").strip()
    if not value:
        raise ValidationError(f"Missing required value for {field_name!r} at row {row_number}.")
    return value
