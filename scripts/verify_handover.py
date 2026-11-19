"""Verify the saved paper tables and compact handover provenance.

This checks included run-level JSON, canonical CSV/JSON/LaTeX tables, and the
deterministic source hash. It does not verify the omitted historical archive
or replay neural inference.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "reproduce_results.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("reproduce_results", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the saved-result aggregation script")
    results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(results)

    artifacts, diagnostics = results.build_artifacts(ROOT)
    checks = 0
    for path, expected in artifacts.items():
        with (ROOT / "outputs" / "assets" / path).open(encoding="utf-8", newline="") as handle:
            checks += results.compare_rows(expected, list(csv.DictReader(handle)), path)

    controlled = results.read_json(
        ROOT, "outputs/assets/multiseed/controlled_conditions_multiseed_assets.json"
    )
    if (
        controlled["run_seeds"] != list(results.SEEDS)
        or controlled["training_repeated"] is not False
        or controlled["checkpoint_frozen"] is not True
    ):
        raise ValueError("Controlled-results protocol metadata mismatch")
    checks += results.compare_rows(
        artifacts[results.CONTROL_CSV], controlled["rows"], "controlled JSON"
    )

    for path, expected in results.latex_tables(artifacts).items():
        actual = (ROOT / "outputs" / "assets" / path).read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if actual != expected.replace("\r\n", "\n").strip():
            raise ValueError(f"{path}: canonical LaTeX table differs from saved evidence")
        checks += 1

    reference = results.read_json(
        ROOT, "outputs/deterministic_reference/controlled_deterministic.json"
    )
    source_path = ROOT / "outputs/provenance/deterministic_source.json"
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != reference["source_sha256"]:
        raise ValueError("Deterministic source SHA-256 mismatch")
    original = json.loads(source_bytes.decode("utf-8"))
    expected_records = [
        row for row in original["results"]
        if row["condition"] in results.DETERMINISTIC_NAMES
    ]
    if reference["results"] != expected_records:
        raise ValueError("Deterministic reference differs from its saved source")
    checks += 2

    manifest = results.read_json(
        ROOT, "outputs/assets/multiseed/multiseed_final_manifest.json"
    )
    for path in manifest["authoritative_outputs"].values():
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
        checks += 1

    print(
        f"PASS: {checks} saved-evidence comparisons; "
        f"{len(results.SEEDS)} expected stochastic seeds; "
        f"{len(manifest['authoritative_outputs'])} canonical output files present."
    )
    print("Scope: historical archive integrity and neural inference replay are excluded.")


if __name__ == "__main__":
    main()