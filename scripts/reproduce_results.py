"""Recompute the final paper artifacts from saved evaluation records.

The default command writes a separate output directory; --check is read-only.
No inference, training, checkpoint, dataset, or third-party package is needed
unless --plots is requested (Matplotlib).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (12345, 23456, 34567)
CONTROL_RAW = "outputs/multiseed_controlled_framewise/controlled_multiseed_framewise_raw.json"
BRANCH_RAW = "outputs/multiseed_branch_ablation_framewise/c4_l3_pairwise_multiseed_raw.json"
DETERMINISTIC = "outputs/deterministic_reference/controlled_deterministic.json"
FOG_RAW = "outputs/real_fog_external/fog_6_0_external_results.json"
CONTROL_CSV = "multiseed/controlled_conditions_multiseed_assets.csv"
BRANCH_CSV = "multiseed/c4_l3_branch_multiseed_assets.csv"
CONTROL_METRICS = {
    "mAP@0.50": "mAP@0.50", "Recall": "recall_macro_gt_present",
    "ECE": "ECE", "NLL": "NLL", "Brier": "Brier",
    "MeanConfidence": "mean_prediction_confidence",
    "EventAccuracy": "prediction_event_accuracy",
    "ConfMinusEventAcc": "confidence_minus_event_accuracy",
}
BRANCH_METRICS = (
    "mAP@0.50", "precision_macro_gt_present", "recall_macro_gt_present",
    "f1_macro_gt_present", "ECE", "NLL", "Brier", "mean_prediction_confidence",
    "prediction_event_accuracy", "confidence_minus_event_accuracy", "total_predictions",
)
BRANCHES = {
    "Full": ("radar", "lidar", "camera"),
    "Radar+LiDAR": ("radar", "lidar"),
    "Radar+Camera": ("radar", "camera"),
    "LiDAR+Camera": ("lidar", "camera"),
}
DETERMINISTIC_NAMES = ("C0-Clean", "C1-L1", "C1-L2", "C1-L3", "C2a-LiDAR-BEV-Ablation")


def read_json(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or non-finite numeric value: {label}")
    return value


def group_runs(runs, expected):
    groups = {name: [] for name in expected}
    for run in runs:
        name = run["condition"]
        if name not in groups:
            raise ValueError(f"Unexpected condition: {name}")
        groups[name].append(run)
    for name, rows in groups.items():
        if sorted(r["run_seed"] for r in rows) != list(SEEDS):
            raise ValueError(f"{name}: require exactly one record for each seed {SEEDS}")
        for row in rows:
            if row.get("seed", row["run_seed"]) != row["run_seed"]:
                raise ValueError(f"{name}: seed fields disagree")
    return groups


def stats(runs, key):
    values = [finite(r[key], f"{r.get('condition', 'record')}/{key}") for r in runs]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def unique_records(records, key):
    indexed = {}
    for row in records:
        name = row[key]
        if name in indexed:
            raise ValueError(f"Duplicate {key}: {name}")
        indexed[name] = row
    return indexed


def branch_record(records, modalities):
    selected = [r for r in records if set(r["active_modalities"]) == set(modalities)]
    if len(selected) != 1:
        raise ValueError(f"Expected one branch record for {modalities}")
    return selected[0]


def build_artifacts(root=ROOT):
    raw = read_json(root, CONTROL_RAW)
    branch_raw = read_json(root, BRANCH_RAW)
    for source in (raw, branch_raw):
        if tuple(source["run_seeds"]) != SEEDS or source["training_repeated"] is not False or source["checkpoint_frozen"] is not True:
            raise ValueError("Saved records do not match the frozen three-realization protocol")
    expected = [f"C{c}-L{level}" for c in range(2, 6) for level in range(1, 4)]
    groups = group_runs(raw["runs"], expected)
    branch_groups = group_runs(branch_raw["runs"], ["C4-L3-" + name.replace("+", "-") for name in list(BRANCHES)[1:]])
    deterministic = unique_records(read_json(root, DETERMINISTIC)["results"], "condition")
    if set(deterministic) != set(DETERMINISTIC_NAMES):
        raise ValueError("Missing or unexpected deterministic reference condition")
    controlled = []
    for name in [*DETERMINISTIC_NAMES, *expected]:
        runs = [deterministic[name]] if name in deterministic else groups[name]
        row = {"condition": name, "aggregation": "deterministic" if name in deterministic else "mean±sample_std", "n_degradation_realizations": len(runs)}
        for column, key in CONTROL_METRICS.items():
            row[column + "_mean"], row[column + "_std"] = stats(runs, key)
        controlled.append(row)
    controlled_by_name = {r["condition"]: r for r in controlled}
    branches = []
    for name, modalities in BRANCHES.items():
        configuration = "C4-L3-" + name.replace("+", "-")
        runs = groups["C4-L3"] if name == "Full" else branch_groups[configuration]
        row = {"configuration": configuration, "active_modalities": "+".join(modalities), "n_runs": len(runs)}
        for key in BRANCH_METRICS:
            row[key + "_mean"], row[key + "_std"] = stats(runs, key)
        row["Configuration"] = name
        branches.append(row)
    table1 = []
    for name in ["C0-Clean", *[f"C{c}-L3" for c in range(1, 6)]]:
        source = controlled_by_name[name]
        row = {"Condition": name}
        for column, key in {"mAP": "mAP@0.50", "Recall": "Recall", "ECE": "ECE", "ConfGap": "ConfMinusEventAcc"}.items():
            for suffix in ("mean", "std"):
                row[column + "_" + suffix] = source[key + "_" + suffix]
        row["Aggregation"] = source["aggregation"]
        table1.append(row)
    fog = read_json(root, FOG_RAW)
    f = fog["primary_evaluation"]
    common_fog = (f["AP_class_0"] + f["AP_class_1"]) / 2
    table2 = [{"Frames": fog["primary_raw_annotated_frames"], **{
        key: round(value, 4) for key, value in {
            "mAP@0.50 (3-class)": f["mAP@0.50"], "Car+Van mAP": common_fog,
            "Car AP": f["AP_class_0"], "Van AP": f["AP_class_1"], "Bus AP": f["AP_class_2"],
            "ECE": f["ECE"], "Recall": f["recall_macro_gt_present"],
            "Mean Conf.": f["mean_prediction_confidence"], "Event Acc.": f["prediction_event_accuracy"],
        }.items()}}]
    clean_branches = read_json(root, "outputs/branch_ablation/clean_pairwise_branch_ablation_corrected.json")["results"]
    fog_branches = read_json(root, "outputs/branch_ablation/real_fog_pairwise_branch_ablation_corrected.json")["results"]
    table3 = []
    for branch in branches:
        name = branch["Configuration"]
        clean = branch_record(clean_branches, BRANCHES[name])
        fog_branch = branch_record(fog_branches, BRANCHES[name])
        table3.append({"Configuration": name, "Clean": (clean["AP_class_0"] + clean["AP_class_1"]) / 2,
                       "C4-L3_mean": branch["mAP@0.50_mean"], "C4-L3_std": branch["mAP@0.50_std"],
                       "Real_Fog": (fog_branch["AP_class_0"] + fog_branch["AP_class_1"]) / 2})
    diagnostics = {
        "C3-L1_all_three_mAP_and_ECE_below_clean": all(
            r["mAP@0.50"] < deterministic["C0-Clean"]["mAP@0.50"] and r["ECE"] < deterministic["C0-Clean"]["ECE"] for r in groups["C3-L1"]),
        "C3-L1_prediction_count_mean_std": stats(groups["C3-L1"], "total_predictions"),
        "C2-L3_prediction_count_mean_std": stats(groups["C2-L3"], "total_predictions"),
        "C4-L3_LiDAR_Camera_prediction_counts": [r["total_predictions"] for r in sorted(branch_groups["C4-L3-LiDAR-Camera"], key=lambda r: r["run_seed"])],
        "fog_empty_road_predictions_per_frame": fog["empty_road_diagnostic"]["total_false_positive_detections"] / fog["empty_road_diagnostic"]["frames"],
    }
    tables = {"tables/table1_controlled_severe.csv": table1, "tables/table2_real_fog.csv": table2, "tables/table3_branch_ablation.csv": table3}
    return {CONTROL_CSV: controlled, BRANCH_CSV: branches, **tables}, diagnostics


def latex_tables(artifacts):
    def pm(row, key):
        return f"${row[key + '_mean']:.4f} \\pm {row[key + '_std']:.4f}$"
    def table(spec, headers, rows):
        return "\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(headers) + " \\\\\n\\midrule\n" + "".join(" & ".join(row) + " \\\\\n" for row in rows) + "\\bottomrule\n\\end{tabular}\n"
    one = artifacts["tables/table1_controlled_severe.csv"]
    rows1 = [[r["Condition"], *[f"{r[k + '_mean']:.4f}" if r["Aggregation"] == "deterministic" else pm(r, k) for k in ("mAP", "Recall", "ECE", "ConfGap")]] for r in one]
    two = artifacts["tables/table2_real_fog.csv"]
    rows2 = [[str(r["Frames"]), *[f"{v:.4f}" for k, v in r.items() if k != "Frames"]] for r in two]
    three = artifacts["tables/table3_branch_ablation.csv"]
    rows3 = [[r["Configuration"], f"{r['Clean']:.4f}", pm(r, "C4-L3"), f"{r['Real_Fog']:.4f}"] for r in three]
    return {
        "tables/table1_controlled_severe.tex": table("lcccc", ["Condition", "mAP@0.50", "Recall", "ECE", "Conf.-Acc."], rows1),
        "tables/table2_real_fog.tex": table("rrrrrrrrrr", list(two[0]), rows2),
        "tables/table3_branch_ablation.tex": table("lccc", ["Configuration", "Clean", "Synthetic C4-L3", "Real Fog"], rows3),
    }


def compare_rows(expected, actual, label):
    if len(expected) != len(actual):
        raise ValueError(f"{label}: row count mismatch")
    checks = 0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if set(left) != set(right):
            raise ValueError(f"{label} row {index}: column mismatch")
        for key, value in left.items():
            if isinstance(value, (int, float)):
                try:
                    equal = math.isclose(value, float(right[key]), rel_tol=1e-10, abs_tol=1e-12)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = value == right[key]
            if not equal:
                raise ValueError(f"{label} row {index}/{key}: expected {value!r}, found {right[key]!r}")
            checks += 1
    return checks


def check_artifacts(root, artifacts):
    checks = 0
    for path, expected in artifacts.items():
        with (root / "outputs/assets" / path).open(encoding="utf-8", newline="") as handle:
            checks += compare_rows(expected, list(csv.DictReader(handle)), path)
    controlled_json = read_json(root, "outputs/assets/multiseed/controlled_conditions_multiseed_assets.json")
    if controlled_json["run_seeds"] != list(SEEDS) or controlled_json["training_repeated"] is not False or controlled_json["checkpoint_frozen"] is not True:
        raise ValueError("Controlled JSON protocol metadata mismatch")
    checks += compare_rows(artifacts[CONTROL_CSV], controlled_json["rows"], "controlled JSON")
    for path, expected in latex_tables(artifacts).items():
        if (root / "outputs/assets" / path).read_text(encoding="utf-8").strip() != expected.strip():
            raise ValueError(f"{path}: LaTeX table differs from regenerated values")
        checks += 1
    archived = read_json(root, "archive/pre_multiseed/manifest.json")
    for record in archived["files"]:
        digest = hashlib.sha256((root / record["archived_path"]).read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Archived file changed: {record['archived_path']}")
        checks += 1
    reference = read_json(root, DETERMINISTIC)
    source_path = root / reference["source_archive_path"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != reference["source_sha256"]:
        raise ValueError("Deterministic source hash mismatch")
    original = json.loads(source_path.read_text(encoding="utf-8"))
    expected = [r for r in original["results"] if r["condition"] in DETERMINISTIC_NAMES]
    if reference["results"] != expected:
        raise ValueError("Deterministic extraction differs from original records")
    checks += 1
    return checks


def write_artifacts(output, artifacts, diagnostics):
    for path, rows in artifacts.items():
        dest = output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for path, text in latex_tables(artifacts).items():
        (output / path).write_text(text, encoding="utf-8")
    summary = {"run_seeds": list(SEEDS), "training_repeated": False, "checkpoint_frozen": True, "rows": artifacts[CONTROL_CSV]}
    (output / "multiseed/controlled_conditions_multiseed_assets.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "discussion_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


def plot_artifacts(output, artifacts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {r["condition"]: r for r in artifacts[CONTROL_CSV]}
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in [("mAP@0.50", "mAP@0.50", "fig1_map_vs_severity"), ("ECE", "Expected Calibration Error (ECE)", "fig2_ece_vs_severity")]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for c in range(1, 6):
            group = [rows["C0-Clean"], *[rows[f"C{c}-L{level}"] for level in range(1, 4)]]
            ax.errorbar(range(4), [r[metric + "_mean"] for r in group], yerr=[r[metric + "_std"] for r in group], marker="o", capsize=3, label=f"C{c}")
        ax.set(xticks=range(4), xticklabels=["Clean", "L1", "L2", "L3"], xlabel="Degradation severity", ylabel=title, ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(ncol=3)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(figure_dir / (filename + "." + suffix), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    branches = artifacts["tables/table3_branch_ablation.csv"]
    for offset, key, label in [(-0.25, "Clean", "Clean"), (0, "C4-L3_mean", "C4-L3"), (0.25, "Real_Fog", "Real fog")]:
        ax.bar([i + offset for i in range(len(branches))], [r[key] for r in branches], width=0.25, yerr=[r["C4-L3_std"] for r in branches] if key == "C4-L3_mean" else None, capsize=3, label=label)
    ax.set(xticks=range(len(branches)), xticklabels=[r["Configuration"] for r in branches], ylabel="Car+Van mAP@0.50", ylim=(0, None))
    ax.legend()
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / ("fig3_branch_ablation_common_class." + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only comparison with committed CSV/JSON/LaTeX tables and archive hashes")
    parser.add_argument("--output-dir", type=Path, help="Separate destination (default: outputs/reproduced)")
    parser.add_argument("--plots", action="store_true", help="Also render plots using Matplotlib")
    args = parser.parse_args()
    if args.check and (args.output_dir or args.plots):
        parser.error("--check is read-only; do not combine with --output-dir or --plots")
    if args.check and not (ROOT / 'archive/pre_multiseed/manifest.json').is_file():
        parser.error('Historical archive is not bundled; use scripts/verify_handover.py for this handover.')
    artifacts, diagnostics = build_artifacts()
    if args.check:
        checks = check_artifacts(ROOT, artifacts)
        print(f"PASS: {checks} field/file comparisons; 36 controlled and 9 pairwise records; exact three-seed coverage.")
        print(json.dumps(diagnostics, indent=2))
    else:
        output = (args.output_dir or ROOT / "outputs/reproduced").resolve()
        canonical = (ROOT / "outputs/assets").resolve()
        if output == canonical or canonical in output.parents:
            parser.error("Choose a separate directory; canonical paper assets are protected")
        write_artifacts(output, artifacts, diagnostics)
        if args.plots:
            plot_artifacts(output, artifacts)
        print(f"Regenerated saved-result summaries and Tables I–III in {output}")


if __name__ == "__main__":
    main()
