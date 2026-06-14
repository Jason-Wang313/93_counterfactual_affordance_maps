import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 930421
SEEDS = list(range(7))
EPISODES = 84
STRESS_EPISODES = 52
CANDIDATES = 12

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


TASKS = {
    "cluttered_pick_and_place": {
        "difficulty": 0.38,
        "pose_sensitivity": 0.48,
        "support_sensitivity": 0.32,
        "semantic_sensitivity": 0.22,
        "contact_noise": 0.34,
        "fragility": 0.30,
    },
    "tool_regrasp_for_use": {
        "difficulty": 0.46,
        "pose_sensitivity": 0.66,
        "support_sensitivity": 0.24,
        "semantic_sensitivity": 0.62,
        "contact_noise": 0.30,
        "fragility": 0.22,
    },
    "support_sensitive_stacking": {
        "difficulty": 0.50,
        "pose_sensitivity": 0.40,
        "support_sensitivity": 0.72,
        "semantic_sensitivity": 0.30,
        "contact_noise": 0.44,
        "fragility": 0.56,
    },
    "occluded_container_opening": {
        "difficulty": 0.44,
        "pose_sensitivity": 0.52,
        "support_sensitivity": 0.46,
        "semantic_sensitivity": 0.58,
        "contact_noise": 0.38,
        "fragility": 0.36,
    },
}

SPLITS = {
    "nominal_affordance": {
        "pose": 0.00,
        "support": 0.00,
        "occlusion": 0.00,
        "semantic": 0.00,
        "contact": 0.00,
    },
    "pose_shift": {
        "pose": 0.26,
        "support": 0.04,
        "occlusion": 0.06,
        "semantic": 0.04,
        "contact": 0.06,
    },
    "support_shift": {
        "pose": 0.06,
        "support": 0.28,
        "occlusion": 0.08,
        "semantic": 0.05,
        "contact": 0.08,
    },
    "occlusion_semantic_shift": {
        "pose": 0.08,
        "support": 0.08,
        "occlusion": 0.26,
        "semantic": 0.24,
        "contact": 0.08,
    },
    "combined_counterfactual_stress": {
        "pose": 0.20,
        "support": 0.20,
        "occlusion": 0.22,
        "semantic": 0.18,
        "contact": 0.16,
    },
}

METHODS = [
    "observed_affordance_map",
    "multi_affordance_grasping",
    "graph_conv_affordance",
    "gaussian_grasp_map",
    "vlm_spatial_affordance",
    "interactive_affordance_probe",
    "ensemble_uncertainty_affordance",
    "proposed_counterfactual_affordance_map",
    "oracle_counterfactual_map",
]

ABLATIONS = [
    "full_counterfactual_affordance",
    "minus_pose_counterfactuals",
    "minus_support_counterfactuals",
    "minus_semantic_counterfactuals",
    "minus_uncertainty_calibration",
    "observed_only_counterfactual_head",
    "support_only_counterfactual_head",
]

PROFILES = {
    "observed_affordance_map": {"pose": 0.28, "support": 0.18, "semantic": 0.20, "cf": 0.05, "noise": 0.22, "probe": 0.00, "conservative": 0.06},
    "multi_affordance_grasping": {"pose": 0.42, "support": 0.25, "semantic": 0.26, "cf": 0.12, "noise": 0.18, "probe": 0.00, "conservative": 0.08},
    "graph_conv_affordance": {"pose": 0.45, "support": 0.50, "semantic": 0.28, "cf": 0.18, "noise": 0.16, "probe": 0.00, "conservative": 0.10},
    "gaussian_grasp_map": {"pose": 0.58, "support": 0.26, "semantic": 0.18, "cf": 0.12, "noise": 0.17, "probe": 0.00, "conservative": 0.06},
    "vlm_spatial_affordance": {"pose": 0.36, "support": 0.28, "semantic": 0.58, "cf": 0.20, "noise": 0.18, "probe": 0.00, "conservative": 0.12},
    "interactive_affordance_probe": {"pose": 0.52, "support": 0.58, "semantic": 0.42, "cf": 0.34, "noise": 0.13, "probe": 0.18, "conservative": 0.16},
    "ensemble_uncertainty_affordance": {"pose": 0.48, "support": 0.46, "semantic": 0.38, "cf": 0.30, "noise": 0.14, "probe": 0.04, "conservative": 0.22},
    "proposed_counterfactual_affordance_map": {"pose": 0.61, "support": 0.62, "semantic": 0.55, "cf": 0.56, "noise": 0.13, "probe": 0.02, "conservative": 0.16},
    "oracle_counterfactual_map": {"pose": 0.92, "support": 0.92, "semantic": 0.90, "cf": 0.88, "noise": 0.04, "probe": 0.00, "conservative": 0.08},
    "full_counterfactual_affordance": {"pose": 0.61, "support": 0.62, "semantic": 0.55, "cf": 0.56, "noise": 0.13, "probe": 0.02, "conservative": 0.16},
    "minus_pose_counterfactuals": {"pose": 0.38, "support": 0.62, "semantic": 0.55, "cf": 0.40, "noise": 0.14, "probe": 0.02, "conservative": 0.14},
    "minus_support_counterfactuals": {"pose": 0.61, "support": 0.36, "semantic": 0.55, "cf": 0.38, "noise": 0.14, "probe": 0.02, "conservative": 0.14},
    "minus_semantic_counterfactuals": {"pose": 0.61, "support": 0.62, "semantic": 0.34, "cf": 0.42, "noise": 0.14, "probe": 0.02, "conservative": 0.14},
    "minus_uncertainty_calibration": {"pose": 0.61, "support": 0.62, "semantic": 0.55, "cf": 0.56, "noise": 0.20, "probe": 0.02, "conservative": 0.04},
    "observed_only_counterfactual_head": {"pose": 0.32, "support": 0.24, "semantic": 0.26, "cf": 0.10, "noise": 0.20, "probe": 0.00, "conservative": 0.08},
    "support_only_counterfactual_head": {"pose": 0.28, "support": 0.66, "semantic": 0.24, "cf": 0.30, "noise": 0.17, "probe": 0.02, "conservative": 0.12},
}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def stable_offset(*parts):
    total = 0
    for part in parts:
        for ch in str(part):
            total = (total * 131 + ord(ch)) % 1_000_003
    return total


def ci95(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(len(values))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = 0
    precisions = []
    for rank, label in enumerate(sorted_labels, start=1):
        if label:
            tp += 1
            precisions.append(tp / rank)
    return float(sum(precisions) / positives)


def ece(labels, scores, bins=10):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    out = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (scores >= lo) & (scores < hi if idx < bins - 1 else scores <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return out


def generate_scene(task_name, split_name, seed, episode, stress_level=None):
    task = TASKS[task_name]
    split = SPLITS[split_name]
    stress = 1.0 if stress_level is None else stress_level
    rng = np.random.default_rng(BASE_SEED + stable_offset(task_name, split_name, seed, episode, stress_level))
    pose_shift = clamp(split["pose"] * stress + rng.normal(0.0, 0.04))
    support_shift = clamp(split["support"] * stress + rng.normal(0.0, 0.04))
    occlusion = clamp(0.08 + split["occlusion"] * stress + rng.normal(0.0, 0.035))
    semantic_ambiguity = clamp(0.08 + split["semantic"] * stress + rng.normal(0.0, 0.035))
    contact_noise = clamp(task["contact_noise"] + split["contact"] * stress + rng.normal(0.0, 0.05))
    candidates = []
    for idx in range(CANDIDATES):
        pose_match = clamp(rng.beta(2.2, 2.4) - 0.30 * pose_shift + rng.normal(0.0, 0.05))
        support_valid = clamp(rng.beta(2.5, 2.0) - 0.38 * support_shift + rng.normal(0.0, 0.06))
        semantic_match = clamp(rng.beta(2.0, 2.3) - 0.32 * semantic_ambiguity + rng.normal(0.0, 0.06))
        contact_stability = clamp(rng.beta(2.4, 2.2) - 0.24 * contact_noise + rng.normal(0.0, 0.05))
        occlusion_penalty = occlusion * rng.uniform(0.55, 1.10)
        current_visible = clamp(0.45 * pose_match + 0.25 * semantic_match + 0.20 * contact_stability - 0.30 * occlusion_penalty)
        counter_pose = clamp(pose_match + 0.34 * pose_shift * rng.uniform(0.2, 1.0) + rng.normal(0.0, 0.04))
        counter_support = clamp(support_valid + 0.40 * support_shift * rng.uniform(0.1, 1.0) + rng.normal(0.0, 0.04))
        counter_semantic = clamp(semantic_match + 0.32 * semantic_ambiguity * rng.uniform(0.1, 1.0) + rng.normal(0.0, 0.04))
        true_utility = clamp(
            0.30 * pose_match
            + 0.30 * support_valid
            + 0.23 * semantic_match
            + 0.25 * contact_stability
            - 0.36 * task["difficulty"]
            - 0.22 * task["fragility"] * (1.0 - support_valid)
            + rng.normal(0.0, 0.04),
            0.0,
            1.0,
        )
        counter_utility = clamp(
            true_utility
            + task["pose_sensitivity"] * 0.22 * (counter_pose - pose_match)
            + task["support_sensitivity"] * 0.24 * (counter_support - support_valid)
            + task["semantic_sensitivity"] * 0.20 * (counter_semantic - semantic_match)
            + rng.normal(0.0, 0.035),
            0.0,
            1.0,
        )
        candidates.append({
            "pose_match": pose_match,
            "support_valid": support_valid,
            "semantic_match": semantic_match,
            "contact_stability": contact_stability,
            "current_visible": current_visible,
            "counter_pose": counter_pose,
            "counter_support": counter_support,
            "counter_semantic": counter_semantic,
            "true_utility": true_utility,
            "counter_utility": counter_utility,
            "affordance_label": int(true_utility > 0.52),
            "counterfactual_label": int(counter_utility > true_utility + 0.09 and counter_utility > 0.54),
            "invalid": int(support_valid < 0.34 or semantic_match < 0.24 or contact_stability < 0.28),
        })
    return {
        "task": task,
        "pose_shift": pose_shift,
        "support_shift": support_shift,
        "occlusion": occlusion,
        "semantic_ambiguity": semantic_ambiguity,
        "contact_noise": contact_noise,
        "candidates": candidates,
    }


def method_scores(method, scene, seed, episode):
    profile = PROFILES[method]
    rng = np.random.default_rng(BASE_SEED + stable_offset(method, seed, episode, "scores"))
    scores = []
    cf_scores = []
    for cand in scene["candidates"]:
        observed_term = (
            0.32 * cand["current_visible"]
            + profile["pose"] * 0.24 * cand["pose_match"]
            + profile["support"] * 0.25 * cand["support_valid"]
            + profile["semantic"] * 0.22 * cand["semantic_match"]
            + 0.18 * cand["contact_stability"]
        )
        cf_term = profile["cf"] * (
            0.26 * cand["counter_pose"]
            + 0.30 * cand["counter_support"]
            + 0.24 * cand["counter_semantic"]
            + 0.20 * cand["counter_utility"]
        )
        if method == "oracle_counterfactual_map":
            raw = 0.72 * cand["true_utility"] + 0.42 * cand["counter_utility"]
            cf_raw = cand["counter_utility"]
        else:
            raw = observed_term + cf_term - profile["conservative"] * scene["occlusion"]
            cf_raw = (
                profile["cf"] * 0.55 * cand["counter_utility"]
                + profile["support"] * 0.20 * cand["counter_support"]
                + profile["pose"] * 0.15 * cand["counter_pose"]
                + profile["semantic"] * 0.10 * cand["counter_semantic"]
            )
        noise_scale = profile["noise"] + 0.18 * scene["occlusion"] + 0.10 * scene["contact_noise"]
        if profile["probe"] > 0:
            noise_scale *= max(0.45, 1.0 - profile["probe"] * 1.8)
            raw += profile["probe"] * (0.30 * cand["support_valid"] + 0.20 * cand["contact_stability"])
            cf_raw += profile["probe"] * 0.20 * cand["counter_support"]
        score = clamp(sigmoid(3.0 * (raw - 0.48) + rng.normal(0.0, noise_scale)))
        cf_score = clamp(sigmoid(3.0 * (cf_raw - 0.35) + rng.normal(0.0, noise_scale * 0.8)))
        scores.append(score)
        cf_scores.append(cf_score)
    return np.asarray(scores), np.asarray(cf_scores)


def simulate_episode(method, task_name, split_name, seed, episode, stress_level=None):
    scene = generate_scene(task_name, split_name, seed, episode, stress_level)
    scores, cf_scores = method_scores(method, scene, seed, episode)
    labels = np.asarray([c["affordance_label"] for c in scene["candidates"]], dtype=int)
    cf_labels = np.asarray([c["counterfactual_label"] for c in scene["candidates"]], dtype=int)
    utilities = np.asarray([c["true_utility"] for c in scene["candidates"]], dtype=float)
    counter_utils = np.asarray([c["counter_utility"] for c in scene["candidates"]], dtype=float)
    invalids = np.asarray([c["invalid"] for c in scene["candidates"]], dtype=int)
    chosen = int(np.argmax(scores))
    oracle = int(np.argmax(utilities))
    chosen_utility = float(utilities[chosen])
    chosen_counter = float(counter_utils[chosen])
    task = scene["task"]
    probe_cost = PROFILES[method]["probe"]
    invalid_action = int(invalids[chosen] == 1)
    support_collapse = int(scene["candidates"][chosen]["support_valid"] < 0.38 and task["support_sensitivity"] > 0.45)
    semantic_violation = int(scene["candidates"][chosen]["semantic_match"] < 0.30 and task["semantic_sensitivity"] > 0.45)
    damage = clamp(
        0.10 * invalid_action
        + 0.22 * support_collapse
        + 0.18 * task["fragility"] * (1.0 - scene["candidates"][chosen]["support_valid"])
        + 0.08 * scene["contact_noise"]
    )
    success_prob = clamp(sigmoid(5.0 * (max(chosen_utility, chosen_counter * 0.86) - 0.50) - 1.2 * invalid_action - 0.8 * semantic_violation - 0.5 * probe_cost))
    rng = np.random.default_rng(BASE_SEED + stable_offset(method, task_name, split_name, seed, episode, stress_level, "outcome"))
    task_success = int(rng.random() < success_prob and damage < 0.42)
    recovery_success = int(rng.random() < clamp(0.20 + 0.65 * cf_scores.max() - 0.35 * invalid_action - 0.20 * scene["occlusion"]))
    positives = max(1, int(labels.sum()))
    cf_positives = max(1, int(cf_labels.sum()))
    cf_top = np.argsort(-cf_scores)[:max(1, min(3, cf_positives))]
    counterfactual_recall = float(cf_labels[cf_top].sum() / cf_positives)
    support_error = abs(float(scores[chosen]) - float(scene["candidates"][chosen]["support_valid"]))
    planning_regret = max(0.0, float(utilities[oracle]) - chosen_utility)
    return {
        "method": method,
        "split": split_name,
        "task": task_name,
        "seed": seed,
        "episode": episode,
        "stress_level": "" if stress_level is None else f"{stress_level:.2f}",
        "affordance_ap": average_precision(labels, scores),
        "counterfactual_recall": counterfactual_recall,
        "calibration_error": ece(labels, scores),
        "support_error": support_error,
        "semantic_violation": semantic_violation,
        "task_success": task_success,
        "invalid_action": invalid_action,
        "damage": damage,
        "planning_regret": planning_regret,
        "probe_cost": probe_cost,
        "recovery_success": recovery_success,
        "chosen_utility": chosen_utility,
        "oracle_utility": float(utilities[oracle]),
        "cf_positive_rate": float(cf_labels.sum() / len(cf_labels)),
        "num_positive": int(labels.sum()),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate_rows(methods, split_names, episodes, stress_level=None):
    rows = []
    for method in methods:
        for split_name in split_names:
            for task_name in TASKS:
                for seed in SEEDS:
                    for episode in range(episodes):
                        rows.append(simulate_episode(method, task_name, split_name, seed, episode, stress_level))
    return rows


def unit_metrics(rows, group_keys):
    metrics = [
        "affordance_ap",
        "counterfactual_recall",
        "calibration_error",
        "support_error",
        "semantic_violation",
        "task_success",
        "invalid_action",
        "damage",
        "planning_regret",
        "probe_cost",
        "recovery_success",
    ]
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, {m: [] for m in metrics})
        for metric in metrics:
            groups[key][metric].append(float(row[metric]))
    out = []
    for key, values in sorted(groups.items()):
        entry = {group_keys[i]: key[i] for i in range(len(group_keys))}
        for metric in metrics:
            entry[metric] = float(np.mean(values[metric]))
        entry["episodes"] = len(next(iter(values.values())))
        out.append(entry)
    return out


def summarize(units, by_keys):
    metrics = [
        "affordance_ap",
        "counterfactual_recall",
        "calibration_error",
        "support_error",
        "semantic_violation",
        "task_success",
        "invalid_action",
        "damage",
        "planning_regret",
        "probe_cost",
        "recovery_success",
    ]
    groups = {}
    for row in units:
        key = tuple(row[k] for k in by_keys)
        groups.setdefault(key, {m: [] for m in metrics})
        for metric in metrics:
            groups[key][metric].append(float(row[metric]))
    summary = []
    for key, values in sorted(groups.items()):
        entry = {by_keys[i]: key[i] for i in range(len(by_keys))}
        for metric in metrics:
            vals = values[metric]
            entry[f"mean_{metric}"] = f"{sum(vals) / len(vals):.5f}"
            entry[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        entry["units"] = len(next(iter(values.values())))
        summary.append(entry)
    return summary


def paired_gate(units, split_name):
    grouped = {}
    for row in units:
        if row["split"] == split_name:
            grouped.setdefault((row["task"], row["seed"]), {})[row["method"]] = row
    means = {}
    for method in METHODS:
        if method == "oracle_counterfactual_map":
            continue
        vals = [float(v[method]["task_success"]) for v in grouped.values() if method in v]
        if vals:
            means[method] = sum(vals) / len(vals)
    best_baseline = max((m for m in means if m != "proposed_counterfactual_affordance_map"), key=lambda m: means[m])
    success_diffs = []
    recall_diffs = []
    invalid_diffs = []
    regret_diffs = []
    cost_diffs = []
    for methods in grouped.values():
        if "proposed_counterfactual_affordance_map" in methods and best_baseline in methods:
            proposed = methods["proposed_counterfactual_affordance_map"]
            baseline = methods[best_baseline]
            success_diffs.append(float(proposed["task_success"]) - float(baseline["task_success"]))
            recall_diffs.append(float(proposed["counterfactual_recall"]) - float(baseline["counterfactual_recall"]))
            invalid_diffs.append(float(baseline["invalid_action"]) - float(proposed["invalid_action"]))
            regret_diffs.append(float(baseline["planning_regret"]) - float(proposed["planning_regret"]))
            cost_diffs.append(float(proposed["probe_cost"]) - float(baseline["probe_cost"]))
    return {
        "best_non_oracle_baseline": best_baseline,
        "paired_success_diff": sum(success_diffs) / len(success_diffs),
        "paired_success_ci95": ci95(success_diffs),
        "paired_cf_recall_diff": sum(recall_diffs) / len(recall_diffs),
        "paired_cf_recall_ci95": ci95(recall_diffs),
        "paired_invalid_reduction": sum(invalid_diffs) / len(invalid_diffs),
        "paired_invalid_ci95": ci95(invalid_diffs),
        "paired_regret_reduction": sum(regret_diffs) / len(regret_diffs),
        "paired_regret_ci95": ci95(regret_diffs),
        "paired_extra_cost": sum(cost_diffs) / len(cost_diffs),
        "paired_extra_cost_ci95": ci95(cost_diffs),
    }


def find_row(summary, method, split):
    for row in summary:
        if row["method"] == method and row["split"] == split:
            return row
    raise KeyError((method, split))


def plot_bars(summary_rows, split, metrics, filename, title):
    rows = [r for r in summary_rows if r["split"] == split]
    labels = [r["method"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    width = 0.75 / len(metrics)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for idx, metric in enumerate(metrics):
        vals = [float(r[f"mean_{metric}"]) for r in rows]
        errs = [float(r[f"ci95_{metric}"]) for r in rows]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, vals, width, yerr=errs, capsize=3, label=metric.replace("_", " "))
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def plot_ablation(ablation_summary):
    labels = [r["method"].replace("_", "\n") for r in ablation_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.24, [float(r["mean_task_success"]) for r in ablation_summary], 0.24, label="task success")
    ax.bar(x, [float(r["mean_counterfactual_recall"]) for r in ablation_summary], 0.24, label="counterfactual recall")
    ax.bar(x + 0.24, [1.0 - float(r["mean_invalid_action"]) for r in ablation_summary], 0.24, label="valid action rate")
    ax.set_title("Paper 93 counterfactual-affordance ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "counterfactual_affordance_ablation.png", dpi=180)
    plt.close(fig)


def plot_stress(stress_summary):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in sorted({r["method"] for r in stress_summary}):
        rows = sorted([r for r in stress_summary if r["method"] == method], key=lambda r: float(r["stress_level"]))
        levels = [float(r["stress_level"]) for r in rows]
        success = [float(r["mean_task_success"]) for r in rows]
        ax.plot(levels, success, marker="o", label=method.replace("_", " "))
    ax.set_title("Paper 93 stress sweep: task success")
    ax.set_xlabel("stress level")
    ax.set_ylabel("task success")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "counterfactual_affordance_stress_sweep.png", dpi=180)
    plt.close(fig)


def main():
    main_rows = simulate_rows(METHODS, list(SPLITS.keys()), EPISODES)
    write_csv(RESULTS / "rollouts.csv", main_rows)
    main_units = unit_metrics(main_rows, ["method", "split", "task", "seed"])
    write_csv(RESULTS / "raw_seed_metrics.csv", main_units)
    summary = summarize(main_units, ["method", "split"])
    write_csv(RESULTS / "metrics.csv", summary)

    ablation_rows = simulate_rows(ABLATIONS, ["combined_counterfactual_stress"], EPISODES)
    write_csv(RESULTS / "ablation_rollouts.csv", ablation_rows)
    ablation_units = unit_metrics(ablation_rows, ["method", "task", "seed"])
    write_csv(RESULTS / "ablation_seed_metrics.csv", ablation_units)
    ablation_summary = summarize(ablation_units, ["method"])
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)

    stress_methods = [
        "graph_conv_affordance",
        "vlm_spatial_affordance",
        "interactive_affordance_probe",
        "ensemble_uncertainty_affordance",
        "proposed_counterfactual_affordance_map",
        "oracle_counterfactual_map",
    ]
    stress_rows = []
    for stress_level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        stress_rows.extend(simulate_rows(stress_methods, ["combined_counterfactual_stress"], STRESS_EPISODES, stress_level))
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_rows)
    stress_units = unit_metrics(stress_rows, ["method", "stress_level", "task", "seed"])
    stress_summary = summarize(stress_units, ["method", "stress_level"])
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)

    gate = paired_gate(main_units, "combined_counterfactual_stress")
    proposed = find_row(summary, "proposed_counterfactual_affordance_map", "combined_counterfactual_stress")
    baseline = find_row(summary, gate["best_non_oracle_baseline"], "combined_counterfactual_stress")
    oracle = find_row(summary, "oracle_counterfactual_map", "combined_counterfactual_stress")
    ablation_success = {r["method"]: float(r["mean_task_success"]) for r in ablation_summary}
    best_ablation = max((m for m in ablation_success if m != "full_counterfactual_affordance"), key=lambda m: ablation_success[m])

    stress_max = [r for r in stress_summary if abs(float(r["stress_level"]) - 1.0) < 1e-9]
    stress_best_baseline = max(
        [r for r in stress_max if r["method"] not in {"proposed_counterfactual_affordance_map", "oracle_counterfactual_map"}],
        key=lambda r: float(r["mean_task_success"]),
    )
    stress_proposed = next(r for r in stress_max if r["method"] == "proposed_counterfactual_affordance_map")

    success_gate = gate["paired_success_diff"] - gate["paired_success_ci95"] > 0.015
    recall_gate = gate["paired_cf_recall_diff"] - gate["paired_cf_recall_ci95"] > 0.015
    invalid_or_regret_gate = (
        gate["paired_invalid_reduction"] - gate["paired_invalid_ci95"] > 0.010
        or gate["paired_regret_reduction"] - gate["paired_regret_ci95"] > 0.010
    )
    cost_ok = gate["paired_extra_cost"] <= 0.10
    ablation_gate = ablation_success["full_counterfactual_affordance"] >= ablation_success[best_ablation] + 0.01
    stress_gate = float(stress_proposed["mean_task_success"]) >= float(stress_best_baseline["mean_task_success"]) - 0.005
    terminal = "STRONG_REVISE" if all([success_gate, recall_gate, invalid_or_regret_gate, cost_ok, ablation_gate, stress_gate]) else "KILL_ARCHIVE"

    pairwise = [{
        "split": "combined_counterfactual_stress",
        "proposed": "proposed_counterfactual_affordance_map",
        "best_non_oracle_baseline": gate["best_non_oracle_baseline"],
        "paired_success_diff": f"{gate['paired_success_diff']:.5f}",
        "paired_success_ci95": f"{gate['paired_success_ci95']:.5f}",
        "paired_cf_recall_diff": f"{gate['paired_cf_recall_diff']:.5f}",
        "paired_cf_recall_ci95": f"{gate['paired_cf_recall_ci95']:.5f}",
        "paired_invalid_reduction": f"{gate['paired_invalid_reduction']:.5f}",
        "paired_invalid_ci95": f"{gate['paired_invalid_ci95']:.5f}",
        "paired_regret_reduction": f"{gate['paired_regret_reduction']:.5f}",
        "paired_regret_ci95": f"{gate['paired_regret_ci95']:.5f}",
        "paired_extra_cost": f"{gate['paired_extra_cost']:.5f}",
        "paired_extra_cost_ci95": f"{gate['paired_extra_cost_ci95']:.5f}",
        "success_gate": success_gate,
        "recall_gate": recall_gate,
        "invalid_or_regret_gate": invalid_or_regret_gate,
        "cost_gate": cost_ok,
        "ablation_gate": ablation_gate,
        "stress_gate": stress_gate,
        "terminal": terminal,
    }]
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)

    negative_cases = [
        {
            "case": "active_probe_matches_action_choice",
            "observed_failure": "interactive probing reaches similar or better task success despite weaker counterfactual map recall",
            "implication": "closed-loop value does not require the full proposed representation in this benchmark",
        },
        {
            "case": "semantic_occlusion_shift",
            "observed_failure": "VLM-style spatial affordances recover semantic matches that the counterfactual map over-penalizes",
            "implication": "counterfactual geometry is not enough for language-conditioned affordance tasks",
        },
        {
            "case": "support_only_ablation",
            "observed_failure": "support-only counterfactuals can match much of the full method on stacking-style tasks",
            "implication": "the broad pose/support/semantic counterfactual mechanism is not cleanly isolated",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", negative_cases)

    plot_bars(summary, "combined_counterfactual_stress", ["affordance_ap", "counterfactual_recall", "calibration_error"], "counterfactual_affordance_map_quality.png", "Paper 93 combined stress: map quality")
    plot_bars(summary, "combined_counterfactual_stress", ["task_success", "invalid_action", "damage"], "counterfactual_affordance_task_outcomes.png", "Paper 93 combined stress: task outcomes")
    plot_bars(summary, "combined_counterfactual_stress", ["planning_regret", "probe_cost", "recovery_success"], "counterfactual_affordance_regret_cost.png", "Paper 93 combined stress: regret and cost")
    plot_ablation(ablation_summary)
    plot_stress(stress_summary)

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 93 counterfactual_affordance_maps v4 rebuild\n")
        handle.write(f"Terminal recommendation: {terminal}\n")
        handle.write("Reason: deterministic manipulation-affordance benchmark added; no robot hardware or accepted high-fidelity manipulation benchmark validation is available.\n")
        handle.write(f"Main rollout rows: {len(main_rows)}\n")
        handle.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        handle.write(f"Stress rollout rows: {len(stress_rows)}\n")
        handle.write(f"Seeds: {SEEDS}\n\n")
        handle.write("Combined counterfactual stress:\n")
        for method in METHODS:
            row = find_row(summary, method, "combined_counterfactual_stress")
            handle.write(
                f"{method} success={row['mean_task_success']} ci95={row['ci95_task_success']} "
                f"ap={row['mean_affordance_ap']} cf_recall={row['mean_counterfactual_recall']} "
                f"invalid={row['mean_invalid_action']} damage={row['mean_damage']} regret={row['mean_planning_regret']} cost={row['mean_probe_cost']}\n"
            )
        handle.write(
            f"paired success diff vs best baseline {gate['best_non_oracle_baseline']}="
            f"{gate['paired_success_diff']:.5f} ci95={gate['paired_success_ci95']:.5f}\n"
        )
        handle.write(
            f"paired cf-recall diff={gate['paired_cf_recall_diff']:.5f} ci95={gate['paired_cf_recall_ci95']:.5f}; "
            f"paired regret reduction={gate['paired_regret_reduction']:.5f} ci95={gate['paired_regret_ci95']:.5f}\n\n"
        )
        handle.write("Ablations:\n")
        for row in ablation_summary:
            handle.write(
                f"{row['method']} success={row['mean_task_success']} cf_recall={row['mean_counterfactual_recall']} "
                f"invalid={row['mean_invalid_action']} regret={row['mean_planning_regret']} cost={row['mean_probe_cost']}\n"
            )
        handle.write("\nCombined stress level 1.0:\n")
        for row in stress_max:
            handle.write(
                f"{row['method']} success={row['mean_task_success']} cf_recall={row['mean_counterfactual_recall']} "
                f"invalid={row['mean_invalid_action']} regret={row['mean_planning_regret']}\n"
            )
        handle.write("\nGate checks:\n")
        handle.write(f"success_gate={success_gate}\n")
        handle.write(f"recall_gate={recall_gate}\n")
        handle.write(f"invalid_or_regret_gate={invalid_or_regret_gate}\n")
        handle.write(f"cost_ok={cost_ok}\n")
        handle.write(f"ablation_gate={ablation_gate} best_ablation={best_ablation}\n")
        handle.write(f"stress_gate={stress_gate} stress_best_baseline={stress_best_baseline['method']}\n")
        handle.write(f"oracle_success={oracle['mean_task_success']}\n")

    print(f"terminal={terminal}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
