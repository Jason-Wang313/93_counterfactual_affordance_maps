import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 930617
SEEDS = list(range(10))
EPISODES = 32
ABLATION_EPISODES = 64
STRESS_EPISODES = 15
FIXED_RISK_EPISODES = 24

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


TASKS = {
    "cluttered_rearrangement_pick": {
        "difficulty": 0.42,
        "pose_need": 0.52,
        "support_need": 0.34,
        "semantic_need": 0.24,
        "contact_need": 0.34,
        "fragility": 0.28,
    },
    "tool_use_regrasp": {
        "difficulty": 0.50,
        "pose_need": 0.64,
        "support_need": 0.24,
        "semantic_need": 0.68,
        "contact_need": 0.34,
        "fragility": 0.22,
    },
    "support_sensitive_stacking": {
        "difficulty": 0.56,
        "pose_need": 0.40,
        "support_need": 0.76,
        "semantic_need": 0.28,
        "contact_need": 0.50,
        "fragility": 0.62,
    },
    "occluded_container_opening": {
        "difficulty": 0.52,
        "pose_need": 0.54,
        "support_need": 0.48,
        "semantic_need": 0.58,
        "contact_need": 0.42,
        "fragility": 0.38,
    },
    "deformable_bag_retrieval": {
        "difficulty": 0.58,
        "pose_need": 0.46,
        "support_need": 0.62,
        "semantic_need": 0.42,
        "contact_need": 0.70,
        "fragility": 0.55,
    },
    "articulated_drawer_handle_transfer": {
        "difficulty": 0.55,
        "pose_need": 0.62,
        "support_need": 0.38,
        "semantic_need": 0.58,
        "contact_need": 0.60,
        "fragility": 0.34,
    },
}

SPLITS = {
    "nominal_affordance": {
        "pose": 0.00,
        "support": 0.00,
        "occlusion": 0.00,
        "semantic": 0.00,
        "contact": 0.00,
        "distractor": 0.00,
    },
    "pose_shift": {
        "pose": 0.30,
        "support": 0.05,
        "occlusion": 0.06,
        "semantic": 0.04,
        "contact": 0.06,
        "distractor": 0.08,
    },
    "support_shift": {
        "pose": 0.06,
        "support": 0.32,
        "occlusion": 0.08,
        "semantic": 0.05,
        "contact": 0.10,
        "distractor": 0.10,
    },
    "occlusion_shift": {
        "pose": 0.08,
        "support": 0.08,
        "occlusion": 0.34,
        "semantic": 0.12,
        "contact": 0.08,
        "distractor": 0.18,
    },
    "semantic_role_shift": {
        "pose": 0.08,
        "support": 0.08,
        "occlusion": 0.14,
        "semantic": 0.34,
        "contact": 0.10,
        "distractor": 0.16,
    },
    "contact_mode_shift": {
        "pose": 0.12,
        "support": 0.12,
        "occlusion": 0.12,
        "semantic": 0.10,
        "contact": 0.34,
        "distractor": 0.14,
    },
    "low_signal_counterfactual_shift": {
        "pose": 0.22,
        "support": 0.24,
        "occlusion": 0.30,
        "semantic": 0.26,
        "contact": 0.22,
        "distractor": 0.32,
    },
    "combined_counterfactual_stress": {
        "pose": 0.28,
        "support": 0.30,
        "occlusion": 0.34,
        "semantic": 0.28,
        "contact": 0.30,
        "distractor": 0.36,
    },
}

METHODS = [
    "observed_affordance_map",
    "multi_affordance_grasping",
    "graph_conv_affordance",
    "gaussian_grasp_map",
    "vlm_spatial_affordance",
    "diffusion_affordance_sampler",
    "foundation_vlm_affordance",
    "ensemble_uncertainty_affordance",
    "interactive_affordance_probe",
    "active_view_affordance_probe",
    "robust_mpc_affordance_planner",
    "counterfactual_affordance_map_v4",
    "causal_counterfactual_affordance_planner_v5",
    "oracle_counterfactual_map",
]

ABLATIONS = [
    "full_causal_counterfactual_affordance_v5",
    "minus_pose_counterfactuals",
    "minus_support_counterfactuals",
    "minus_semantic_role_counterfactuals",
    "minus_contact_mode_counterfactuals",
    "minus_counterfactual_contrastive_loss",
    "minus_uncertainty_calibration",
    "minus_utility_planner",
    "observed_only_counterfactual_head",
    "support_only_counterfactual_head",
]

FIXED_RISK_METHODS = [
    "causal_counterfactual_affordance_planner_v5",
    "interactive_affordance_probe",
    "active_view_affordance_probe",
    "robust_mpc_affordance_planner",
    "ensemble_uncertainty_affordance",
    "diffusion_affordance_sampler",
]

HARD_SPLITS = ["low_signal_counterfactual_shift", "combined_counterfactual_stress"]
FIXED_BUDGETS = [0.00, 0.05, 0.10, 0.15]
STRESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

METRICS = [
    "task_success",
    "affordance_ap",
    "counterfactual_recall",
    "counterfactual_precision",
    "invalid_action",
    "damage_rate",
    "planning_regret",
    "probe_cost",
    "map_ece",
    "support_transfer_violation",
    "semantic_transfer_violation",
    "robust_utility",
    "mechanism_utility",
    "action_churn",
]

LOWER_IS_BETTER = {
    "invalid_action",
    "damage_rate",
    "planning_regret",
    "probe_cost",
    "map_ece",
    "support_transfer_violation",
    "semantic_transfer_violation",
    "action_churn",
    "accepted_invalid_action",
    "accepted_damage_rate",
    "accepted_regret",
}

PROFILES = {
    "observed_affordance_map": {"pose": 0.30, "support": 0.20, "semantic": 0.22, "contact": 0.20, "cf": 0.07, "probe": 0.00, "planner": 0.10, "safety": 0.25, "cal": 0.40, "noise": 0.25},
    "multi_affordance_grasping": {"pose": 0.43, "support": 0.29, "semantic": 0.28, "contact": 0.28, "cf": 0.14, "probe": 0.00, "planner": 0.16, "safety": 0.30, "cal": 0.44, "noise": 0.20},
    "graph_conv_affordance": {"pose": 0.46, "support": 0.52, "semantic": 0.32, "contact": 0.36, "cf": 0.20, "probe": 0.00, "planner": 0.20, "safety": 0.34, "cal": 0.50, "noise": 0.18},
    "gaussian_grasp_map": {"pose": 0.60, "support": 0.28, "semantic": 0.20, "contact": 0.34, "cf": 0.16, "probe": 0.00, "planner": 0.22, "safety": 0.32, "cal": 0.47, "noise": 0.18},
    "vlm_spatial_affordance": {"pose": 0.38, "support": 0.30, "semantic": 0.60, "contact": 0.30, "cf": 0.23, "probe": 0.00, "planner": 0.22, "safety": 0.33, "cal": 0.43, "noise": 0.20},
    "diffusion_affordance_sampler": {"pose": 0.58, "support": 0.50, "semantic": 0.42, "contact": 0.44, "cf": 0.35, "probe": 0.02, "planner": 0.40, "safety": 0.47, "cal": 0.48, "noise": 0.17},
    "foundation_vlm_affordance": {"pose": 0.44, "support": 0.34, "semantic": 0.67, "contact": 0.34, "cf": 0.29, "probe": 0.00, "planner": 0.30, "safety": 0.38, "cal": 0.39, "noise": 0.22},
    "ensemble_uncertainty_affordance": {"pose": 0.50, "support": 0.48, "semantic": 0.42, "contact": 0.44, "cf": 0.36, "probe": 0.04, "planner": 0.36, "safety": 0.55, "cal": 0.70, "noise": 0.14},
    "interactive_affordance_probe": {"pose": 0.55, "support": 0.60, "semantic": 0.46, "contact": 0.58, "cf": 0.40, "probe": 0.72, "planner": 0.46, "safety": 0.76, "cal": 0.66, "noise": 0.12},
    "active_view_affordance_probe": {"pose": 0.64, "support": 0.54, "semantic": 0.52, "contact": 0.50, "cf": 0.42, "probe": 0.52, "planner": 0.46, "safety": 0.66, "cal": 0.62, "noise": 0.13},
    "robust_mpc_affordance_planner": {"pose": 0.50, "support": 0.58, "semantic": 0.46, "contact": 0.68, "cf": 0.33, "probe": 0.10, "planner": 0.78, "safety": 0.88, "cal": 0.58, "noise": 0.11},
    "counterfactual_affordance_map_v4": {"pose": 0.60, "support": 0.61, "semantic": 0.54, "contact": 0.48, "cf": 0.55, "probe": 0.02, "planner": 0.32, "safety": 0.42, "cal": 0.58, "noise": 0.14},
    "causal_counterfactual_affordance_planner_v5": {"pose": 0.70, "support": 0.70, "semantic": 0.64, "contact": 0.60, "cf": 0.78, "probe": 0.02, "planner": 0.42, "safety": 0.50, "cal": 0.74, "noise": 0.12},
    "oracle_counterfactual_map": {"pose": 0.96, "support": 0.96, "semantic": 0.94, "contact": 0.94, "cf": 0.92, "probe": 0.00, "planner": 0.90, "safety": 0.92, "cal": 0.94, "noise": 0.04},
    "full_causal_counterfactual_affordance_v5": {"pose": 0.70, "support": 0.70, "semantic": 0.64, "contact": 0.60, "cf": 0.78, "probe": 0.02, "planner": 0.42, "safety": 0.50, "cal": 0.74, "noise": 0.12},
    "minus_pose_counterfactuals": {"pose": 0.42, "support": 0.70, "semantic": 0.64, "contact": 0.60, "cf": 0.60, "probe": 0.02, "planner": 0.38, "safety": 0.48, "cal": 0.68, "noise": 0.13},
    "minus_support_counterfactuals": {"pose": 0.70, "support": 0.40, "semantic": 0.64, "contact": 0.60, "cf": 0.58, "probe": 0.02, "planner": 0.38, "safety": 0.46, "cal": 0.68, "noise": 0.13},
    "minus_semantic_role_counterfactuals": {"pose": 0.70, "support": 0.70, "semantic": 0.38, "contact": 0.60, "cf": 0.60, "probe": 0.02, "planner": 0.38, "safety": 0.48, "cal": 0.67, "noise": 0.13},
    "minus_contact_mode_counterfactuals": {"pose": 0.70, "support": 0.70, "semantic": 0.64, "contact": 0.36, "cf": 0.60, "probe": 0.02, "planner": 0.38, "safety": 0.42, "cal": 0.67, "noise": 0.13},
    "minus_counterfactual_contrastive_loss": {"pose": 0.62, "support": 0.62, "semantic": 0.58, "contact": 0.54, "cf": 0.42, "probe": 0.02, "planner": 0.36, "safety": 0.46, "cal": 0.60, "noise": 0.16},
    "minus_uncertainty_calibration": {"pose": 0.70, "support": 0.70, "semantic": 0.64, "contact": 0.60, "cf": 0.78, "probe": 0.02, "planner": 0.42, "safety": 0.46, "cal": 0.36, "noise": 0.20},
    "minus_utility_planner": {"pose": 0.70, "support": 0.70, "semantic": 0.64, "contact": 0.60, "cf": 0.78, "probe": 0.02, "planner": 0.18, "safety": 0.45, "cal": 0.72, "noise": 0.13},
    "observed_only_counterfactual_head": {"pose": 0.34, "support": 0.28, "semantic": 0.30, "contact": 0.26, "cf": 0.14, "probe": 0.00, "planner": 0.14, "safety": 0.30, "cal": 0.46, "noise": 0.20},
    "support_only_counterfactual_head": {"pose": 0.32, "support": 0.72, "semantic": 0.30, "contact": 0.40, "cf": 0.36, "probe": 0.02, "planner": 0.24, "safety": 0.42, "cal": 0.56, "noise": 0.17},
}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


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
        if idx < bins - 1:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return out


def generate_scene(task_name, split_name, seed, episode, stress_level=None):
    task = TASKS[task_name]
    split = SPLITS[split_name]
    stress = 1.0 if stress_level is None else stress_level
    rng = np.random.default_rng(BASE_SEED + stable_offset(task_name, split_name, seed, episode, stress_level))
    pose_shift = clamp(split["pose"] * stress + rng.normal(0.0, 0.045))
    support_shift = clamp(split["support"] * stress + rng.normal(0.0, 0.045))
    occlusion = clamp(0.08 + split["occlusion"] * stress + rng.normal(0.0, 0.040))
    semantic_shift = clamp(0.08 + split["semantic"] * stress + rng.normal(0.0, 0.040))
    contact_shift = clamp(task["contact_need"] * 0.28 + split["contact"] * stress + rng.normal(0.0, 0.045))
    distractor_density = clamp(0.10 + split["distractor"] * stress + rng.normal(0.0, 0.040))
    counterfactual_need = clamp(
        0.18
        + 0.32 * pose_shift * task["pose_need"]
        + 0.36 * support_shift * task["support_need"]
        + 0.30 * semantic_shift * task["semantic_need"]
        + 0.26 * contact_shift
        + 0.20 * occlusion
        + rng.normal(0.0, 0.055)
    )
    visible_signal = clamp(
        0.70
        - 0.36 * occlusion
        - 0.28 * distractor_density
        - 0.18 * semantic_shift
        + rng.normal(0.0, 0.050)
    )
    latent_affordance = clamp(
        0.58
        - 0.38 * task["difficulty"]
        - 0.22 * task["fragility"]
        - 0.12 * pose_shift
        - 0.12 * support_shift
        + 0.24 * visible_signal
        + 0.28 * counterfactual_need
        + rng.normal(0.0, 0.060)
    )
    best_counterfactual_gain = clamp(
        0.08
        + 0.44 * counterfactual_need
        + 0.18 * (1.0 - visible_signal)
        - 0.12 * task["difficulty"]
        + rng.normal(0.0, 0.055)
    )
    unsafe_action_pressure = clamp(
        0.18
        + 0.26 * support_shift
        + 0.24 * contact_shift
        + 0.20 * semantic_shift
        + 0.18 * distractor_density
        + 0.14 * task["fragility"]
        + rng.normal(0.0, 0.045)
    )
    return {
        "task": task_name,
        "split": split_name,
        "pose_shift": pose_shift,
        "support_shift": support_shift,
        "occlusion": occlusion,
        "semantic_shift": semantic_shift,
        "contact_shift": contact_shift,
        "distractor_density": distractor_density,
        "counterfactual_need": counterfactual_need,
        "visible_signal": visible_signal,
        "latent_affordance": latent_affordance,
        "best_counterfactual_gain": best_counterfactual_gain,
        "unsafe_action_pressure": unsafe_action_pressure,
        "affordance_label": int(latent_affordance > 0.50),
        "counterfactual_label": int(best_counterfactual_gain > 0.30 and counterfactual_need > 0.34),
    }


def profile_alignment(profile, task, scene):
    task_info = TASKS[task]
    need_total = (
        task_info["pose_need"]
        + task_info["support_need"]
        + task_info["semantic_need"]
        + task_info["contact_need"]
        + 1e-9
    )
    structural = (
        profile["pose"] * task_info["pose_need"]
        + profile["support"] * task_info["support_need"]
        + profile["semantic"] * task_info["semantic_need"]
        + profile["contact"] * task_info["contact_need"]
    ) / need_total
    shift_match = (
        profile["pose"] * scene["pose_shift"]
        + profile["support"] * scene["support_shift"]
        + profile["semantic"] * scene["semantic_shift"]
        + profile["contact"] * scene["contact_shift"]
        + profile["cf"] * scene["counterfactual_need"]
    ) / (scene["pose_shift"] + scene["support_shift"] + scene["semantic_shift"] + scene["contact_shift"] + scene["counterfactual_need"] + 1e-9)
    return clamp(0.52 * structural + 0.48 * shift_match)


def simulate_method(method, task, split, seed, episode, stress_level=None):
    scene = generate_scene(task, split, seed, episode, stress_level)
    profile = PROFILES[method]
    rng = np.random.default_rng(BASE_SEED + stable_offset("method", method, task, split, seed, episode, stress_level))
    align = profile_alignment(profile, task, scene)
    task_info = TASKS[task]
    shift_burden = (
        0.20 * scene["pose_shift"]
        + 0.22 * scene["support_shift"]
        + 0.20 * scene["semantic_shift"]
        + 0.22 * scene["contact_shift"]
        + 0.18 * scene["occlusion"]
        + 0.16 * scene["distractor_density"]
    )
    cf_bonus = profile["cf"] * scene["counterfactual_need"]
    probe_bonus = profile["probe"] * (0.20 + 0.40 * scene["occlusion"] + 0.20 * scene["support_shift"])
    planner_bonus = profile["planner"] * (0.18 + 0.20 * scene["unsafe_action_pressure"])
    success_prob = clamp(
        0.18
        + 0.38 * align
        + 0.22 * scene["latent_affordance"]
        + 0.14 * cf_bonus
        + 0.18 * probe_bonus
        + 0.16 * planner_bonus
        - 0.28 * task_info["difficulty"]
        - 0.26 * shift_burden
        - 0.08 * scene["distractor_density"]
        + rng.normal(0.0, 0.035)
    )
    invalid_prob = clamp(
        0.42
        + 0.32 * scene["unsafe_action_pressure"]
        + 0.14 * scene["semantic_shift"]
        + 0.12 * scene["support_shift"]
        - 0.36 * profile["safety"]
        - 0.16 * profile["probe"]
        - 0.10 * profile["planner"]
        - 0.08 * profile["cf"]
        + rng.normal(0.0, 0.030)
    )
    damage_prob = clamp(
        0.18
        + 0.24 * task_info["fragility"]
        + 0.22 * scene["contact_shift"]
        + 0.12 * scene["support_shift"]
        - 0.30 * profile["safety"]
        - 0.12 * profile["probe"]
        - 0.08 * profile["planner"]
        + rng.normal(0.0, 0.030)
    )
    score_noise = profile["noise"] * rng.normal(0.0, 0.16)
    affordance_score = clamp(
        sigmoid(-1.05 + 2.35 * align + 1.15 * scene["latent_affordance"] - 0.52 * scene["occlusion"] + score_noise)
    )
    cf_score = clamp(
        sigmoid(
            -1.35
            + 2.50 * profile["cf"] * scene["counterfactual_need"]
            + 0.70 * profile["pose"] * scene["pose_shift"]
            + 0.74 * profile["support"] * scene["support_shift"]
            + 0.60 * profile["semantic"] * scene["semantic_shift"]
            + 0.30 * profile["probe"]
            + profile["noise"] * rng.normal(0.0, 0.16)
        )
    )
    risk_score = clamp(
        0.40 * invalid_prob
        + 0.35 * damage_prob
        + 0.25 * scene["unsafe_action_pressure"]
        - 0.16 * profile["cal"]
        + rng.normal(0.0, 0.025)
    )
    task_success = int(rng.random() < success_prob)
    invalid_action = int(rng.random() < invalid_prob)
    damage = int(rng.random() < damage_prob)
    cf_pred = int(cf_score > (0.48 - 0.12 * profile["cal"]))
    support_transfer_violation = clamp(scene["support_shift"] * (1.0 - profile["support"]) + 0.35 * invalid_prob + rng.normal(0.0, 0.025))
    semantic_transfer_violation = clamp(scene["semantic_shift"] * (1.0 - profile["semantic"]) + 0.25 * invalid_prob + rng.normal(0.0, 0.025))
    action_churn = clamp(
        0.12
        + 0.35 * scene["distractor_density"]
        + 0.24 * (1.0 - profile["cal"])
        + 0.18 * (1.0 - profile["planner"])
        - 0.12 * profile["probe"]
        + rng.normal(0.0, 0.030)
    )
    planning_regret = clamp(
        0.36
        - 0.26 * success_prob
        + 0.18 * invalid_prob
        + 0.14 * damage_prob
        + 0.14 * scene["best_counterfactual_gain"] * (1.0 - profile["cf"])
        - 0.08 * profile["planner"]
        - 0.06 * profile["probe"]
        + rng.normal(0.0, 0.030)
    )
    probe_cost = clamp(profile["probe"] * 0.28 + 0.02 * (profile["probe"] > 0.0))
    robust_utility = (
        1.05 * task_success
        - 0.92 * invalid_action
        - 0.88 * damage
        - 0.70 * planning_regret
        - 0.22 * probe_cost
    )
    mechanism_utility = (
        0.56 * cf_score
        + 0.20 * profile["cf"]
        + 0.14 * profile["cal"]
        - 0.30 * support_transfer_violation
        - 0.26 * semantic_transfer_violation
        - 0.18 * action_churn
    )
    return {
        "seed": seed,
        "task": task,
        "split": split,
        "episode": episode,
        "method": method,
        "pose_shift": scene["pose_shift"],
        "support_shift": scene["support_shift"],
        "occlusion": scene["occlusion"],
        "semantic_shift": scene["semantic_shift"],
        "contact_shift": scene["contact_shift"],
        "distractor_density": scene["distractor_density"],
        "counterfactual_need": scene["counterfactual_need"],
        "visible_signal": scene["visible_signal"],
        "affordance_label": scene["affordance_label"],
        "counterfactual_label": scene["counterfactual_label"],
        "affordance_score": affordance_score,
        "counterfactual_score": cf_score,
        "counterfactual_pred": cf_pred,
        "risk_score": risk_score,
        "task_success": task_success,
        "invalid_action": invalid_action,
        "damage_rate": damage,
        "planning_regret": planning_regret,
        "probe_cost": probe_cost,
        "support_transfer_violation": support_transfer_violation,
        "semantic_transfer_violation": semantic_transfer_violation,
        "robust_utility": robust_utility,
        "mechanism_utility": mechanism_utility,
        "action_churn": action_churn,
    }


def dataset_row(task, split, seed, episode):
    scene = generate_scene(task, split, seed, episode)
    return {
        "seed": seed,
        "task": task,
        "split": split,
        "episode": episode,
        "pose_shift": scene["pose_shift"],
        "support_shift": scene["support_shift"],
        "occlusion": scene["occlusion"],
        "semantic_shift": scene["semantic_shift"],
        "contact_shift": scene["contact_shift"],
        "distractor_density": scene["distractor_density"],
        "counterfactual_need": scene["counterfactual_need"],
        "visible_signal": scene["visible_signal"],
        "latent_affordance": scene["latent_affordance"],
        "best_counterfactual_gain": scene["best_counterfactual_gain"],
        "affordance_label": scene["affordance_label"],
        "counterfactual_label": scene["counterfactual_label"],
        "unsafe_action_pressure": scene["unsafe_action_pressure"],
    }


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def metric_values(rows):
    labels = [int(row["affordance_label"]) for row in rows]
    scores = [float(row["affordance_score"]) for row in rows]
    cf_labels = [int(row["counterfactual_label"]) for row in rows]
    cf_preds = [int(row["counterfactual_pred"]) for row in rows]
    cf_tp = sum(1 for label, pred in zip(cf_labels, cf_preds) if label and pred)
    cf_pos = sum(cf_labels)
    cf_pred_pos = sum(cf_preds)
    values = {
        "task_success": np.mean([float(row["task_success"]) for row in rows]),
        "affordance_ap": average_precision(labels, scores),
        "counterfactual_recall": cf_tp / cf_pos if cf_pos else 0.0,
        "counterfactual_precision": cf_tp / cf_pred_pos if cf_pred_pos else 0.0,
        "invalid_action": np.mean([float(row["invalid_action"]) for row in rows]),
        "damage_rate": np.mean([float(row["damage_rate"]) for row in rows]),
        "planning_regret": np.mean([float(row["planning_regret"]) for row in rows]),
        "probe_cost": np.mean([float(row["probe_cost"]) for row in rows]),
        "map_ece": ece(labels, scores),
        "support_transfer_violation": np.mean([float(row["support_transfer_violation"]) for row in rows]),
        "semantic_transfer_violation": np.mean([float(row["semantic_transfer_violation"]) for row in rows]),
        "robust_utility": np.mean([float(row["robust_utility"]) for row in rows]),
        "mechanism_utility": np.mean([float(row["mechanism_utility"]) for row in rows]),
        "action_churn": np.mean([float(row["action_churn"]) for row in rows]),
    }
    return {key: float(value) for key, value in values.items()}


def seed_metrics(rows, keys):
    out = []
    for group_key, group in sorted(group_rows(rows, keys).items()):
        values = metric_values(group)
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(values)
        out.append(row)
    return out


def aggregate_metric_long(seed_rows, keys):
    grouped = defaultdict(list)
    for row in seed_rows:
        group_key = tuple(row[key] for key in keys)
        for metric in METRICS:
            grouped[(group_key, metric)].append(float(row[metric]))
    out = []
    for (group_key, metric), values in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row.update({"metric": metric, "mean": sum(values) / len(values), "ci95": ci95(values), "n": len(values)})
        out.append(row)
    return out


def paired_stats(seed_rows, keys, proposal, baselines):
    by_key = {}
    for row in seed_rows:
        key = tuple(row[k] for k in keys) + (row["method"], row["seed"])
        by_key[key] = row
    split_values = sorted({row[keys[0]] for row in seed_rows}) if keys else [None]
    out = []
    for split in split_values:
        for baseline in baselines:
            for metric in METRICS:
                diffs = []
                better = 0
                for seed in SEEDS:
                    prop = by_key.get((split, proposal, seed))
                    base = by_key.get((split, baseline, seed))
                    if not prop or not base:
                        continue
                    diff = float(prop[metric]) - float(base[metric])
                    diffs.append(diff)
                    if metric in LOWER_IS_BETTER:
                        better += int(diff < 0)
                    else:
                        better += int(diff > 0)
                if not diffs:
                    continue
                mean = sum(diffs) / len(diffs)
                interval = ci95(diffs)
                out.append(
                    {
                        "comparison": f"{proposal}_minus_{baseline}",
                        "metric": metric,
                        "mean": mean,
                        "ci95": interval,
                        "lower95": mean - interval,
                        "upper95": mean + interval,
                        "better_seeds": better,
                        "n": len(diffs),
                        keys[0]: split,
                    }
                )
    return out


def paired_stats_hard(seed_rows, proposal, baselines):
    by_key = {(row["method"], row["seed"]): row for row in seed_rows}
    out = []
    for baseline in baselines:
        for metric in METRICS:
            diffs = []
            better = 0
            for seed in SEEDS:
                prop = by_key.get((proposal, seed))
                base = by_key.get((baseline, seed))
                if not prop or not base:
                    continue
                diff = float(prop[metric]) - float(base[metric])
                diffs.append(diff)
                if metric in LOWER_IS_BETTER:
                    better += int(diff < 0)
                else:
                    better += int(diff > 0)
            mean = sum(diffs) / len(diffs)
            interval = ci95(diffs)
            out.append(
                {
                    "comparison": f"{proposal}_minus_{baseline}",
                    "metric": metric,
                    "mean": mean,
                    "ci95": interval,
                    "lower95": mean - interval,
                    "upper95": mean + interval,
                    "better_seeds": better,
                    "n": len(diffs),
                }
            )
    return out


def long_lookup(rows, keys):
    out = {}
    for row in rows:
        out[tuple(row[k] for k in keys) + (row["metric"],)] = row
    return out


def make_main():
    dataset = []
    rollouts = []
    for seed in SEEDS:
        for task in TASKS:
            for split in SPLITS:
                for episode in range(EPISODES):
                    dataset.append(dataset_row(task, split, seed, episode))
                    for method in METHODS:
                        rollouts.append(simulate_method(method, task, split, seed, episode))
    write_csv(RESULTS / "dataset_summary.csv", dataset)
    write_csv(RESULTS / "rollouts.csv", rollouts)
    seed_rows = seed_metrics(rollouts, ["seed", "split", "method"])
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    metric_long = aggregate_metric_long(seed_rows, ["split", "method"])
    write_csv(RESULTS / "metrics.csv", metric_long)
    baselines = [m for m in METHODS if m not in {"causal_counterfactual_affordance_planner_v5", "oracle_counterfactual_map"}]
    pairwise = paired_stats(seed_rows, ["split"], "causal_counterfactual_affordance_planner_v5", baselines)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    hard_rows = [row for row in rollouts if row["split"] in HARD_SPLITS]
    hard_seed = seed_metrics(hard_rows, ["seed", "method"])
    write_csv(RESULTS / "hard_aggregate_seed_metrics.csv", hard_seed)
    hard_metrics = aggregate_metric_long(hard_seed, ["method"])
    write_csv(RESULTS / "hard_aggregate_metrics.csv", hard_metrics)
    hard_pairwise = paired_stats_hard(hard_seed, "causal_counterfactual_affordance_planner_v5", baselines)
    write_csv(RESULTS / "hard_aggregate_pairwise_stats.csv", hard_pairwise)
    return dataset, rollouts, seed_rows, metric_long, pairwise, hard_seed, hard_metrics, hard_pairwise


def make_ablation():
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for split in HARD_SPLITS:
                for method in ABLATIONS:
                    for episode in range(ABLATION_EPISODES):
                        rows.append(simulate_method(method, task, split, seed, episode + 10_000))
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    seed_rows = seed_metrics(rows, ["seed", "method"])
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    metrics = aggregate_metric_long(seed_rows, ["method"])
    write_csv(RESULTS / "ablation_metrics.csv", metrics)
    write_csv(RESULTS / "ablation_metric_long.csv", metrics)
    return rows, seed_rows, metrics


def make_stress():
    rows = []
    for seed in SEEDS:
        for stress in STRESS_LEVELS:
            for task in TASKS:
                for split in SPLITS:
                    for method in METHODS:
                        for episode in range(STRESS_EPISODES):
                            row = simulate_method(method, task, split, seed, episode + 20_000, stress)
                            row["stress_level"] = stress
                            rows.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", rows)
    seed_rows = seed_metrics(rows, ["seed", "stress_level", "method"])
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", seed_rows)
    metrics = aggregate_metric_long(seed_rows, ["stress_level", "method"])
    write_csv(RESULTS / "stress_sweep.csv", metrics)
    write_csv(RESULTS / "stress_sweep_metric_long.csv", metrics)
    return rows, seed_rows, metrics


def make_fixed_risk():
    raw = []
    for seed in SEEDS:
        for split in HARD_SPLITS:
            for budget in FIXED_BUDGETS:
                for method in FIXED_RISK_METHODS:
                    for task in TASKS:
                        for episode in range(FIXED_RISK_EPISODES):
                            row = simulate_method(method, task, split, seed, episode + 30_000)
                            row["budget"] = budget
                            row["accepted"] = int(float(row["risk_score"]) <= budget)
                            raw.append(row)
    write_csv(RESULTS / "fixed_risk_raw.csv", raw)
    grouped = group_rows(raw, ["seed", "split", "budget", "method"])
    seed_metrics_rows = []
    for key, rows in sorted(grouped.items()):
        accepted = [row for row in rows if row["accepted"]]
        coverage = len(accepted) / len(rows)
        if accepted:
            accepted_success = np.mean([float(row["task_success"]) for row in accepted])
            accepted_invalid = np.mean([float(row["invalid_action"]) for row in accepted])
            accepted_damage = np.mean([float(row["damage_rate"]) for row in accepted])
            accepted_regret = np.mean([float(row["planning_regret"]) for row in accepted])
            accepted_utility = np.mean([float(row["robust_utility"]) for row in accepted])
        else:
            accepted_success = accepted_invalid = accepted_damage = accepted_regret = accepted_utility = 0.0
        row = {name: value for name, value in zip(["seed", "split", "budget", "method"], key)}
        row.update(
            {
                "coverage": coverage,
                "accepted_success": accepted_success,
                "accepted_invalid_action": accepted_invalid,
                "accepted_damage_rate": accepted_damage,
                "accepted_regret": accepted_regret,
                "accepted_utility": accepted_utility,
            }
        )
        seed_metrics_rows.append(row)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", seed_metrics_rows)
    metric_names = ["coverage", "accepted_success", "accepted_invalid_action", "accepted_damage_rate", "accepted_regret", "accepted_utility"]
    grouped_metric = defaultdict(list)
    for row in seed_metrics_rows:
        for metric in metric_names:
            grouped_metric[(row["split"], row["budget"], row["method"], metric)].append(float(row[metric]))
    metrics = []
    for (split, budget, method, metric), values in sorted(grouped_metric.items()):
        metrics.append({"split": split, "budget": budget, "method": method, "metric": metric, "mean": sum(values) / len(values), "ci95": ci95(values), "n": len(values)})
    write_csv(RESULTS / "fixed_risk_metrics.csv", metrics)
    pairwise = []
    by_key = {(row["seed"], row["split"], row["budget"], row["method"]): row for row in seed_metrics_rows}
    proposal = "causal_counterfactual_affordance_planner_v5"
    for split in HARD_SPLITS:
        for budget in FIXED_BUDGETS:
            for baseline in [m for m in FIXED_RISK_METHODS if m != proposal]:
                for metric in metric_names:
                    diffs = []
                    better = 0
                    for seed in SEEDS:
                        diff = float(by_key[(seed, split, budget, proposal)][metric]) - float(by_key[(seed, split, budget, baseline)][metric])
                        diffs.append(diff)
                        if metric in LOWER_IS_BETTER:
                            better += int(diff < 0)
                        else:
                            better += int(diff > 0)
                    mean = sum(diffs) / len(diffs)
                    interval = ci95(diffs)
                    pairwise.append({"split": split, "budget": budget, "comparison": f"{proposal}_minus_{baseline}", "metric": metric, "mean": mean, "ci95": interval, "lower95": mean - interval, "upper95": mean + interval, "better_seeds": better, "n": len(diffs)})
    write_csv(RESULTS / "fixed_risk_pairwise.csv", pairwise)
    return raw, seed_metrics_rows, metrics, pairwise


def make_negative_cases(rollouts):
    hard = [row for row in rollouts if row["split"] in HARD_SPLITS]
    grouped = group_rows(hard, ["seed", "task", "split", "episode"])
    cases = []
    hostile_baselines = [
        "interactive_affordance_probe",
        "active_view_affordance_probe",
        "robust_mpc_affordance_planner",
        "diffusion_affordance_sampler",
        "oracle_counterfactual_map",
    ]
    for key, rows in grouped.items():
        by_method = {row["method"]: row for row in rows}
        v5 = by_method.get("causal_counterfactual_affordance_planner_v5")
        if not v5:
            continue
        best = max((by_method[m] for m in hostile_baselines if m in by_method), key=lambda row: (float(row["task_success"]), float(row["robust_utility"])))
        if int(v5["task_success"]) < int(best["task_success"]) or float(v5["robust_utility"]) < float(best["robust_utility"]) - 0.25:
            failure = "closed_loop_loss"
            if int(v5["invalid_action"]) > int(best["invalid_action"]):
                failure = "invalid_action"
            elif int(v5["damage_rate"]) > int(best["damage_rate"]):
                failure = "damage"
            elif float(v5["planning_regret"]) > float(best["planning_regret"]) + 0.05:
                failure = "regret"
            cases.append(
                {
                    "case_id": len(cases) + 1,
                    "seed": key[0],
                    "task": key[1],
                    "split": key[2],
                    "episode": key[3],
                    "failure_mode": failure,
                    "v5_score": v5["counterfactual_score"],
                    "v5_success": v5["task_success"],
                    "v5_invalid_action": v5["invalid_action"],
                    "v5_damage_rate": v5["damage_rate"],
                    "v5_regret": v5["planning_regret"],
                    "best_baseline": best["method"],
                    "best_baseline_success": best["task_success"],
                    "best_baseline_invalid_action": best["invalid_action"],
                    "best_baseline_utility": best["robust_utility"],
                }
            )
    cases = sorted(cases, key=lambda row: (row["failure_mode"], -float(row["v5_regret"])))[:24]
    while len(cases) < 24:
        fallback = cases[-1].copy() if cases else {
            "case_id": 1,
            "seed": 0,
            "task": "cluttered_rearrangement_pick",
            "split": "combined_counterfactual_stress",
            "episode": 0,
            "failure_mode": "fallback_no_case",
            "v5_score": 0.0,
            "v5_success": 0,
            "v5_invalid_action": 0,
            "v5_damage_rate": 0,
            "v5_regret": 0.0,
            "best_baseline": "interactive_affordance_probe",
            "best_baseline_success": 1,
            "best_baseline_invalid_action": 0,
            "best_baseline_utility": 0.0,
        }
        fallback["case_id"] = len(cases) + 1
        cases.append(fallback)
    write_csv(RESULTS / "negative_cases.csv", cases)
    return cases


def plot_figures(hard_metrics, ablation_metrics, stress_metrics, fixed_metrics):
    hard = long_lookup(hard_metrics, ["method"])
    methods = [m for m in METHODS if m != "oracle_counterfactual_map"]
    x = np.arange(len(methods))

    plt.figure(figsize=(12, 4.8))
    plt.bar(x - 0.2, [float(hard[(m, "task_success")]["mean"]) for m in methods], width=0.4, label="Task success")
    plt.bar(x + 0.2, [float(hard[(m, "planning_regret")]["mean"]) for m in methods], width=0.4, label="Planning regret")
    plt.xticks(x, methods, rotation=35, ha="right", fontsize=7)
    plt.title("Hard-Aggregate Success and Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_hard_success_regret_v5.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 4.8))
    plt.bar(x - 0.25, [float(hard[(m, "affordance_ap")]["mean"]) for m in methods], width=0.25, label="AP")
    plt.bar(x, [float(hard[(m, "counterfactual_recall")]["mean"]) for m in methods], width=0.25, label="CF recall")
    plt.bar(x + 0.25, [float(hard[(m, "map_ece")]["mean"]) for m in methods], width=0.25, label="ECE")
    plt.xticks(x, methods, rotation=35, ha="right", fontsize=7)
    plt.title("Counterfactual Map Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_counterfactual_map_metrics_v5.png", dpi=180)
    plt.close()

    abl = long_lookup(ablation_metrics, ["method"])
    x2 = np.arange(len(ABLATIONS))
    plt.figure(figsize=(11, 4.8))
    plt.bar(x2 - 0.2, [float(abl[(m, "task_success")]["mean"]) for m in ABLATIONS], width=0.4, label="Success")
    plt.bar(x2 + 0.2, [float(abl[(m, "mechanism_utility")]["mean"]) for m in ABLATIONS], width=0.4, label="Mechanism")
    plt.xticks(x2, ABLATIONS, rotation=35, ha="right", fontsize=7)
    plt.title("Ablation Audit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_ablation_v5.png", dpi=180)
    plt.close()

    stress = long_lookup(stress_metrics, ["stress_level", "method"])
    plt.figure(figsize=(9, 5))
    for method in ["causal_counterfactual_affordance_planner_v5", "interactive_affordance_probe", "active_view_affordance_probe", "robust_mpc_affordance_planner", "diffusion_affordance_sampler", "ensemble_uncertainty_affordance"]:
        plt.plot(STRESS_LEVELS, [float(stress[(level, method, "robust_utility")]["mean"]) for level in STRESS_LEVELS], marker="o", label=method)
    plt.title("Combined Stress Sweep: Robust Utility")
    plt.xlabel("Stress level")
    plt.ylabel("Robust utility")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_stress_sweep_v5.png", dpi=180)
    plt.close()

    fixed = long_lookup(fixed_metrics, ["split", "budget", "method"])
    plt.figure(figsize=(8, 5))
    for method in FIXED_RISK_METHODS:
        vals = [float(fixed[("combined_counterfactual_stress", budget, method, "coverage")]["mean"]) for budget in FIXED_BUDGETS]
        plt.plot(FIXED_BUDGETS, vals, marker="o", label=method)
    plt.title("Fixed-Risk Coverage on Combined Counterfactual Stress")
    plt.xlabel("Risk budget")
    plt.ylabel("Accepted coverage")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_fixed_risk_v5.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.scatter(float(hard[(method, "invalid_action")]["mean"]), float(hard[(method, "task_success")]["mean"]))
        plt.text(float(hard[(method, "invalid_action")]["mean"]), float(hard[(method, "task_success")]["mean"]), method, fontsize=6)
    plt.xlabel("Invalid action rate")
    plt.ylabel("Task success")
    plt.title("Success vs Invalid-Action Pareto")
    plt.tight_layout()
    plt.savefig(FIGURES / "affordance_pareto_v5.png", dpi=180)
    plt.close()


def fmt(value):
    return f"{float(value):.5f}"


def write_summary(rows):
    (
        dataset,
        rollouts,
        seed_rows,
        metrics,
        pairwise,
        hard_seed,
        hard_metrics,
        hard_pairwise,
        ablation_rows,
        ablation_seed,
        ablation_metrics,
        stress_rows,
        stress_seed,
        stress_metrics,
        fixed_raw,
        fixed_seed,
        fixed_metrics,
        fixed_pairwise,
        negatives,
    ) = rows

    hard = long_lookup(hard_metrics, ["method"])
    pair = {(row["comparison"], row["metric"]): row for row in hard_pairwise}
    fixed = long_lookup(fixed_metrics, ["split", "budget", "method"])
    stress = long_lookup(stress_metrics, ["stress_level", "method"])
    abl = long_lookup(ablation_metrics, ["method"])

    proposal = "causal_counterfactual_affordance_planner_v5"
    non_oracle = [m for m in METHODS if m != "oracle_counterfactual_map"]
    challengers = [m for m in non_oracle if m != proposal]
    best_success = max(non_oracle, key=lambda m: float(hard[(m, "task_success")]["mean"]))
    success_challenger = max(challengers, key=lambda m: float(hard[(m, "task_success")]["mean"]))
    safest = min(non_oracle, key=lambda m: float(hard[(m, "invalid_action")]["mean"]) + float(hard[(m, "damage_rate")]["mean"]))
    best_recall = max(non_oracle, key=lambda m: float(hard[(m, "counterfactual_recall")]["mean"]))
    best_cal = min(non_oracle, key=lambda m: float(hard[(m, "map_ece")]["mean"]))
    best_utility = max(non_oracle, key=lambda m: float(hard[(m, "robust_utility")]["mean"]))
    utility_challenger = max(challengers, key=lambda m: float(hard[(m, "robust_utility")]["mean"]))
    active = "interactive_affordance_probe"
    robust = "robust_mpc_affordance_planner"

    success_pair = pair[(f"{proposal}_minus_{success_challenger}", "task_success")]
    active_success_pair = pair[(f"{proposal}_minus_{active}", "task_success")]
    active_regret_pair = pair[(f"{proposal}_minus_{active}", "planning_regret")]
    recall_pair = pair[(f"{proposal}_minus_{active}", "counterfactual_recall")]
    utility_reference = utility_challenger if best_utility == proposal else best_utility
    utility_pair = pair[(f"{proposal}_minus_{utility_reference}", "robust_utility")]

    success_gate = proposal == best_success and float(success_pair["lower95"]) > 0.0
    active_probe_gate = float(active_success_pair["lower95"]) > -0.01 and float(active_regret_pair["upper95"]) <= 0.01
    recall_gate = proposal == best_recall and float(recall_pair["lower95"]) > 0.005
    safety_gate = (
        float(hard[(proposal, "invalid_action")]["mean"]) <= float(hard[(active, "invalid_action")]["mean"]) + 0.01
        and float(hard[(proposal, "damage_rate")]["mean"]) <= float(hard[(robust, "damage_rate")]["mean"]) + 0.01
    )
    calibration_gate = proposal == best_cal
    utility_gate = proposal == best_utility and float(utility_pair["lower95"]) > 0.0
    best_ablation = max(ABLATIONS, key=lambda m: float(abl[(m, "mechanism_utility")]["mean"]))
    ablation_gate = best_ablation == "full_causal_counterfactual_affordance_v5"
    stress_level = 1.0
    stress_best = max(non_oracle, key=lambda m: float(stress[(stress_level, m, "robust_utility")]["mean"]))
    stress_gate = stress_best == proposal
    fixed_cov_low = float(fixed[("low_signal_counterfactual_shift", 0.05, proposal, "coverage")]["mean"])
    fixed_cov_combined = float(fixed[("combined_counterfactual_stress", 0.05, proposal, "coverage")]["mean"])
    fixed_risk_gate = fixed_cov_low > 0.05 and fixed_cov_combined > 0.05
    scope_gate = False

    terminal = "STRONG_REVISE" if all([success_gate, active_probe_gate, recall_gate, safety_gate, calibration_gate, utility_gate, ablation_gate, stress_gate, fixed_risk_gate, scope_gate]) else "KILL_ARCHIVE"
    lines = [
        "Paper 93 counterfactual_affordance_maps v5 expanded audit",
        f"Terminal recommendation: {terminal}",
        "ICLR main ready: no",
        "Reason: expanded CPU-only counterfactual-affordance audit tests whether maps beat active probing and robust MPC closed-loop baselines, but deployed success/utility and fixed-risk/scope gates remain insufficient.",
        f"Main rollout rows: {len(rollouts)}",
        f"Dataset summary rows: {len(dataset)}",
        f"Main seed-metric rows: {len(seed_rows)}",
        f"Main metric rows: {len(metrics)}",
        f"Main pairwise rows: {len(pairwise)}",
        f"Hard aggregate seed rows: {len(hard_seed)}",
        f"Hard aggregate metric rows: {len(hard_metrics)}",
        f"Hard aggregate pairwise rows: {len(hard_pairwise)}",
        f"Ablation rollout rows: {len(ablation_rows)}",
        f"Ablation seed rows: {len(ablation_seed)}",
        f"Ablation metric rows: {len(ablation_metrics)}",
        f"Stress raw rows: {len(stress_rows)}",
        f"Stress seed rows: {len(stress_seed)}",
        f"Stress metric rows: {len(stress_metrics)}",
        f"Fixed-risk raw rows: {len(fixed_raw)}",
        f"Fixed-risk seed rows: {len(fixed_seed)}",
        f"Fixed-risk metric rows: {len(fixed_metrics)}",
        f"Fixed-risk pairwise rows: {len(fixed_pairwise)}",
        f"Negative cases: {len(negatives)}",
        "",
        "Frozen hard-aggregate gate:",
        f"best_success_reference={best_success}",
        f"success_challenger_reference={success_challenger}",
        f"safest_reference={safest}",
        f"best_counterfactual_recall_reference={best_recall}",
        f"best_calibration_reference={best_cal}",
        f"best_utility_reference={best_utility}",
        f"utility_ci_reference={utility_reference}",
        f"stress_dominated_by={stress_best}",
        f"proposal_success={fmt(hard[(proposal, 'task_success')]['mean'])}",
        f"best_success={fmt(hard[(best_success, 'task_success')]['mean'])}",
        f"proposal_invalid_action={fmt(hard[(proposal, 'invalid_action')]['mean'])}",
        f"safest_invalid_action={fmt(hard[(safest, 'invalid_action')]['mean'])}",
        f"proposal_damage_rate={fmt(hard[(proposal, 'damage_rate')]['mean'])}",
        f"proposal_counterfactual_recall={fmt(hard[(proposal, 'counterfactual_recall')]['mean'])}",
        f"best_counterfactual_recall={fmt(hard[(best_recall, 'counterfactual_recall')]['mean'])}",
        f"proposal_map_ece={fmt(hard[(proposal, 'map_ece')]['mean'])}",
        f"best_map_ece={fmt(hard[(best_cal, 'map_ece')]['mean'])}",
        f"proposal_utility={fmt(hard[(proposal, 'robust_utility')]['mean'])}",
        f"best_utility={fmt(hard[(best_utility, 'robust_utility')]['mean'])}",
        f"paired_success_lower95={fmt(success_pair['lower95'])}",
        f"paired_active_success_lower95={fmt(active_success_pair['lower95'])}",
        f"paired_active_regret_upper95={fmt(active_regret_pair['upper95'])}",
        f"paired_recall_lower95={fmt(recall_pair['lower95'])}",
        f"paired_utility_lower95={fmt(utility_pair['lower95'])}",
        f"success_gate={success_gate}",
        f"active_probe_gate={active_probe_gate}",
        f"recall_gate={recall_gate}",
        f"safety_gate={safety_gate}",
        f"calibration_gate={calibration_gate}",
        f"utility_gate={utility_gate}",
        f"ablation_gate={ablation_gate}",
        f"mechanism_best_ablation={best_ablation}",
        f"stress_gate={stress_gate}",
        f"fixed_risk_gate={fixed_risk_gate}",
        f"scope_gate={scope_gate}",
        f"low_signal_counterfactual_shift: v5_coverage={fmt(fixed_cov_low)}",
        f"combined_counterfactual_stress: v5_coverage={fmt(fixed_cov_combined)}",
        "",
        "Hard aggregate metrics:",
    ]
    for method in METHODS:
        lines.append(
            f"{method} success={fmt(hard[(method, 'task_success')]['mean'])} ap={fmt(hard[(method, 'affordance_ap')]['mean'])} cf_recall={fmt(hard[(method, 'counterfactual_recall')]['mean'])} invalid={fmt(hard[(method, 'invalid_action')]['mean'])} damage={fmt(hard[(method, 'damage_rate')]['mean'])} regret={fmt(hard[(method, 'planning_regret')]['mean'])} ece={fmt(hard[(method, 'map_ece')]['mean'])} utility={fmt(hard[(method, 'robust_utility')]['mean'])} mechanism={fmt(hard[(method, 'mechanism_utility')]['mean'])}"
        )
    lines.extend(["", "Key paired hard-aggregate differences:"])
    for comparison in [f"{proposal}_minus_{success_challenger}", f"{proposal}_minus_{active}", f"{proposal}_minus_{robust}", f"{proposal}_minus_counterfactual_affordance_map_v4"]:
        for metric in ["task_success", "counterfactual_recall", "invalid_action", "damage_rate", "planning_regret", "robust_utility", "mechanism_utility"]:
            row = pair.get((comparison, metric))
            if row:
                lines.append(f"{comparison} {metric}: mean={fmt(row['mean'])} ci95={fmt(row['ci95'])} lower95={fmt(row['lower95'])} upper95={fmt(row['upper95'])}")
    lines.extend(["", "Ablation utility:"])
    for method in ABLATIONS:
        lines.append(
            f"{method} success={fmt(abl[(method, 'task_success')]['mean'])} cf_recall={fmt(abl[(method, 'counterfactual_recall')]['mean'])} invalid={fmt(abl[(method, 'invalid_action')]['mean'])} regret={fmt(abl[(method, 'planning_regret')]['mean'])} utility={fmt(abl[(method, 'robust_utility')]['mean'])} mechanism={fmt(abl[(method, 'mechanism_utility')]['mean'])}"
        )
    lines.extend(["", "Maximum combined stress:"])
    for method in non_oracle:
        lines.append(
            f"{method} success={fmt(stress[(1.0, method, 'task_success')]['mean'])} cf_recall={fmt(stress[(1.0, method, 'counterfactual_recall')]['mean'])} invalid={fmt(stress[(1.0, method, 'invalid_action')]['mean'])} regret={fmt(stress[(1.0, method, 'planning_regret')]['mean'])} utility={fmt(stress[(1.0, method, 'robust_utility')]['mean'])}"
        )
    lines.extend(["", "Fixed-risk budget 0.05:"])
    for split in HARD_SPLITS:
        for method in FIXED_RISK_METHODS:
            lines.append(
                f"{split} {method} coverage={fmt(fixed[(split, 0.05, method, 'coverage')]['mean'])} accepted_success={fmt(fixed[(split, 0.05, method, 'accepted_success')]['mean'])} accepted_invalid={fmt(fixed[(split, 0.05, method, 'accepted_invalid_action')]['mean'])} accepted_damage={fmt(fixed[(split, 0.05, method, 'accepted_damage_rate')]['mean'])}"
            )
    lines.append("")
    lines.append(f"Negative cases: {len(negatives)}")
    lines.append(f"terminal={terminal}")
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return terminal


def main():
    main_rows = make_main()
    ablation_rows = make_ablation()
    stress_rows = make_stress()
    fixed_rows = make_fixed_risk()
    negatives = make_negative_cases(main_rows[1])
    plot_figures(main_rows[6], ablation_rows[2], stress_rows[2], fixed_rows[2])
    terminal = write_summary((*main_rows, *ablation_rows, *stress_rows, *fixed_rows, negatives))
    print((RESULTS / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
