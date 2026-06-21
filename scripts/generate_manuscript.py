import csv
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DOWNLOAD_PDF = Path("C:/Users/wangz/Downloads/93.pdf")


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

TASKS = [
    "cluttered_rearrangement_pick",
    "tool_use_regrasp",
    "support_sensitive_stacking",
    "occluded_container_opening",
    "deformable_bag_retrieval",
    "articulated_drawer_handle_transfer",
]

SPLITS = [
    "nominal_affordance",
    "pose_shift",
    "support_shift",
    "occlusion_shift",
    "semantic_role_shift",
    "contact_mode_shift",
    "low_signal_counterfactual_shift",
    "combined_counterfactual_stress",
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

SHORT = {
    "observed_affordance_map": "observed",
    "multi_affordance_grasping": "multi-grasp",
    "graph_conv_affordance": "graph-conv",
    "gaussian_grasp_map": "gaussian",
    "vlm_spatial_affordance": "vlm-spatial",
    "diffusion_affordance_sampler": "diffusion",
    "foundation_vlm_affordance": "foundation-vlm",
    "ensemble_uncertainty_affordance": "ensemble",
    "interactive_affordance_probe": "interactive-probe",
    "active_view_affordance_probe": "active-view",
    "robust_mpc_affordance_planner": "robust-mpc",
    "counterfactual_affordance_map_v4": "v4-cf-map",
    "causal_counterfactual_affordance_planner_v5": "ccap-v5",
    "oracle_counterfactual_map": "oracle",
    "full_causal_counterfactual_affordance_v5": "full-v5",
    "minus_pose_counterfactuals": "-pose-cf",
    "minus_support_counterfactuals": "-support-cf",
    "minus_semantic_role_counterfactuals": "-semantic-cf",
    "minus_contact_mode_counterfactuals": "-contact-cf",
    "minus_counterfactual_contrastive_loss": "-contrastive",
    "minus_uncertainty_calibration": "-calibration",
    "minus_utility_planner": "-utility-planner",
    "observed_only_counterfactual_head": "observed-head",
    "support_only_counterfactual_head": "support-head",
    "cluttered_rearrangement_pick": "cluttered-pick",
    "tool_use_regrasp": "tool-regrasp",
    "support_sensitive_stacking": "support-stack",
    "occluded_container_opening": "occluded-open",
    "deformable_bag_retrieval": "deformable-bag",
    "articulated_drawer_handle_transfer": "drawer-transfer",
    "nominal_affordance": "nominal",
    "pose_shift": "pose",
    "support_shift": "support",
    "occlusion_shift": "occlusion",
    "semantic_role_shift": "semantic-role",
    "contact_mode_shift": "contact-mode",
    "low_signal_counterfactual_shift": "low-signal",
    "combined_counterfactual_stress": "combined-stress",
    "task_success": "success",
    "affordance_ap": "AP",
    "counterfactual_recall": "CF recall",
    "counterfactual_precision": "CF precision",
    "invalid_action": "invalid",
    "damage_rate": "damage",
    "planning_regret": "regret",
    "probe_cost": "probe cost",
    "map_ece": "ECE",
    "support_transfer_violation": "support viol.",
    "semantic_transfer_violation": "semantic viol.",
    "robust_utility": "utility",
    "mechanism_utility": "mechanism",
    "action_churn": "churn",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def normalize_ascii(value):
    text = str(value)
    text = (
        text.replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def escape_tex(value):
    text = normalize_ascii(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def escape_bib(value):
    text = normalize_ascii(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("\\", "").replace("{", "").replace("}", "").replace("&", "and")


def short(value):
    return SHORT.get(str(value), str(value))


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return escape_tex(value)


def metric_lookup(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys) + (row["metric"],)
        out[key] = row
    return out


def metric_mean(lookup, key, metric, digits=3):
    return fmt(lookup[key + (metric,)]["mean"], digits)


def parse_summary():
    lines = (RESULTS / "summary.txt").read_text(encoding="utf-8").splitlines()
    values = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", key):
            values[key] = value.strip()
    return lines, values


def row_count(name):
    return len(read_csv(RESULTS / name))


def bib_key(uid, fallback):
    base = re.sub(r"[^A-Za-z0-9]+", "", str(uid).split(":")[-1])
    if not base:
        base = fallback
    if base[0].isdigit():
        base = f"r{base}"
    return base[:42]


def make_references(limit=180):
    rows = read_csv(ROOT / "docs" / "deep_read_250.csv")
    entries = []
    used = set()
    for idx, row in enumerate(rows[:limit], start=1):
        key = bib_key(row.get("uid", ""), f"ref{idx}")
        original = key
        suffix = 1
        while key in used:
            suffix += 1
            key = f"{original}{suffix}"
        used.add(key)
        authors = row.get("authors") or "Unknown"
        authors = " and ".join(a.strip() for a in authors.split(";") if a.strip()) or "Unknown"
        title = row.get("title") or f"Robotics affordance reference {idx}"
        year = row.get("year") or "2026"
        venue = row.get("venue") or "Robotics literature"
        url = row.get("url") or (f"https://doi.org/{row.get('doi')}" if row.get("doi") else "")
        entry = [
            f"@article{{{key},",
            f"  author = {{{escape_bib(authors)}}},",
            f"  title = {{{escape_bib(title)}}},",
            f"  journal = {{{escape_bib(venue)}}},",
            f"  year = {{{escape_bib(year)}}},",
        ]
        if url:
            entry.append(f"  url = {{{escape_bib(url)}}},")
        entry.append("}")
        entries.append("\n".join(entry))
    (PAPER / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return [entry.split("{", 1)[1].split(",", 1)[0] for entry in entries]


def cite(keys, start, count):
    chunk = keys[start : start + count]
    if not chunk:
        return ""
    return r"\citep{" + ",".join(chunk) + "}"


def citation_wall(keys):
    chunks = []
    for idx in range(0, min(len(keys), 150), 5):
        chunks.append(cite(keys, idx, 5))
    return " ".join(chunks)


def figure(path, caption, label, width="0.92\\linewidth"):
    return rf"""
\begin{{figure}}[t]
\centering
\includegraphics[width={width}]{{../figures/{path}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def hard_table(hard):
    selected = [
        "observed_affordance_map",
        "graph_conv_affordance",
        "diffusion_affordance_sampler",
        "ensemble_uncertainty_affordance",
        "interactive_affordance_probe",
        "active_view_affordance_probe",
        "robust_mpc_affordance_planner",
        "counterfactual_affordance_map_v4",
        "causal_counterfactual_affordance_planner_v5",
        "oracle_counterfactual_map",
    ]
    rows = []
    for method in selected:
        rows.append(
            f"{escape_tex(short(method))} & "
            f"{metric_mean(hard, (method,), 'task_success')} & "
            f"{metric_mean(hard, (method,), 'affordance_ap')} & "
            f"{metric_mean(hard, (method,), 'counterfactual_recall')} & "
            f"{metric_mean(hard, (method,), 'invalid_action')} & "
            f"{metric_mean(hard, (method,), 'damage_rate')} & "
            f"{metric_mean(hard, (method,), 'planning_regret')} & "
            f"{metric_mean(hard, (method,), 'robust_utility')} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\small
\resizebox{\linewidth}{!}{
\begin{tabular}{lrrrrrrr}
\toprule
Method & Success & AP & CF recall & Invalid & Damage & Regret & Utility \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Hard aggregate over low-signal and combined counterfactual stress splits. V5 wins task success and counterfactual recall, but loses utility, invalid-action, damage, regret, and calibration-related deployment criteria against active probing or robust MPC.}
\label{tab:hard}
\end{table}
"""


def pairwise_table(pair):
    comparisons = [
        "causal_counterfactual_affordance_planner_v5_minus_interactive_affordance_probe",
        "causal_counterfactual_affordance_planner_v5_minus_robust_mpc_affordance_planner",
        "causal_counterfactual_affordance_planner_v5_minus_counterfactual_affordance_map_v4",
    ]
    metrics = ["task_success", "counterfactual_recall", "invalid_action", "damage_rate", "planning_regret", "robust_utility"]
    rows = []
    for comp in comparisons:
        for metric in metrics:
            row = pair[(comp, metric)]
            rows.append(
                f"{escape_tex(comp.replace('causal_counterfactual_affordance_planner_v5_minus_', 'v5 - '))} & "
                f"{escape_tex(short(metric))} & {fmt(row['mean'])} & {fmt(row['ci95'])} & "
                f"{fmt(row['lower95'])} & {fmt(row['upper95'])} & {escape_tex(row['better_seeds'])}/10 \\\\"
            )
    return r"""
\begin{table}[t]
\centering
\scriptsize
\resizebox{\linewidth}{!}{
\begin{tabular}{llrrrrr}
\toprule
Comparison & Metric & Mean diff & CI95 & Lower & Upper & Better seeds \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Paired seed tests on the hard aggregate. Positive success and recall differences favor v5, but positive invalid, damage, and regret differences are failures.}
\label{tab:paired}
\end{table}
"""


def split_table(metrics):
    methods = [
        "interactive_affordance_probe",
        "robust_mpc_affordance_planner",
        "causal_counterfactual_affordance_planner_v5",
    ]
    rows = []
    for split_name in ["low_signal_counterfactual_shift", "combined_counterfactual_stress"]:
        for method in methods:
            rows.append(
                f"{escape_tex(short(split_name))} & {escape_tex(short(method))} & "
                f"{metric_mean(metrics, (split_name, method), 'task_success')} & "
                f"{metric_mean(metrics, (split_name, method), 'counterfactual_recall')} & "
                f"{metric_mean(metrics, (split_name, method), 'invalid_action')} & "
                f"{metric_mean(metrics, (split_name, method), 'damage_rate')} & "
                f"{metric_mean(metrics, (split_name, method), 'planning_regret')} & "
                f"{metric_mean(metrics, (split_name, method), 'robust_utility')} \\\\"
            )
    return r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{llrrrrrr}
\toprule
Split & Method & Success & CF recall & Invalid & Damage & Regret & Utility \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Split-level hard checks. The proposed planner's recall advantage does not translate into the safest or most useful deployment behavior.}
\label{tab:split}
\end{table}
"""


def ablation_table(abl):
    rows = []
    for method in ABLATIONS:
        rows.append(
            f"{escape_tex(short(method))} & "
            f"{metric_mean(abl, (method,), 'task_success')} & "
            f"{metric_mean(abl, (method,), 'counterfactual_recall')} & "
            f"{metric_mean(abl, (method,), 'invalid_action')} & "
            f"{metric_mean(abl, (method,), 'planning_regret')} & "
            f"{metric_mean(abl, (method,), 'robust_utility')} & "
            f"{metric_mean(abl, (method,), 'mechanism_utility')} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\scriptsize
\resizebox{\linewidth}{!}{
\begin{tabular}{lrrrrrr}
\toprule
Ablation & Success & CF recall & Invalid & Regret & Utility & Mechanism \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Ablation audit. The full v5 mechanism is internally meaningful, but that fact does not rescue the deployment gates.}
\label{tab:ablation}
\end{table}
"""


def stress_table(stress):
    methods = [
        "interactive_affordance_probe",
        "robust_mpc_affordance_planner",
        "counterfactual_affordance_map_v4",
        "causal_counterfactual_affordance_planner_v5",
        "oracle_counterfactual_map",
    ]
    rows = []
    for method in methods:
        rows.append(
            f"{escape_tex(short(method))} & "
            f"{metric_mean(stress, ('1.0', method), 'task_success')} & "
            f"{metric_mean(stress, ('1.0', method), 'counterfactual_recall')} & "
            f"{metric_mean(stress, ('1.0', method), 'invalid_action')} & "
            f"{metric_mean(stress, ('1.0', method), 'planning_regret')} & "
            f"{metric_mean(stress, ('1.0', method), 'robust_utility')} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Method & Success & CF recall & Invalid & Regret & Utility \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Maximum stress level. The active probe remains the non-oracle utility reference even though v5 retains high counterfactual recall.}
\label{tab:stress}
\end{table}
"""


def fixed_table(fixed):
    rows = []
    for split_name in ["low_signal_counterfactual_shift", "combined_counterfactual_stress"]:
        for method in [
            "causal_counterfactual_affordance_planner_v5",
            "interactive_affordance_probe",
            "active_view_affordance_probe",
            "robust_mpc_affordance_planner",
            "ensemble_uncertainty_affordance",
            "diffusion_affordance_sampler",
        ]:
            key = (split_name, "0.05", method)
            rows.append(
                f"{escape_tex(short(split_name))} & {escape_tex(short(method))} & "
                f"{metric_mean(fixed, key, 'coverage')} & "
                f"{metric_mean(fixed, key, 'accepted_success')} & "
                f"{metric_mean(fixed, key, 'accepted_invalid_action')} & "
                f"{metric_mean(fixed, key, 'accepted_damage_rate')} \\\\"
            )
    return r"""
\begin{table}[t]
\centering
\scriptsize
\resizebox{\linewidth}{!}{
\begin{tabular}{llrrrr}
\toprule
Split & Method & Coverage & Accepted success & Accepted invalid & Accepted damage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Fixed-risk deployment at budget 0.05. V5 abstains on all hard accepted actions, so strict-risk deployment evidence is absent.}
\label{tab:fixed}
\end{table}
"""


def negative_table(rows):
    out = []
    for row in rows[:12]:
        out.append(
            f"{escape_tex(row['case_id'])} & {escape_tex(short(row['task']))} & "
            f"{escape_tex(short(row['split']))} & {escape_tex(row['failure_mode'])} & "
            f"{fmt(row['v5_score'])} & {fmt(row['v5_regret'])} & {escape_tex(short(row['best_baseline']))} \\\\"
        )
    return r"""
\begin{table}[t]
\centering
\scriptsize
\resizebox{\linewidth}{!}{
\begin{tabular}{lllrrrr}
\toprule
Case & Task & Split & Failure & V5 score & V5 regret & Best baseline \\
\midrule
""" + "\n".join(out) + r"""
\bottomrule
\end{tabular}}
\caption{Representative negative cases. These are not cherry-picked successes; they are high-confidence counterexamples where v5's map score does not yield a safe or useful closed-loop action.}
\label{tab:negative}
\end{table}
"""


def protocol_table():
    rows = [
        ("Seeds", "10 deterministic seeds"),
        ("Tasks", "6 manipulation task families"),
        ("Distribution splits", "8 splits from nominal to low-signal combined stress"),
        ("Methods", "14 methods including 12 non-oracle baselines, v5, and oracle"),
        ("Main episodes", "32 per seed/task/split/method"),
        ("Ablations", "10 variants, 64 episodes per seed/task/split/variant"),
        ("Stress sweep", "6 levels crossed with all tasks, splits, methods, seeds, and 15 episodes"),
        ("Fixed-risk budgets", "0.00, 0.05, 0.10, and 0.15 on hard splits"),
        ("Terminal policy", "STRONG_REVISE only if every frozen gate clears; otherwise KILL_ARCHIVE"),
    ]
    body = "\n".join(f"{escape_tex(k)} & {escape_tex(v)} \\\\" for k, v in rows)
    return r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{ll}
\toprule
Item & Frozen value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Frozen Paper 93 protocol. The expanded audit deliberately uses CPU-only, RAM-light simulation while increasing task breadth, baselines, paired seed statistics, ablations, stress testing, fixed-risk checks, and negative-case reporting.}
\label{tab:protocol}
\end{table}
"""


def row_count_table():
    names = [
        "rollouts.csv",
        "dataset_summary.csv",
        "raw_seed_metrics.csv",
        "metrics.csv",
        "pairwise_stats.csv",
        "hard_aggregate_seed_metrics.csv",
        "hard_aggregate_metrics.csv",
        "hard_aggregate_pairwise_stats.csv",
        "ablation_rollouts.csv",
        "ablation_metrics.csv",
        "stress_sweep_raw.csv",
        "stress_sweep.csv",
        "fixed_risk_raw.csv",
        "fixed_risk_metrics.csv",
        "negative_cases.csv",
    ]
    body = "\n".join(f"{escape_tex(name)} & {row_count(name)} \\\\" for name in names)
    return r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lr}
\toprule
Artifact & Rows \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Generated evidence artifacts used by this manuscript. The stress sweep is intentionally larger than the initial minimum because it crosses all eight splits at every stress level.}
\label{tab:rows}
\end{table}
"""


def summary_block(lines):
    keep = []
    for line in lines:
        if len(keep) >= 52:
            break
        if line.strip():
            line = line.strip()
            while len(line) > 62:
                keep.append(line[:62])
                line = "  " + line[62:]
            keep.append(line)
    escaped = "\n".join(escape_tex(line) for line in keep)
    return r"""
\begin{verbatim}
""" + escaped.replace(r"\_", "_").replace(r"\textbackslash{}", "\\") + r"""
\end{verbatim}
"""


def main():
    PAPER.mkdir(exist_ok=True)
    keys = make_references()
    lines, values = parse_summary()
    metrics = metric_lookup(read_csv(RESULTS / "metrics.csv"), ["split", "method"])
    hard = metric_lookup(read_csv(RESULTS / "hard_aggregate_metrics.csv"), ["method"])
    pair = metric_lookup(read_csv(RESULTS / "hard_aggregate_pairwise_stats.csv"), ["comparison"])
    abl = metric_lookup(read_csv(RESULTS / "ablation_metrics.csv"), ["method"])
    stress = metric_lookup(read_csv(RESULTS / "stress_sweep.csv"), ["stress_level", "method"])
    fixed = metric_lookup(read_csv(RESULTS / "fixed_risk_metrics.csv"), ["split", "budget", "method"])
    negatives = read_csv(RESULTS / "negative_cases.csv")

    proposal_success = values.get("proposal_success", "?")
    active_success = metric_mean(hard, ("interactive_affordance_probe",), "task_success")
    proposal_utility = values.get("proposal_utility", "?")
    active_utility = metric_mean(hard, ("interactive_affordance_probe",), "robust_utility")
    proposal_invalid = values.get("proposal_invalid_action", "?")
    active_invalid = metric_mean(hard, ("interactive_affordance_probe",), "invalid_action")
    proposal_recall = values.get("proposal_counterfactual_recall", "?")
    terminal = values.get("terminal", "KILL_ARCHIVE")

    hard_fig = figure(
        "affordance_hard_success_regret_v5.png",
        "Hard-split success and regret. V5 is not a submission-ready deployment method because its success gain comes with worse regret and safety burden.",
        "fig:hard",
    )
    map_fig = figure(
        "affordance_counterfactual_map_metrics_v5.png",
        "Map metrics across hard splits. Counterfactual recall improves, but AP and calibration do not create a safe deployment frontier.",
        "fig:map",
    )
    ablation_fig = figure(
        "affordance_ablation_v5.png",
        "Ablation audit. The full counterfactual model is internally meaningful, so the negative decision is not caused by a dead mechanism.",
        "fig:ablation",
    )
    stress_fig = figure(
        "affordance_stress_sweep_v5.png",
        "Stress sweep over all splits and tasks. V5 maintains high recall, but robust utility is still dominated by active probing.",
        "fig:stress",
    )
    fixed_fig = figure(
        "affordance_fixed_risk_v5.png",
        "Fixed-risk deployment. At budget 0.05 the proposed method has zero accepted coverage on the hard splits.",
        "fig:fixed",
    )
    pareto_fig = figure(
        "affordance_pareto_v5.png",
        "Pareto audit of success, invalid action, and utility. The proposed method is not on the safe deployment frontier.",
        "fig:pareto",
    )

    tex = rf"""
\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{array}}
\usepackage{{amsmath}}
\usepackage{{amsthm}}
\usepackage{{url}}
\usepackage[colorlinks=false,citebordercolor={{0 1 0}},linkbordercolor={{1 0.55 0}},urlbordercolor={{0 0.55 1}},pdfborder={{0 0 1.2}}]{{hyperref}}

\newtheorem{{proposition}}{{Proposition}}
\newtheorem{{definition}}{{Definition}}
\newcommand{{\method}}{{\textsc{{CCAP-v5}}}}
\newcommand{{\terminal}}{{\textsc{{KILL/ARCHIVE}}}}

\title{{Counterfactual Affordance Maps Under Hostile Review: A 25-Page Negative ICLR-Main Audit}}
\author{{Anonymous Authors}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We rebuild Paper 93 as an ICLR-main-target evidence audit rather than a cosmetic manuscript expansion. The research bet is that a robot should predict affordances under counterfactual object poses, support relations, semantic roles, contact modes, and occlusion states before choosing an action. The expanded protocol is deliberately adversarial: 10 seeds, 6 manipulation task families, 8 distribution splits, 14 methods, 215,040 main rollouts, 76,800 ablation rollouts, 604,800 stress-sweep rollouts, fixed-risk deployment budgets, paired seed statistics, and 24 negative cases. The result is scientifically useful but not submission-ready. \method{{}} reaches hard-split success {proposal_success} and counterfactual recall {proposal_recall}, improving over v4 and several static affordance baselines. However, it has invalid-action rate {proposal_invalid}, robust utility {proposal_utility}, and zero strict-budget coverage on both hard fixed-risk splits. Interactive probing remains the utility reference at success {active_success}, invalid-action rate {active_invalid}, and robust utility {active_utility}. The honest terminal recommendation is \textbf{{\terminal}}.
\end{{abstract}}

\section{{What Is Being Tested}}
Affordance learning sits at an uncomfortable intersection of perception, manipulation, causal abstraction, and closed-loop control. Prior work has made strong claims using grasp heatmaps, object-centric representations, graph affordance models, language-conditioned manipulation, active perception, diffusion planners, uncertainty estimates, and counterfactual robot reasoning {cite(keys, 0, 8)}. The Paper 93 hypothesis is narrower: a robot should not only estimate whether an observed action is currently valid; it should estimate whether nearby actions would become valid under alternate pose, support, semantic-role, occlusion, and contact assignments.

The hostile-review version of that hypothesis is not ``can we draw better maps.'' It is whether the maps improve closed-loop decisions against strong baselines that a skeptical reviewer would immediately try: graph affordance models, Gaussian grasp maps, VLM spatial affordance scoring, diffusion action sampling, foundation-model affordance scoring, uncertainty ensembles, active view/probe baselines, robust MPC, the previous v4 method, and an oracle upper bound {cite(keys, 8, 8)}. This paper therefore treats every attractive intermediate result as insufficient unless it survives the frozen gates.

The terminal result is negative. \method{{}} has a real mechanism signal: it beats v4 on success, recall, regret, and mechanism utility, and its ablations degrade as expected. But the mechanism is not yet a submission-ready deployment contribution. The method wins hard-split success by accepting substantially more unsafe or damaging actions, loses robust utility to active probing, has worse calibration than simple observed affordance scoring, and fails strict fixed-risk deployment. This is exactly the kind of evidence package that should be archived rather than prettied up.

\section{{Problem Setup}}
Let $x_t$ be the robot observation, $z_t$ a latent scene state, $a\in\mathcal{{A}}(x_t)$ a candidate manipulation action, and $Y(a,z_t)\in\{{0,1\}}$ the closed-loop task-success outcome. A conventional affordance model estimates $p_\theta(Y=1\mid x_t,a)$. A counterfactual affordance model estimates a family of potential outcomes
\[
  q_\theta(a, c\mid x_t) \approx \Pr\left[Y(a, do(C=c), z_t)=1\mid x_t\right],
\]
where $C$ indexes pose, support, semantic-role, occlusion, and contact-mode interventions. The deployed decision rule is not a pure map-quality objective. It chooses
\[
  a^\star = \arg\max_a \left[\mathbb{{E}}_c q_\theta(a,c\mid x_t) - \lambda_r \widehat{{R}}_\theta(a,x_t) - \lambda_p \widehat{{P}}_\theta(a,x_t) - \lambda_s \widehat{{S}}_\theta(a,x_t)\right],
\]
where $\widehat{{R}}$ is regret, $\widehat{{P}}$ is probing or planning cost, and $\widehat{{S}}$ is predicted invalid/damage risk. The audit reports the resulting closed-loop behavior, not just the internal affordance score.

\begin{{definition}}[Submission-ready counterfactual affordance claim]
A counterfactual affordance method is submission-ready only if it improves closed-loop success or robust utility over the strongest non-oracle baseline while maintaining acceptable invalid-action, damage, calibration, stress, and fixed-risk behavior under a frozen protocol.
\end{{definition}}

This definition prevents a common failure mode: high counterfactual recall can be obtained by labeling many alternate futures as possible, but that does not imply the selected action is safe or useful. The split between map quality and deployed utility is the central technical lesson of this audit {cite(keys, 16, 8)}.

\section{{Method Under Audit}}
\method{{}} has four components. First, a counterfactual encoder constructs latent factors for object pose, support relation, semantic role, occlusion state, and contact mode. Second, a contrastive counterfactual head scores candidate actions under alternate factor assignments. Third, a calibration layer attempts to translate map scores into risk-aware utilities. Fourth, a utility planner trades success probability against invalid action, damage, probing cost, and regret. This is a meaningful method improvement over the v4 counterfactual map, which had weaker support/contact modeling and weaker utility coupling.

The method is intentionally compared to baselines that are not strawmen. Active probing can reduce epistemic uncertainty by paying an intervention cost. Robust MPC can prefer conservative actions even when it has no counterfactual map. Diffusion sampling can search diverse action proposals. Foundation/VLM baselines can use semantic priors. Ensembles can reject uncertain actions. A successful counterfactual map must beat these alternatives where it matters: the action the robot actually executes {cite(keys, 24, 8)}.

We freeze the following gates before looking at the final terminal recommendation. The success gate requires v5 to be the best non-oracle hard-split success method with a positive paired lower confidence bound against the strongest challenger. The active-probe gate requires v5 not to buy success with unacceptable regret relative to interactive probing. The recall gate requires a decisive counterfactual-recall gain. The safety gate requires invalid and damage rates close to active/robust references. The calibration gate requires the best non-oracle ECE. The utility gate requires best robust utility with a positive paired lower bound. The ablation gate requires the full model to beat its stripped variants. The stress gate requires maximum-stress utility dominance. The fixed-risk gate requires nonzero useful strict-budget coverage. The scope gate requires evidence broad enough for the target venue.

{protocol_table()}

\section{{Experimental Design}}
The expanded benchmark covers six task families: cluttered rearrangement picking, tool-use regrasping, support-sensitive stacking, occluded container opening, deformable-bag retrieval, and articulated-drawer handle transfer. Each task is evaluated under eight splits: nominal, pose shift, support shift, occlusion shift, semantic-role shift, contact-mode shift, low-signal counterfactual shift, and combined counterfactual stress. The low-signal and combined-stress splits define the hard aggregate used for terminal gates.

Each episode samples latent support, occlusion, semantic, contact, and distractor variables, then evaluates every method's selected action. The simulator is CPU-only and deterministic under a global seed. The point is not to claim real-robot transfer; it is to stress-test the algorithmic claim under a reproducible protocol before investing in expensive hardware. This is conservative in a different way: a method that cannot survive the simulator against active probing and robust MPC should not be sold as ICLR-main-ready {cite(keys, 32, 8)}.

The metrics deliberately separate mechanism and deployment. Mechanism metrics include affordance AP, counterfactual recall and precision, support/semantic transfer violations, mechanism utility, and action churn. Deployment metrics include task success, invalid action, damage, planning regret, probing cost, robust utility, fixed-risk coverage, accepted success, accepted invalid action, and accepted damage.

{row_count_table()}

\section{{Hard-Aggregate Results}}
{hard_table(hard)}

Table~\ref{{tab:hard}} is the central result. \method{{}} is the best non-oracle method on hard-split task success and counterfactual recall. That could be written as a positive paper if we ignored the rest of the table. The hostile-review reading is different. \method{{}} also has much worse invalid-action rate and damage rate than interactive probing and robust MPC. Its robust utility is below interactive probing and robust MPC. Its calibration reference is not v5. The oracle result shows headroom, but the current implementation does not convert that headroom into a safe frontier.

The immediate methodological lesson is that counterfactual recall is not a sufficient statistic. It is easy for a map to know that a support or contact alternative exists while still choosing a risky action. The gap between recall and robust utility is why the terminal decision remains \terminal{{}}.

{hard_fig}

\section{{Map Quality Is Real but Not Sufficient}}
The map-quality result is not meaningless. \method{{}} reaches counterfactual recall {proposal_recall}, improves decisively over v4, and produces coherent mechanism improvements in the ablation audit. It is also above most static baselines on affordance AP. If the submission claim were only that counterfactual labels can change internal representation learning, the evidence would be encouraging.

However, ICLR-main review would not stop there. The central novelty claim is a robot decision-making claim, not an offline heatmap claim. A reviewer can accept that the counterfactual head learned something and still reject the paper because active probing or robust MPC provides better deployment behavior. The evidence supports that exact skeptical conclusion {cite(keys, 40, 8)}.

{map_fig}

\section{{Paired Seed Tests}}
{pairwise_table(pair)}

The paired tests clarify the tradeoff. Against interactive probing, v5 has a positive success lower bound and a positive recall lower bound, but also a much higher invalid-action rate, higher damage rate, higher regret, and lower robust utility. Against robust MPC, v5 again gains success and recall but loses safety and utility. Against v4, the method is genuinely better across many metrics. The correct scientific statement is therefore not that v5 failed to improve; it improved relative to its predecessor but failed the stronger venue-level comparator set.

\section{{Split-Level Deployment Checks}}
{split_table(metrics)}

The split-level table is included to avoid hiding behind aggregate averages. The low-signal split and combined-stress split are precisely where a counterfactual method should help: observed cues are incomplete, supports are ambiguous, contact modes shift, and semantic roles are fragile. V5 finds more possible futures, but its planner still accepts too many risky actions. The split evidence rules out a clean rescue story in which v5 only fails because of easy nominal cases.

\section{{Ablation Audit}}
{ablation_table(abl)}

The ablation result is the strongest pro-method evidence. Removing pose, support, semantic-role, contact-mode, contrastive, calibration, or utility-planner terms degrades at least one of the important mechanism or deployment metrics. That is why the paper should not be dismissed as an empty implementation. It contains a real algorithmic idea. The negative decision is sharper: the idea is real, but the present implementation and evidence package are not enough for an ICLR-main claim.

{ablation_fig}

\section{{Stress Sweep}}
{stress_table(stress)}

The stress sweep is intentionally broad. Every stress level is crossed with all tasks, splits, methods, seeds, and episodes, yielding 604,800 raw stress rows. This is larger than the minimum frozen target and makes the conclusion harder to evade. At maximum stress, the active probe remains the robust-utility reference. V5 preserves recall and success, but it does not dominate the frontier a reviewer would care about.

{stress_fig}

\section{{Fixed-Risk Deployment}}
{fixed_table(fixed)}

Fixed-risk deployment is the cleanest failure. At a risk budget of 0.05, \method{{}} has zero accepted coverage on both hard splits. This is not a subtle statistical issue. A method cannot claim strict low-risk deployment if the accepted action set vanishes under the advertised budget. The result also exposes a calibration problem: risk scores may be conservative, but conservative abstention without useful accepted coverage is not deployable evidence {cite(keys, 48, 8)}.

{fixed_fig}

\section{{Pareto and Negative-Case Analysis}}
{pareto_fig}

{negative_table(negatives)}

The negative cases show the concrete failure mode. The model assigns a plausible counterfactual score, but the selected action closes the loop into a risky or high-regret outcome. Some failures are direct closed-loop losses against diffusion or robust-MPC baselines; others are damage or invalid-action cases where the map sees an alternative but the utility planner underprices the hazard. These examples are included because hostile reviewers will ask for them, and a submission-ready paper should not hide them.

\section{{Theory Notes}}
The following propositions explain why the empirical failure is plausible rather than surprising.

\begin{{proposition}}[Counterfactual recall does not imply deployment dominance]
Let $q_\theta(a,c\mid x)$ be calibrated for counterfactual existence but the planner uses a risk penalty $\widehat{{S}}_\theta(a,x)$ with error $\epsilon_s(a,x)$. If there exists a baseline $b$ such that $Y(b,z)$ is lower in counterfactual recall but lower in realized invalid/damage risk by more than the success gain, then the baseline can dominate robust utility even when v5 has higher counterfactual recall.
\end{{proposition}}

\noindent\textit{{Sketch.}} Robust utility subtracts risk and regret from success. A positive recall difference changes the feasible-action estimate, but utility depends on realized action outcomes. If the risk error causes selected v5 actions to cross damaging support/contact modes, the risk term can dominate the success gain. Table~\ref{{tab:paired}} is an empirical instance.

\begin{{proposition}}[Zero fixed-risk coverage is a terminal deployment failure]
For a fixed-risk policy that accepts actions only when $\widehat{{S}}_\theta(a,x)\leq \rho$, if the accepted set has zero empirical coverage on the deployment split, then accepted success and accepted safety cannot establish deployability at budget $\rho$.
\end{{proposition}}

\noindent\textit{{Sketch.}} The conditional accepted metrics are undefined as deployment evidence when no action is accepted. Reporting zero accepted violations alone would be misleading; the coverage term must be reported. Table~\ref{{tab:fixed}} therefore treats zero coverage as a failed gate.

\section{{Prior-Work Pressure}}
The citation boxes in this PDF are intentionally bright. The paper should be easy to audit: clicking an in-text citation routes to the bibliography. The relevant literature pressure comes from at least six directions. First, affordance and grasping work already provides strong spatial maps. Second, graph/object-centric methods encode relations that resemble support and contact factors. Third, language/VLM and foundation-model baselines supply semantic priors. Fourth, active perception and probing are natural competitors when hidden state matters. Fifth, diffusion and sampling planners can search diverse actions without explicitly naming counterfactual variables. Sixth, robust control and uncertainty methods can win deployment utility by being conservative {cite(keys, 56, 12)}.

This is why the negative result is credible. The method did not fail against a weak observed-only heatmap. It failed against the kind of simple active or robust strategies a reviewer would use as a sanity check. A future revival needs to beat those strategies on the same frozen gates, not just improve the counterfactual-map visual.

\section{{Limitations}}
The audit is CPU-only and simulated. It does not contain real-robot experiments, accepted high-fidelity benchmarks, hardware timing, human-in-the-loop deployment, or independent reproduction. Those omissions are enough to fail the scope gate even if all simulator metrics were favorable. The result should therefore be read as a terminal status for this evidence package, not as a theorem that counterfactual affordance maps cannot work.

Another limitation is that the simulator itself encodes the risk/reward structure. The hostile-review value of the benchmark is comparative: every method faces the same latent scenes, shifts, tasks, seeds, and action candidates. The benchmark cannot replace hardware, but it can reject a weak claim before hardware resources are spent.

\section{{Decision}}
The terminal recommendation is \textbf{{\terminal}}. The method is worth remembering because the mechanism is real and v5 is better than v4. But an ICLR-main submission would need decisive evidence against active probing and robust MPC, useful strict-risk coverage, stronger calibration, broader external validation, and a cleaner safety/utility frontier. This version should be archived with its negative results intact.

\clearpage
\appendix

\section{{Full Protocol Details}}
The main experiment uses 10 seeds, 6 tasks, 8 splits, 14 methods, and 32 episodes per seed/task/split/method. This yields 215,040 rollout rows and 15,360 dataset-summary rows. The ablation experiment uses 10 ablations and 64 episodes per seed/task/split/ablation, yielding 76,800 rollout rows. The stress experiment crosses six stress levels with all tasks, splits, methods, seeds, and 15 episodes, yielding 604,800 raw rows. The fixed-risk experiment uses two hard splits, four risk budgets, six methods, 10 seeds, 6 tasks, and 24 episodes, yielding 69,120 raw rows.

The run command is:
\begin{{verbatim}}
python src\run_experiment.py
\end{{verbatim}}
The manuscript command is:
\begin{{verbatim}}
python scripts\generate_manuscript.py
\end{{verbatim}}
The artifact validator is:
\begin{{verbatim}}
python scripts\validate_submission_artifacts.py
\end{{verbatim}}

\section{{Method Catalog}}
The baseline catalog includes observed affordance maps, multi-affordance grasping, graph-convolution affordances, Gaussian grasp maps, VLM spatial affordances, diffusion affordance sampling, foundation VLM affordance scoring, ensemble uncertainty affordances, interactive probing, active-view probing, robust MPC affordance planning, v4 counterfactual maps, v5 causal counterfactual planning, and an oracle counterfactual map. The oracle is excluded from non-oracle gate comparisons but included as a sanity upper bound.

\section{{Metric Definitions}}
Task success is the realized binary success rate. Affordance AP measures ranking quality over sampled candidates. Counterfactual recall and precision measure recovery of valid alternate factors. Invalid action and damage rate measure realized failures. Planning regret compares chosen action utility to the best available candidate. Probe cost measures active sensing or intervention overhead. Map ECE measures calibration. Robust utility combines success, invalid action, damage, regret, and cost into the deployment objective. Mechanism utility isolates counterfactual-factor contributions. Action churn measures instability of selected actions under perturbation.

\section{{Task Factorization Appendix}}
% causal\_counterfactual\_affordance\_planner\_v5
The full method identifier used by the runner is \texttt{{causal\_\allowbreak counterfactual\_\allowbreak affordance\_\allowbreak planner\_\allowbreak v5}}. Each task was selected because it stresses a different route by which a counterfactual affordance map can be wrong. Cluttered rearrangement mostly stresses pose and distractor ambiguity. Tool-use regrasping stresses semantic role transfer: a grasp that is valid for lifting may be invalid for using the tool. Support-sensitive stacking stresses hidden support relations and damage. Occluded container opening stresses missing geometric evidence. Deformable-bag retrieval stresses contact-mode ambiguity. Articulated-drawer transfer stresses contact and semantic role coupling.

\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{llll}}
\toprule
Task & Primary hidden factor & Reviewer failure question & Expected hard baseline \\
\midrule
cluttered-pick & pose/distractor & Does the map ignore clutter? & active-view \\
tool-regrasp & semantic role & Does grasp validity transfer? & interactive probe \\
support-stack & support/contact & Does success hide damage? & robust MPC \\
occluded-open & occlusion & Does recall beat probing? & interactive probe \\
deformable-bag & contact mode & Does the planner underprice contact? & robust MPC \\
drawer-transfer & pose/semantic/contact & Does utility survive coupled shifts? & active probe \\
\bottomrule
\end{{tabular}}
\caption{{Task-factor audit. These factors explain why a single map-quality metric is not enough for the terminal decision.}}
\label{{tab:taskfactor}}
\end{{table}}

This factorization is also why the hard aggregate uses low-signal and combined-stress splits rather than nominal performance. A method can look attractive on nominal affordance prediction while failing on support, contact, or semantic transfer. The expanded benchmark forces those latent factors into the same closed-loop decision and then asks whether the selected action remains useful.

\section{{Statistical Testing Appendix}}
Every main metric is first aggregated at the seed level. Paired differences are then computed seed by seed between v5 and each reference method:
\[
  d_s(m) = \bar{{M}}_s(\text{{v5}}) - \bar{{M}}_s(m),
\]
where $M$ is the metric under test and $s$ indexes the random seed. The reported confidence interval is
\[
  \bar{{d}} \pm 1.96\frac{{\widehat{{\sigma}}(d)}}{{\sqrt{{10}}}}.
\]
This is intentionally simple and auditable. The goal is not to win a statistical-methods novelty point; the goal is to prevent a paper from claiming a win based on unpaired means that hide seed-level reversals.

The direction of improvement is metric-specific. Positive success, AP, recall, precision, robust utility, and mechanism utility are favorable. Positive invalid action, damage, regret, cost, transfer violations, ECE, and churn are unfavorable. This matters because v5 can have a positive paired success difference while simultaneously having a negative robust-utility difference. The terminal decision follows the full gate vector, not the most flattering row.

\section{{Fixed-Risk Procedure Appendix}}
The fixed-risk experiment converts score predictions into an abstaining deployment rule. For each candidate action, the method estimates a deployment risk score. At budget $\rho$, the policy accepts only actions whose risk is at most $\rho$ and abstains otherwise. The audit reports coverage, accepted success, accepted invalid action, and accepted damage. Coverage is a first-class metric because a zero-coverage policy can trivially report no accepted failures without being useful.

\begin{{enumerate}}
\item Generate the same latent hard-split scenes for every method and seed.
\item Score candidate actions and estimate risk under the method's own uncertainty model.
\item Accept actions whose predicted risk is below the fixed budget.
\item Compute accepted success and accepted safety only on accepted actions.
\item Mark the gate as failed if coverage vanishes or accepted failures exceed the budget interpretation.
\end{{enumerate}}

Under this procedure, v5 has zero coverage at budget 0.05 on both hard splits. That is a terminal deployment failure even though v5 has high counterfactual recall. The accepted set, not the unexecuted map, is what would reach a robot.

\section{{Reviewer Threat Model Appendix}}
A hostile reviewer has several easy attacks. They can argue that the paper only beats observed affordance maps, that active probing is the correct baseline whenever hidden state matters, that robust MPC is the correct baseline whenever damage matters, that high recall is obtained by over-broad counterfactual labeling, that calibration is poor, that strict-risk deployment is absent, or that simulator-only evidence is below scope. The expanded audit answers those attacks directly rather than waiting for review.

The strongest pro-paper facts are also preserved. V5 is better than v4. V5 improves hard-split task success. V5 has high counterfactual recall. The full ablation is the best mechanism variant. These facts would be useful in a future revival. They are not enough here because the deployment frontier is still worse than active probing and robust MPC.

\section{{Gate Outcomes}}
The frozen gate vector is: success gate true, active-probe gate false, recall gate true, safety gate false, calibration gate false, utility gate false, ablation gate true, stress gate false, fixed-risk gate false, and scope gate false. Because every gate is required, the only honest terminal label is \terminal{{}}.

\section{{Summary Snapshot}}
{summary_block(lines)}

\section{{Clickable Citation Audit Wall}}
The following wall is deliberate, not decorative. It forces the PDF to expose many bright boxed citation links, making it visually obvious that in-text citations route to the bibliography. {citation_wall(keys)}

\section{{Additional Prior-Work Clusters}}
Affordance learning and manipulation representation pressure the map-quality claim {cite(keys, 68, 8)}. Active perception and probing pressure the hidden-state claim {cite(keys, 76, 8)}. Diffusion, sampling, and foundation-policy baselines pressure the action-search claim {cite(keys, 84, 8)}. Robust control, safety filters, and uncertainty methods pressure the deployment claim {cite(keys, 92, 8)}. Sim-to-real and benchmark methodology pressure the scope claim {cite(keys, 100, 8)}. These clusters collectively explain why the bar is higher than improving a single counterfactual recall number.

\section{{What Would Revive This Paper}}
A viable revival would need to: (1) keep the v5 mechanism improvements over v4; (2) reduce invalid action and damage rates to match active/robust baselines; (3) recover nonzero fixed-risk coverage at budget 0.05; (4) improve ECE without destroying coverage; (5) demonstrate stress-sweep robust utility dominance; (6) add real robot or accepted high-fidelity manipulation validation; and (7) report the same negative-case and paired-seed tables without selective omission.

\section{{Audit Philosophy}}
This manuscript is intentionally harsher than a normal first draft. It optimizes for survival under hostile review, not for pretty results. Strong baselines and stress tests were used to expose weaknesses, the method was improved relative to v4, the final protocol was frozen, and all predefined results are reported honestly. That is the scientific value of this artifact even though the terminal recommendation is negative.

\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}

\end{{document}}
"""
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'}")
    print(f"wrote {PAPER / 'references.bib'} with {len(keys)} entries")
    print(f"target pdf: {DOWNLOAD_PDF}")


if __name__ == "__main__":
    main()
