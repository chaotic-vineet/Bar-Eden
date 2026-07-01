"""
Plotting and evaluation utilities for the Act 1.2 models.

PLOT_BASE is the root directory; every plotting function takes a
model_name argument that creates a subfolder:
    D:\\Bar-Eden\\Act 1\\Act 1.2 Transformer\\Plots\\<model_name>\\

get_named_parameters / get_parameter_groups walk any Act 1.2 model
and return human-readable names grouped by role:
    Embedding + Pos Enc | Block N Attention | Block N FFN |
    LayerNorms | Output (W_out)

Supports both the current fused-attention + weight-tied Transformer
and the museum models (loop-over-heads, separate LMHead).

All plots save automatically and close the figure (no plt.show()).
"""

import json
import os
from collections import OrderedDict
import numpy as np
import torch
import matplotlib.pyplot as plt
import barkeep_style as bks

bks.apply_style()

PLOT_BASE = r"D:\Bar-Eden\Act 1\Act 1.2 Transformer\Plots"


def _save(fig, filename, model_name=None, save_path=None):
    """Save fig and close it. Uses PLOT_BASE/<model_name>/ by default."""
    if save_path:
        path = save_path
    elif model_name:
        path = os.path.join(PLOT_BASE, model_name, filename)
    else:
        path = os.path.join(PLOT_BASE, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  NAMED PARAMETERS + GROUPING
# ═══════════════════════════════════════════════════════════════

def get_named_parameters(model):
    """
    Walk a model and return [(name, param)] in the same order as
    model.parameters().

    Handles both:
      - Current Transformer (fused attention, weight-tied, no LMHead)
      - Museum models (loop-over-heads, separate LMHead)

    Names are human-readable:
        embedding, pos_enc,
        b0_W_Q, b0_W_K, b0_W_V, b0_W_out,
        b0_ln1_γ, b0_ln1_β, b0_ffn_up_w, b0_ffn_up_b, ...
        ln_f_γ, ln_f_β
    """
    named = []

    # ── embedding ──
    named.append(("embedding", model.embedding.embedding_matrix))

    # ── positional encoding ──
    if hasattr(model, "positional_encoding"):
        named.append(("pos_enc", model.positional_encoding.positional_encoding_matrix))

    # ── stacked blocks ──
    blocks = getattr(model, "blocks", None)
    if blocks is None and hasattr(model, "block"):
        blocks = [model.block]

    if blocks is not None:
        for bi, block in enumerate(blocks):
            prefix = f"b{bi}"
            mha = block.multihead

            # fused attention (current): W_Query, W_Key, W_Value, W_out
            if hasattr(mha, "W_Query"):
                named.append((f"{prefix}_W_Q", mha.W_Query.weight))
                named.append((f"{prefix}_W_K", mha.W_Key.weight))
                named.append((f"{prefix}_W_V", mha.W_Value.weight))
                named.append((f"{prefix}_W_out", mha.W_out.weight))
            # museum: loop-over-heads
            elif hasattr(mha, "heads"):
                for hi, head in enumerate(mha.heads):
                    named.append((f"{prefix}_h{hi}_Q", head.W_Query.weight))
                    named.append((f"{prefix}_h{hi}_K", head.W_Key.weight))
                    named.append((f"{prefix}_h{hi}_V", head.W_Value.weight))
                named.append((f"{prefix}_W_out", mha.W_out.weight))

            # ln1
            named.append((f"{prefix}_ln1_γ", block.ln1.gamma))
            named.append((f"{prefix}_ln1_β", block.ln1.beta))

            # ffn
            named.append((f"{prefix}_ffn_up_w", block.ffn.linear1.weight))
            if block.ffn.linear1.bias is not None:
                named.append((f"{prefix}_ffn_up_b", block.ffn.linear1.bias))
            named.append((f"{prefix}_ffn_down_w", block.ffn.linear2.weight))
            if block.ffn.linear2.bias is not None:
                named.append((f"{prefix}_ffn_down_b", block.ffn.linear2.bias))

            # ln2
            named.append((f"{prefix}_ln2_γ", block.ln2.gamma))
            named.append((f"{prefix}_ln2_β", block.ln2.beta))

    # ── flat models (no blocks) ──
    elif hasattr(model, "multihead"):
        mha = model.multihead
        if hasattr(mha, "W_Query"):
            named.append(("W_Q", mha.W_Query.weight))
            named.append(("W_K", mha.W_Key.weight))
            named.append(("W_V", mha.W_Value.weight))
            named.append(("W_out", mha.W_out.weight))
        elif hasattr(mha, "heads"):
            for hi, head in enumerate(mha.heads):
                named.append((f"h{hi}_Q", head.W_Query.weight))
                named.append((f"h{hi}_K", head.W_Key.weight))
                named.append((f"h{hi}_V", head.W_Value.weight))
            named.append(("W_out", mha.W_out.weight))

        if hasattr(model, "ffn"):
            named.append(("ffn_up_w", model.ffn.linear1.weight))
            if model.ffn.linear1.bias is not None:
                named.append(("ffn_up_b", model.ffn.linear1.bias))
            named.append(("ffn_down_w", model.ffn.linear2.weight))
            if model.ffn.linear2.bias is not None:
                named.append(("ffn_down_b", model.ffn.linear2.bias))

    # ── lm head (museum models only — current uses weight tying) ──
    if hasattr(model, "lmhead"):
        named.append(("lm_head", model.lmhead.W_projection.weight))

    # ── final layer norm ──
    if hasattr(model, "ln"):
        named.append(("ln_f_γ", model.ln.gamma))
        named.append(("ln_f_β", model.ln.beta))

    return named


def get_parameter_groups(named_params):
    """
    Group named parameters by architectural role.

    Returns OrderedDict:
        "Embedding + Pos Enc"      -> [(name, param), ...]
        "Block 0 Attention"        -> [(name, param), ...]  (fused)
        "Block 0 Head 0"           -> [(name, param), ...]  (museum)
        "Block 0 FFN"              -> [(name, param), ...]
        "LayerNorms"               -> [(name, param), ...]
        "Output (W_out)"           -> [(name, param), ...]
    """
    groups = OrderedDict()

    for name, param in named_params:
        if name in ("embedding", "pos_enc"):
            group = "Embedding + Pos Enc"
        elif name.endswith(("_W_Q", "_W_K", "_W_V")):
            # fused: b0_W_Q -> "Block 0 Attention"
            if name.startswith("b"):
                group = f"Block {name[1]} Attention"
            else:
                group = "Attention"
        elif "_h" in name and ("_Q" in name or "_K" in name or "_V" in name):
            # museum per-head: b0_h2_Q -> "Block 0 Head 2"
            if name.startswith("b"):
                parts = name.split("_")
                group = f"Block {parts[0][1:]} Head {parts[1][1:]}"
            else:
                parts = name.split("_")
                group = f"Head {parts[0][1:]}"
        elif "ffn" in name:
            if name.startswith("b"):
                group = f"Block {name[1]} FFN"
            else:
                group = "FFN"
        elif "ln" in name:
            group = "LayerNorms"
        elif "W_out" in name or "lm_head" in name:
            group = "Output (W_out)"
        else:
            group = "Other"

        groups.setdefault(group, []).append((name, param))

    return groups


# ═══════════════════════════════════════════════════════════════
#  CROSS-RUN COMPARISON
# ═══════════════════════════════════════════════════════════════

def load_runs(log_path):
    """
    Read a JSONL log and group "metric" records by run_name.

    Returns {run_name: {"steps": [...], "train_loss": [...],
    "dev_loss": [...], "lr": [...]}}
    """
    runs = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "metric":
                continue
            run = runs.setdefault(rec["run_name"],
                                  {"steps": [], "train_loss": [],
                                   "dev_loss": [], "lr": []})
            run["steps"].append(rec["step"])
            run["train_loss"].append(rec["train_loss"])
            run["dev_loss"].append(rec["dev_loss"])
            run["lr"].append(rec["lr"])
    return runs


def plot_run_comparison(runs, metric="dev_loss", model_name=None, save_path=None):
    """Overlay `metric` across multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"]]

    for i, (name, run) in enumerate(runs.items()):
        color = palette[i % len(palette)]
        ax.plot(run["steps"], run[metric], marker="D", color=color, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — run comparison")
    ax.legend()

    _save(fig, "run_comparison.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  SINGLE-RUN TRAINING CURVES
# ═══════════════════════════════════════════════════════════════

def plot_training_curve(results, title="Training Loss", model_name=None, save_path=None):
    """Full per-step training loss (smoothed) + dev-loss checkpoints."""
    loss_per_itrn = results["loss_per_itrn"]
    dev_itrns = results["dev_itrns"]
    dev_losses = results["dev_losses"]

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = range(len(loss_per_itrn))
    ax.plot(steps, loss_per_itrn, color=bks.COLORS["amber"],
            alpha=0.08, linewidth=0.5)

    window = 1000
    arr = np.asarray(loss_per_itrn)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    idx = np.arange(len(arr))
    lo = np.maximum(0, idx - window)
    smoothed = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    ax.plot(steps, smoothed, color=bks.COLORS["amber"], linewidth=1.5,
            label="Train loss (smoothed)")

    ax.scatter(dev_itrns, dev_losses, color=bks.COLORS["red"], s=40,
               zorder=5, marker="D", label="Dev loss")
    for s, v in zip(dev_itrns, dev_losses):
        ax.text(s, v + 0.03, f"{v:.3f}", ha="center", fontsize=9,
                color=bks.COLORS["red"])

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    _save(fig, "training_curve.png", model_name, save_path)


def plot_lr_schedule(results, model_name=None, save_path=None):
    """Plot the learning-rate schedule."""
    lr_per_itrn = results["lr_per_itrn"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(len(lr_per_itrn)), lr_per_itrn,
            color=bks.COLORS["teal"], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    _save(fig, "lr_schedule.png", model_name, save_path)


def plot_train_vs_dev(results, model_name=None, save_path=None):
    """Bar chart: train loss (avg of 5K steps before checkpoint) vs dev."""
    loss_per_itrn = results["loss_per_itrn"]
    dev_itrns = results["dev_itrns"]
    dev_losses = results["dev_losses"]

    trn_at_checkpoints = []
    for s in dev_itrns:
        w = max(0, s - 5000)
        trn_at_checkpoints.append(sum(loss_per_itrn[w:s]) / (s - w))

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos, bar_w = range(len(dev_itrns)), 0.35
    ax.bar([p - bar_w / 2 for p in x_pos], trn_at_checkpoints, bar_w,
           color=bks.COLORS["amber"], label="Train")
    ax.bar([p + bar_w / 2 for p in x_pos], dev_losses, bar_w,
           color=bks.COLORS["red"], label="Dev")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f"{s // 1000}K" for s in dev_itrns])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Dev Loss at Checkpoints")
    ax.legend()

    _save(fig, "train_vs_dev.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(model, data, name=""):
    """Cross-entropy loss under no_grad."""
    with torch.no_grad():
        logits = model(data.inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), data.targets,
            ignore_index=data.pad_idx)
    if name:
        print(f"{name:>5}: {loss.item():.4f}")
    return loss.item()


def evaluate_all(model, trn, dev, test):
    """Evaluate on train, dev, test splits."""
    return {
        "train": evaluate(model, trn, "train"),
        "dev": evaluate(model, dev, "dev"),
        "test": evaluate(model, test, "test"),
    }


# ═══════════════════════════════════════════════════════════════
#  WEIGHT / ACTIVATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def weight_histogram(model, bins=50, model_name=None, save_path=None):
    """Histogram of all weights flattened together."""
    all_vals = torch.cat([p.detach().flatten().cpu()
                          for p in model.parameters()])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_vals.numpy(), bins=bins, color=bks.COLORS["amber"])
    ax.set_title(f"Weight distribution — mean {all_vals.mean():.4f},"
                 f" std {all_vals.std():.4f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

    _save(fig, "weight_histogram.png", model_name, save_path)


def activation_saturation(model, x, bins=50, model_name=None, save_path=None):
    """
    Histogram the FFN's GeLU activations after a forward pass.
    For multi-block models, shows all blocks side by side.
    """
    blocks = getattr(model, "blocks", None)
    if blocks:
        ffns = [(f"Block {i}", b.ffn) for i, b in enumerate(blocks)]
    elif hasattr(model, "ffn"):
        ffns = [("FFN", model.ffn)]
    else:
        print("model has no .ffn — nothing to check")
        return

    with torch.no_grad():
        model(x)

    n = len(ffns)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, ffn) in zip(axes, ffns):
        out = ffn.gelu.out.detach().cpu().flatten()
        suppressed = (out < -0.1).float().mean().item()
        ax.hist(out.numpy(), bins=bins, color=bks.COLORS["teal"])
        ax.set_title(f"{label} GeLU — "
                     f"{suppressed * 100:.1f}% suppressed")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count")

    fig.suptitle("FFN GeLU activations")
    plt.tight_layout()
    _save(fig, "activation_saturation.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC PASS (forward + backward, all models)
# ═══════════════════════════════════════════════════════════════

def diagnostic_pass(model, x, y):
    """
    One forward+backward pass retaining gradients on every stage.

    Handles both weight-tied Transformer (current) and museum models
    with separate LMHead.

    Returns (stages, loss) where stages is a dict {name: tensor}
    with .grad populated.
    """
    for p in model.parameters():
        p.grad = None

    stages = {}

    # embedding (+ positional if present)
    emb = model.embedding(x)
    if hasattr(model, "positional_encoding"):
        emb = emb + model.positional_encoding()
    emb.retain_grad()
    stages["embedding"] = emb

    # N stacked blocks (Transformer)
    if hasattr(model, "blocks"):
        h = emb
        for i, block in enumerate(model.blocks):
            attn_out = h + block.multihead(block.ln1(h))
            attn_out.retain_grad()
            stages[f"block_{i}_attn"] = attn_out

            ffn_out = attn_out + block.ffn(block.ln2(attn_out))
            ffn_out.retain_grad()
            stages[f"block_{i}_ffn"] = ffn_out

            h = ffn_out

        normalized = model.ln(h)
        normalized.retain_grad()
        stages["normalized"] = normalized
        pre_logits = normalized

    # flat models
    else:
        attn_out = model.multihead(emb)
        attn_out.retain_grad()
        stages["attn_out"] = attn_out

        if hasattr(model, "ffn"):
            ffn_out = model.ffn(attn_out)
            ffn_out.retain_grad()
            stages["ffn_out"] = ffn_out
            pre_logits = ffn_out
        else:
            pre_logits = attn_out

    # lm head — weight-tied or separate
    if hasattr(model, "lmhead"):
        logits = model.lmhead(pre_logits)
    else:
        logits = pre_logits @ model.embedding.embedding_matrix.T
    logits.retain_grad()
    stages["logits"] = logits

    pad_idx = y.max().item()
    loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1), y, ignore_index=pad_idx)
    loss.backward()

    return stages, loss


def plot_activation_distributions(stages, model_name=None, save_path=None):
    """Histogram forward-pass values at each stage."""
    names = list(stages.keys())
    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.2 * len(names), 3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = stages[name].detach().cpu().flatten().numpy()
        ax.hist(vals, bins=40, color=bks.COLORS["amber"])
        ax.set_title(f"{name}\nμ={vals.mean():.3f} σ={vals.std():.3f}",
                     fontsize=9)
    fig.suptitle("Activation distributions (forward pass)")
    plt.tight_layout()

    _save(fig, "activation_distributions.png", model_name, save_path)


def plot_activation_gradients(stages, model_name=None, save_path=None):
    """Histogram dL/d(activation) at each stage."""
    names = list(stages.keys())
    fig, axes = plt.subplots(1, len(names),
                             figsize=(3.2 * len(names), 3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = stages[name].grad.detach().cpu().flatten().numpy()
        ax.hist(vals, bins=40, color=bks.COLORS["red"])
        ax.set_title(f"{name} grad\n"
                     f"μ={vals.mean():.1e} σ={vals.std():.1e}",
                     fontsize=9)
    fig.suptitle("Activation gradient distributions (backward pass)")
    plt.tight_layout()

    _save(fig, "activation_gradients.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  GROUPED WEIGHT GRADIENTS
# ═══════════════════════════════════════════════════════════════

def plot_weight_gradients(model, model_name=None, save_path=None):
    """
    dL/dW histograms grouped by architectural role.
    One subplot per group; parameters within a group overlaid.
    """
    named = get_named_parameters(model)
    groups = get_parameter_groups(named)

    # filter to params with gradients
    groups = OrderedDict(
        (g, [(n, p) for n, p in members if p.grad is not None])
        for g, members in groups.items()
    )
    groups = OrderedDict((g, m) for g, m in groups.items() if m)

    n_groups = len(groups)
    cols = min(n_groups, 3)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 3.5 * rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"],
               "#5B9BD5", "#A0856C", "#D97706", "#4FB3C9"]

    for idx, (group_name, members) in enumerate(groups.items()):
        ax = axes[idx]
        for mi, (name, param) in enumerate(members):
            vals = param.grad.detach().cpu().flatten().numpy()
            color = palette[mi % len(palette)]
            ax.hist(vals, bins=50, alpha=0.6, density=True,
                    color=color, label=name)
        ax.set_title(group_name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Weight gradient distributions (grouped)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    _save(fig, "weight_gradients.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  GROUPED UPDATE-TO-DATA RATIOS
# ═══════════════════════════════════════════════════════════════

def plot_update_ratios(ud, model, model_name=None, save_path=None):
    """
    log10(std(update)/std(data)) per parameter, grouped by role.
    One subplot per group with the −3 target line.

    `ud` is the list-of-lists from train_model, one inner list per
    checkpoint, one value per parameter (in model.parameters() order).
    """
    named = get_named_parameters(model)
    groups = get_parameter_groups(named)

    # build a param id → index map
    params_list = list(model.parameters())
    param_to_idx = {id(p): i for i, p in enumerate(params_list)}

    def has_ud(param):
        idx = param_to_idx.get(id(param))
        return (idx is not None and param.ndim >= 2
                and idx < len(ud[0]))

    groups_filtered = OrderedDict()
    for g, members in groups.items():
        valid = [(n, p) for n, p in members if has_ud(p)]
        if valid:
            groups_filtered[g] = valid

    n_groups = len(groups_filtered)
    if n_groups == 0:
        print("No 2D parameters with UD data found.")
        return

    cols = min(n_groups, 3)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5.5 * cols, 3.5 * rows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    palette = [bks.COLORS["amber"], bks.COLORS["red"],
               bks.COLORS["teal"], bks.COLORS["violet"],
               "#5B9BD5", "#A0856C", "#D97706", "#4FB3C9"]

    n_checkpoints = len(ud)
    x_ckpts = range(n_checkpoints)

    for idx, (group_name, members) in enumerate(groups_filtered.items()):
        ax = axes[idx]
        for mi, (name, param) in enumerate(members):
            pi = param_to_idx[id(param)]
            values = [ud[c][pi] for c in range(n_checkpoints)]
            color = palette[mi % len(palette)]
            ax.plot(x_ckpts, values, color=color, label=name,
                    linewidth=1.5, marker=".", markersize=3)

        ax.axhline(-3, color="white", linestyle="--", alpha=0.3,
                   linewidth=1, label="target (−3)")
        ax.set_title(group_name, fontsize=10)
        ax.set_xlabel("Checkpoint", fontsize=8)
        ax.set_ylabel("log₁₀(update/data)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Update-to-data ratio (grouped)", fontsize=12, y=1.02)
    plt.tight_layout()

    _save(fig, "update_ratios.png", model_name, save_path)


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE: run all diagnostics for one model
# ═══════════════════════════════════════════════════════════════

def run_all_diagnostics(model, results, x, y, model_name):
    """
    Run every plot for a single model, saving to
    PLOT_BASE/<model_name>/.

    model:      the trained model object
    results:    the dict from train_model()
    x, y:       a batch of inputs/targets for diagnostic_pass
    model_name: subfolder name (e.g. "3-Block Transformer")
    """
    print(f"── {model_name} ──")

    # switch to eval for diagnostics
    if hasattr(model, "eval"):
        model.eval()

    # training curves
    plot_training_curve(results, title=f"Training Loss — {model_name}",
                        model_name=model_name)
    plot_lr_schedule(results, model_name=model_name)
    plot_train_vs_dev(results, model_name=model_name)

    # weight histogram
    weight_histogram(model, model_name=model_name)

    # activation saturation (FFN models only)
    activation_saturation(model, x, model_name=model_name)

    # diagnostic pass → activation/gradient distributions
    stages, loss = diagnostic_pass(model, x, y)
    plot_activation_distributions(stages, model_name=model_name)
    plot_activation_gradients(stages, model_name=model_name)

    # grouped weight gradients
    plot_weight_gradients(model, model_name=model_name)

    # grouped update-to-data ratios
    if "ud" in results and results["ud"]:
        plot_update_ratios(results["ud"], model,
                           model_name=model_name)

    # restore training mode
    if hasattr(model, "train"):
        model.train()

    print(f"  saved to {os.path.join(PLOT_BASE, model_name)}")