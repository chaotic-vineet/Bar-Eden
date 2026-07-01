"""
Hand-rolled training loop for the Act 1.2 attention models — no
torch.optim. Parameters are updated in place via plain SGD with a
cosine learning-rate schedule.

Works with any model exposing __call__, parameters(), and config_dict()
(see model.py).

Logging: if config.log_path is set, train_model writes JSONL records —
    - one "config" record at the start of the run: model.config_dict(),
      total parameter count, and config.__dict__
    - one "metric" record every config.log_interval steps: step,
      train_loss, dev_loss, lr

Multiple runs can share one log file and be compared via
analysis.load_runs / analysis.plot_run_comparison.
"""

import torch
import time
import math
import json
from dataclasses import dataclass

@dataclass
class TrainConfig:
    """
    Shared training hyperparameters. For an ablation, reuse one
    TrainConfig unchanged across runs — only run_name and the model
    should differ.

    lr follows a cosine schedule from lr_start down to lr_end over
    `iterations` steps. Dev loss is evaluated every `log_interval`
    steps. log_path=None disables JSONL logging.
    """
    lr_start: float
    lr_end: float
    iterations: int
    log_interval: int
    warmup_steps: int
    batch_size: int = 4096
    log_path: str = None
    max_norm: float = 1.0

def log_run_config(log_path, run_name, model, config):
    """
    Write one "config" record describing this run: model.config_dict(),
    total parameter count, and every TrainConfig field. No-op if
    log_path is None.
    """
    if log_path is None:
        return
    record = {
        "type": "config",
        "run_name": run_name,
        "num_params": sum(p.numel() for p in model.parameters()),
        **config.__dict__,
        **model.config_dict(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def train_model(SEED, model, optimizer,train, dev, config, run_name, device, pause_time):
    """
    Train `model` on `train` for config.iterations steps via SGD with a
    cosine LR schedule, evaluating on `dev` every config.log_interval steps.

    Batches are drawn from a generator seeded with dataset.SEED, created
    fresh inside this function — so every call sees an identical batch
    sequence regardless of how much random state model construction
    consumed beforehand.

    If config.log_path is set, writes one "config" record (see
    log_run_config) and one "metric" record per checkpoint.

    Returns a dict:
        lr_per_itrn:   list[float], length config.iterations
        loss_per_itrn: list[float], length config.iterations
        dev_losses:    list[float], one per checkpoint
        dev_itrns:     list[int],   step numbers of each checkpoint
        elapsed:       float, wall-clock seconds
    """
    log_run_config(config.log_path, run_name, model, config)

    batch_generator = torch.Generator(device=device).manual_seed(SEED)
    dev_batch_generator = torch.Generator(device=device).manual_seed(SEED)

    lr_start = config.lr_start
    lr_end = config.lr_end

    lr_per_itrn   = []
    checkpoint_times = []
    loss_per_itrn = torch.zeros(config.iterations, device=device)
    dev_losses    = []
    dev_itrns     = []
    ud            = []

    if 'xpu' in device:
        torch.xpu.synchronize()
    t0 = time.time()

    parameters = list(model.parameters())

    max_norm = config.max_norm

    warmup_steps = config.warmup_steps
    iterations = config.iterations
    batch_size = config.batch_size

    train_inputs_shape = train.inputs.shape[0]
    dev_inputs_shape = dev.inputs.shape[0]
    pad_idx = train.pad_idx

    for itrn in range(1, iterations+1):
        idx = torch.randint(0, train_inputs_shape, (batch_size,), device=device, generator=batch_generator)

        logits = model(train.inputs[idx])
        loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), train.targets[idx], ignore_index=pad_idx)

        if itrn < warmup_steps:
            lr = lr_start * itrn / warmup_steps
        else:
            cosine_itrn = itrn - warmup_steps
            cosine_itrns = iterations - warmup_steps
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * cosine_itrn / cosine_itrns))

        for parameter in parameters:
            parameter.grad = None
        loss.backward()

        global_norm_squared = 0
        for parameter in parameters:
            global_norm_squared += parameter.grad.pow(2).sum()

        global_norm = global_norm_squared ** 0.5
        clip_coeff = torch.clamp(max_norm / (global_norm + 1e-8), max=1.0)

        for parameter in parameters:
            parameter.grad = parameter.grad * clip_coeff

        optimizer(lr)

        lr_per_itrn.append(lr)
        loss_per_itrn[itrn-1] = loss.detach()

        if itrn % config.log_interval == 0:
            time.sleep(pause_time*60)

            model.eval()

            dev_idx = torch.randint(0, dev_inputs_shape, (batch_size,), device=device, generator=dev_batch_generator)

            with torch.no_grad():
                dev_logits = model(dev.inputs[dev_idx])
                dev_loss = torch.nn.functional.cross_entropy(dev_logits.permute(0, 2, 1), dev.targets[dev_idx], ignore_index=pad_idx)

            trn_loss_val = loss.item()
            dev_loss_val = dev_loss.item()

            dev_losses.append(dev_loss_val)
            dev_itrns.append(itrn)

            print(f"  step {itrn:>7,} | dev {dev_loss_val:.4f} vs train {trn_loss_val:.4f}| lr {lr:.4f}")

            magnitudes = optimizer.step_magnitudes(lr)

            ud.append([
                ((magnitude.std()) / parameter.data.std()).log10().item()
                for magnitude, parameter in zip(magnitudes, parameters)
            ])

            if 'xpu' in device:
                torch.xpu.synchronize()
            checkpoint_times.append(time.time() - t0)

            if config.log_path is not None:
                record = {
                    "type": "metric",
                    "run_name": run_name,
                    "step": itrn,
                    "train_loss": trn_loss_val,
                    "dev_loss": dev_loss_val,
                    "lr": lr,
                }

                with open(config.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            model.train()


    if 'xpu' in device:
        torch.xpu.synchronize()
    elapsed = time.time() - t0

    loss_per_itrn = loss_per_itrn.cpu().tolist()

    return {
        "lr_per_itrn": lr_per_itrn,
        "loss_per_itrn": loss_per_itrn,
        "dev_losses": dev_losses,
        "dev_itrns": dev_itrns,
        "elapsed": elapsed,
        "ud": ud
    }