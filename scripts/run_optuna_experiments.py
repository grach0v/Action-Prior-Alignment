#!/usr/bin/env python3
"""
Optuna-based hyperparameter search over the A2 training pipeline.

This runner expects the dataset to be split into train/test .npy files that
follow the structure produced by `scripts/split_a2_data.py`. For every trial we
train the policy network for a user-provided number of epochs (the repository
only supports `batch_size=1`) and evaluate on the held-out test split using the
cross-entropy loss and top-1 accuracy. The Optuna study minimizes the test loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import random
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from helpers.data_loader import unified_data_loader
from models.networks import (
    CLIPAction,
    CLIPLangEmbAction,
    AdaptFeatCLIPAction,
    AdaptPolicyCLIPAction,
)
from models.efficient_attention import (
    list_efficient_attention_choices,
    normalize_efficient_attention_choice,
)
import utils.utils as utils
from env.constants import WORKSPACE_LIMITS

from tqdm.auto import tqdm

@dataclasses.dataclass
class BaseArgs:
    device: torch.device
    seed: int
    batch_size: int = 1
    normalize: bool = False
    adaptive: bool = False
    adaptive_type: str = "policy"
    adaptive_way: str = "residual"
    fusion_sa: bool = False
    layer_norm: bool = False
    lang_emb: bool = False
    task_emb: bool = False
    lang_enc: str = "clip"
    agent: str = "unified"
    width: int = 768
    layers: int = 1
    heads: int = 8
    hidden_size: int = 384
    use_rope: bool = False
    no_feat_rope: bool = False
    no_rgb_feat: bool = False
    loss: str = "ce"
    lr: float = 3e-4
    epoch_num: int = 5
    adjust_lr: bool = False
    step_size: int = 50
    step_ratio: float = 0.5
    sample_num: int | None = None
    evaluate: bool = False
    resume: bool = False
    load_model: bool = False
    model_path: str = ""
    task_num: int | None = None
    log_suffix: str = "optuna"
    efficient_attn: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data_train.npy"),
        help="Path to the train split produced by scripts/split_a2_data.py.",
    )
    parser.add_argument(
        "--test-path",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data_test.npy"),
        help="Path to the test split produced by scripts/split_a2_data.py.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to execute.",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=10,
        help="Number of training epochs to run within each trial.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="a2_optuna_search",
        help="Name assigned to the Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=pathlib.Path,
        default=pathlib.Path("results/optuna_a2.db"),
        help="SQLite storage path for the Optuna study.",
    )
    parser.add_argument(
        "--csv-output",
        type=pathlib.Path,
        default=pathlib.Path("results/optuna_a2_trials.csv"),
        help="CSV path where the trial dataframe will be exported.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for training/evaluation.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Enable Optuna median pruner.",
    )
    parser.add_argument(
        "--train-sample-num",
        type=int,
        default=None,
        help="Optional cap on the number of training samples (debugging aid).",
    )
    parser.add_argument(
        "--test-sample-num",
        type=int,
        default=None,
        help="Optional cap on the number of test samples (debugging aid).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed (a unique offset is used per trial).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display tqdm-based progress bars (requires tqdm to be installed).",
    )

    return parser.parse_args()


def create_agent(args: BaseArgs) -> torch.nn.Module:
    if not args.adaptive:
        if not args.lang_emb:
            return CLIPAction(action_dim=7, args=args)
        return CLIPLangEmbAction(action_dim=7, args=args)

    if args.adaptive_type == "policy":
        if args.adaptive_way == "full":
            return CLIPAction(action_dim=7, args=args)
        agent = AdaptPolicyCLIPAction(action_dim=7, args=args)
        for name, param in agent.named_parameters():
            param.requires_grad = False
            if "residual_policy" in name:
                param.requires_grad = True
        return agent

    agent = AdaptFeatCLIPAction(action_dim=7, args=args)
    for name, param in agent.named_parameters():
        param.requires_grad = False
        if "feat_adapter" in name:
            param.requires_grad = True
    return agent


def prepare_dataloader(
    path: pathlib.Path,
    *,
    shuffle: bool,
    sample_num: int | None,
    seed: int,
) -> DataLoader:
    loader = unified_data_loader(str(path), sample_num=sample_num, shuffle=shuffle)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

        sampler = torch.utils.data.RandomSampler(
            loader.dataset,
            replacement=False,
            generator=generator,
        )
        return DataLoader(
            loader.dataset,
            batch_size=1,
            sampler=sampler,
        )
    return loader


def prepare_batch(
    batch: List,
    device: torch.device,
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, bool]:
    sequence, lang_goal, pts_pos, pts_feat, pts_sim, actions, action_idx, done = batch

    if not bool(done[0].item()):
        return None

    pts_pos = pts_pos.to(device)
    pts_feat = pts_feat.to(device)
    pts_sim = pts_sim.to(device)
    actions = actions.to(device)
    action_idx = action_idx.to(device)
    lang_goal_str = lang_goal[0]

    if normalize:
        pts_pos = utils.normalize_pos(pts_pos, WORKSPACE_LIMITS.T, device=device)
        actions[:, :, :3] = utils.normalize_pos(actions[:, :, :3], WORKSPACE_LIMITS.T, device=device)

    return pts_pos, pts_feat, pts_sim, actions, action_idx, lang_goal_str, True


def forward_and_loss(
    model: torch.nn.Module,
    batch: List,
    args: BaseArgs,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    processed = prepare_batch(batch, device, normalize=args.normalize)
    if processed is None:
        return None

    pts_pos, pts_feat, pts_sim, actions, action_idx, lang_goal_str, _ = processed

    if args.task_emb:
        mode = "grasp"
        place_verbs = ["put", "place", "move"]
        for verb in place_verbs:
            if verb in lang_goal_str:
                mode = "place"
                break
    else:
        mode = None

    if args.adaptive:
        raise NotImplementedError("Adaptive mode is not supported in the Optuna runner yet.")

    if not args.lang_emb:
        pred_logits, _ = model(pts_pos, pts_feat, pts_sim, actions, mode)
    else:
        pred_logits, _ = model(pts_pos, pts_feat, actions, lang_goal_str, mode)

    loss = criterion(pred_logits, action_idx)
    preds = torch.argmax(pred_logits, dim=-1)
    correct = torch.eq(preds, action_idx).float()
    return loss, correct


def train_for_epochs(
    model: torch.nn.Module,
    train_loader: DataLoader,
    args: BaseArgs,
    epochs: int,
    *,
    show_progress: bool = False,
) -> List[float]:
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=2e-5, betas=(0.9, 0.99))
    if args.adjust_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.step_ratio,
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    device = args.device
    history: List[float] = []

    use_tqdm = show_progress and tqdm is not None
    epoch_iter = tqdm(range(epochs), desc="train epochs", leave=False) if use_tqdm else range(epochs)

    for epoch in epoch_iter:
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            result = forward_and_loss(model, batch, args, criterion=criterion, device=device)
            if result is None:
                continue
            loss, _ = result
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            sample_count += 1

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / sample_count if sample_count else math.inf
        history.append(epoch_loss)
        if use_tqdm:
            epoch_iter.set_postfix(loss=f"{epoch_loss:.4f}", samples=sample_count)

    if use_tqdm:
        epoch_iter.close()
    return history


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    args: BaseArgs,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    device = args.device
    total_loss = 0.0
    total_correct = 0.0
    evaluated = 0

    with torch.no_grad():
        for batch in data_loader:
            result = forward_and_loss(model, batch, args, criterion=criterion, device=device)
            if result is None:
                continue
            loss, correct = result
            total_loss += loss.detach().item()
            total_correct += correct.sum().detach().item()
            evaluated += correct.numel()

    avg_loss = total_loss / evaluated if evaluated else math.inf
    accuracy = total_correct / evaluated if evaluated else 0.0
    return {"loss": avg_loss, "accuracy": accuracy, "evaluated": evaluated}


def trial_objective(
    trial: optuna.trial.Trial,
    *,
    base_args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> float:
    seed = base_args.seed + trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    experiment_args = BaseArgs(
        device=torch.device(base_args.device),
        seed=seed,
        epoch_num=base_args.epochs_per_trial,
    )

    experiment_args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    experiment_args.hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512])
    experiment_args.layers = trial.suggest_int("layers", 1, 3)
    width_choices = getattr(base_args, "width_choices", [768])
    experiment_args.width = trial.suggest_categorical("width", width_choices)
    experiment_args.heads = trial.suggest_categorical("heads", [4, 8])
    if experiment_args.width % experiment_args.heads != 0:
        trial.set_user_attr(
            "skipped_combination",
            f"width={experiment_args.width}, heads={experiment_args.heads}",
        )
        raise optuna.exceptions.TrialPruned()
    attn_choice = trial.suggest_categorical(
        "efficient_attn",
        list_efficient_attention_choices(),
    )
    experiment_args.efficient_attn = normalize_efficient_attention_choice(attn_choice)
    experiment_args.lang_emb = trial.suggest_categorical("lang_emb", [False, True])
    experiment_args.normalize = trial.suggest_categorical("normalize", [False, True])
    experiment_args.use_rope = trial.suggest_categorical("use_rope", [False, True])
    experiment_args.adjust_lr = trial.suggest_categorical("adjust_lr", [False, True])
    if experiment_args.adjust_lr:
        experiment_args.step_size = trial.suggest_int("step_size", 5, 50, step=5)
        experiment_args.step_ratio = trial.suggest_float("step_ratio", 0.1, 0.9)

    experiment_args.task_emb = False
    experiment_args.loss = "ce"
    experiment_args.task_num = 2 if experiment_args.task_emb else None

    model = create_agent(experiment_args).to(experiment_args.device)

    try:
        train_history = train_for_epochs(
            model,
            train_loader,
            experiment_args,
            epochs=base_args.epochs_per_trial,
            show_progress=getattr(base_args, "progress", False),
        )

        evaluation = evaluate_model(model, test_loader, experiment_args)
    except RuntimeError as exc:
        trial.set_user_attr("failure", str(exc))
        raise optuna.exceptions.TrialPruned() from exc

    trial.report(evaluation["loss"], step=base_args.epochs_per_trial)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("train_history", train_history)
    trial.set_user_attr("test_accuracy", evaluation["accuracy"])
    trial.set_user_attr("evaluated_samples", evaluation["evaluated"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return evaluation["loss"]


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = device.type

    train_loader = prepare_dataloader(
        args.train_path,
        shuffle=True,
        sample_num=args.train_sample_num,
        seed=args.seed,
    )
    test_loader = prepare_dataloader(
        args.test_path,
        shuffle=False,
        sample_num=args.test_sample_num,
        seed=args.seed,
    )

    storage_url = f"sqlite:///{args.storage}"
    ensure_parent(args.storage)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner() if args.prune else None,
    )

    if getattr(args, "progress", False) and tqdm is None:
        print("Progress requested but tqdm is not installed. Install tqdm to enable progress bars.")

    default_width_choices = [512, 640, 768]
    stored_width_choices = set()
    for existing_trial in study.get_trials(deepcopy=False):
        dist = existing_trial.distributions.get("width")
        if dist is not None and hasattr(dist, "choices"):
            stored_width_choices.update(int(choice) for choice in dist.choices)
        elif "width" in existing_trial.params:
            stored_width_choices.add(int(existing_trial.params["width"]))
    if stored_width_choices:
        args.width_choices = sorted(stored_width_choices)
    else:
        args.width_choices = default_width_choices

    optuna_callbacks: List = []
    if getattr(args, "progress", False) and tqdm is not None:
        try:
            from optuna.progress_bar import TQDMCallback

            optuna_callbacks.append(TQDMCallback(n_trials=args.n_trials))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to enable Optuna tqdm callback: {exc}")

    study.optimize(
        lambda trial: trial_objective(
            trial,
            base_args=args,
            train_loader=train_loader,
            test_loader=test_loader,
        ),
        n_trials=args.n_trials,
        callbacks=optuna_callbacks or None,
    )

    ensure_parent(args.csv_output)
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(args.csv_output, index=False)

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  value (test loss): {best_trial.value}")
    print(f"  params: {best_trial.params}")
    print(f"  test accuracy: {best_trial.user_attrs.get('test_accuracy')}")


if __name__ == "__main__":
    main()
