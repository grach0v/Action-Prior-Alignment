#!/usr/bin/env python3
import argparse
import datetime
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.constants import WORKSPACE_LIMITS
from helpers.data_loader import unified_data_loader, unified_adaptive_data_loader
from models.networks import (
    CLIPAction,
    AdaptPolicyCLIPAction,
    AdaptFeatCLIPAction,
    CLIPLangEmbAction,
)
import utils.utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute training losses for a saved checkpoint on the training dataset."
    )
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--seed", type=int, default=1234, metavar="N", help="random seed")
    parser.add_argument(
        "--normalize", action="store_true", default=False, help="apply workspace normalization"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/a2_pp_data.npy", help="path to training data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="checkpoint path produced by the training script",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="optional number of samples to load from the dataset",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "mse", "nll"],
        help="loss type to mirror training",
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        default="BC_PP_loss_eval",
        help="suffix for tensorboard log directory",
    )
    parser.add_argument("--fusion_sa", action="store_true", default=False)
    parser.add_argument("--layer_norm", action="store_true", default=False)
    parser.add_argument("--lang_emb", action="store_true", default=False)
    parser.add_argument("--lang_enc", action="store", type=str, default="clip")
    parser.add_argument("--task_emb", action="store_true", default=False)
    parser.add_argument(
        "--efficient_attn",
        action="store",
        type=str,
        default=None,
        help="efficient attention choice (see models/efficient_attention.py)",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--no_feat_rope", action="store_true", default=False)
    parser.add_argument("--no_rgb_feat", action="store_true", default=False)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument(
        "--adaptive_type",
        action="store",
        type=str,
        default="policy",
        choices=["policy", "feat"],
    )
    parser.add_argument(
        "--adaptive_way",
        action="store",
        type=str,
        default="residual",
        choices=["residual", "full"],
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(args):
    if not args.adaptive:
        if not args.lang_emb:
            return CLIPAction(action_dim=7, args=args)
        return CLIPLangEmbAction(action_dim=7, args=args)

    if args.adaptive_type == "policy":
        if args.adaptive_way == "full":
            return CLIPAction(action_dim=7, args=args)
        return AdaptPolicyCLIPAction(action_dim=7, args=args)
    if args.adaptive_type == "feat":
        return AdaptFeatCLIPAction(action_dim=7, args=args)
    raise ValueError(f"Unsupported adaptive_type: {args.adaptive_type}")


def build_criterion(args):
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "nll":
        return nn.NLLLoss()
    if args.adaptive:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss type: {args.loss}")


def determine_mode(args, lang_goal):
    if not args.task_emb:
        return None
    mode = "grasp"
    for verb in ["put", "place", "move"]:
        if verb in lang_goal:
            mode = "place"
    return mode


def compute_batch_loss(model, batch, args, criterion):
    if args.adaptive:
        sequence, lang_goal, pts_pos, pts_feat, pts_sim, actions, gt_action_idxs = batch
    else:
        sequence, lang_goal, pts_pos, pts_feat, pts_sim, actions, action_idx, done = batch
        if not done:
            return None

    pts_pos = pts_pos.to(args.device)
    pts_feat = pts_feat.to(args.device)
    pts_sim = pts_sim.to(args.device)
    actions = actions.to(args.device)
    lang_goal = lang_goal[0]

    if args.normalize:
        pts_pos = utils.normalize_pos(pts_pos, WORKSPACE_LIMITS.T, device=pts_pos.device)
        actions[:, :, :3] = utils.normalize_pos(
            actions[:, :, :3], WORKSPACE_LIMITS.T, device=pts_pos.device
        )

    mode = determine_mode(args, lang_goal)

    # Mirror the training loss computation without gradient updates.
    if args.adaptive:
        gt_action_idxs = torch.as_tensor(gt_action_idxs, device=args.device)
        gt_action_logits = torch.zeros(actions.shape[1], device=args.device)
        gt_action_logits[gt_action_idxs] = 1
        gt_action_logits = gt_action_logits.unsqueeze(0)
        pred_action_logits, _ = model(pts_pos, pts_feat, pts_sim, actions, mode)
        loss = criterion(pred_action_logits, gt_action_logits)
    else:
        action_idx = torch.from_numpy(np.array(action_idx)).to(args.device)
        if not args.lang_emb:
            pred_action_logits, _ = model(pts_pos, pts_feat, pts_sim, actions, mode)
        else:
            pred_action_logits, _ = model(pts_pos, pts_feat, actions, lang_goal, mode)

        if args.loss == "ce":
            loss = criterion(pred_action_logits, action_idx)
        elif args.loss == "mse":
            pred_action_logits_softmax = F.softmax(pred_action_logits, dim=-1)
            pred_logits = pred_action_logits_softmax[0][action_idx.item()].unsqueeze(0)
            loss = criterion(pred_logits, torch.ones(1, device=args.device))
        elif args.loss == "nll":
            pred_action_logits_sigmoid = F.logsigmoid(pred_action_logits)
            loss = criterion(pred_action_logits_sigmoid, action_idx)
        else:
            raise ValueError(f"Unsupported loss type: {args.loss}")

    loss = loss.sum()
    return loss


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    args.task_num = 2 if args.task_emb else None
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"tensorlogs/{timestamp}_{args.log_suffix}"
    writer = SummaryWriter(log_dir)

    model = build_model(args)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    criterion = build_criterion(args)
    train_dl = (
        unified_adaptive_data_loader(args.data_path, args.sample_num, shuffle=False)
        if args.adaptive
        else unified_data_loader(args.data_path, args.sample_num, shuffle=False)
    )

    losses = []
    iteration = 0
    with torch.no_grad():
        for batch in train_dl:
            loss = compute_batch_loss(model, batch, args, criterion)
            if loss is None:
                continue
            loss_value = loss.item()
            losses.append(loss_value)
            writer.add_scalar("loss/iteration", loss_value, global_step=iteration)
            iteration += 1

    avg_loss = float(np.mean(losses)) if losses else 0.0
    writer.add_scalar("loss/epoch", avg_loss, global_step=0)
    writer.close()

    print(f"Logged {len(losses)} losses to {log_dir}. Average loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
