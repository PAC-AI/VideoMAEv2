# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             average_precision_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             precision_recall_curve,
                             roc_curve,
                             confusion_matrix)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import wandb
from tqdm import tqdm

import utils


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    use_wandb=False,
                    bin_cls=False,
                    cls_wt_scale=1,
                    global_rank=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    for step,batch in enumerate(tqdm(data_loader,desc=f'train epoch {epoch}')):
    # for step, batch in enumerate(
    #         metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # NOTE: When the decoder mask ratio is 0,
        # in other words, when decoder masking is not used,
        # decode_masked_pos = ~bool_masked_pos
        images, bool_masked_pos, decode_masked_pos, labels_bin, _ = batch

        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        labels_bin = labels_bin.to(device,non_blocking=True)
        if cls_wt_scale > 0:
            labels_bin_wt = (labels_bin*cls_wt_scale)+((1-labels_bin)*1)
            labels_bin_wt /= cls_wt_scale+1
        else:
            labels_bin_wt = torch.ones_like(labels_bin)/labels_bin.shape[0]

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                     None,
                                                                     None,
                                                                     None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                   None, None,
                                                                   None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)

            B, N, C = images_patch.shape
            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)

        if loss_scaler is None:
            outputs = model(images, bool_masked_pos, decode_masked_pos)
            loss = (outputs - labels)**2
            loss = loss.mean(dim=-1)
            cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
            loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(images, bool_masked_pos, decode_masked_pos)
                if bin_cls:
                    probs = torch.sigmoid(outputs)
                    loss = F.binary_cross_entropy_with_logits(outputs,labels_bin,
                                                              labels_bin_wt)
                    acc = binary_accuracy(probs,labels_bin)
                else:
                    loss = (outputs - labels)**2
                    loss = loss.mean(dim=-1)
                    cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(
                        B, -1)
                    loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()

        if loss_scaler is None:
            loss.backward()
            if max_norm is None:
                grad_norm = utils.get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()
            loss_scale_value = 0
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if use_wandb and global_rank == 0:
            wandb.log({'epoch'      : epoch,
                       'train_loss' : loss_value,
                       'train_acc'  : acc.item(),
                       'loss_scale' : loss_scale_value,
                       'max_lr'     : max_lr,
                       'min_lr'     : min_lr,
                       'wd'         : weight_decay_value,
                       'grad_norm'  : grad_norm},
                       step=it)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable,
                  device: torch.device,
                  epoch: int,
                  use_wandb=False,
                  cls_wt_scale=1,
                  global_rank=0,
                  val_data_path=None):
    model.eval()
    video_names = list()
    labels = list()
    probs = list()
    loss = list()
    acc = list()
    for step,batch in enumerate(tqdm(data_loader,desc=f'val epoch {epoch}')):
        images, bool_masked_pos, decode_masked_pos, labels_bin, video_name = batch

        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        labels_bin = labels_bin.to(device,non_blocking=True)
        if cls_wt_scale > 0:
            labels_bin_wt = (labels_bin*cls_wt_scale)+((1-labels_bin)*1)
            labels_bin_wt /= cls_wt_scale+1
        else:
            labels_bin_wt = torch.ones_like(labels_bin)/labels_bin.shape[0]
        labels.append(labels_bin)
        video_names.extend(video_name)

        with torch.cuda.amp.autocast(), torch.no_grad():
            outputs = model(images, bool_masked_pos, decode_masked_pos)
            probs.append(torch.sigmoid(outputs))
            loss.append(F.binary_cross_entropy_with_logits(outputs,labels_bin,
                                                           labels_bin_wt))
    labels = torch.cat(labels).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    loss = torch.stack(loss).cpu().numpy()

    # clip_cnt = Counter(l.split(' ')[0]
    #                    for l in open(val_data_path).readlines())
    # video_name_cnt = Counter(video_names)
    # for k,v in video_name_cnt.items():
    #     if v != clip_cnt[k]:
    #         pass
    # assert clip_cnt == video_name_cnt
    # assert labels.shape[0] == probs.shape[0]

    res = pd.DataFrame({'video_name':video_names,
                        'label'     :labels,
                        'prob'      :probs})
    res['pred'] = res.prob >= 0.5
    maj_vote = res.groupby('video_name').mean().reset_index()
    maj_vote.pred = maj_vote.pred >= 0.5
    labels = maj_vote.label
    probs = maj_vote.prob
    preds = maj_vote.pred

    acc = accuracy_score(labels,preds)
    bal_acc = balanced_accuracy_score(labels,preds)
    ap = average_precision_score(labels,probs)
    f1 = f1_score(labels,preds)
    p = precision_score(labels,preds)
    r = recall_score(labels,preds)
    auc = roc_auc_score(labels,probs)

    pr = precision_recall_curve(labels,probs)
    fig_pr = plt.figure()
    plt.plot(pr[1],pr[0])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')
    plt.close()

    roc = roc_curve(labels,probs)
    fig_roc = plt.figure()
    plt.plot(roc[0],roc[1])    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.close()

    cf = confusion_matrix(labels,preds,normalize='true')
    fig_cm = plt.figure()    
    ax = sns.heatmap(cf,fmt='.2%',annot=True,
                     xticklabels=['NF','F'],
                     yticklabels=['NF','F'])
    ax.set(xlabel='PRED',ylabel='TRUE')
    plt.close()

    probs = np.stack([1-probs,probs],axis=1)

    if use_wandb and global_rank == 0:
        wandb.log({'val_loss'    : loss.mean(),
                   'val_acc'     : acc,
                   'val_bal_acc' : bal_acc,
                   'val_ap'      : ap,
                   'val_f1'      : f1,
                   'val_p'       : p,
                   'val_r'       : r,
                   'val_auc'     : auc,
                   'val_pr'      : wandb.Image(fig_pr),
                   'val_roc'     : wandb.Image(fig_roc),
                   'val_cm'      : wandb.Image(fig_cm)},
                  commit=False)
