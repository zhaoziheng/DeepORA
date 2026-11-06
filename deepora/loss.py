"""Masked focal, foreground Dice/BCE and multi-positive contrastive losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BinaryDiceLoss(nn.Module):
    """Foreground channel-wise squared-denominator Dice loss."""

    def __init__(self, smooth=1e-07, p=2, reduction="none"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape == target.shape, (
            f"predict {predict.shape} & target {target.shape} do not match"
        )
        predict = rearrange(predict.contiguous(), "b c h w d -> (b c) (h w d)")
        target = rearrange(target.contiguous(), "b c h w d -> (b c) (h w d)")
        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - intersection / union
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class MultiTaskLoss(nn.Module):
    """Combine observed-label focal loss, auxiliary masks and report alignment."""

    def __init__(
        self,
        device,
        do_pretrain: bool = False,
        target_labels: list = [],
        use_focal: bool = False,
        focal_gamma: float = 5.0,
        focal_alpha: float = None,
        brainstructure_seg_loss_weight: float = 0.1,
        use_brainstructure_seg_loss: bool = False,
        vessel_seg_loss_weight: float = 0.1,
        use_vessel_seg_loss: bool = False,
        use_masked_loss: bool = True,
        clip_loss_weight: float = 0.1,
        use_clip_loss: bool = False,
    ):
        super().__init__()
        self.device = device
        self.do_pretrain = do_pretrain
        self.target_labels = target_labels
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_masked_loss = use_masked_loss
        self.brainstructure_seg_loss_weight = brainstructure_seg_loss_weight
        self.use_brainstructure_seg_loss = use_brainstructure_seg_loss
        self.vessel_seg_loss_weight = vessel_seg_loss_weight
        self.use_vessel_seg_loss = use_vessel_seg_loss
        self.clip_loss_weight = clip_loss_weight
        self.use_clip_loss = use_clip_loss
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss = BinaryDiceLoss(reduction="none")
        self.seg_bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.label_to_idx = {label: idx for (idx, label) in enumerate(target_labels)}

    def forward(
        self,
        result: dict,
        labels: dict,
        label_mask=None,
        ct_mask: torch.Tensor = None,
        mr_mask: torch.Tensor = None,
        ct_segs: torch.Tensor = None,
        mr_segs: torch.Tensor = None,
        ct_seg_mask: torch.Tensor = None,
        mr_seg_mask: torch.Tensor = None,
        ct_vessel_segs: torch.Tensor = None,
        mr_vessel_segs: torch.Tensor = None,
        ct_vessel_seg_mask: torch.Tensor = None,
        mr_vessel_seg_mask: torch.Tensor = None,
        report_mask: torch.Tensor = None,
    ):
        detailed_loss = {}
        cls_loss = torch.tensor(0.0, device=self.device)
        if not self.do_pretrain:
            pred_dict = result["predictions"]
            if self.use_masked_loss:
                (cls_loss, cls_detailed) = self._compute_masked_classification_loss(
                    pred_dict, labels, label_mask
                )
            else:
                (cls_loss, cls_detailed) = self._compute_full_classification_loss(
                    pred_dict, labels
                )
            detailed_loss.update(cls_detailed)
            detailed_loss["cls_loss"] = (
                cls_loss.item() if cls_loss.numel() == 1 else cls_loss.mean().item()
            )
        ct_brainstructure_dice_loss = torch.tensor(0.0, device=self.device)
        ct_brainstructure_bce_loss = torch.tensor(0.0, device=self.device)
        mr_brainstructure_dice_loss = torch.tensor(0.0, device=self.device)
        mr_brainstructure_bce_loss = torch.tensor(0.0, device=self.device)
        if self.use_brainstructure_seg_loss:
            ct_seg_pred = result.get("ct_seg_pred")
            if (
                ct_seg_pred is not None
                and ct_segs is not None
                and (ct_mask is not None)
                and (ct_seg_mask is not None)
            ):
                (ct_brainstructure_dice_loss, ct_brainstructure_bce_loss) = (
                    self._compute_segmentation_loss(
                        seg_pred=ct_seg_pred,
                        seg_gt=ct_segs,
                        img_mask=ct_mask,
                        seg_mask=ct_seg_mask,
                    )
                )
            mr_seg_pred = result.get("mr_seg_pred")
            if (
                mr_seg_pred is not None
                and mr_segs is not None
                and (mr_mask is not None)
                and (mr_seg_mask is not None)
            ):
                (mr_brainstructure_dice_loss, mr_brainstructure_bce_loss) = (
                    self._compute_segmentation_loss(
                        seg_pred=mr_seg_pred,
                        seg_gt=mr_segs,
                        img_mask=mr_mask,
                        seg_mask=mr_seg_mask,
                    )
                )
        detailed_loss["ct_brainstructure_dice_loss"] = (
            ct_brainstructure_dice_loss.item()
        )
        detailed_loss["ct_brainstructure_bce_loss"] = ct_brainstructure_bce_loss.item()
        detailed_loss["mr_brainstructure_dice_loss"] = (
            mr_brainstructure_dice_loss.item()
        )
        detailed_loss["mr_brainstructure_bce_loss"] = mr_brainstructure_bce_loss.item()
        ct_vessel_dice_loss = torch.tensor(0.0, device=self.device)
        ct_vessel_bce_loss = torch.tensor(0.0, device=self.device)
        mr_vessel_dice_loss = torch.tensor(0.0, device=self.device)
        mr_vessel_bce_loss = torch.tensor(0.0, device=self.device)
        if self.use_vessel_seg_loss:
            ct_vessel_seg_pred = result.get("ct_vessel_seg_pred")
            if ct_vessel_seg_pred is not None and ct_vessel_segs is not None:
                (ct_vessel_dice_loss, ct_vessel_bce_loss) = (
                    self._compute_segmentation_loss(
                        seg_pred=ct_vessel_seg_pred,
                        seg_gt=ct_vessel_segs,
                        img_mask=ct_mask,
                        seg_mask=ct_vessel_seg_mask,
                    )
                )
            mr_vessel_seg_pred = result.get("mr_vessel_seg_pred")
            if mr_vessel_seg_pred is not None and mr_vessel_segs is not None:
                (mr_vessel_dice_loss, mr_vessel_bce_loss) = (
                    self._compute_segmentation_loss(
                        seg_pred=mr_vessel_seg_pred,
                        seg_gt=mr_vessel_segs,
                        img_mask=mr_mask,
                        seg_mask=mr_vessel_seg_mask,
                    )
                )
        detailed_loss["ct_vessel_dice_loss"] = ct_vessel_dice_loss.item()
        detailed_loss["ct_vessel_bce_loss"] = ct_vessel_bce_loss.item()
        detailed_loss["mr_vessel_dice_loss"] = mr_vessel_dice_loss.item()
        detailed_loss["mr_vessel_bce_loss"] = mr_vessel_bce_loss.item()
        ct_clip_loss = torch.tensor(0.0, device=self.device)
        mr_clip_loss = torch.tensor(0.0, device=self.device)
        clip_loss = torch.tensor(0.0, device=self.device)
        if self.use_clip_loss:
            report_sim = result.get("report_sim")
            ct_logits_per_image = result.get("ct_logits_per_image")
            ct_logits_per_text = result.get("ct_logits_per_text")
            if ct_logits_per_image is not None and ct_logits_per_text is not None:
                ct_clip_mask = ct_mask
                if report_mask is not None:
                    ct_clip_mask = ct_mask & report_mask
                ct_clip_loss = self._compute_clip_loss(
                    logits_per_image=ct_logits_per_image,
                    logits_per_text=ct_logits_per_text,
                    report_sim=report_sim,
                    mask=ct_clip_mask,
                    positive_threshold=0.9,
                    tau=0.1,
                )
            mr_logits_per_image = result.get("mr_logits_per_image")
            mr_logits_per_text = result.get("mr_logits_per_text")
            if mr_logits_per_image is not None and mr_logits_per_text is not None:
                mr_clip_mask = mr_mask
                if report_mask is not None:
                    mr_clip_mask = mr_mask & report_mask
                mr_clip_loss = self._compute_clip_loss(
                    logits_per_image=mr_logits_per_image,
                    logits_per_text=mr_logits_per_text,
                    report_sim=report_sim,
                    mask=mr_clip_mask,
                    positive_threshold=0.9,
                    tau=0.1,
                )
            clip_loss = ct_clip_loss + mr_clip_loss
        detailed_loss["ct_clip_loss"] = ct_clip_loss.item()
        detailed_loss["mr_clip_loss"] = mr_clip_loss.item()
        detailed_loss["clip_loss"] = clip_loss.item()
        brainstructure_seg_loss = (
            ct_brainstructure_dice_loss
            + ct_brainstructure_bce_loss
            + mr_brainstructure_dice_loss
            + mr_brainstructure_bce_loss
        )
        vessel_seg_loss = (
            ct_vessel_dice_loss
            + ct_vessel_bce_loss
            + mr_vessel_dice_loss
            + mr_vessel_bce_loss
        )
        total_loss = (
            cls_loss
            + self.brainstructure_seg_loss_weight * brainstructure_seg_loss
            + self.vessel_seg_loss_weight * vessel_seg_loss
            + self.clip_loss_weight * clip_loss
        )
        detailed_loss["total_loss"] = total_loss.item()
        return (total_loss, detailed_loss)

    def _compute_clip_loss(
        self,
        logits_per_image,
        logits_per_text,
        report_sim=None,
        mask=None,
        positive_threshold=0.95,
        tau=0.1,
    ):
        batch_size = logits_per_image.shape[0]
        if mask is not None and (not mask.any()):
            return torch.tensor(0.0, device=self.device)
        if mask is not None:
            valid_indices = mask.nonzero(as_tuple=True)[0]
            num_valid = valid_indices.shape[0]
            if num_valid == 0:
                return torch.tensor(0.0, device=self.device)
            valid_logits_image = logits_per_image[valid_indices][:, valid_indices]
            valid_logits_text = logits_per_text[valid_indices][:, valid_indices]
            if report_sim is not None:
                valid_report_sim = report_sim[valid_indices][:, valid_indices]
            else:
                valid_report_sim = torch.eye(num_valid, device=self.device)
        else:
            valid_logits_image = logits_per_image
            valid_logits_text = logits_per_text
            num_valid = batch_size
            if report_sim is not None:
                valid_report_sim = report_sim
            else:
                valid_report_sim = torch.eye(num_valid, device=self.device)
        pos_mask = (valid_report_sim >= positive_threshold).float()
        loss_i2t = self._multi_positive_infonce(valid_logits_image, pos_mask, tau)
        loss_t2i = self._multi_positive_infonce(valid_logits_text, pos_mask.t(), tau)
        clip_loss = (loss_i2t + loss_t2i) / 2.0
        return clip_loss

    def _multi_positive_infonce(self, pred_sim, pos_mask, tau=0.1):
        scores = pred_sim / tau
        max_score = torch.max(scores, dim=1, keepdim=True)[0].detach()
        scores_sub = scores - max_score
        exp_scores = torch.exp(scores_sub)
        denominator = exp_scores.sum(dim=1, keepdim=True)
        log_probs = scores_sub - torch.log(denominator + 1e-10)
        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1.0)
        log_probs_sum = (log_probs * pos_mask).sum(dim=1)
        loss_per_sample = -(log_probs_sum / pos_count)
        return loss_per_sample.mean()

    def _compute_masked_classification_loss(self, pred_dict, gt_dict, label_mask):
        total_loss = torch.tensor(0.0, device=self.device)
        detailed_loss = {}
        for label, pred in pred_dict.items():
            if label not in gt_dict:
                detailed_loss[label] = None
                continue
            gt = gt_dict[label].to(self.device)
            mask = label_mask[:, self.label_to_idx[label]]
            if mask.sum() > 0:
                bce_loss = self.bce_with_logits(pred, gt)
                if self.use_focal:
                    p = torch.sigmoid(pred)
                    p_t = p * gt + (1 - p) * (1 - gt)
                    focal_weight = (1 - p_t) ** self.focal_gamma
                    loss = focal_weight * bce_loss
                else:
                    loss = bce_loss
                masked_loss = loss[mask]
                label_loss = masked_loss.mean()
                total_loss = total_loss + label_loss
                detailed_loss[label] = label_loss.item()
            else:
                detailed_loss[label] = None
        num_valid = sum((1 for v in detailed_loss.values() if v is not None))
        if num_valid == 0:
            raise ValueError(
                "num_valid is 0! no valid labels for classification loss in this batch"
            )
        total_loss = total_loss / num_valid
        return (total_loss, detailed_loss)

    def _compute_full_classification_loss(
        self, pred_dict, gt_dict, target_class_idx=None
    ):
        device = pred_dict[list(pred_dict.keys())[0]].device
        total_loss = torch.tensor(0.0, device=self.device)
        detailed_loss = {}
        num_valid_labels = 0
        for label, pred in pred_dict.items():
            if label not in gt_dict:
                detailed_loss[label] = None
                continue
            gt = gt_dict[label].to(device)
            bce_loss = self.bce_with_logits(pred, gt)
            if self.use_focal:
                p = torch.sigmoid(pred)
                p_t = p * gt + (1 - p) * (1 - gt)
                focal_weight = (1 - p_t) ** self.focal_gamma
                if self.focal_alpha is not None:
                    alpha_t = self.focal_alpha * gt + (1 - self.focal_alpha) * (1 - gt)
                    loss = alpha_t * focal_weight * bce_loss
                else:
                    loss = focal_weight * bce_loss
            else:
                loss = bce_loss
            label_loss = loss.mean()
            total_loss = total_loss + label_loss
            detailed_loss[label] = label_loss.item()
            num_valid_labels += 1
            if torch.isnan(label_loss) or torch.isinf(label_loss):
                print(f"Warning: {label} loss is nan/inf: pred={pred}, gt={gt}")
        if num_valid_labels > 0:
            total_loss = total_loss / num_valid_labels
        return (total_loss, detailed_loss)

    def _compute_segmentation_loss(self, seg_pred, seg_gt, img_mask, seg_mask):
        valid_mask = img_mask & seg_mask
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            dummy_loss = (seg_pred * 0.0).sum()
            return (dummy_loss, dummy_loss)
        valid_pred = seg_pred[valid_mask]
        valid_gt = seg_gt[valid_mask]
        num_classes = valid_pred.shape[1]
        valid_gt_squeeze = valid_gt.squeeze(1).long()
        valid_gt_onehot = F.one_hot(valid_gt_squeeze, num_classes=num_classes + 1)
        valid_gt_onehot = valid_gt_onehot[..., 1:]
        valid_gt_onehot = valid_gt_onehot.permute(0, 4, 1, 2, 3).float()
        pred_sigmoid = torch.sigmoid(valid_pred)
        dice = self.dice_loss(pred_sigmoid, valid_gt_onehot)
        dice = rearrange(dice, "(b c) -> b c", b=valid_pred.shape[0])
        dice_loss = dice.mean()
        bce = self.seg_bce_loss(valid_pred, valid_gt_onehot)
        bce = bce.mean(dim=(2, 3, 4))
        bce_loss = bce.mean()
        return (dice_loss, bce_loss)
