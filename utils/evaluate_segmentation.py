import numpy as np
from sklearn.metrics import confusion_matrix

def compute_iou(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt > 0, mask_pred > 0)
    union = np.logical_or(mask_gt > 0, mask_pred > 0)
    return intersection.sum() / (union.sum() + 1e-8)

def compute_dice(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt > 0, mask_pred > 0).sum()
    total = (mask_gt > 0).sum() + (mask_pred > 0).sum()
    return 2 * intersection / (total + 1e-8)

def compute_pixel_accuracy(mask_gt, mask_pred):
    gt_bin = (mask_gt > 0).astype(np.uint8).flatten()
    pred_bin = (mask_pred > 0).astype(np.uint8).flatten()
    tn, fp, fn, tp = confusion_matrix(gt_bin, pred_bin, labels=[0, 1]).ravel()
    return {
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8)
    }

def evaluate_masks(mask_gt, mask_pred):
    return {
        "iou": compute_iou(mask_gt, mask_pred),
        "dice": compute_dice(mask_gt, mask_pred),
        **compute_pixel_accuracy(mask_gt, mask_pred)
    }
