import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from dataset import get_dataloaders, MedicalImageDataset
from model import ViTUNETRSegmentationModel  # Use original model class
from monai.transforms import AsDiscrete, Compose

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for saving results
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy Loss for better segmentation performance."""
    def __init__(self, lambda_dice=0.8, lambda_ce=0.2, class_weights=None, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.smooth = smooth
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.clone().detach().to(DEVICE)
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        else:
            self.class_weights = None
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        intersection = (tumor_probs * targets_float).sum()
        union = tumor_probs.sum() + targets_float.sum()
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        if self.class_weights is not None:
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs.squeeze(1), targets_float, 
                pos_weight=self.class_weights[1] / self.class_weights[0]
            )
        else:
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs.squeeze(1), targets_float
            )
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets = targets.float()
        intersection = (tumor_probs * targets).sum()
        union = tumor_probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets = targets.float()
        pt = tumor_probs * targets + (1 - tumor_probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs.squeeze(1), targets, reduction='none')
        focal_loss = self.alpha * focal_weight * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss: Binary CrossEntropy + Dice Loss."""
    def __init__(self, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs.squeeze(1), targets.float())
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = bce_loss + self.dice_weight * dice_loss
        return total_loss

class BalancedLoss(nn.Module):
    """Balanced loss function that handles class imbalance without extreme penalties."""
    def __init__(self, dice_weight=0.3):
        super(BalancedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs.squeeze(1), targets.float())
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = bce_loss + self.dice_weight * dice_loss
        return total_loss

class TumorFocusedLoss(nn.Module):
    """Loss function specifically designed to force tumor detection."""
    def __init__(self, tumor_weight=1.0):
        super(TumorFocusedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.tumor_weight = tumor_weight
    
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs.squeeze(1), targets.float())
        dice_loss = self.dice_loss(inputs, targets)
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        false_negative_penalty = torch.mean(targets_float * (1 - tumor_probs))
        false_positive_penalty = torch.mean((1 - targets_float) * tumor_probs * 0.1)
        total_loss = bce_loss + dice_loss + self.tumor_weight * false_negative_penalty + false_positive_penalty
        return total_loss

class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and Dice Loss for better tumor detection."""
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        pt = tumor_probs * targets_float + (1 - tumor_probs) * (1 - targets_float)
        focal_weight = (1 - pt) ** self.gamma
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs.squeeze(1), targets_float, reduction='none')
        focal_loss = self.alpha * focal_weight * bce_loss
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = focal_loss.mean() + self.dice_weight * dice_loss
        return total_loss

class TumorDetectionLoss(nn.Module):
    """Loss function specifically designed for extreme class imbalance in tumor detection."""
    def __init__(self, tumor_weight=15.0):
        super(TumorDetectionLoss, self).__init__()
        self.tumor_weight = tumor_weight
        self.dice_losses = DiceLoss()
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs.squeeze(1), targets_float, reduction='none'
        )
        weighted_loss = bce_loss * (targets_float * self.tumor_weight + (1 - targets_float))
        dice_loss = self.dice_losses(inputs, targets)
        total_loss = weighted_loss.mean() + 0.3 * dice_loss
        return total_loss
    
    def dice_loss(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        intersection = (tumor_probs * targets_float).sum()
        union = tumor_probs.sum() + targets_float.sum()
        dice = (2.0 * intersection) / (union + 1e-6)
        return 1.0 - dice

class ExtremeImbalanceLoss(nn.Module):
    """Loss function designed for extreme class imbalance scenarios."""
    def __init__(self, tumor_weight=20.0, focal_gamma=2.0):
        super(ExtremeImbalanceLoss, self).__init__()
        self.tumor_weight = tumor_weight
        self.focal_gamma = focal_gamma
        self.dice_losses = DiceLoss()
    
    def forward(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        pt = tumor_probs * targets_float + (1 - tumor_probs) * (1 - targets_float)
        focal_weight = (1 - pt) ** self.focal_gamma
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs.squeeze(1), targets_float, reduction='none'
        )
        focal_bce = focal_weight * bce_loss
        weighted_loss = focal_bce * (targets_float * self.tumor_weight + (1 - targets_float))
        dice_loss = self.dice_losses(inputs, targets)
        total_loss = weighted_loss.mean() + 0.5 * dice_loss
        return total_loss
    
    def dice_loss(self, inputs, targets):
        tumor_probs = torch.sigmoid(inputs).squeeze(1)
        targets_float = targets.float()
        intersection = (tumor_probs * targets_float).sum()
        union = tumor_probs.sum() + targets_float.sum()
        dice = (2.0 * intersection) / (union + 1e-6)
        return 1.0 - dice

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    total_dice = 0
    total_tumor_predictions = 0
    total_tumor_targets = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)   # (B, 1, 96, 96, 96)
            labels = labels.to(device)  # (B, 96, 96, 96)
            outputs = model(imgs)  # (B, 1, 96, 96, 96)
            loss = criterion(outputs, labels)
            batch_metrics = calculate_metrics_with_threshold(outputs, labels, threshold=0.3)
            tumor_probs = torch.sigmoid(outputs).squeeze(1)
            predictions = (tumor_probs > 0.3).float()
            batch_tumor_predictions = predictions.sum().item()
            batch_tumor_targets = (labels == 1).sum().item()
            batch_pixels = labels.numel()
            total_loss += loss.item()
            total_dice += batch_metrics['dice_score']
            total_tumor_predictions += batch_tumor_predictions
            total_tumor_targets += batch_tumor_targets
            total_pixels += batch_pixels
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0
    tumor_pred_rate = 100 * total_tumor_predictions / total_pixels if total_pixels > 0 else 0
    tumor_target_rate = 100 * total_tumor_targets / total_pixels if total_pixels > 0 else 0
    
    return avg_loss, {
        'dice_score': avg_dice,
        'tumor_pred_rate': tumor_pred_rate,
        'tumor_target_rate': tumor_target_rate
    }

def calculate_metrics_with_threshold(outputs, targets, threshold=0.3):
    """Calculate Dice score with a lower threshold to encourage tumor detection."""
    tumor_probs = torch.sigmoid(outputs).squeeze(1)
    predictions = (tumor_probs > threshold).float()
    targets_float = targets.float()
    num_predictions = predictions.numel()
    num_targets = targets_float.numel()
    tumor_predictions = predictions.sum().item()
    tumor_targets = targets_float.sum().item()
    pred_percent = 100 * tumor_predictions / num_predictions
    target_percent = 100 * tumor_targets / num_targets
    if hasattr(calculate_metrics_with_threshold, 'debug_count'):
        calculate_metrics_with_threshold.debug_count += 1
    else:
        calculate_metrics_with_threshold.debug_count = 0
    if calculate_metrics_with_threshold.debug_count < 3:
        print(f"[DEBUG] Threshold={threshold} - Predictions: {tumor_predictions}/{num_predictions} ({pred_percent:.2f}%)")
        print(f"[DEBUG] Threshold={threshold} - Targets: {tumor_targets}/{num_targets} ({target_percent:.2f}%)")
        print(f"[DEBUG] Threshold={threshold} - Tumor probabilities range: {tumor_probs.min().item():.4f} to {tumor_probs.max().item():.4f}")
    intersection = (predictions * targets_float).sum()
    union = predictions.sum() + targets_float.sum()
    dice_score = (2.0 * intersection) / (union + 1e-6)
    return {
        'dice_score': dice_score.item(),
        'tumor_pred_rate': pred_percent,
        'tumor_target_rate': target_percent
    }

def calculate_metrics(outputs, targets):
    """Calculate Dice score."""
    tumor_probs = torch.sigmoid(outputs).squeeze(1)
    predictions = (tumor_probs > 0.5).float()
    targets = targets.float()
    num_predictions = predictions.numel()
    num_targets = targets.numel()
    tumor_predictions = predictions.sum().item()
    tumor_targets = targets.sum().item()
    if hasattr(calculate_metrics, 'debug_count'):
        calculate_metrics.debug_count += 1
    else:
        calculate_metrics.debug_count = 0
    if calculate_metrics.debug_count < 3:
        pred_percent = 100 * tumor_predictions / num_predictions
        target_percent = 100 * tumor_targets / num_targets
        print(f"[DEBUG] Predictions: {tumor_predictions}/{num_predictions} ({pred_percent:.2f}%)")
        print(f"[DEBUG] Targets: {tumor_targets}/{num_targets} ({target_percent:.2f}%)")
        print(f"[DEBUG] Class distribution in predictions: {torch.bincount(predictions.long().flatten())}")
        print(f"[DEBUG] Class distribution in targets: {torch.bincount(targets.long().flatten())}")
        tumor_logits = outputs.squeeze(1).mean().item()
        tumor_probs_mean = tumor_probs.mean().item()
        print(f"[DEBUG] Raw logits - Tumor: {tumor_logits:.4f}")
        print(f"[DEBUG] Probabilities - Tumor: {tumor_probs_mean:.4f}")
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    dice_score = (2.0 * intersection) / (union + 1e-6)
    return {
        'dice_score': dice_score.item()
    }

def freeze_encoder(model, freeze=True):
    """Freeze or unfreeze the ViT encoder."""
    for param in model.unetr.vit.parameters():
        param.requires_grad = not freeze
    print(f"Encoder {'frozen' if freeze else 'unfrozen'}")

def unfreeze_encoder_layers(model, num_layers_to_unfreeze):
    """Gradually unfreeze the last N layers of the ViT encoder."""
    encoder_params = list(model.unetr.vit.named_parameters())
    for name, param in encoder_params:
        param.requires_grad = False
    total_layers = 12
    layers_to_unfreeze = list(range(total_layers - num_layers_to_unfreeze, total_layers))
    for name, param in encoder_params:
        for layer_idx in layers_to_unfreeze:
            if f'layers.{layer_idx}' in name:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
    print(f"Unfrozen last {num_layers_to_unfreeze} encoder layers")

def get_trainable_params_count(model):
    """Get the number of trainable parameters."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return sum(p.numel() for p in trainable_params)

def create_optimizer_with_differential_lrs(model, backbone_lr=1e-5, decoder_lr=1e-4, 
                                          backbone_weight_decay=1e-5, decoder_weight_decay=1e-4):
    """Create optimizer with differential learning rates for backbone vs decoder components."""
    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'unetr.vit' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    print(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}")
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay},
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': decoder_weight_decay}
    ])
    print(f"Optimizer created with backbone_lr={backbone_lr}, decoder_lr={decoder_lr}")
    print(f"Backbone weight decay: {backbone_weight_decay}, Decoder weight decay: {decoder_weight_decay}")
    return optimizer

def create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6):
    """Create cosine annealing scheduler with warm restarts."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    print(f"Scheduler created: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    return scheduler

def plot_training_metrics(training_history, save_path='plots/training_metrics.png'):
    """Plot training and validation metrics in real-time."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = training_history['epochs']
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, training_history['val_dice'], 'g-', label='Validation Dice', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax3.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax4.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Validation Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics(training_history, save_path='plots/combined_metrics.png'):
    """Plot combined metrics in a single figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs = training_history['epochs']
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, training_history['val_dice'], 'g-', label='Validation Dice', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train():
    print("Testing loss functions...")
    test_inputs = torch.randn(2, 1, 96, 96, 96).to(DEVICE)
    test_targets = torch.zeros(2, 96, 96, 96, dtype=torch.long).to(DEVICE)
    test_targets[0, 40:60, 40:60, 40:60] = 1
    test_class_weights = torch.tensor([1.0, 5.0]).to(DEVICE)
    ce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    tumor_focused_loss = TumorFocusedLoss()
    focal_dice_loss = FocalDiceLoss(alpha=1, gamma=2, dice_weight=0.5)
    tumor_detection_loss = TumorDetectionLoss(tumor_weight=10.0)
    extreme_imbalance_loss = ExtremeImbalanceLoss(tumor_weight=20.0, focal_gamma=2.0)
    dice_ce_loss = DiceCELoss(lambda_dice=0.8, lambda_ce=0.2, class_weights=test_class_weights)
    print(f"BCE Loss: {ce_loss(test_inputs.squeeze(1), test_targets.float()):.4f}")
    print(f"Dice Loss: {dice_loss(test_inputs, test_targets):.4f}")
    print(f"Combined Loss: {combined_loss(test_inputs, test_targets):.4f}")
    print(f"Tumor Focused Loss: {tumor_focused_loss(test_inputs, test_targets):.4f}")
    print(f"FocalDiceLoss: {focal_dice_loss(test_inputs, test_targets):.4f}")
    print(f"TumorDetectionLoss: {tumor_detection_loss(test_inputs, test_targets):.4f}")
    print(f"ExtremeImbalanceLoss: {extreme_imbalance_loss(test_inputs, test_targets):.4f}")
    print(f"DiceCELoss: {dice_ce_loss(test_inputs, test_targets):.4f}")

    print("Loading dataset...")
    data_dir = '/home/ubuntu/projects/finetune_npy/Task002_FOMO2'
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        train_ratio=0.8,
        batch_size=8,
        num_workers=4
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} training patches, {len(val_loader.dataset)} validation patches")
    print(f"Training batch size: 8 (patches)")
    print(f"Each patch: 96x96x96 voxels")

    simclr_ckpt_path = '/home/ubuntu/projects/nasrin_brainaic/checkpoints/BrainIAC.ckpt'
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=simclr_ckpt_path,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1
    ).to(DEVICE)

    training_phases = [
        {
            'name': 'Phase 1: Decoder Only',
            'start_epoch': 0,
            'end_epoch': 200,
            'encoder_frozen': True,
            'backbone_lr': 0.0,
            'decoder_lr': 1e-4,
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Freeze encoder, train decoder only on high-resolution patches'
        },
        {
            'name': 'Phase 2: Last 2 Encoder Layers',
            'start_epoch': 200,
            'end_epoch': 600,
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 2,
            'backbone_lr': 5e-6,
            'decoder_lr': 1e-4,
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Unfreeze last 2 encoder layers with conservative LR'
        },
        {
            'name': 'Phase 3: Last 4 Encoder Layers',
            'start_epoch': 600,
            'end_epoch': 1200,
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 4,
            'backbone_lr': 1e-6,
            'decoder_lr': 1e-4,
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Unfreeze last 4 encoder layers with conservative LR'
        },
        {
            'name': 'Phase 4: Full Fine-tuning',
            'start_epoch': 1200,
            'end_epoch': 2000,
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 12,
            'backbone_lr': 5e-7,
            'decoder_lr': 1e-4,
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Fine-tune entire model with minimal backbone LR on high-res patches'
        }
    ]

    current_phase_idx = 0
    current_phase = training_phases[current_phase_idx]
    print(f"\n{'='*80}")
    print(f"STARTING {current_phase['name']}")
    print(f"{'='*80}")
    print(f"Description: {current_phase['description']}")

    if current_phase['encoder_frozen']:
        freeze_encoder(model, freeze=True)
    else:
        unfreeze_encoder_layers(model, current_phase['encoder_layers_unfrozen'])

    optimizer = create_optimizer_with_differential_lrs(
        model, 
        backbone_lr=current_phase['backbone_lr'], 
        decoder_lr=current_phase['decoder_lr'],
        backbone_weight_decay=current_phase['backbone_weight_decay'],
        decoder_weight_decay=current_phase['decoder_weight_decay']
    )

    scheduler = create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    criterion = DiceCELoss(
        lambda_dice=0.8, 
        lambda_ce=0.2, 
        class_weights=[0.1, 0.9]
    )
    print(f"Using DiceCELoss with lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9]")

    trainable_params = get_trainable_params_count(model)
    print(f"Initial trainable parameters: {trainable_params:,}")

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'epochs': [],
        'phases': []
    }

    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0

    model.train()
    print(f"\n{'='*80}")
    print(f"STARTING PATCH-BASED TRAINING FOR 2000 EPOCHS")
    print(f"Using 96x96x96 patches from original volumes")
    print(f"Loss: DiceCELoss (lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9])")
    print(f"Optimizer: AdamW with differential learning rates (backbone: 1e-5, decoder: 1e-4)")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6)")
    print(f"Checkpoints will be saved every 100 epochs")
    print(f"Best model will be saved based on validation loss")
    print(f"Plots will be updated after each epoch")
    print(f"{'='*80}\n")

    for epoch in range(2000):
        if epoch >= current_phase['end_epoch'] and current_phase_idx < len(training_phases) - 1:
            current_phase_idx += 1
            current_phase = training_phases[current_phase_idx]
            print(f"\n{'='*80}")
            print(f"TRANSITIONING TO {current_phase['name']}")
            print(f"{'='*80}")
            print(f"Description: {current_phase['description']}")
            if current_phase['encoder_frozen']:
                freeze_encoder(model, freeze=True)
            else:
                unfreeze_encoder_layers(model, current_phase['encoder_layers_unfrozen'])
            optimizer = create_optimizer_with_differential_lrs(
                model, 
                backbone_lr=current_phase['backbone_lr'], 
                decoder_lr=current_phase['decoder_lr'],
                backbone_weight_decay=current_phase['backbone_weight_decay'],
                decoder_weight_decay=current_phase['decoder_weight_decay']
            )
            scheduler = create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
            trainable_params = get_trainable_params_count(model)
            print(f"Updated trainable parameters: {trainable_params:,}")
            training_history['phases'].append({
                'epoch': epoch,
                'phase_name': current_phase['name'],
                'trainable_params': trainable_params
            })

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/2000 - {current_phase['name']}")
        total_train_loss = 0
        num_train_batches = 0

        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if num_train_batches < 5:
                with torch.no_grad():
                    tumor_probs = torch.sigmoid(outputs).squeeze(1)
                    predictions = (tumor_probs > 0.5).float()
                    tumor_pred_rate = predictions.mean().item()
                    tumor_target_rate = labels.float().mean().item()
                    print(f"[DEBUG] Batch {num_train_batches+1}: Tumor pred rate: {tumor_pred_rate:.4f}, Target rate: {tumor_target_rate:.4f}")
            total_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_dice'].append(val_metrics['dice_score'])
        training_history['epochs'].append(epoch + 1)
        current_backbone_lr = optimizer.param_groups[0]['lr']
        current_decoder_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1}/2000: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {val_metrics['dice_score']:.4f}")
        print(f"  Val Tumor Pred Rate: {val_metrics['tumor_pred_rate']:.2f}%, Target Rate: {val_metrics['tumor_target_rate']:.2f}%")
        print(f"  Current Phase: {current_phase['name']}")
        print(f"  Current LRs - Backbone: {current_backbone_lr:.2e}, Decoder: {current_decoder_lr:.2e}")
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            best_model_path = 'checkpoints/segmentation_best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'training_history': training_history,
                'current_phase': current_phase,
                'current_phase_idx': current_phase_idx
            }, best_model_path)
            print(f"  ✓ Saved best model to {best_model_path}")
        if val_metrics['dice_score'] > best_val_dice:
            best_val_dice = val_metrics['dice_score']
            print(f"  ✓ New best Dice score: {best_val_dice:.4f}")
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f'checkpoints/segmentation_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'training_history': training_history,
                'current_phase': current_phase,
                'current_phase_idx': current_phase_idx
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
        plot_training_metrics(training_history, 'plots/training_metrics.png')
        plot_combined_metrics(training_history, 'plots/combined_metrics.png')
        if (epoch + 1) % 100 == 0:
            print(f"\n{'='*80}")
            print(f"PROGRESS SUMMARY - Epoch {epoch+1}/2000")
            print(f"{'='*80}")
            print(f"Current Phase: {current_phase['name']}")
            print(f"Current Backbone LR: {current_backbone_lr:.2e}")
            print(f"Current Decoder LR: {current_decoder_lr:.2e}")
            print(f"Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch})")
            print(f"Best Dice Score: {best_val_dice:.4f} (epoch {best_epoch})")
            print(f"Current Tumor Pred Rate: {val_metrics['tumor_pred_rate']:.2f}%")
            print(f"Target Tumor Rate: {val_metrics['tumor_target_rate']:.2f}%")
            print(f"Training Progress: {epoch+1}/2000 epochs completed ({((epoch+1)/2000)*100:.1f}%)")
            print(f"Loss Function: DiceCELoss (lambda_dice=0.8, lambda_ce=0.2)")
            print(f"Optimizer: AdamW with differential learning rates")
            print(f"Scheduler: CosineAnnealingWarmRestarts")
            print(f"Plots updated: plots/training_metrics.png, plots/combined_metrics.png")
            print(f"{'='*80}\n")

    final_model_path = 'checkpoints/segmentation_final_model_2000epochs_patches.pth'
    torch.save({
        'epoch': 2000,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
        'training_history': training_history,
        'final_phase': current_phase
    }, final_model_path)
    results_path = 'results/segmentation_training_results_2000epochs_patches.json'
    with open(results_path, 'w') as f:
        json.dump({
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'final_val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'best_epoch': best_epoch,
            'training_history': training_history,
            'training_phases': training_phases,
            'final_phase': current_phase
        }, f, indent=2)
    plot_training_metrics(training_history, 'plots/final_training_metrics_patches.png')
    plot_combined_metrics(training_history, 'plots/final_combined_metrics_patches.png')
    print(f"\n{'='*80}")
    print(f"PATCH-BASED TRAINING COMPLETED - 2000 EPOCHS FINISHED!")
    print(f"{'='*80}")
    print(f"✓ Best model saved to: checkpoints/segmentation_best_model.pth")
    print(f"✓ Final model saved to: {final_model_path}")
    print(f"✓ Training results saved to: {results_path}")
    print(f"✓ Final plots saved to: plots/final_training_metrics_patches.png, plots/final_combined_metrics_patches.png")
    print(f"✓ Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"✓ Best validation dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"✓ Final validation loss: {avg_val_loss:.4f}")
    print(f"✓ Final validation dice: {val_metrics['dice_score']:.4f}")
    print(f"✓ Final training phase: {current_phase['name']}")
    print(f"✓ Training method: 96x96x96 patches from original volumes")
    print(f"✓ Loss function: DiceCELoss (lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9])")
    print(f"✓ Optimizer: AdamW with differential learning rates")
    print(f"✓ Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6)")
    print(f"✓ Checkpoints saved every 100 epochs")
    print(f"✓ Best model saved based on validation loss")
    print(f"{'='*80}")

if __name__ == "__main__":
    train()