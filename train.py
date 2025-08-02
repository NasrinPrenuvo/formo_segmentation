import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

from dataset import BrainVolumeDataset
from model import ViTUNETRSegmentationModel  # Updated model import
from dataset import BrainVolumePatchesDataset # Added patch-based dataset import

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
                self.class_weights = class_weights.clone().detach()
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W, D) - logits for tumor
        # targets: (N, H, W, D) - labels (0 or 1)
        
        # Apply sigmoid to get tumor probability
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets_float = targets.float()
        
        # Dice Loss component
        intersection = (tumor_probs * targets_float).sum()
        union = tumor_probs.sum() + targets_float.sum()
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        
        # Cross-Entropy Loss component
        if self.class_weights is not None:
            # Apply class weights to BCE loss
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs.squeeze(1), targets_float, 
                pos_weight=self.class_weights[1] / self.class_weights[0]  # tumor_weight / background_weight
            )
        else:
            ce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs.squeeze(1), targets_float
            )
        
        # Combine losses
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        
        return total_loss

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W, D) - logits for tumor
        # targets: (N, H, W, D) - labels (0 or 1)
        
        # Apply sigmoid to get tumor probability
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        
        # Convert targets to float
        targets = targets.float()
        
        # Calculate intersection and union
        intersection = (tumor_probs * targets).sum()
        union = tumor_probs.sum() + targets.sum()
        
        # Dice coefficient
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
        # inputs: (N, 1, H, W, D) - logits for tumor
        # targets: (N, H, W, D) - labels (0 or 1)
        
        # Apply sigmoid to get tumor probability
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets = targets.float()
        
        # Calculate focal loss
        pt = tumor_probs * targets + (1 - tumor_probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Binary cross entropy term
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs.squeeze(1), targets, reduction='none')
        
        # Apply focal weight
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
        
        # Simple weighted combination
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
        # Standard losses
        bce_loss = self.bce_loss(inputs.squeeze(1), targets.float())
        dice_loss = self.dice_loss(inputs, targets)
        
        # More balanced penalty for missing tumors (false negatives)
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets_float = targets.float()
        
        # Penalty: when target is tumor (1) but prediction is low
        # Use a more moderate penalty that doesn't explode
        false_negative_penalty = torch.mean(
            targets_float * (1 - tumor_probs)  # Linear penalty instead of exponential
        )
        
        # Penalty: when prediction is high but target is background (0) - but with lower weight
        false_positive_penalty = torch.mean(
            (1 - targets_float) * tumor_probs * 0.1  # Linear penalty instead of exponential
        )
        
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
        # Focal Loss component
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets_float = targets.float()
        
        # Calculate focal loss
        pt = tumor_probs * targets_float + (1 - tumor_probs) * (1 - targets_float)
        focal_weight = (1 - pt) ** self.gamma
        
        # Binary cross entropy term
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs.squeeze(1), targets_float, reduction='none')
        focal_loss = self.alpha * focal_weight * bce_loss
        
        # Dice Loss component
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combine losses
        total_loss = focal_loss.mean() + self.dice_weight * dice_loss
        return total_loss

class TumorDetectionLoss(nn.Module):
    """Loss function specifically designed for extreme class imbalance in tumor detection."""
    def __init__(self, tumor_weight=15.0):
        super(TumorDetectionLoss, self).__init__()
        self.tumor_weight = tumor_weight
    
    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W, D) - logits for tumor
        # targets: (N, H, W, D) - labels
        
        # Apply sigmoid to get tumor probability
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets_float = targets.float()
        
        # Binary cross entropy with logits for tumor class
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs.squeeze(1), targets_float, reduction='none'
        )
        
        # Weight the loss heavily for tumor pixels
        weighted_loss = bce_loss * (targets_float * self.tumor_weight + (1 - targets_float))
        
        # Add Dice loss component
        dice_loss = self.dice_loss(inputs, targets)
        
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
    
    def forward(self, inputs, targets):
        # inputs: (N, 1, H, W, D) - logits for tumor
        # targets: (N, H, W, D) - labels
        
        # Apply sigmoid to get tumor probability
        tumor_probs = torch.sigmoid(inputs).squeeze(1)  # (N, H, W, D)
        targets_float = targets.float()
        
        # Focal loss component to handle class imbalance
        pt = tumor_probs * targets_float + (1 - tumor_probs) * (1 - targets_float)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs.squeeze(1), targets_float, reduction='none'
        )
        
        # Apply focal weight and tumor weighting
        focal_bce = focal_weight * bce_loss
        weighted_loss = focal_bce * (targets_float * self.tumor_weight + (1 - targets_float))
        
        # Dice loss for better overlap
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combine losses
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
    
    # Metrics tracking
    total_dice = 0
    total_tumor_predictions = 0
    total_tumor_targets = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            imgs = batch['image'].to(device)   # (B, 1, 96, 96, 96) - resampled volumes
            labels = batch['label'].to(device)  # (B, 96, 96, 96) - volume labels
            
            outputs = model(imgs)  # (B, num_classes, 96, 96, 96)
            
            loss = criterion(outputs, labels)
            
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics_with_threshold(outputs, labels, threshold=0.3)
            
            # Track tumor prediction statistics
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
    
    # Calculate overall tumor prediction rate
    tumor_pred_rate = 100 * total_tumor_predictions / total_pixels if total_pixels > 0 else 0
    tumor_target_rate = 100 * total_tumor_targets / total_pixels if total_pixels > 0 else 0
    
    return avg_loss, {
        'dice_score': avg_dice,
        'tumor_pred_rate': tumor_pred_rate,
        'tumor_target_rate': tumor_target_rate
    }

def calculate_metrics_with_threshold(outputs, targets, threshold=0.3):
    """Calculate Dice score with a lower threshold to encourage tumor detection."""
    # outputs: (N, 1, H, W, D) - logits for tumor
    # targets: (N, H, W, D) - labels
    
    # Get predictions with lower threshold
    tumor_probs = torch.sigmoid(outputs).squeeze(1)  # (N, H, W, D)
    
    # Use lower threshold for tumor prediction
    predictions = (tumor_probs > threshold).float()
    targets_float = targets.float()
    
    # Debug information
    num_predictions = predictions.numel()
    num_targets = targets_float.numel()
    tumor_predictions = predictions.sum().item()
    tumor_targets = targets_float.sum().item()
    
    # Calculate percentages (always needed for return)
    pred_percent = 100 * tumor_predictions / num_predictions
    target_percent = 100 * tumor_targets / num_targets
    
    # Print debug info for first few batches
    if hasattr(calculate_metrics_with_threshold, 'debug_count'):
        calculate_metrics_with_threshold.debug_count += 1
    else:
        calculate_metrics_with_threshold.debug_count = 0
    
    if calculate_metrics_with_threshold.debug_count < 3:  # Only print for first 3 calls
        print(f"[DEBUG] Threshold={threshold} - Predictions: {tumor_predictions}/{num_predictions} ({pred_percent:.2f}%)")
        print(f"[DEBUG] Threshold={threshold} - Targets: {tumor_targets}/{num_targets} ({target_percent:.2f}%)")
        print(f"[DEBUG] Threshold={threshold} - Tumor probabilities range: {tumor_probs.min().item():.4f} to {tumor_probs.max().item():.4f}")
    
    # Calculate Dice score for tumor class
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
    # outputs: (N, 1, H, W, D) - logits for tumor
    # targets: (N, H, W, D) - labels
    
    # Get predictions
    tumor_probs = torch.sigmoid(outputs).squeeze(1)  # (N, H, W, D)
    predictions = (tumor_probs > 0.5).float()  # (N, H, W, D)
    
    # Convert to float for calculations
    targets = targets.float()
    
    # Debug information
    num_predictions = predictions.numel()
    num_targets = targets.numel()
    tumor_predictions = predictions.sum().item()
    tumor_targets = targets.sum().item()
    
    # Print debug info for first few batches
    if hasattr(calculate_metrics, 'debug_count'):
        calculate_metrics.debug_count += 1
    else:
        calculate_metrics.debug_count = 0
    
    if calculate_metrics.debug_count < 3:  # Only print for first 3 calls
        pred_percent = 100 * tumor_predictions / num_predictions
        target_percent = 100 * tumor_targets / num_targets
        print(f"[DEBUG] Predictions: {tumor_predictions}/{num_predictions} ({pred_percent:.2f}%)")
        print(f"[DEBUG] Targets: {tumor_targets}/{num_targets} ({target_percent:.2f}%)")
        print(f"[DEBUG] Class distribution in predictions: {torch.bincount(predictions.long().flatten())}")
        print(f"[DEBUG] Class distribution in targets: {torch.bincount(targets.long().flatten())}")
        
        # Check raw logits and probabilities
        tumor_logits = outputs.squeeze(1).mean().item()
        tumor_probs_mean = tumor_probs.mean().item()
        
        print(f"[DEBUG] Raw logits - Tumor: {tumor_logits:.4f}")
        print(f"[DEBUG] Probabilities - Tumor: {tumor_probs_mean:.4f}")
    
    # Calculate Dice score for tumor class
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    dice_score = (2.0 * intersection) / (union + 1e-6)
    
    return {
        'dice_score': dice_score.item()
    }

def calculate_class_weights(dataset):
    """Calculate class weights based on dataset statistics."""
    print("Calculating class weights from dataset...")
    
    total_pixels = 0
    tumor_pixels = 0
    
    # Sample a few batches to estimate class distribution
    sample_size = min(10, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in sample_indices:
        sample = dataset[idx]
        # Volume-based dataset
        label = sample['label']  # (96, 96, 96)
        total_pixels += label.numel()
        tumor_pixels += (label == 1).sum().item()
    
    background_pixels = total_pixels - tumor_pixels
    
    # Calculate weights (inverse frequency)
    background_weight = 1.0
    tumor_weight = background_pixels / tumor_pixels if tumor_pixels > 0 else 1.0
    
    # Cap the tumor weight to avoid extreme values
    tumor_weight = min(tumor_weight, 10.0)
    
    print(f"Class distribution: Background={background_pixels}, Tumor={tumor_pixels}")
    print(f"Class weights: Background={background_weight:.3f}, Tumor={tumor_weight:.3f}")
    
    return torch.tensor([background_weight, tumor_weight], dtype=torch.float32)

def analyze_dataset_tumor_distribution(dataset):
    """Analyze tumor distribution in the dataset."""
    print("Analyzing tumor distribution in dataset...")
    
    total_samples = len(dataset)
    tumor_positive_samples = 0
    total_tumor_pixels = 0
    total_pixels = 0
    
    for i in range(min(100, total_samples)):  # Sample first 100 for analysis
        sample = dataset[i]
        # Volume-based dataset
        label = sample['label']  # (96, 96, 96)
        tumor_pixels = (label == 1).sum().item()
        total_pixels += label.numel()
        total_tumor_pixels += tumor_pixels
        if tumor_pixels > 0:
            tumor_positive_samples += 1
    
    tumor_ratio = total_tumor_pixels / total_pixels if total_pixels > 0 else 0
    positive_ratio = tumor_positive_samples / min(100, total_samples)
    
    print(f"Tumor pixel ratio: {tumor_ratio:.6f} ({tumor_ratio*100:.4f}%)")
    print(f"Tumor-positive samples: {tumor_positive_samples}/{min(100, total_samples)} ({positive_ratio*100:.2f}%)")
    
    return {
        'tumor_ratio': tumor_ratio,
        'positive_ratio': positive_ratio,
        'total_tumor_pixels': total_tumor_pixels,
        'total_pixels': total_pixels
    }

def plot_training_metrics(training_history, save_path='plots/training_metrics.png'):
    """Plot training and validation metrics in real-time."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = training_history['epochs']
    
    # Plot training and validation loss
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation dice score
    ax2.plot(epochs, training_history['val_dice'], 'g-', label='Validation Dice', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot training loss only (for better visibility)
    ax3.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot validation loss only (for better visibility)
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
    
    # Combined loss plot
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice score plot
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

def freeze_encoder(model, freeze=True):
    """Freeze or unfreeze the ViT encoder."""
    for param in model.unetr.vit.parameters():
        param.requires_grad = not freeze
    print(f"Encoder {'frozen' if freeze else 'unfrozen'}")

def unfreeze_encoder_layers(model, num_layers_to_unfreeze):
    """Gradually unfreeze the last N layers of the ViT encoder."""
    # Get all encoder parameters
    encoder_params = list(model.unetr.vit.named_parameters())
    
    # Freeze all encoder layers first
    for name, param in encoder_params:
        param.requires_grad = False
    
    # Unfreeze the last N layers
    total_layers = 12  # Assuming 12 transformer layers in ViT
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
    # Separate parameters for backbone (ViT) vs decoder components
    backbone_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'unetr.vit' in name:  # ViT backbone parameters
            backbone_params.append(param)
        else:  # Decoder/head parameters
            decoder_params.append(param)
    
    print(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}")
    
    # Use significantly lower learning rate for pretrained backbone
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': backbone_weight_decay},  # Lower LR for pretrained
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': decoder_weight_decay}      # Higher LR for decoder
    ])
    
    print(f"Optimizer created with backbone_lr={backbone_lr}, decoder_lr={decoder_lr}")
    print(f"Backbone weight decay: {backbone_weight_decay}, Decoder weight decay: {decoder_weight_decay}")
    return optimizer

def create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6):
    """Create cosine annealing scheduler with warm restarts."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0,  # Restart every N epochs
        T_mult=T_mult,  # Multiply restart period
        eta_min=eta_min  # Minimum learning rate
    )
    
    print(f"Scheduler created: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    return scheduler

def create_optimizer_with_different_lrs(model, encoder_lr=1e-5, decoder_lr=1e-4):
    """Create optimizer with different learning rates for encoder and decoder."""
    # Get encoder parameters by name
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'unetr.vit' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ])
    
    print(f"Optimizer created with encoder_lr={encoder_lr}, decoder_lr={decoder_lr}")
    print(f"Encoder parameters: {len(encoder_params)}, Decoder parameters: {len(decoder_params)}")
    return optimizer

def train():
    # Test the loss functions to ensure they work correctly
    print("Testing loss functions...")
    test_inputs = torch.randn(2, 1, 96, 96, 96)  # 2 volumes, 1 channel for tumor
    test_targets = torch.zeros(2, 96, 96, 96, dtype=torch.long)
    test_targets[0, 40:60, 40:60, 40:60] = 1  # Add some tumor in first volume
    
    test_class_weights = torch.tensor([1.0, 5.0])  # Higher weight for tumor
    
    # Test different loss functions
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
    
    # Load full dataset using comprehensive patchification approach
    print("Loading dataset with comprehensive patchification (preserves spatial resolution)...")
    full_dataset = BrainVolumePatchesDataset(
        '/home/ubuntu/projects/finetune_npy/Task002_FOMO2',
        patch_size=96,
        patch_strategy='comprehensive',  # Use comprehensive patchification strategy
        overlap=0.5,                    # 50% overlap between patches
        max_patches_per_volume=25,      # Extract up to 25 patches per volume
        tumor_ratio=0.4                 # 40% tumor patches, 60% background patches
    )
    
    # Analyze dataset tumor distribution
    tumor_stats = analyze_dataset_tumor_distribution(full_dataset)
    
    # Calculate class weights with higher tumor weight
    class_weights = calculate_class_weights(full_dataset).to(DEVICE)
    # Reduce the extreme class weight to prevent overfitting
    class_weights[1] = min(class_weights[1] * 1.5, 8.0)  # Cap at 8 to avoid extreme values
    print(f"Adjusted class weights: {class_weights}")
    
    # Create train/validation split (80% train, 20% validation)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Create indices for splitting
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create training and validation datasets (using patchified data)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Print study IDs for training and validation sets
    print("=== Dataset Split Information ===")
    print(f"Total dataset size: {dataset_size} volumes")
    print(f"Training set size: {len(train_dataset)} volumes")
    print(f"Validation set size: {len(val_dataset)} volumes")
    
    # Get unique file names from patches
    train_files = set()
    val_files = set()
    
    for idx in train_indices:
        sample = full_dataset[idx]
        train_files.add(sample['file_name'])
    
    for idx in val_indices:
        sample = full_dataset[idx]
        val_files.add(sample['file_name'])
    
    print(f"\n=== Training Files ({len(train_files)} files) ===")
    for i, file_name in enumerate(sorted(train_files)):
        print(f"{i+1:3d}. {file_name}")
    
    print(f"\n=== Validation Files ({len(val_files)} files) ===")
    for i, file_name in enumerate(sorted(val_files)):
        print(f"{i+1:3d}. {file_name}")
    
    # Create data loaders with batch size for patch-based training
    batch_size = 8  # Patch-based training allows larger batch size due to smaller memory footprint
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset split: {len(train_dataset)} training patches, {len(val_dataset)} validation patches")
    print(f"Training batch size: {batch_size} (patches)")
    print(f"Each patch: {96}x{96}x{96} voxels")
    
    # Initialize model and load pretrained weights
    simclr_ckpt_path = '/home/ubuntu/projects/nasrin_brainaic/checkpoints/BrainIAC.ckpt'
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=simclr_ckpt_path,
        img_size=(96, 96, 96),  # Fixed patch size
        in_channels=1,
        out_channels=1  # 1 channel for binary segmentation (tumor probability)
    ).to(DEVICE)
    
    # Define training phases for gradual unfreezing (optimized for patch-based training)
    training_phases = [
        {
            'name': 'Phase 1: Decoder Only',
            'start_epoch': 0,
            'end_epoch': 200,  # Extended for longer training
            'encoder_frozen': True,
            'backbone_lr': 0.0,
            'decoder_lr': 1e-4,  # Higher LR for decoder
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Freeze encoder, train decoder only on high-resolution patches'
        },
        {
            'name': 'Phase 2: Last 2 Encoder Layers',
            'start_epoch': 200,
            'end_epoch': 600,  # Extended for longer training
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 2,
            'backbone_lr': 5e-6,  # Conservative backbone LR
            'decoder_lr': 1e-4,  # Keep decoder LR stable
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Unfreeze last 2 encoder layers with conservative LR'
        },
        {
            'name': 'Phase 3: Last 4 Encoder Layers',
            'start_epoch': 600,
            'end_epoch': 1200,  # Extended for longer training
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 4,
            'backbone_lr': 1e-6,  # Conservative backbone LR
            'decoder_lr': 1e-4,  # Keep decoder LR stable
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Unfreeze last 4 encoder layers with conservative LR'
        },
        {
            'name': 'Phase 4: Full Fine-tuning',
            'start_epoch': 1200,
            'end_epoch': 2000,  # Extended to 2000 epochs
            'encoder_frozen': False,
            'encoder_layers_unfrozen': 12,  # All layers
            'backbone_lr': 5e-7,  # Minimal backbone LR for final fine-tuning
            'decoder_lr': 1e-4,  # Keep decoder LR stable
            'backbone_weight_decay': 1e-5,
            'decoder_weight_decay': 1e-4,
            'description': 'Fine-tune entire model with minimal backbone LR on high-res patches'
        }
    ]
    
    # Initialize training phase
    current_phase_idx = 0
    current_phase = training_phases[current_phase_idx]
    
    # Set up initial phase
    print(f"\n{'='*80}")
    print(f"STARTING {current_phase['name']}")
    print(f"{'='*80}")
    print(f"Description: {current_phase['description']}")
    
    # Initialize encoder state based on current phase
    if current_phase['encoder_frozen']:
        freeze_encoder(model, freeze=True)
    else:
        unfreeze_encoder_layers(model, current_phase['encoder_layers_unfrozen'])
    
    # Create optimizer with phase-specific learning rates
    optimizer = create_optimizer_with_differential_lrs(
        model, 
        backbone_lr=current_phase['backbone_lr'], 
        decoder_lr=current_phase['decoder_lr'],
        backbone_weight_decay=current_phase['backbone_weight_decay'],
        decoder_weight_decay=current_phase['decoder_weight_decay']
    )
    
    # Add learning rate scheduler for better convergence
    scheduler = create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # Use the new DiceCELoss with friend's configuration
    criterion = DiceCELoss(
        lambda_dice=0.8, 
        lambda_ce=0.2, 
        class_weights=[0.1, 0.9]  # Background: 0.1, Tumor: 0.9
    )
    print(f"Using DiceCELoss with lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9]")
    
    # Print initial trainable parameters
    trainable_params = get_trainable_params_count(model)
    print(f"Initial trainable parameters: {trainable_params:,}")
    
    # Training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'epochs': [],
        'train_files': list(train_files),
        'val_files': list(val_files),
        'phases': []
    }

    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    
    model.train()
    print(f"\n{'='*80}")
    print(f"STARTING PATCH-BASED TRAINING FOR 2000 EPOCHS WITH IMPROVED CONFIGURATION")
    print(f"Using 96x96x96 patches from original volumes (preserves spatial resolution)")
    print(f"Loss: DiceCELoss (lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9])")
    print(f"Optimizer: AdamW with differential learning rates (backbone: 1e-5, decoder: 1e-4)")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6)")
    print(f"Checkpoints will be saved every 100 epochs")
    print(f"Best model will be saved based on validation loss")
    print(f"Plots will be updated after each epoch")
    print(f"{'='*80}\n")
    
    for epoch in range(2000):  # Train for 2000 epochs
        # Check if we need to transition to next phase
        if epoch >= current_phase['end_epoch'] and current_phase_idx < len(training_phases) - 1:
            current_phase_idx += 1
            current_phase = training_phases[current_phase_idx]
            
            print(f"\n{'='*80}")
            print(f"TRANSITIONING TO {current_phase['name']}")
            print(f"{'='*80}")
            print(f"Description: {current_phase['description']}")
            
            # Update encoder state
            if current_phase['encoder_frozen']:
                freeze_encoder(model, freeze=True)
            else:
                unfreeze_encoder_layers(model, current_phase['encoder_layers_unfrozen'])
            
            # Create new optimizer with updated learning rates
            optimizer = create_optimizer_with_differential_lrs(
                model, 
                backbone_lr=current_phase['backbone_lr'], 
                decoder_lr=current_phase['decoder_lr'],
                backbone_weight_decay=current_phase['backbone_weight_decay'],
                decoder_weight_decay=current_phase['decoder_weight_decay']
            )
            
            # Reset scheduler
            scheduler = create_scheduler_with_cosine_annealing(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
            
            # Print updated trainable parameters
            trainable_params = get_trainable_params_count(model)
            print(f"Updated trainable parameters: {trainable_params:,}")
            
            # Record phase transition
            training_history['phases'].append({
                'epoch': epoch,
                'phase_name': current_phase['name'],
                'trainable_params': trainable_params
            })
        
        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/2000 - {current_phase['name']}")
        total_train_loss = 0
        num_train_batches = 0

        for batch in pbar:
            imgs = batch['image'].to(DEVICE)   # (B, 1, 96, 96, 96) - patches from original volumes
            labels = batch['label'].to(DEVICE)  # (B, 96, 96, 96) - patch labels

            optimizer.zero_grad()
            outputs = model(imgs)  # (B, num_classes, 96, 96, 96)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Debug: Monitor tumor prediction rates
            if num_train_batches < 5:  # Only for first few batches
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
        
        # Validation phase
        avg_val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_dice'].append(val_metrics['dice_score'])
        training_history['epochs'].append(epoch + 1)
        
        # Get current learning rates
        current_backbone_lr = optimizer.param_groups[0]['lr']
        current_decoder_lr = optimizer.param_groups[1]['lr']
        
        print(f"Epoch {epoch+1}/2000: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {val_metrics['dice_score']:.4f}")
        print(f"  Val Tumor Pred Rate: {val_metrics['tumor_pred_rate']:.2f}%, Target Rate: {val_metrics['tumor_target_rate']:.2f}%")
        print(f"  Current Phase: {current_phase['name']}")
        print(f"  Current LRs - Backbone: {current_backbone_lr:.2e}, Decoder: {current_decoder_lr:.2e}")
        
        # Step the scheduler (cosine annealing scheduler steps every epoch)
        scheduler.step()
        
        # Track best metrics and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            
            # Save best model
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
        
        # Save checkpoint every 100 epochs
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
        
        # Update plots after each epoch
        plot_training_metrics(training_history, 'plots/training_metrics.png')
        plot_combined_metrics(training_history, 'plots/combined_metrics.png')
        
        # Progress summary every 100 epochs
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
    
    # Save final model and results
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
    
    # Save training results as JSON
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
    
    # Create final comprehensive plots
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
    print(f"✓ Training method: 96x96x96 patches from original volumes (preserved spatial resolution)")
    print(f"✓ Loss function: DiceCELoss (lambda_dice=0.8, lambda_ce=0.2, class_weights=[0.1, 0.9])")
    print(f"✓ Optimizer: AdamW with differential learning rates")
    print(f"✓ Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2, eta_min=1e-6)")
    print(f"✓ Checkpoints saved every 100 epochs")
    print(f"✓ Best model saved based on validation loss")
    print(f"{'='*80}")

if __name__ == "__main__":
    train()
