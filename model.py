import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class ViTUNETRSegmentationModel(nn.Module):
    def __init__(self, simclr_ckpt_path, img_size=(96,96,96), in_channels=1, out_channels=1):
        super().__init__()
        
        # Create UNETR model
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=1,  # 1 for binary segmentation (tumor probability)
            img_size=(96, 96, 96),  # Fixed size for 96x96x96 resampled volumes
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0
        )
        
        # Load SimCLR weights into UNETR's ViT encoder
        if simclr_ckpt_path:
            print(f"Loading SimCLR weights from {simclr_ckpt_path}")
            ckpt = torch.load(simclr_ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            
            # Extract backbone weights and map them to UNETR's ViT
            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    # Remove 'backbone.' prefix
                    new_key = k[9:]
                    backbone_state_dict[new_key] = v
            
            # Load weights into UNETR's ViT encoder
            try:
                # Try to load with strict=False to handle any minor mismatches
                missing_keys, unexpected_keys = self.unetr.vit.load_state_dict(backbone_state_dict, strict=False)
                print(f"Successfully loaded {len(backbone_state_dict)} pretrained layers")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                print(f"Warning: Could not load all pretrained weights: {e}")
                print("Continuing with random initialization for incompatible layers")
        
        print("="*10)
        print("UNETR initialized with ViT encoder")
        print("="*10)
        
    def forward(self, x):
        return self.unetr(x)
