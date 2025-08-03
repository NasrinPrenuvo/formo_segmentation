#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
from model import ViTUNETRSegmentationModel
from inference_dataloader import create_inference_dataloader, stitch_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def test_inference():
    input_file = "/home/ubuntu/projects/finetune_npy/Task002_FOMO2/FOMO2_sub_1.npy"
    checkpoint_path = "/home/ubuntu/projects/nasrin_brainaic/segmentation/checkpoints/segmentation_best_model.pth"
    output_dir = "./output"
    
    print("Step 1: Creating dataloader...")
    try:
        dataloader, original_shape = create_inference_dataloader(
            input_file=input_file,
            crop_size=96,
            stride=48,
            batch_size=1
        )
        print(f"✓ Dataloader created successfully. Original shape: {original_shape}")
        print(f"✓ Number of crops: {len(dataloader)}")
    except Exception as e:
        print(f"✗ Error creating dataloader: {e}")
        return
    
    print("\nStep 2: Loading model...")
    try:
        model = ViTUNETRSegmentationModel(
            simclr_ckpt_path=None,
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1
        ).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("\nStep 3: Testing inference on first crop...")
    try:
        # Get first batch
        batch = next(iter(dataloader))
        crop_tensor = batch['crop']
        coords = batch['coords']
        
        print(f"Crop tensor shape: {crop_tensor.shape}")
        print(f"Crop tensor dtype: {crop_tensor.dtype}")
        print(f"Coords: {coords}")
        
        # Move to device and handle batch dimension
        crop_tensor = crop_tensor.to(DEVICE)
        # Keep the original shape [1, 1, 96, 96, 96] as it's correct for the model
        
        print(f"After preprocessing - shape: {crop_tensor.shape}")
        
        # Check if the tensor values are reasonable
        print(f"Crop tensor range: [{crop_tensor.min():.4f}, {crop_tensor.max():.4f}]")
        print(f"Crop tensor mean: {crop_tensor.mean():.4f}")
        print(f"Crop tensor std: {crop_tensor.std():.4f}")
        
        # Try to understand the patch embedding issue
        print("\nDebugging patch embedding...")
        try:
            # Access the patch embedding layer directly
            patch_embed = model.unetr.vit.patch_embedding
            print(f"Patch embedding type: {type(patch_embed)}")
            
            # Check the patch embedding configuration
            if hasattr(patch_embed, 'patch_size'):
                print(f"Patch embedding patch_size: {patch_embed.patch_size}")
            if hasattr(patch_embed, 'hidden_size'):
                print(f"Patch embedding hidden_size: {patch_embed.hidden_size}")
            if hasattr(patch_embed, 'position_embeddings'):
                print(f"Position embeddings shape: {patch_embed.position_embeddings.shape}")
            
            # Try to see what happens in patch embedding
            with torch.no_grad():
                # This should give us the patch embeddings
                patch_output = patch_embed(crop_tensor)
                print(f"Patch embedding output shape: {patch_output.shape}")
                
        except Exception as e:
            print(f"Error in patch embedding debug: {e}")
            import traceback
            traceback.print_exc()
        
        # Run inference with the correct shape
        with torch.no_grad():
            output = model(crop_tensor)
            prob = torch.sigmoid(output)
            prob = prob.squeeze(0).squeeze(0).cpu().numpy()
        
        print(f"✓ Inference successful! Output shape: {prob.shape}")
        print(f"Output range: [{prob.min():.4f}, {prob.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStep 4: Running full inference...")
    try:
        predictions = []
        crop_coords = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                crop_tensor = batch['crop']
                coords = batch['coords']
                
                # Preprocess - keep the original shape [1, 1, 96, 96, 96]
                crop_tensor = crop_tensor.to(DEVICE)
                
                # Inference
                output = model(crop_tensor)
                prob = torch.sigmoid(output)
                prob = prob.squeeze(0).squeeze(0).cpu().numpy()
                
                predictions.append(prob)
                crop_coords.append((coords[0].item(), coords[1].item(), coords[2].item()))
                
                print(f"Processed crop {i+1}/{len(dataloader)}")
        
        print("✓ Full inference completed!")
        
        # Stitch predictions
        print("\nStep 5: Stitching predictions...")
        stitched_prediction = stitch_predictions(
            predictions=predictions,
            crop_coords=crop_coords,
            output_shape=original_shape,
            crop_size=96,
            threshold=0.5
        )
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_filename = "FOMO2_sub_1_prediction.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, stitched_prediction)
        
        print(f"✓ Prediction saved to {output_path}")
        print(f"Final prediction shape: {stitched_prediction.shape}")
        print(f"Final prediction range: [{stitched_prediction.min()}, {stitched_prediction.max()}]")
        
    except Exception as e:
        print(f"✗ Error during full inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference() 