import os
import numpy as np
import torch
from model import ViTUNETRSegmentationModel
import argparse
from inference_dataloader import create_inference_dataloader, stitch_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """Load the trained ViTUNETRSegmentationModel from checkpoint."""
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=None,  # No SimCLR weights needed for inference
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def process_single_crop(model, crop_tensor):
    """
    Process a single crop through the model.
    
    Args:
        model: The loaded model
        crop_tensor (torch.Tensor): Input crop tensor [1, 1, 96, 96, 96] from dataloader
    
    Returns:
        np.ndarray: Processed prediction [96, 96, 96]
    """
    # Ensure tensor is on correct device and has correct shape
    crop_tensor = crop_tensor.to(DEVICE)
    
    # The model expects [batch, channels, depth, height, width]
    # If we have [1, 96, 96, 96], we need to add the channel dimension
    if crop_tensor.dim() == 4 and crop_tensor.shape[0] == 1:
        # [1, 96, 96, 96] -> [1, 1, 96, 96, 96]
        crop_tensor = crop_tensor.unsqueeze(1)
    
    # Verify tensor shape
    if crop_tensor.shape != (1, 1, 96, 96, 96):
        raise ValueError(f"Expected crop tensor shape (1, 1, 96, 96, 96), got {crop_tensor.shape}")
    
    with torch.no_grad():
        # Run inference
        output = model(crop_tensor)
        
        # Apply sigmoid to get probabilities
        prob = torch.sigmoid(output)
        
        # Remove batch and channel dimensions
        prob = prob.squeeze(0).squeeze(0).cpu().numpy()  # [96, 96, 96]
        
        return prob

def infer(input_file, checkpoint_path, output_dir):
    """Perform inference on a single .npy file and save the output."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create inference dataloader
    dataloader, original_shape = create_inference_dataloader(
        input_file=input_file,
        crop_size=96,
        stride=48,
        batch_size=1
    )
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Process each crop separately
    predictions = []
    crop_coords = []
    
    print(f"Processing {len(dataloader)} crops...")
    
    for i, batch in enumerate(dataloader):
        crop_tensor = batch['crop']  # [1, 1, 96, 96, 96] from dataloader
        coords = batch['coords']
        
        # The dataloader already provides the correct shape [1, 1, 96, 96, 96]
        # No need to modify the tensor shape
        
        # Process the crop
        try:
            prob = process_single_crop(model, crop_tensor)
            predictions.append(prob)
            crop_coords.append((coords[0].item(), coords[1].item(), coords[2].item()))
            
            print(f"[DEBUG] Processed crop {i+1}/{len(dataloader)}, "
                  f"shape: {prob.shape}, range: [{prob.min():.4f}, {prob.max():.4f}]")
                  
        except Exception as e:
            print(f"Error processing crop {i+1}: {e}")
            # Continue with other crops
            continue
    
    if not predictions:
        raise ValueError("No crops were successfully processed!")
    
    # Stitch predictions back together
    print("Stitching predictions...")
    stitched_prediction = stitch_predictions(
        predictions=predictions,
        crop_coords=crop_coords,
        output_shape=original_shape,
        crop_size=96,
        threshold=0.5
    )
    
    # Save output
    output_filename = os.path.basename(input_file).replace('.npy', '_prediction.npy')
    output_path = os.path.join(output_dir, output_filename)
    np.save(output_path, stitched_prediction)
    print(f"[DEBUG] Saved prediction to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Perform segmentation inference on a .npy file")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input .npy file")
    parser.add_argument('--checkpoint_path', type=str, 
                       default='/home/ubuntu/projects/nasrin_brainaic/segmentation/checkpoints/segmentation_best_model.pth', 
                       help="Path to model checkpoint")
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ubuntu/projects/nasrin_brainaic/segmentation/outputs/', 
                       help="Directory to save output .npy files")
    args = parser.parse_args()
    
    infer(args.input_file, args.checkpoint_path, args.output_dir)

if __name__ == "__main__":
    main()