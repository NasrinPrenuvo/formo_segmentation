import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class InferenceDataset(Dataset):
    def __init__(self, input_file, crop_size=96, stride=48):
        """
        Dataset for inference that generates crops from a single input file.
        
        Args:
            input_file (str): Path to the input .npy file
            crop_size (int): Size of the crops (default: 96)
            stride (int): Stride for crop generation (default: 48)
        """
        self.input_file = input_file
        self.crop_size = crop_size
        self.stride = stride
        
        # Load the input data
        self.data = np.load(input_file, allow_pickle=True)
        if self.data.dtype == np.object_:
            self.data = np.array(self.data, dtype=np.float32)
        
        # Extract the image (channel 1)
        self.image = self.data[1, :, :, :].astype(np.float32)
        self.original_shape = self.image.shape
        
        # Generate crop coordinates
        self.crop_coords = self._generate_crop_coordinates()
        
        print(f"[DEBUG] Input file: {input_file}")
        print(f"[DEBUG] Image shape: {self.image.shape}, dtype: {self.image.dtype}")
        print(f"[DEBUG] Generated {len(self.crop_coords)} crops with shape [{crop_size}, {crop_size}, {crop_size}]")
    
    def _generate_crop_coordinates(self):
        """Generate coordinates for crops that fit within the image."""
        d, h, w = self.image.shape
        coords = []
        
        for z in range(0, d - self.crop_size + 1, self.stride):
            for y in range(0, h - self.crop_size + 1, self.stride):
                for x in range(0, w - self.crop_size + 1, self.stride):
                    coords.append((z, y, x))
        
        return coords
    
    def __len__(self):
        return len(self.crop_coords)
    
    def __getitem__(self, idx):
        z, y, x = self.crop_coords[idx]
        
        # Extract crop
        crop = self.image[z:z+self.crop_size, y:y+self.crop_size, x:x+self.crop_size]
        
        # Normalize the crop (optional - adjust based on your training preprocessing)
        # crop = (crop - crop.mean()) / (crop.std() + 1e-8)
        
        # Add channel dimension and convert to tensor
        crop = np.expand_dims(crop, axis=0)  # [1, 96, 96, 96]
        crop = torch.from_numpy(crop).float()
        
        return {
            'crop': crop,
            'coords': (z, y, x),
            'idx': idx
        }

def create_inference_dataloader(input_file, crop_size=96, stride=48, batch_size=1):
    """
    Create a dataloader for inference.
    
    Args:
        input_file (str): Path to the input .npy file
        crop_size (int): Size of the crops
        stride (int): Stride for crop generation
        batch_size (int): Batch size for the dataloader
    
    Returns:
        DataLoader: Dataloader for inference
    """
    dataset = InferenceDataset(input_file, crop_size, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Use 0 for inference to avoid multiprocessing issues
        pin_memory=True
    )
    return dataloader, dataset.original_shape

def stitch_predictions(predictions, crop_coords, output_shape, crop_size=96, threshold=0.5):
    """
    Stitch predicted crops back into the original volume shape, averaging overlaps.
    
    Args:
        predictions (list): List of predicted crops
        crop_coords (list): List of crop coordinates
        output_shape (tuple): Original image shape
        crop_size (int): Size of the crops
        threshold (float): Threshold for binary segmentation
    
    Returns:
        np.ndarray: Stitched prediction
    """
    d, h, w = output_shape
    prediction = np.zeros((d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)  # Track number of contributions per voxel
    
    for pred, (z, y, x) in zip(predictions, crop_coords):
        # Convert prediction to binary
        binary_crop = (pred > threshold).astype(np.float32)
        
        # Add to the prediction volume
        prediction[z:z+crop_size, y:y+crop_size, x:x+crop_size] += binary_crop
        count_map[z:z+crop_size, y:y+crop_size, x:x+crop_size] += 1
    
    # Avoid division by zero
    count_map = np.where(count_map == 0, 1, count_map)
    stitched = prediction / count_map
    
    # Ensure binary output (0 or 1)
    stitched = (stitched > 0.5).astype(np.float32)
    
    print(f"[DEBUG] Stitched prediction shape: {stitched.shape}, dtype: {stitched.dtype}")
    return stitched 