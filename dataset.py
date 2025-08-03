import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import train_test_split

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, file_paths, transform=None):
        self.data_dir = data_dir
        self.file_paths = file_paths
        self.transform = transform
        self.crops = self._generate_crops()

    def _extract_subject_id(self, file_path):
        filename = os.path.basename(file_path)
        match = re.match(r'(subject\d+)_.*\.npy', filename)
        if match:
            return match.group(1)
        return filename

    def _generate_crops(self):
        crops = []
        for file_path in self.file_paths:
            data = np.load(file_path, allow_pickle=True)
            # Convert object array to numeric if necessary
            if data.dtype == np.object_:
                data = np.array(data, dtype=np.float32)
            image = data[1, :, :, :]  # Channel 1 as input
            label = data[3, :, :, :]  # Channel 4 as binary mask
            # Ensure numeric dtypes
            image = np.asarray(image, dtype=np.float32)
            label = np.asarray(label, dtype=np.float32)
            print(f"[DEBUG] File: {file_path}, Image dtype: {image.dtype}, Label dtype: {label.dtype}, Shape: {image.shape}")
            d, h, w = image.shape
            tumor_crops = []
            background_crops = []
            for z in range(0, d - 96 + 1, 48):
                for y in range(0, h - 96 + 1, 48):
                    for x in range(0, w - 96 + 1, 48):
                        crop_label = label[z:z+96, y:y+96, x:x+96]
                        if crop_label.sum() > 0:
                            tumor_crops.append((file_path, z, y, x))
                        else:
                            background_crops.append((file_path, z, y, x))
            num_tumor_crops = len(tumor_crops)
            num_background_crops = min(len(background_crops), max(1, num_tumor_crops * 2))
            if num_background_crops > 0:
                indices = np.random.choice(len(background_crops), size=num_background_crops, replace=False)
                selected_background_crops = [background_crops[i] for i in indices]
                crops.extend(tumor_crops)
                crops.extend(selected_background_crops)
            else:
                crops.extend(tumor_crops)
        print(f"[DEBUG] Total crops: {len(crops)} (Tumor: {len(tumor_crops)}, Background: {num_background_crops})")
        return crops

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        file_path, z, y, x = self.crops[idx]
        data = np.load(file_path, allow_pickle=True)
        # Convert object array to numeric if necessary
        if data.dtype == np.object_:
            data = np.array(data, dtype=np.float32)
        image = data[1, :, :, :]  # Channel 1
        label = data[3, :, :, :]  # Channel 4
        # Ensure numeric dtypes
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        image_crop = image[z:z+96, y:y+96, x:x+96]
        label_crop = label[z:z+96, y:y+96, x:x+96]
        image_crop = np.expand_dims(image_crop, axis=0)  # Add channel dim: [1, 96, 96, 96]
        # Debug shapes and dtypes
        print(f"[DEBUG] Crop {idx}: Image shape: {image_crop.shape}, dtype: {image_crop.dtype}, "
              f"Label shape: {label_crop.shape}, dtype: {label_crop.dtype}")
        image_crop = torch.from_numpy(image_crop).float()
        label_crop = torch.from_numpy(label_crop).float()
        if self.transform:
            image_crop = self.transform(image_crop)
        return image_crop, label_crop

def get_dataloaders(data_dir, train_ratio=0.8, batch_size=8, num_workers=4):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    train_files, val_files = train_test_split(all_files, train_size=train_ratio, random_state=42)
    train_dataset = MedicalImageDataset(data_dir, train_files)
    val_dataset = MedicalImageDataset(data_dir, val_files)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader