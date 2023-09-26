
"""
Contains functionalitiy for creating Pytorch Datalaoder's for 
image classification
"""
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pydicom
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path

try:
    import pydicom
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'pydicom'])
    import pydicom
    
NUM_WORKER = os.cpu_count()


def Create_datalaoder(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKER  
):
    """
    Create train and testing dataloaders.
    
      Takes in a training directory and testing directroy path and turns them into 
      PyTorch Datasets and then into PyTorch DataLoaders.

      Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

      Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
          train_dataloader, test_dataloader, class_names = Create_dataloaders(train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            transform=some_transform,
            batch_size=32,
            num_workers=4)
  """
    # Creating train and test datasets.
    train_data = torchvision.datasets.ImageFolder(root = train_dir,
                                                  transform = transform)
    test_data = torchvision.datasets.ImageFolder(root = test_dir,
                                                 transform = transform)
    # Creating Dataloader
    train_datalaoder = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  shuffle = True)
    test_datalaoder = DataLoader(dataset = test_data,
                                 batch_size = batch_size,
                                 shuffle = False)
    
    return train_datalaoder, test_datalaoder, train_data.classes


class Custom_dataset_DICOM(Dataset):
    def __init__(self, targ_dir: str, transform = None):
        
        self.paths = list(Path(targ_dir).glob('*/*.dcm'))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    
    def load_image(self, Index):
        image_path = self.paths[Index]
        ds = pydicom.dcmread(image_path)
        return ds.pixel_array.astype(np.float32) / 255.0
        
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_names = self.path[index].parent.name # expects path in data_folder/class_name/image.dcm
        class_idx = self.class_to_idx[class_name]
        
        #transform 
        if self.transform:
            return self.transform(img), class_names
        else:
            img, class_idx

            
# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def Creat_datalaoders_DICOM(train_dir: str,
                           test_dir: str,
                           transform: transforms.Compose,
                           batch_size: int,
                           num_workers: int = NUM_WORKER ):

    """
    Create train and testing dataloaders for DICOM images.

      Takes in a training directory and testing directroy path and turns them into 
      PyTorch Datasets and then into PyTorch DataLoaders.

      Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

      Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
          train_dataloader, test_dataloader, class_names = Create_dataloaders_DICOM(train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            transform=some_transform,
            batch_size=32,
            num_workers=4)
  """
    # Creating training and test dataset
    train_data = Custom_dataset_DICOM(train_dir, transform = transform)
    test_data = Custom_dataset_DICOM(test_dir, transform = transform)
    
    
    # Creating Dataloader
    train_datalaoder_DICOM = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  shuffle = True)
    test_datalaoder_DICOM = DataLoader(dataset = test_data,
                                 batch_size = batch_size,
                                 shuffle = False)
    
    return train_datalaoder_DICOM, test_datalaoder_DICOM, train_data.classes
