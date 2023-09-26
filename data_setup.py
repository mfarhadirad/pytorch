"""
Contains functionalitiy for creating Pytorch Datalaoder's for 
image classification
"""
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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
          train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
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
    
