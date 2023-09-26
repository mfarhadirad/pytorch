
def train(URL: str,
          data_path: str,
          file_name: str,
          model_name: str
          
    

):
    # load or Download data
    # Create dataloader using data_setup
    # build the model using model builder
    # start training with engine
    # save model with utils
    """
    Trains a PyTorch image classification model using device-agnostic code.
    Be aware that the data should be a zip file which contains a file and this file 
    contains two files nemed train and test 
    """
    import os
    import torch

    from torchvision import transforms
    from timeit import default_timer as timer 
    from going_modular import data_setup, engine, model_builder, utils, get_data
    # import data_setup, engine, model_builder, utils

    # Setup hyperparameters
    NUM_EPOCHS = 5 
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # data_path = "C:/Users/ACER/AppData/Roaming/jupyter/kernels/going_modular/data"
    # file_name = 'pizza_steak_sushi.zip'
    # file_dir_or_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    
    # load or Download data
    Download_data = get_data.get_data(data_path= data_path,
                             file_name = file_name,
                             file_dir_or_url = URL)
    
    filename_withoutsuffix = str(file_name.removesuffix('.zip'))
    # Setup directories
    train_dir = data_path + filename_withoutsuffix + '/train'
    test_dir = data_path + filename_withoutsuffix + '/test'

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
                                         transforms.Resize((64, 64)),
                                         transforms.ToTensor()
    ])

    # Create DataLoader's and get class_names
    train_dataloader, test_dataloader, class_names = data_setup.Create_datalaoder(train_dir=train_dir, 
                                                                                   test_dir=test_dir,
                                                                                   transform=data_transform,
                                                                                   batch_size=BATCH_SIZE)

    # Create model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=HIDDEN_UNITS,
                                  output_shape=len(class_names)).to(device)

    # Setup loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start the timer

    start_time = timer()

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS, 
                 device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model to file
    utils.save_model(model=model,
                     target_dir='C:/Users/ACER/AppData/Roaming/jupyter/kernels/going_modular/models',
                     model_name= model_name)

