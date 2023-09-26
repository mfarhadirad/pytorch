import os
import zipfile

from pathlib import Path

import requests

def get_data(data_path: str,
             file_name: str,
             file_dir_or_url: str):


    # Setup path to data folder
    data_path = Path(data_path)
    image_path = data_path / file_name.removesuffix('.zip')

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / file_name, "wb") as f:
        request = requests.get(file_dir_or_url)
        print(f"Downloading {file_name} data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / file_name, "r") as zip_ref:
        print(f"Unzipping {file_name} data...") 
        zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(data_path / file_name)
