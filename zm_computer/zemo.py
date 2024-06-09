import os
import cv2
import numpy as np
import pandas as pd
from ZEMO import zemo

def calculate_zernike_moments(image_dir, csv_output_path, image_size, zernike_order):

    """
    Calculate the Zernike moments for images in a directory and save to a CSV file.
    
    Parameters:
        image_dir (str): Path to the directory containing image files.
        csv_output_path (str): Path to save the output CSV file.
        image_size (tuple): Size to resize images to. Default is (200, 200).
        zernike_order (int): Zernike polynomial order. Default is 45.

    Returns:
        pd.DataFrame: DataFrame containing the Zernike moments for each image.
    """

    ZBFSTR = zemo.zernike_bf(image_size[0], zernike_order, 1)
    
    image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
    
    zernike_moments = []
    
    for img_path in image_files:
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, image_size=(sz, sz))
        im = resized_image[:, :, 0]
        Z = np.abs(zemo.zernike_mom(np.array(im), ZBFSTR))
        zernike_moments.append(Z)
    
    df = pd.DataFrame(zernike_moments)
    df.to_csv(csv_output_path, index=False)
    
    return df

# Example usage
image_directory = '/path/to/your/directory'
output_csv = '/path/to/your/directory/galaxy_45_0.csv'
zernike_df = calculate_zernike_moments(image_directory, output_csv, image_size=(sz, sz), zernike_order=order)
