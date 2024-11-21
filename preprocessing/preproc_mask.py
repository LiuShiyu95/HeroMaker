import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

root_dir = 'video_reconstruction/all_sequences'
name = 'final'

msk_folder = f'{root_dir}/{name}/{name}_masks'
frg_mask_folder = f'{root_dir}/{name}/{name}_masks_0'
bkg_mask_folder = f'{root_dir}/{name}/{name}_masks_1'
os.makedirs(frg_mask_folder, exist_ok=True)
os.makedirs(bkg_mask_folder, exist_ok=True)

files = glob(msk_folder + '/*.png')
num = len(files)

for i in tqdm(range(num)):
    file_n = os.path.basename(files[i])
    mask = cv2.imread(os.path.join(msk_folder, file_n), 0)
    mask[mask > 0] = 1
    cv2.imwrite(os.path.join(frg_mask_folder, file_n), mask * 255)

    bg_mask = mask.copy()
    bg_mask[bg_mask == 0] = 127
    bg_mask[bg_mask == 255] = 0
    bg_mask[bg_mask == 127] = 255
    cv2.imwrite(os.path.join(bkg_mask_folder, file_n), bg_mask)