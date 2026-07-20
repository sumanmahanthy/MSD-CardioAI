import sys
import os
from PIL import Image
import torch
from utils.grad_cam import generate_heatmap
print('Imports done.')
if __name__ == '__main__':
    img = Image.new('RGB', (384, 384), color='gray')
    tensor = torch.zeros((1, 3, 384, 384))
    try:
        generate_heatmap(img, tensor, 'test_out.jpg', target_label='Cardiomegaly')
        print('Success!')
    except Exception as e:
        print(f'Failed: {e}')
