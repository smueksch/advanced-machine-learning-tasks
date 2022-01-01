import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import torch


def vertical_wave(img):
    img_output = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            offset_x = int(25.0 * np.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j + offset_x < img.shape[0]:
                img_output[i, j] = img[i, (j + offset_x) % img.shape[1]]
            else:
                img_output[i, j] = 0

    return(img_output)


def horizontal_wave(img):
    img_output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            offset_x = 0
            offset_y = int(16.0 * np.sin(2 * 3.14 * j / 150))
            if i + offset_y < img.shape[0]:
                img_output[i, j] = img[(i + offset_y) % img.shape[0], j]
            else:
                img_output[i, j] = 0

    return(img_output)


def horizontal_vertical(img):
    img_output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            offset_x = int(20.0 * np.sin(2 * 3.14 * i / 150))
            offset_y = int(20.0 * np.cos(2 * 3.14 * j / 150))
            if i + offset_y < img.shape[0] and j + offset_x < img.shape[1]:
                img_output[i, j] = img[(i + offset_y) %
                                       img.shape[0], (j + offset_x) % img.shape[1]]
            else:
                img_output[i, j] = 0
    return(img_output)


def concave(img):
    img_output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            offset_x = int(128 * np.sin(2 * 3.14 * i / (2 * img.shape[1])))
            offset_y = 0
            if j + offset_x < img.shape[1]:
                img_output[i, j] = img[i, (j + offset_x) % img.shape[1]]
            else:
                img_output[i, j] = 0
    return(img_output)

# taken from https://www.py4u.net/discuss/184760


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(
        new_height, height), min(
        new_width, width)
    pad_height1, pad_width1 = (
        height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (
        height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1,
                                             pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def generate_augmented_data(img, label, box):

    image = Image.fromarray(np.uint8(img))
    image_label = Image.fromarray(np.uint8(label))
    image_box = Image.fromarray(np.uint8(box))

    ## Original. ##

    original_Image = torch.tensor(img, dtype=torch.uint8)
    original_Image = torch.unsqueeze(original_Image, 0)

    original_Label = torch.tensor(label, dtype=torch.uint8)

    original_Box = torch.tensor(box, dtype=torch.uint8)

    ## Rotations. ##

    rotated_Image = transforms.functional.rotate(
        image, angle=20)  # 20 deg counter clockwise rotation
    rotated_Image = torch.tensor(np.array(rotated_Image))
    rotated_Image = torch.unsqueeze(rotated_Image, 0)

    rotated_Label = transforms.functional.rotate(
        image_label, angle=20)  # 20 deg counter clockwise rotation
    rotated_Label = torch.tensor(np.array(rotated_Label))

    rotated_Box = transforms.functional.rotate(
        image_box, angle=20)  # 20 deg counter clockwise rotation
    rotated_Box = torch.tensor(np.array(rotated_Box))

    ## Translation. ##

    translated_Image = transforms.functional.affine(
        image, angle=0, translate=[-20, -20],
        scale=1, shear=0)  # rotation left and up
    translated_Image = torch.tensor(np.array(translated_Image))
    translated_Image = torch.unsqueeze(translated_Image, 0)

    translated_Label = transforms.functional.affine(
        image_label, angle=0, translate=[-20, -20], scale=1, shear=0)
    translated_Label = torch.tensor(np.array(translated_Label))

    translated_Box = transforms.functional.affine(
        image_box, angle=0, translate=[-20, -20], scale=1, shear=0)
    translated_Box = torch.tensor(np.array(translated_Box))

    ## Sheared. ##

    sheared_Image = transforms.functional.affine(
        image, angle=0, translate=[0, 0],
        scale=1, shear=(10, 20))
    sheared_Image = torch.tensor(np.array(sheared_Image))
    sheared_Image = torch.unsqueeze(sheared_Image, 0)

    sheared_Label = transforms.functional.affine(
        image_label, angle=0, translate=[0, 0],
        scale=1, shear=(10, 20))
    sheared_Label = torch.tensor(np.array(sheared_Label))

    sheared_Box = transforms.functional.affine(
        image_box, angle=0, translate=[0, 0],
        scale=1, shear=(10, 20))
    sheared_Box = torch.tensor(np.array(sheared_Box))

    ## Zoomed. ##

    zoomed_Image = cv2_clipped_zoom(img, zoom_factor=1.5)
    zoomed_Image = torch.tensor(zoomed_Image)
    zoomed_Image = torch.unsqueeze(zoomed_Image, 0)

    zoomed_Label = cv2_clipped_zoom(label.astype(np.uint8), zoom_factor=1.5)
    zoomed_Label = torch.tensor(zoomed_Label)

    zoomed_Box = cv2_clipped_zoom(box.astype(np.uint8), zoom_factor=1.5)
    zoomed_Box = torch.tensor(zoomed_Box)

    ## Wave deformations. ##

    vertical_wave_Image = torch.tensor(vertical_wave(img))
    vertical_wave_Image = torch.unsqueeze(vertical_wave_Image, 0)
    vertical_wave_Label = torch.tensor(vertical_wave(label))
    vertical_wave_Box = torch.tensor(vertical_wave(box))

    horizontal_wave_Image = torch.tensor(horizontal_wave(img))
    horizontal_wave_Image = torch.unsqueeze(horizontal_wave_Image, 0)
    horizontal_wave_Label = torch.tensor(horizontal_wave(label))
    horizontal_wave_Box = torch.tensor(horizontal_wave(box))

    horizontal_vertical_Image = torch.tensor(horizontal_vertical(img))
    horizontal_vertical_Image = torch.unsqueeze(horizontal_vertical_Image, 0)
    horizontal_vertical_Label = torch.tensor(horizontal_vertical(label))
    horizontal_vertical_Box = torch.tensor(horizontal_vertical(box))

    concave_Image = torch.tensor(concave(img))
    concave_Image = torch.unsqueeze(concave_Image, 0)
    concave_Label = torch.tensor(concave(label))
    concave_Box = torch.tensor(concave(box))

    value_img = {
        'original': original_Image,
        'rotated': rotated_Image,
        'translated': translated_Image,
        'sheared': sheared_Image,
        'zoomed': zoomed_Image,
        'vertical_wave': vertical_wave_Image,
        'horizontal_wave': horizontal_wave_Image,
        'horizontal_vertical': horizontal_vertical_Image,
        'concave': concave_Image
        }

    value_label = {
        'original': original_Label,
        'rotated': rotated_Label,
        'translated': translated_Label,
        'sheared': sheared_Label,
        'zoomed': zoomed_Label,
        'vertical_wave': vertical_wave_Label,
        'horizontal_wave': horizontal_wave_Label,
        'horizontal_vertical': horizontal_vertical_Label,
        'concave': concave_Label
        }

    value_box = {
        'original': original_Box,
        'rotated': rotated_Box,
        'translated': translated_Box,
        'sheared': sheared_Box,
        'zoomed': zoomed_Box,
        'vertical_wave': vertical_wave_Box,
        'horizontal_wave': horizontal_wave_Box,
        'horizontal_vertical': horizontal_vertical_Box,
        'concave': concave_Box
        }

    return (value_img, value_label, value_box)


def pad_to_multiple(array: np.ndarray, n: int) -> np.ndarray:
    """Pad array to have width and height as multiple of given number."""
    original_height, original_width = array.shape

    def next_multiple(x): return ((x // n) + 1) * n

    new_height = next_multiple(original_height) - original_height
    new_width = next_multiple(original_width) - original_width

    return np.pad(array, ((0, new_height), (0, new_width)))
