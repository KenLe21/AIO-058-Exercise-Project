import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc và thay đổi kích thước hình nền 1
bg1_image = cv2.imread('week 2 picture/GreenBackground.png', 1)
bg1_image = cv2.resize(bg1_image, (678, 381))

# Đọc và thay đổi kích thước hình đối tượng
ob_image = cv2.imread('week 2 picture/Object.png', 1)
ob_image = cv2.resize(ob_image, (678, 381))

# Đọc và thay đổi kích thước hình nền 2
bg2_image = cv2.imread('week 2 picture/NewBackground.jpg', 1)
bg2_image = cv2.resize(bg2_image, (678, 381))


def compute_difference(bg_img, input_img):
    if bg_img.shape != input_img.shape:
        raise ValueError("Must in the same shape")
    difference_single_channel = cv2.absdiff(bg_img, input_img)
    return difference_single_channel

difference_single_channel = compute_difference(bg1_image, ob_image)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(difference_single_channel, cv2.COLOR_BGR2RGB))
plt.title('Difference Image')
plt.axis('off')
plt.show()


def compute_binary_mask(difference_single_channel, threshold=30):
    _, difference_binary = cv2.threshold(difference_single_channel, threshold, 255, cv2.THRESH_BINARY)
    return difference_binary

difference_single_channel = compute_difference(bg1_image, ob_image)
binary_mask = compute_binary_mask(difference_single_channel)
plt.figure(figsize=(10, 6))
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')
plt.show()


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask == 255, ob_image, bg2_image)
    return output
result_image = replace_background(bg1_image, bg2_image, ob_image)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Result Image with Replaced Background')
plt.axis('off')
plt.show()

