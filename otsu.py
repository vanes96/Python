import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sys import argv
import os.path
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2125, 0.7154,0.0721])

def otsu(src_path = 'lena.jpg', dst_path = 'output.jpg'):
    #== Перед применением алгоритма Оцу, целесообразно выполнить сглаживание ==
    # img_orig = cv2.imread(src_path)
    # width, height = img_orig.shape[1], img_orig.shape[0]
    # img_sum = np.cumsum(np.cumsum(img_orig, 0), 1)
    # img_blur = np.copy(img_orig)
    # w, h = 9, 9
    #
    # for i in range(h // 2 + 1, height - h):
    #     i1, i2 = i - (h // 2) - 1, i + (h // 2)
    #     for j in range(w // 2 + 1, width - w):
    #         j1, j2 = j - (w // 2) - 1, j + (w // 2)
    #         sum_ = img_sum[i2][j2] + img_sum[i1][j1] - img_sum[i1][j2] - img_sum[i2][j1]
    #         img_blur[i][j] = sum_ // (w * h)
    # =====================================================================
    img_orig = plt.imread(src_path)
    #img_gray = rgb2gray(img_blur)
    img_gray = rgb2gray(img_orig)
    hist = np.histogram(img_gray, bins = range(257))[0]
    min, max = 0, 0

    for i in range(256):
        if hist[i] > 0:
            min = i
            break
    for i in range(255, -1, -1):
        if hist[i] > 0:
            max = i
            break

    hist_size = max - min + 1
    num, sum_val = sum(hist), 0
    for i in range(min, max + 1):
        sum_val += hist[i] * i

    sig_max, thresh, sum1, num1 = -1, 0, 0, 0
    for i in range(min, max):
        num1 += hist[i]
        sum1 += hist[i] * i
        p1 = num1 / num
        avg1 = sum1 / num1
        avg2 = (sum_val - sum1) / (num - num1)
        sig = p1 * (1 - p1) * (avg1 - avg2) ** 2
        if sig > sig_max:
            sig_max = sig
            thresh = i

    for i in range(np.size(img_gray, 0)):
        for j in range(np.size(img_gray, 1)):
            if img_gray[i][j] <= thresh:
                img_gray[i][j] = 0
            else:
                img_gray[i][j] = 255

    fig = plt.figure(figsize=(15, 8))
    fig.add_subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_orig)
    #plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    fig.add_subplot(1, 2, 2)
    plt.title('Rendered Image')
    plt.imshow(img_gray, cmap='gray')
    #fig.savefig(dst_path)
    plt.show()
    img.imsave(dst_path, img_gray, cmap = 'gray')

#otsu('lena.jpg')
#======= MAIN =======
if __name__ == '__main__':
    otsu(*argv[1:])