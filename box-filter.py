from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import os.path
import cv2

def box_flter(src_path, dst_path = 'output.jpg', w = 3, h = 3):
    if w % 2 == 0:
        w += 1
    if h % 2 == 0:
        h += 1

    img_orig = cv2.imread(src_path)
    width, height = img_orig.shape[1], img_orig.shape[0]
    img_sum = np.cumsum(np.cumsum(img_orig, 0), 1)
    img_blur = np.zeros((height, width, 3), np.uint8)

    for i in range(0, height):
        i1, i2 = i - (h // 2) - 1, i + (h // 2)
        dh = 0

        if i1 < 0:
            dh += -i1
            i1 = 0
        if i2 >= height:
            dh += i2 - (height - 1)
            i2 = height - 1

        for j in range(0, width):
            j1, j2 = j - (w // 2) - 1, j + (w // 2)
            dw = 0

            if j1 < 0:
                dw += -j1
                j1 = 0
            if j2 >= width:
                dw += j2 - (width - 1)
                j2 = width - 1

            sum = img_sum[i2][j2] + img_sum[i1][j1] - img_sum[i1][j2] - img_sum[i2][j1]
            img_blur[i][j] = sum // ((w - dw) * (h - dh))

    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    fig.add_subplot(1, 2, 2)
    plt.title('Blurred Image. Box: width = ' + str(w) + ', height = ' + str(h))
    #cv2.boxFilter(src=img_orig, ddepth=0, dst=img_blur,ksize=(30, 30), borderType=cv2.BORDER_REFLECT)
    plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
    plt.show()
    fig.savefig(dst_path)

if __name__ == '__main__':
    assert os.path.exists(argv[1])
    assert len(argv) != 4
    if len(argv) == 5:
        argv[3], argv[4] = int(argv[3]), int(argv[4])
        assert argv[3] > 0
        assert argv[4] > 0
    box_flter(*argv[1:])
    #box_flter('lena.jpg', dst_path = 'output.jpg', w = 3, h = 3)