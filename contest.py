from sys import argv
import matplotlib.image as img
import csv
import os.path
import cv2
from os import listdir
from os.path import isfile, join

def show_image(img):
    cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey()

def save_image(img_name, Class):
    img_orig = cv2.imread("Merged\\" + img_name)
    path = "Results\\" + str(Class) + "\\" + img_name
    img.imsave(path, img_orig)

def check_border(img):
    height, width = len(img), len(img[0])
    sides, sums = [0, 0, 0, 0], [0, 0, 0, 0]

    for y in range(height):
        sums[0] += img[y][0][0]
    for x in range(width):
        sums[1] += img[0][x][0]
    for y in range(height):
        sums[2] += img[y][width - 1][0]
    for x in range(width):
        sums[3] += img[height - 1][x][0]

    for i in range(len(sides)):
        if sums[i] == 0:
            sides[i] = 1
    return sides

def is_class0(img_name):
    img = cv2.imread(img_name)
    height, width = len(img), len(img[0])
    offsets = check_border(img)
    yl, xl = (height + offsets[1] - offsets[3]) // 2 - 2, (width + offsets[0] - offsets[2]) // 2 - 2
    sums, d = [0, 0, 0], 4

    for i in range(3):
        for y in range(yl - d * i, yl + d * (i + 1)):
            for x in range(xl - d * i, xl + d * (i + 1)):
                sums[i] += img[y][x][0]

    if sums[1] // 255 == 128 and 0 == sums[0] == sums[2] - sums[1]:
        return True
    else:
        return False

def is_class1(img_name):
    img = cv2.imread(img_name)
    height, width = len(img), len(img[0])
    count_lines_vert, count_lines_horiz = 0, 0
    ver, hor = [], []
    xc1, xc2, yc1, yc2 = 0, 0, 0, 0
    xt1, xt2, yt1, yt2 = 0, 0, 0, 0
    for x in range(width):
        sum = 0
        for y in range(height):
            sum += img[y][x][0]
        h = sum // 255
        if 0.3 * height < h < 0.4 * height:
            #print(h / height)
            count_lines_vert += 1
            ver.append(h)
            if xc1 == 0:
                xc1 = x
            xc2 = x
            n = ver.__len__()
            if xt1 == 0 and ver[n - 2] != h:
                xt1 = x
            if h == ver[0] and ver[n - 2] != h:
                xt2 = x

    for y in range(height):
        sum = 0
        for x in range(width):
            sum += img[y][x][0]
        w = sum // 255
        if 0.3 * width < w < 0.4 * width:
            #print(w / width)
            count_lines_horiz += 1
            hor.append(w)
            if yc1 == 0:
                yc1 = y
            yc2 = y
            n = hor.__len__()
            if yt1 == 0 and hor[n - 2] != w:
                yt1 = y
            if w == hor[0] and hor[n - 2] != w:
                yt2 = y

    code_width, code_height = xc2 - xc1, yc2 - yc1
    text_width, text_height = xt2 - xt1, yt2 - yt1

    if count_lines_vert > 200 and code_width * 0.24 < text_width < code_width * 0.32 or \
       count_lines_horiz > 200 and code_height * 0.24 < text_height < code_height * 0.32:
        return True
    else:
        return False

def is_class2(img_name):
    img = cv2.imread(img_name)
    height, width = len(img), len(img[0])
    count_lines_vert, count_lines_horiz = 0, 0

    for x in range(width):
        sum = 0
        for y in range(height):
            sum += img[y][x][0]
        if sum == 0:
            count_lines_vert += 1

    for y in range(height):
        sum = 0
        for x in range(width):
            sum += img[y][x][0]
        if sum == 0:
            count_lines_horiz += 1

    if count_lines_vert > 150 or count_lines_horiz > 150:
        return True
    else:
        return False

def is_class3(img_name):
    img = cv2.imread(img_name)
    height, width = len(img), len(img[0])
    w1_ver, w2_ver, w_ver = 0, 0, 0
    h1_hor, h2_hor, h_hor = 0, 0, 0

    for x in range(width):
        sum = 0
        for y in range(height):
            sum += img[y][x][0]
        h = sum // 255
        if h == 40:
            w_ver += 1
        else:
            if w_ver == 24:
                w1_ver = w_ver
            elif w_ver == 21:
                w2_ver = w_ver
            w_ver = 0

    for y in range(height):
        sum = 0
        for x in range(width):
            sum += img[y][x][0]
        w = sum // 255
        if w == 40:
            h_hor += 1
        else:
            if h_hor == 24:
                h1_hor = h_hor
            elif h_hor == 21:
                h2_hor = h_hor
            h_hor = 0

    if w1_ver == 24 and w2_ver == 21 or h1_hor == 24 and h2_hor == 21:
        return True
    else:
        return False

def is_rect(img, corner):
    sums = [0, 0, 0]
    rect_size, d = 28, 0

    for i in range(3):
        for y in range(corner[1] + d, corner[1] + rect_size):
            for x in range(corner[0] + d, corner[0] + rect_size):
                sums[i] += img[y][x][0]
        d += 4
        rect_size -= d

    if 0 == sums[2] == sums[0] - sums[1] and sums[0] // 255 == 256:
        return True
    else:
        return False

def is_class4(img_name):
    img = cv2.imread(img_name)
    height, width = len(img), len(img[0])
    offsets = check_border(img)
    n_corners, rect_size, field = 0, 28, 16
    corners = [(field + offsets[0], field + offsets[1]), (width - (field + offsets[2] + rect_size), field + offsets[1]), \
                                         (width - (field + offsets[2] + rect_size), height - (field + offsets[3] + rect_size)), \
                                         (field + offsets[0], height - (field + offsets[3] + rect_size))]
    for i in range(4):
        if is_rect(img, corners[i]):
            n_corners += 1

    if n_corners == 3:
        return True
    else:
        return False

def classify(img_name, folder):
    path = folder + "\\" + img_name
    #img_orig = cv2.imread(img_name)
    if is_class0(path):
        #save_image(img_name, 0)
        return 0
    if is_class4(path):
        #save_image(img_name, 4)
        return 4
    elif is_class3(path):
        #save_image(img_name, 3)
        return 3
    elif is_class2(path):
        #save_image(img_name, 2)
        return 2
    elif is_class1(path):
        #save_image(img_name, 1)
        return 1
    else:
        #save_image(img_name, 5)
        return 5

    #show_image(img)
    #save_image(img,"image_.png")

#========================== MAIN ===============================
folder = "Merged"
dir = os.getcwd() + "\\" + folder
images = [f for f in listdir(dir) if isfile(join(dir, f))]

result = open('results.csv', 'w')
result.write("# fname,class\n")
images = os.listdir(folder)
for image in images:
    result.write("{0},{1}\n".format(image, str(classify(image, folder))))
result.close()


