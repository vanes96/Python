from sys import argv
import os.path, json
from scipy import stats
import random
import numpy as np
import cv2
import math
import time

def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    height, width = img_size
    a, b, c = line_params
    data = np.zeros((height, width, 3), np.uint8)
    n_inlier = int(n_points * inlier_ratio)
    n_even = n_points - n_inlier
    delta = 0.01

    for y in range(height):
        if n_inlier == 0:
            break
        for x in range(width):
            if abs(a * (x - width // 2) + b * (y - height // 2) + c) < delta:
                dy, dx = random.normalvariate(0, sigma), random.normalvariate(0, sigma)
                while not(height > int(y + dy) >= 0 and width > int(x + dx) >= 0):
                    dy, dx = random.normalvariate(0, sigma), random.normalvariate(0, sigma)
                data[int(y + dy), int(x + dx)] = (255,255,255)
                n_inlier -= 1
                if n_inlier == 0:
                    break

    for i in range(n_even):
        yr, xr = random.randint(0, height - 1), random.randint(0, width - 1)
        data[yr, xr] = (255,255,255)
    return data

def compute_ransac_thresh(alpha, sigma):
    x = 0
    delta = 0.01
    chi = stats.chi2(sigma)
    while abs(chi.cdf(x) - alpha) > delta:
        x += 1.0
    return x

def compute_ransac_iter_count(conv_prob, inlier_ratio, n_points):
    n = math.log2(1 - conv_prob) / math.log2(1 - inlier_ratio ** n_points)
    n = int(math.ceil(n))
    return n

def compute_line_ransac(data, thresh, n_iter, n_points):
    points = np.zeros((n_points, 2), np.int)
    bestLine = -1, -1
    maxR = -1
    height, width = len(data), len(data[0])
    data_points = []
    for y in range(height):
        for x in range(width):
            if data[y][x][0] != 0:
                data_points.append((x, y))
    count_points = len(data_points)

    for i in range(n_iter):
        for j in range(n_points):
            ir = random.randint(0, count_points - 1)
            while points.tolist().count(data_points[ir]) != 0:
                ir = random.randint(0, count_points - 1)
            points[j] = data_points[ir]

        r = R(points, data_points, thresh)
        if r[0] > maxR:
            maxR = r[0]
            bestLine = r[1], r[2]
            #drawLine(bestLine, data)
        #print("cost = " + str(r[0]) + "; isBest = " + str(True if r[0] == maxR else False))
    return bestLine

def R(points, data_points, thresh):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    k, b = stats.linregress(x=X, y=Y)[:2]
    sum = 0
    for point in data_points:
        sum += cost(dist(point=point, line=(k, b)), thresh)
    return sum, k, b

def cost(dist, thresh):
    if dist ** 2 <= thresh ** 2:
        return 1
    else:
        return 0

def dist(point, line):
    a, b, c = -line[0], 1.0, -line[1]
    x, y = point
    return abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)

def drawLine(line, data, is_best = False):
    k, b = line
    height, width = len(data), len(data[0])
    for x in range(width):
        y = int(k * x + b)
        if height > y >= 0:
            if is_best:
                data[y][x] = (0,255,0)
            else:
                data[y][x] = (255, 0, 0)
    cv2.imshow("RANSAC", cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

def main_debug():
    #time.clock()
    data = generate_data(img_size=(512, 512), line_params=(1.0, 3.0, 2.0), n_points=1000, sigma=1, inlier_ratio=0.3)

    thresh = compute_ransac_thresh(alpha=0.95, sigma=1)

    n_points = 2
    n_iter = compute_ransac_iter_count(conv_prob=0.95, inlier_ratio=0.3, n_points=n_points)

    detected_line = compute_line_ransac(data, thresh, n_iter, n_points)

    print("Detected_Line:  Y = " + str(round(detected_line[0], 2)) + "*X " +
          ("- " if detected_line[1] < 0 else "+ ") + str(round(abs(detected_line[1]), 2)))
    drawLine(detected_line, data, is_best=True)
    #print(time.clock())
    cv2.waitKey()

def main():
    assert len(argv) == 2
    assert os.path.exists(argv[1])
    with open(argv[1]) as fin:
        params = json.load(fin)

    data = generate_data((params['w'], params['h']), (params['a'], params['b'], params['c']),
                          params['n_points'], params['sigma'], params['inlier_ratio'])


    thresh = compute_ransac_thresh(params['alpha'], params['sigma'])

    n_points = 2
    n_iter = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'], n_points)

    detected_line = compute_line_ransac(data, thresh, n_iter, n_points)

    print("Detected_Line:  Y = " + str(round(detected_line[0], 2)) + "*X " +
          ("- " if detected_line[1] < 0 else "+ ") + str(round(abs(detected_line[1]), 2)))
    drawLine(detected_line, data, is_best=True)
    cv2.waitKey()

if __name__ == '__main__':
    main()

