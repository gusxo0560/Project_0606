import Testing
from scipy.spatial import distance
def Check(a, b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    # print((a[0] - b[0] ** 2 + 550))
    # print(((a[1] + b[1]) /2) * (a[1] - b[1]) ** 2 ** 0.5)
    calibration = (a[1] + b[1]) / 2
    # print(' ----------------- : ', dist)
    # print('0.25 * calibration : ', 0.25 * calibration)
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False
def Checking(a, b):
    dist = distance.euclidean(a, b)
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False


bbox_xyxy = [[ 847,6, 890,182],
             [ 824, 1, 856, 82],
             [1125, 73, 1170, 179],
             [ 415, 102, 456, 199],
             [ 193, 268, 256, 398],
             [ 963, 69,1001, 160],
             [ 256, 156, 296, 257],
             [ 798, 71, 837, 177],
             [  73, 191, 113, 281],
             [ 678, 35, 720,126],
             [1125, 73,1170, 179],
             [ 336, 120,380, 214],
             [ 672, 51,714, 150],
             [1096, 88,1136, 189],
             [ 496, 96, 536, 180],
             [ 487, 85, 521, 171],
             [ 446, 95, 484, 187],
             [ 986, 77, 1031, 188],
             [ 435, 597, 513, 717]]
# bbox_xyxy = [[1125, 73, 1170, 179],
# [1110, 70,1165, 170]]

#x1, y1, x2, y2 = [int(i) for i in box]
result = []
result2 = []
for i, de in enumerate(bbox_xyxy):
    rr = [(de[0] + de[2]) / 2, (de[1] + de[3]) / 2]
    # print('rr - > ', rr)
    for i_, de_ in enumerate(bbox_xyxy[i+1:]):
        rr2 = [(de_[0] + de_[2]) / 2, (de_[1] + de_[3]) / 2]
        # print('rr2 - > ', rr2)
        if Check(rr, rr2):
            result.append([rr, rr2])
            result2.append([i, i_])


print(result)
print('result2 :')
print(result2) #name 나오게
print('checking.....................\n')
result = []
result2 = []
for i, de in enumerate(bbox_xyxy):
    rr = [(de[0] + de[2]) / 2, (de[1] + de[3]) / 2]
    # print('rr - > ', rr)
    for i_, de_ in enumerate(bbox_xyxy[i+1:]):
        rr2 = [(de_[0] + de_[2]) / 2, (de_[1] + de_[3]) / 2]
        # print('rr2 - > ', rr2)
        if Checking(rr, rr2):
            result.append([rr, rr2])
            result2.append([i, i_])

print(result)
print('result2 :')
print(result2) #name 나오게
print('end...........')
