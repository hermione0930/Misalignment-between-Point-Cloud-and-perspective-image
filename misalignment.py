import numpy as np
import math
import cv2

######################################################################################
#                        Section 1: Transfer LLA to Image Coordinate                 #
######################################################################################
'''
Convert LLA to ECEF
Input: latitude, longtitude, altitude
Output: x = ECEF X-coordinate (m)
        y = ECEF Y-coordinate (m)
        z = ECEF Z-coordinate (m)
        cos_phi = cosine of geodetic latitude
        sin_phi = sine of geodetic latitude
        cos_lambda = cosine of longtitude
        sin_lambda = sine of longtitude
Reference: https://ww2.mathworks.cn/matlabcentral/fileexchange/7942-covert-lat--lon--alt-to-ecef-cartesian
'''
def lla_to_ecef(lat, lon, alt):
    a = 6378137.0
    b = 6356752.314245

    f = (a - b)/a
    e = math.sqrt(f*(2-f))

    sin_phi = (np.sin(lat*math.pi/180))
    cos_phi = (np.cos(lat*math.pi/180))
    sin_lambda = (np.sin(lon*math.pi/180))
    cos_lambda = (np.cos(lon*math.pi/180))

    N = a/(math.sqrt(1-(e**2)*(sin_phi**2)))

    x = (alt + N)*cos_lambda*cos_phi
    y = (alt + N)*cos_phi*sin_lambda
    z = (alt + (1-e**2)*N)*sin_phi

    return x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda

'''
Convert ECEF to ENU
Input: The same as the output of previous function x, y, z, cos_phi, sin_phi, cos_lambda, sin_lambda
Output: e = ENU east-coordinate
        n = ENU north-coordinate
        u = ENU up-coordinate
Reference: https://ww2.mathworks.cn/help/map/ref/ecef2enu.html
'''
def ecef_to_enu(x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda):
    x0, y0, z0, cos_phi0, sin_phi0, cos_lambda0, sin_lambda0 = lla_to_ecef(45.90414414, 11.02845385,227.5819)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    e = -sin_lambda0*dx          + cos_lambda0*dy
    n = -cos_lambda0*sin_phi0*dx - sin_phi0*sin_lambda0*dy + cos_phi0*dz
    u = cos_phi0*cos_lambda0*dx  + cos_phi0*sin_lambda0*dy + sin_phi0*dz

    return e, n, u

'''
Convert ENU to Camera Coordinate
Input: The same as the output of previous function e, n, u
Output: x_c = Camera X-coordinate
        y_c = Camera Y-coordinate
        z_c = Camera Z-coordinate
'''
def enu_to_cc(e, n, u):
    qs, qx, qy, qz = 0.362114, 0.374050, 0.592222, 0.615007 # Camera parameters

    P = [n,e,-u]
    Rq = [
    		[1-2*qy**2-2*qz**2 , 2*qx*qy+2*qs*qz   , 2*qx*qz-2*qs*qy],
    		[2*qx*qy-2*qs*qz   , 1-2*qx**2-2*qz**2 , 2*qy*qz+2*qs*qx],
    		[2*qx*qz+2*qs*qy   , 2*qy*qz-2*qs*qx   , 1-2*qx**2-2*qy**2]
    	 ]
    x_c = np.dot(Rq,P)[0]
    y_c = np.dot(Rq,P)[1]
    z_c = np.dot(Rq,P)[2]

    return x_c, y_c, z_c

'''
Convert Camera Coordinate to Image Coordinate
Input: The same as the output of previous function x_c, y_c, z_c
Output: x_i = Image X-coordinate
        y_i = Image Y-coordinate
        direction = 1 represents front image
                    2 represents back image
                    3 represents left image
                    4 represents right image
'''
def cc_to_ic(x_c, y_c, z_c):
    global x_i
    global y_i
    global direction
    # front
    if z_c > 0 and z_c > abs(x_c) and z_c > abs(y_c):
        x_i = int(y_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(x_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 1
    # back
    if z_c < 0 and z_c < -abs(x_c) and z_c < -abs(y_c):
        x_i = int(-y_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(x_c/z_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 2
    # left
    if x_c < 0 and x_c < -abs(z_c) and x_c < -abs(y_c):
        x_i = int(-y_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(-z_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 3
    # right
    if x_c > 0 and x_c > abs(y_c) and x_c > abs(z_c):
        x_i = int(y_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        y_i = int(-z_c/x_c*(resolution - 1)/2+(resolution + 1)/2)
        direction = 4

    return x_i, y_i, direction


######################################################################################
#                       Section 2: Draw Point Cloud on image                         #
######################################################################################
resolution = 2048
data = open('final_project_point_cloud.fuse', 'rb')

Img_front = np.zeros((resolution,resolution))
Img_back = np.zeros((resolution,resolution))
Img_left = np.zeros((resolution,resolution))
Img_right = np.zeros((resolution,resolution))
Img_front_with_intensity = np.zeros((resolution,resolution))
Img_back_with_intensity = np.zeros((resolution,resolution))
Img_left_with_intensity = np.zeros((resolution,resolution))
Img_right_with_intensity = np.zeros((resolution,resolution))

for line in data:
    line = line.decode('utf8').strip().split(' ')
    intensity = float(line[3])
    x, y, z, cos_phi, sin_phi, cos_lambda, sin_lambda = lla_to_ecef(float(line[0]), float(line[1]), float(line[2]))
    e, n, u = ecef_to_enu(x,y,z,cos_phi, sin_phi, cos_lambda, sin_lambda)
    x_c, y_c, z_c = enu_to_cc(e, n, u)
    x_i, y_i, direction = cc_to_ic(x_c, y_c, z_c)
    if direction == 1:
        Img_front[x_i][y_i] = 255
        Img_front_with_intensity[x_i][y_i] = intensity
    if direction == 2:
        Img_back[x_i][y_i] = 255
        Img_back_with_intensity[x_i][y_i] = intensity
    if direction == 3:
        Img_left[x_i][y_i] = 255
        Img_left_with_intensity[x_i][y_i] = intensity
    if direction == 4:
        Img_right[x_i][y_i] = 255
        Img_right_with_intensity[x_i][y_i] = intensity

cv2.imwrite('front.png',Img_front)
cv2.imwrite('back.png',Img_back)
cv2.imwrite('right.png',Img_right)
cv2.imwrite('left.png',Img_left)
cv2.imwrite('front_with_intensity.png',Img_front_with_intensity)
cv2.imwrite('back_with_intensity.png',Img_back_with_intensity)
cv2.imwrite('right_with_intensity.png',Img_right_with_intensity)
cv2.imwrite('left_with_intensity.png',Img_left_with_intensity)

# Calculate Histogram Equalization
img1 = cv2.imread('front_with_intensity.png',0)
equ1 = cv2.equalizeHist(img1)
img2 = cv2.imread('back_with_intensity.png',0)
equ2 = cv2.equalizeHist(img2)
img3 = cv2.imread('left_with_intensity.png',0)
equ3 = cv2.equalizeHist(img3)
img4 = cv2.imread('right_with_intensity.png',0)
equ4 = cv2.equalizeHist(img4)

cv2.imwrite('front_equ.png',equ1)
cv2.imwrite('back_equ.png',equ2)
cv2.imwrite('left_equ.png',equ3)
cv2.imwrite('right_equ.png',equ4)


######################################################################################
#       Section 3: Find Misalignment between Point Cloud and perspective image       #
######################################################################################
'''
References: 
https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''
img_front0 = cv2.imread("front.jpg")
img_front = cv2.imread("front.png")
img_back0 = cv2.imread("back.jpg")
img_back = cv2.imread("back.png")
img_left0 =cv2.imread("left.jpg")
img_left =cv2.imread("left.png")
img_right0 = cv2.imread("right.jpg")
img_right = cv2.imread("right.png")

# Calculate misalignment by computing the average of the sum of radians
def calculate_misalignment_by_angle1(keypoints1, keypoints2, matches1to2):
    angles = []
    for m in matches1to2:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        point1 = tuple(np.round(keypoints1[img1_idx].pt).astype(int))
        point2 = tuple(np.round(keypoints2[img2_idx].pt).astype(int) + np.array([2048, 0]))
        diff_point1_point2 = math.atan((float)(point2[1] - point1[1]) / (point1[0] - point2[0])) * (180 / math.pi)
        diff_point1_point2 = abs(diff_point1_point2)
        angles.append(diff_point1_point2)
    return (sum(angles) / len(angles))

# Calculate misalignment by computing the average of the euclidean distance
def calculate_misalignment_by_distance1(keypoints1, keypoints2, matches1to2):
    euclidean = []
    for m in matches1to2:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        point1 = tuple(np.round(keypoints1[img1_idx].pt).astype(int))
        point2 = tuple(np.round(keypoints2[img2_idx].pt).astype(int))
        euclidean_distance = math.sqrt((float)(point2[1] - point1[1])**2 + (point2[0] - point1[0])**2)
        euclidean.append(euclidean_distance)
    return (sum(euclidean) / len(euclidean))

def calculate_misalignment_by_angle2(keypoints1, keypoints2, matches1to2):
    angles = []
    for m, n in matches1to2:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
        x2 = x2 + 2048
        diff_point1_point2 = math.atan((float)(y2 - y1) / (x2 - x1)) * (180 / math.pi)
        diff_point1_point2 = abs(diff_point1_point2)
        angles.append(diff_point1_point2)
    return (sum(angles) / len(angles))

def calculate_misalignment_by_distance2(keypoints1, keypoints2, matches1to2):
    euclidean = []
    for m, n in matches1to2:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
        euclidean_distance = math.sqrt((float)(y2 - y1)**2 + (x2 - x1)**2)
        euclidean.append(euclidean_distance)
    return (sum(euclidean) / len(euclidean))

######################################################
# Method1: Brute-Force Matching with ORB Descriptors #
######################################################
print ("--------------------------------------------------")
print ("Method1: Brute-Force Matching with ORB Descriptors")
orb = cv2.ORB_create(1500)
# Create matcher
bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Front matching
(kp_f_orb, des_f_orb) = orb.detectAndCompute(img_front, None)
(kp_f0_orb, des_f0_orb) = orb.detectAndCompute(img_front0 , None)

matches1_orb = bf1.match(des_f_orb, des_f0_orb)
matches1_orb = sorted(matches1_orb, key=lambda val: val.distance)

front_matching_orb = cv2.drawMatches(img_front, kp_f_orb, img_front0, kp_f0_orb, matches1_orb[:100], None, flags = 2)
cv2.imwrite('front_matching_orb.png',front_matching_orb)
front_result_orb1 = calculate_misalignment_by_angle1(kp_f_orb, kp_f0_orb, matches1_orb)
front_result_orb2 = calculate_misalignment_by_distance1(kp_f_orb, kp_f0_orb, matches1_orb)
print ("front")
print ("The misalignment result for front image calculated by angle is", front_result_orb1)
print ("The misalignment result for front image calculated by distance is", front_result_orb2)

# Back matching
(kp_b_orb, des_b_orb) = orb.detectAndCompute(img_back, None)
(kp_b0_orb, des_b0_orb) = orb.detectAndCompute(img_back0, None)

matches2_orb = bf1.match(des_b_orb, des_b0_orb)
matches2_orb = sorted(matches2_orb, key=lambda val: val.distance)

back_matching_orb = cv2.drawMatches(img_back, kp_b_orb, img_back0, kp_b0_orb, matches2_orb[:100], None, flags = 2)
cv2.imwrite('back_matching_orb.png',back_matching_orb)
back_result_orb1 = calculate_misalignment_by_angle1(kp_b_orb, kp_b0_orb, matches2_orb)
back_result_orb2 = calculate_misalignment_by_distance1(kp_b_orb, kp_b0_orb, matches2_orb)
print ("back")
print ("The misalignment result for back image calculated by angle is", back_result_orb1 )
print ("The misalignment result for back image calculated by distance is", back_result_orb2)

# Left matching
(kp_l_orb, des_l_orb) = orb.detectAndCompute(img_left, None)
(kp_l0_orb, des_l0_orb) = orb.detectAndCompute(img_left0, None)

matches3_orb = bf1.match(des_l_orb, des_l0_orb)
matches3_orb = sorted(matches3_orb, key=lambda val: val.distance)

left_matching_orb = cv2.drawMatches(img_left, kp_l_orb, img_left0, kp_l0_orb, matches2_orb[:100], None, flags = 2)
cv2.imwrite('left_matching_orb.png',left_matching_orb)
left_result_orb1 = calculate_misalignment_by_angle1(kp_l_orb, kp_l0_orb, matches3_orb)
left_result_orb2 = calculate_misalignment_by_distance1(kp_l_orb, kp_l0_orb, matches3_orb)
print("left")
print ("The misalignment result for left image calculated by angle is", left_result_orb1 )
print ("The misalignment result for left image calculated by distance is", left_result_orb2)

# Right matching
(kp_r_orb, des_r_orb) = orb.detectAndCompute(img_right, None)
(kp_r0_orb, des_r0_orb) = orb.detectAndCompute(img_right0, None)

matches4_orb = bf1.match(des_r_orb, des_l0_orb)
matches4_orb = sorted(matches4_orb, key=lambda val: val.distance)

right_matching_orb = cv2.drawMatches(img_right, kp_r_orb, img_right0, kp_r0_orb, matches2_orb[:100], None, flags = 2)
cv2.imwrite('right_matching_orb.png',right_matching_orb)
right_result_orb1 = calculate_misalignment_by_angle1(kp_r_orb, kp_r0_orb, matches4_orb)
right_result_orb2 = calculate_misalignment_by_distance1(kp_r_orb, kp_r0_orb, matches4_orb)
print("right")
print ("The misalignment result for right image calculated by angle is", right_result_orb1 )
print ("The misalignment result for right image calculated by distance is", right_result_orb2)

######################################################################
# Method2: Brute-Force Matching with SIFT Descriptors and Ratio Test #
######################################################################
print ("------------------------------------------------------------------")
print ("Method2: Brute-Force Matching with SIFT Descriptors and Ratio Test")
# create sift extracter to extract sift features
sift = cv2.xfeatures2d.SIFT_create()
# create bf matcher
bf2 = cv2.BFMatcher()

# Front matching
kp_f_si, des_f_si = sift.detectAndCompute(img_front, None)
kp_f0_si, des_f0_si = sift.detectAndCompute(img_front0, None)
matches1_si = bf2.knnMatch(des_f_si, des_f0_si, k=2)

good1 = []
for m,n in matches1_si:
    if m.distance < 0.85 * n.distance:
        good1.append([m])

front_matching_si = cv2.drawMatchesKnn(img_front,kp_f_si,img_front0,kp_f0_si,good1, None, flags=2)
cv2.imwrite('front_matching_sift.png',front_matching_si)
front_result_si1 = calculate_misalignment_by_angle2(kp_f_si, kp_f0_si, matches1_si)
front_result_si2 = calculate_misalignment_by_distance2(kp_f_si, kp_f0_si, matches1_si)
print ("front")
print ("The misalignment result for front image calculated by angle is", front_result_si1)
print ("The misalignment result for front image calculated by distance is", front_result_si2)


# Back matching
kp_b_si, des_b_si = sift.detectAndCompute(img_back, None)
kp_b0_si, des_b0_si = sift.detectAndCompute(img_back0, None)
matches2_si = bf2.knnMatch(des_b_si, des_b0_si, k=2)
good2 = []
for m,n in matches2_si:
    if m.distance < 0.85 * n.distance:
        good2.append([m])

back_matching_si = cv2.drawMatchesKnn(img_back,kp_b_si,img_back0,kp_b0_si,good2, None, flags=2)
cv2.imwrite('back_matching_sift.png',back_matching_si)
back_result_si1 = calculate_misalignment_by_angle2(kp_b_si, kp_b0_si, matches2_si)
back_result_si2 = calculate_misalignment_by_distance2(kp_b_si, kp_b0_si, matches2_si)
print ("back")
print ("The misalignment result for back image calculated by angle is", back_result_si1 )
print ("The misalignment result for back image calculated by distance is", back_result_si2)

# Left matching
kp_l_si, des_l_si = sift.detectAndCompute(img_left, None)
kp_l0_si, des_l0_si = sift.detectAndCompute(img_left0, None)
matches3_si = bf2.knnMatch(des_l_si, des_l0_si, k=2)
good3 = []
for m,n in matches3_si:
    if m.distance < 0.85 * n.distance:
        good3.append([m])

left_matching_si = cv2.drawMatchesKnn(img_left,kp_l_si,img_left0,kp_l0_si,good3, None, flags=2)
cv2.imwrite('left_matching_sift.png',left_matching_si)
left_result_si1 = calculate_misalignment_by_angle2(kp_l_si, kp_l0_si, matches3_si)
left_result_si2 = calculate_misalignment_by_distance2(kp_l_si, kp_l0_si, matches3_si)
print("left")
print ("The misalignment result for left image calculated by angle is", left_result_si1 )
print ("The misalignment result for left image calculated by distance is", left_result_si2)

# Right matching
kp_r_si, des_r_si = sift.detectAndCompute(img_right, None)
kp_r0_si, des_r0_si = sift.detectAndCompute(img_right0, None)
matches4_si = bf2.knnMatch(des_r_si, des_r0_si, k=2)
good4 = []
for m,n in matches4_si:
    if m.distance < 0.85 * n.distance:
        good4.append([m])

right_matching_si = cv2.drawMatchesKnn(img_right,kp_r_si,img_right0,kp_r0_si,good4, None, flags=2)
cv2.imwrite('right_matching_sift.png',right_matching_si)
right_result_si1 = calculate_misalignment_by_angle2(kp_r_si, kp_r0_si, matches4_si)
right_result_si2 = calculate_misalignment_by_distance2(kp_r_si, kp_r0_si, matches4_si)
print("right")
print ("The misalignment result for right image calculated by angle is", right_result_si1 )
print ("The misalignment result for right image calculated by distance is", right_result_si2)

