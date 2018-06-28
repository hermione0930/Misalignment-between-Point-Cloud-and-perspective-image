Project2: Investigate the misalignment between Point Cloud and perspective image

## Personal Information:
Name: Dan Wu, Lingfei Cui, Sheng Ma


## Program Information:
OS: Mac
Environment: Python 3.5+


## How to run the project:
1. Put front.jpg, back.jpg, left.jpg, right.jpg, final_project_point_cloud.fuse, misalignment.py into the same directory
2. Input the following commands in Terminal:
	python misalignment.py


## Result:
1. Point Cloud image:
front.png, back.png, left.png, right.png
2. Point Cloud image using intensity information with histogram equalization:
front_with_intensity.png, back_with_intensity.png, left_with_intensity.png, right_with_intensity.png
3. Matching images generated with method SIFT:
front_matching_sift.png, back_matching_sift.png, left_matching_sift.png, right_matching_sift.png
4. Matching images generated with method ORB:
front_matching_orb.png, back_matching_orb.png, left_matching_orb.png, right_matching_orb.png
5. Calculated misalignment for matching images(8 images) by angle.
6. Calculated misalignment for matching images(8 images) by distance.



## References:
Revised_Coordinate_Transformations.pptx
https://ww2.mathworks.cn/matlabcentral/fileexchange/7942-covert-lat--lon--alt-to-ecef-cartesian
https://ww2.mathworks.cn/help/map/ref/ecef2enu.html
https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
https://www.programcreek.com/python/example/89342/cv2.drawMatchesKnn


