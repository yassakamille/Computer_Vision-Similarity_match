import cv2
import numpy as np

def fun(img1,img2):

 sift = cv2.SIFT_create()

 kp1, des1 = sift.detectAndCompute(img1, None)
 kp2, des2 = sift.detectAndCompute(img2, None)

 bf = cv2.BFMatcher()

 matches = bf.knnMatch(des1, des2, k=2)

 good_matches = []
 for m, n in matches:
     if m.distance < 0.75 * n.distance:
         good_matches.append(m)


 src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
 dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

 matches_mask = mask.ravel().tolist()
 inlier_matches = [good_matches[i] for i in range(len(matches_mask)) if matches_mask[i] == 1]

 img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

 similarity_score = len(good_matches) / min(len(kp1), len(kp2))

 resized_img_matches = cv2.resize(img_matches, (700, 700))

 cv2.imshow('Matches', resized_img_matches)

#Deciding similarity threshold
 print("Similarity Score "+":", similarity_score)
 if similarity_score >= 0.018858307849133536:
    print("The two images are Similar !")
 else:
    print("The two images are NOT Similar !")

 cv2.waitKey(0)
 cv2.destroyAllWindows()

img1 = cv2.imread('assignment data/image4a.jpeg', 0)
img2 = cv2.imread('assignment data/image4b.jpeg', 0)


fun(img1=img1,img2=img2)


