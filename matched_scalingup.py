import cv2
import matplotlib.pyplot as plt
image1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('resized_double.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)
matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], image2, flags=2) # draw first 50 matches
cv2.imshow('image', matched_img)
cv2.imwrite("matched_images_scaledup.jpg", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
