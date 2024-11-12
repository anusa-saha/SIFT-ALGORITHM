import cv2
import matplotlib.pyplot as plt
image = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(image_with_keypoints, cmap='gray')
plt.title("SIFT Keypoints")
plt.axis('off')
plt.show()
cv2.imwrite('image_with_keypoints.jpg',image_with_keypoints )
