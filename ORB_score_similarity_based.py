import cv2

# Load the two images
image1 = cv2.imread('WIN_20241217_11_26_06_Pro.jpg')
image2 = cv2.imread('WIN_20241007_14_43_42_Pro.jpg')

# Resize images to 640x480
image1_resized = cv2.resize(image1, (640, 480), interpolation=cv2.INTER_AREA)
image2_resized = cv2.resize(image2, (640, 480), interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Initialize the brute force matcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Calculate similarity score based on the number of matches
num_matches = len(matches)
similarity_score = num_matches / max(len(keypoints1), len(keypoints2))  # Normalize by the number of keypoints

# Draw the top matches
matched_image = cv2.drawMatches(image1_resized, keypoints1, image2_resized, keypoints2, matches[:50], None, flags=2)
print(f"Number of Matches: {num_matches}")
print(f"Similarity Score: {similarity_score:.4f}")
# Display the matched image
cv2.imshow("Matches", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of matches and similarity score
#print(f"Number of Matches: {num_matches}")
#print(f"Similarity Score: {similarity_score:.4f}")
# Print the number of matches and similarity score

