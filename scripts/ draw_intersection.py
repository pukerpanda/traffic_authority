import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import cv2

import IPython

cam_image_path = 'data/antene_cross.bmp'
map_image_path = "data/antene_cross_osm.png"
camera_image = plt.imread(cam_image_path)

antene_cross = (45.76484,21.23196)
nearest = 200  # meters
G = ox.graph_from_point(antene_cross, dist=nearest, network_type="all")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("OpenStreetMap Graph and Image")

axes[0, 0].imshow(camera_image)
axes[0, 0].set_title("Image")

graph = ox.project_graph(G)

ox.plot_graph(graph, show=False, close=False, ax=axes[0, 1], save=True, filepath=map_image_path, bgcolor="white", edge_color="black", node_color="none",)
map_image = cv2.imread(map_image_path)
axes[0, 1].set_title("OpenStreetMap Graph")

# Apply edge detection to the grayscale image
gray_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
edges = cv2.Canny(gray_image, threshold1=50, threshold2=150, apertureSize=3)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_with_lines = gray_image.copy()
#IPython.embed()

height, width, _ = camera_image.shape
# Create an empty image of the same size and color as the original
image_with_lines = np.full((height, width, 3), (255,255,255), dtype=np.uint8)
for contour in contours:
    cv2.drawContours(image_with_lines, [contour], -1, (0, 0, 255), 2)




# Feature detection (you can choose a feature detector that suits your needs)
# In this example, we'll use ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()

# Find keypoints and descriptors in both images
kp1, des1 = orb.detectAndCompute(image_with_lines, None)
kp2, des2 = orb.detectAndCompute(map_image, None)


# Draw keypoints on the image
image_with_lines_with_keypoints = cv2.drawKeypoints(image_with_lines, kp1, outImage=None)
map_image_with_keypoints = cv2.drawKeypoints(map_image, kp2, outImage=None)


axes[1, 0].imshow(image_with_lines_with_keypoints, cmap="gray")
axes[1, 0].set_title("Contours")


# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
matches = bf.match(des1, des2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)
# Extract matching points

camera_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
map_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix using RANSAC
homography_matrix, mask = cv2.findHomography(camera_points, map_points, cv2.RANSAC, 5.0)
print(homography_matrix)

# Apply the homography to the camera image to align it with the map image
aligned_image = cv2.warpPerspective(camera_image, homography_matrix, (map_image.shape[1], map_image.shape[0]))


# merged_image = cv2.addWeighted(gray_image, 0.7, fig, 0.3, 0)
axes[1, 1].imshow(map_image_with_keypoints, cmap="gray")
axes[1, 1].set_title("Image with Contours")

plt.show()

