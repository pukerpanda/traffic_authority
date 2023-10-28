import cv2
import numpy as np
from osgeo import gdal, osr

# Load the oblique image and the reference geospatial image
oblique_image = cv2.imread('oblique_image.jpg')
reference_image = cv2.imread('reference_image.jpg')

# Define corresponding points in the oblique image and the reference image
# You'll need to manually specify these matching points based on your data
oblique_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
reference_points = np.array([[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]], dtype=np.float32)

# Calculate the homography matrix using OpenCV
homography_matrix, _ = cv2.findHomography(oblique_points, reference_points, cv2.RANSAC, 5.0)

# Define the transformation function
def transform_coordinates(homography_matrix, point):
    point = np.array([point], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0]

# Test the transformation with a point from the oblique image
oblique_point = [x1, y1]
geospatial_point = transform_coordinates(homography_matrix, oblique_point)

# Initialize GDAL for geospatial transformations
gdal.UseExceptions()
gdal.AllRegister()

# Open the reference geospatial image for transformation information
reference_ds = gdal.Open('reference_image.jpg')

# Create spatial reference objects for the source and target coordinate systems
source_srs = osr.SpatialReference()
source_srs.ImportFromWkt(reference_ds.GetProjection())

target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(4326)  # WGS84

# Create a coordinate transformation object
transform = osr.CoordinateTransformation(source_srs, target_srs)

# Transform the geospatial point to WGS84
geospatial_point = transform.TransformPoint(geospatial_point[0], geospatial_point[1])

# Print the georeferenced coordinates
print("Georeferenced Point (WGS84):", geospatial_point[0], geospatial_point[1])
