#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sklearn.cluster import DBSCAN

class BoxDetectionNode:
    def __init__(self):
        rospy.init_node('box_detection_node', anonymous=True)
        self.pc_sub = rospy.Subscriber('/planning/obstacle/surround_cloud_map', PointCloud2, self.pc_callback)
        self.marker_pub = rospy.Publisher('/detected_lines', MarkerArray, queue_size=10)
        rospy.loginfo("Box Detection Node Initialized")

    def pc_callback(self, msg):
        # Convert PointCloud2 to numpy array
        pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points_3d = np.array(list(pc_data))

        # Project to 2D (XY plane)
        points_2d = points_3d[:, :2]

        # Detect edges (e.g., using Canny on a 2D image)
        edges = self.detect_edges(points_2d)

        # Cluster edge points and fit lines
        lines = self.cluster_and_fit_lines(edges)

        # Publish lines as MarkerArray
        self.publish_lines(lines)

    def detect_edges(self, points_2d):
        # Create a 2D image from the points
        image_size = (640, 480)
        image = np.zeros(image_size, dtype=np.uint8)
        for point in points_2d:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                image[y, x] = 255

        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        cv2.imshow("Fitted Ellipses", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return edges

    def cluster_and_fit_lines(self, edges):
        # Find edge points
        edge_points = np.column_stack(np.where(edges > 0))

        # Cluster edge points using DBSCAN
        db = DBSCAN(eps=5, min_samples=10).fit(edge_points)
        labels = db.labels_

        # Fit lines to each cluster
        lines = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            cluster_points = edge_points[labels == label]
            if len(cluster_points) > 1:
                [vx, vy, x, y] = cv2.fitLine(cluster_points, cv2.DIST_L2, 0, 0.01, 0.01)
                lines.append((vx, vy, x, y))
        return lines

    def publish_lines(self, lines):
        marker_array = MarkerArray()
        for i, line in enumerate(lines):
            vx, vy, x, y = line
            marker = Marker()
            marker.header.frame_id = "map"  # Replace with your frame ID
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Line width
            marker.color.a = 1.0  # Alpha
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0
            marker.color.b = 1.0

            # Define line endpoints
            p1 = Point()
            p1.x = float(x - 100 * vx)
            p1.y = float(y - 100 * vy)
            p1.z = 0.0
            p2 = Point()
            p2.x = float(x + 100 * vx)
            p2.y = float(y + 100 * vy)
            p2.z = 0.0

            marker.points.append(p1)
            marker.points.append(p2)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        node = BoxDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass