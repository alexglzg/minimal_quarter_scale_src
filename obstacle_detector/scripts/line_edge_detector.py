#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

class LidarEdgeDetector:
    def __init__(self):
        rospy.init_node("lidar_edge_detector", anonymous=True)

        # Subscribe to LiDAR point cloud topic
        self.pc_sub = rospy.Subscriber("/planning/obstacle/surround_cloud_map", PointCloud2, self.pointcloud_callback)

        # Publisher for visualization
        self.marker_pub = rospy.Publisher("/lidar/edges", MarkerArray, queue_size=1)

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to a list of (x, y) points
        points = self.pointcloud_to_2d(msg)

        if len(points) == 0:
            rospy.logwarn("No valid 2D points extracted!")
            return

        # Create a blank image for edge detection
        img_size = 500  # Size of the image
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Normalize and map points to image space
        points = np.array(points)
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        points = ((points - min_vals) / (max_vals - min_vals) * (img_size - 1)).astype(int)

        # Draw points onto the image
        for x, y in points:
            cv2.circle(img, (x, y), 1, 255, -1)

        # img_blurred = cv2.GaussianBlur(img, (5, 5), 1)

        # Edge detection using Canny
        edges = cv2.Canny(img, 100, 200)
        centerline = self.extract_centerline(edges)

        #  Standard Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        # Draw the lines
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        # cv2.imshow("Fitted Ellipses", centerline)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Line detection using Hough Transform
        # lines = cv2.HoughLinesP(edges_thin, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=60) 

        # # Convert detected lines into ROS markers
        # if lines is not None:
        #     lines = self.merge_lines(np.array(lines))
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.imshow("canny", img)
        #     cv2.imshow("Fitted Ellipses", output)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     self.publish_lines(lines, min_vals, max_vals, img_size)

    def extract_centerline(self, edges):
        """Reduces thick edges to a single-pixel width centerline."""
        thin_edges = cv2.ximgproc.thinning(edges)  # Skeletonization
        return thin_edges

    def merge_lines(self, lines, angle_threshold=10, distance_threshold=10):
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            new_line = True
            for idx, m_line in enumerate(merged_lines):
                mx1, my1, mx2, my2 = m_line[0]
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angle2 = np.arctan2(my2 - my1, mx2 - mx1) * 180 / np.pi

                # If angles are similar and endpoints are close, merge them
                if abs(angle1 - angle2) < angle_threshold and (
                    np.linalg.norm([x1 - mx1, y1 - my1]) < distance_threshold or
                    np.linalg.norm([x2 - mx2, y2 - my2]) < distance_threshold
                ):
                    merged_lines[idx][0] = [min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2)]
                    new_line = False
                    break

            if new_line:
                merged_lines.append(line)
    
        return np.array(merged_lines)

    def pointcloud_to_2d(self, cloud_msg):
        """Extracts 2D (x, y) points from a 3D PointCloud2 message."""
        points = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            points.append((x, y))  # Project onto XY plane
        return points

    def publish_lines(self, lines, min_vals, max_vals, img_size):
        """Publishes detected lines as a MarkerArray for visualization in RViz."""
        marker_array = MarkerArray()
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            # Convert back to original coordinate space
            x1 = min_vals[0] + (x1 / (img_size - 1)) * (max_vals[0] - min_vals[0])
            y1 = min_vals[1] + (y1 / (img_size - 1)) * (max_vals[1] - min_vals[1])
            x2 = min_vals[0] + (x2 / (img_size - 1)) * (max_vals[0] - min_vals[0])
            y2 = min_vals[1] + (y2 / (img_size - 1)) * (max_vals[1] - min_vals[1])

            # Create a line marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "edges"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width
            marker.color.r = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.points.append(self.create_point(x1, y1))
            marker.points.append(self.create_point(x2, y2))

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    @staticmethod
    def create_point(x, y):
        """Helper function to create a geometry_msgs/Point."""
        from geometry_msgs.msg import Point
        p = Point()
        p.x = x
        p.y = y
        p.z = 0
        return p

if __name__ == "__main__":
    try:
        detector = LidarEdgeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
