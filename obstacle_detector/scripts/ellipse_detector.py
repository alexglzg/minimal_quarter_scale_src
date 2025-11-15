#!/usr/bin/env python3

import math
import time
import os

import cv2
import numpy as np
import rospy

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

class EllipseDetector:
    def __init__(self):

        self.map = 0


        rospy.Subscriber("planning/obstacle/map", OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher("/ellipses", MarkerArray, queue_size=10)


    def map_callback(self, data):
        width = data.info.width
        height = data.info.height
        resolution = data.info.resolution
        x_origin = data.info.origin.position.x
        y_origin = data.info.origin.position.y

        occupancy_data = np.array(data.data).reshape((height, width))

        marker_array = MarkerArray()

        # Convert to grayscale image values
        image_data = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if occupancy_data[i, j] == -1:
                    image_data[height-1-i, j] = 127  # Unknown
                elif occupancy_data[i, j] == 0:
                    image_data[height-1-i, j] = 0    # Free space
                else:
                    image_data[height-1-i, j] = 255  # Occupied space

        # Create the OpenCV image
        # Create the grayscale OpenCV image
        grayscale_image = image_data

        # If you need to convert it to a BGR image and then back to grayscale
        bgr_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        grayscale_again = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(grayscale_again, 252, 255, cv2.THRESH_BINARY)[1]

        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        result = bgr_image.copy()

        if len(contours) != 0:
            for i in range(len(contours)):
                if len(contours[i]) >= 5:
                    #cv2.drawContours(thresh,contours[i],-1,(150,10,255),3)
                    #ellipse=cv2.fitEllipse(contours[i])
                    ((p_centx,p_centy), (p_width,p_height), d_angle) = cv2.fitEllipse(contours[i])

                    e_centx = x_origin + (p_centx)*resolution
                    e_centy = y_origin + (height-p_centy)*resolution
                    e_width = p_width*resolution/2
                    e_height = p_height*resolution/2
                    e_angle = np.deg2rad(-d_angle)

                    marker = self.create_ellipse_marker(e_centx, e_centy, e_width, e_height, e_angle,i)
                    marker_array.markers.append(marker)
                    #print("center x,y:",e_centx,e_centy)
                    #print("diameters:",e_width,e_height)
                    #print("orientation angle:",e_angle)
                    cv2.ellipse(result, (int(p_centx),int(p_centy)), (int(p_width/2),int(p_height/2)), d_angle, 0, 360, (0,0,255), 2)
               
        #cv2.imshow("Perfectlyfittedellipses",result)
        #cv2.waitKey(1)

        self.marker_pub.publish(marker_array)

    def create_ellipse_marker(self, center_x, center_y, semi_major_axis, semi_minor_axis, orientation, id, frame_id="map"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = str(id)
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set the color and scale of the marker
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        # Calculate points on the ellipse
        num_points = 100
        angle_step = 2 * math.pi / num_points
        for i in range(num_points + 1):
            angle = i * angle_step
            x = semi_major_axis * math.cos(angle)
            y = semi_minor_axis * math.sin(angle)

            # Rotate the point based on the ellipse orientation
            x_rot = x * math.cos(orientation) - y * math.sin(orientation)
            y_rot = x * math.sin(orientation) + y * math.cos(orientation)

            point = Point()
            point.x = center_x + x_rot
            point.y = center_y + y_rot
            point.z = 0
            marker.points.append(point)

        return marker

    def detect(self):
        self.map = 0

def main():
    rospy.init_node('ellipse_detector')
    rate = rospy.Rate(10)
    ellipseDetector = EllipseDetector()

    while not rospy.is_shutdown():
        ellipseDetector.detect()
        rate.sleep()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
