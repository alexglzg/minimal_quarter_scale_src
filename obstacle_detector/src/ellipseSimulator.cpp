#include <iostream>
#include <math.h>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <roboat_core/Ellipse.h>
#include <roboat_core/EllipseList.h>

class EllipseDetector {
public:
    EllipseDetector() {
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/ellipses", 10);
        ellipse_pub = nh_.advertise<roboat_core::EllipseList>("/ellipse_msg", 10);
    }

    

    void detect() {
        
        visualization_msgs::MarkerArray marker_array;
        roboat_core::EllipseList ellipse_list;
        int len = 0;
        
        // double a = 0.5;
        // double b = 0.2;
        // double obs_x = 2.0;
        // double obs_y = 0.45;
        // double obs_theta = 0.0;
        double a = 0.5;
        double b = 0.1;
        double obs_x = 3.0;
        double obs_y = 0.35;
        double obs_theta = 0.0;
        len = 1;
        
        visualization_msgs::Marker marker = createEllipseMarker(obs_x, -obs_y, a, b, -obs_theta, len);
        marker_array.markers.push_back(marker);

        roboat_core::Ellipse ellipseMsg;
        ellipseMsg.x = obs_x;
        ellipseMsg.y = obs_y;
        ellipseMsg.theta = obs_theta;
        ellipseMsg.a = a;
        ellipseMsg.b = b;
        ellipse_list.ellipse.push_back(ellipseMsg);

        // a = 0.5;
        // b = 0.2;
        // obs_x = 2.0;
        // obs_y = -0.45;
        // obs_theta = 0.0;
        // len = len + 1;
        a = 0.5;
        b = 0.1;
        obs_x = 3.0;
        obs_y = -0.35;
        obs_theta = 0.0;
        len = len + 1;
        
        marker = createEllipseMarker(obs_x, -obs_y, a, b, -obs_theta, len);
        marker_array.markers.push_back(marker);

        ellipseMsg.x = obs_x;
        ellipseMsg.y = obs_y;
        ellipseMsg.theta = obs_theta;
        ellipseMsg.a = a;
        ellipseMsg.b = b;
        ellipse_list.ellipse.push_back(ellipseMsg);

        // a = 0.2;
        // b = 0.5;
        // obs_x = 3.8;//4.92;//4.7;
        // obs_y = 0.1;//0.15;//0.1;
        // obs_theta = 0.0;
        // len = len + 1;

        a = 0.1;
        b = 0.5;
        obs_x = 4.7;//4.92;//4.7;
        obs_y = 0.1;//0.15;//0.1;
        obs_theta = 0.0;
        len = len + 1;
        
        marker = createEllipseMarker(obs_x, -obs_y, a, b, -obs_theta, len);
        marker_array.markers.push_back(marker);

        ellipseMsg.x = obs_x;
        ellipseMsg.y = obs_y;
        ellipseMsg.theta = obs_theta;
        ellipseMsg.a = a;
        ellipseMsg.b = b;
        ellipse_list.ellipse.push_back(ellipseMsg);

        // a = 0.5;
        // b = 0.2;
        // obs_x = 4.2;//4.92;//4.7;
        // obs_y = -0.3;//0.15;//0.1;
        // obs_theta = 0.0;
        // len = len + 1;
        
        // marker = createEllipseMarker(obs_x, -obs_y, a, b, -obs_theta, len);
        // marker_array.markers.push_back(marker);

        // ellipseMsg.x = obs_x;
        // ellipseMsg.y = obs_y;
        // ellipseMsg.theta = obs_theta;
        // ellipseMsg.a = a;
        // ellipseMsg.b = b;
        // ellipse_list.ellipse.push_back(ellipseMsg);

        ellipse_list.len = len;
        ellipse_pub.publish(ellipse_list);
        marker_pub_.publish(marker_array);

    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher ellipse_pub;

    visualization_msgs::Marker createEllipseMarker(double center_x, double center_y, double semi_major_axis, double semi_minor_axis, double orientation, int id, const std::string& frame_id = "map") {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = std::to_string(id);
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;  // Line width
        marker.color.a = 1.0;  // Alpha
        marker.color.r = 1.0;  // Red
        marker.color.g = 0.0;  // Green
        marker.color.b = 0.0;  // Blue

        int num_points = 100;
        double angle_step = 2 * M_PI / num_points;
        for (int i = 0; i <= num_points; ++i) {
            double angle = i * angle_step;
            double x = semi_major_axis * cos(angle);
            double y = semi_minor_axis * sin(angle);

            double x_rot = x * cos(orientation) - y * sin(orientation);
            double y_rot = x * sin(orientation) + y * cos(orientation);

            geometry_msgs::Point point;
            point.x = center_x + x_rot;
            point.y = center_y + y_rot;
            point.z = 0;
            marker.points.push_back(point);
        }

        return marker;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ellipse_detector");
    EllipseDetector ellipseDetector;
    ros::Rate rate(10);

    while (ros::ok()) {
        ellipseDetector.detect();
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}