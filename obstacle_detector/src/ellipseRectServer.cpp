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
        map_sub_ = nh_.subscribe("planning/obstacle/map", 10, &EllipseDetector::mapCallback, this);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/ellipses", 10);
        ellipse_pub = nh_.advertise<roboat_core::EllipseList>("/ellipse_msg", 10);
    }

    roboat_core::Ellipse ellipseMsg;

    /*void detect() {
        // No need to implement as it is used only for placeholder in Python
    }*/

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher ellipse_pub;

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        int width = msg->info.width;
        int height = msg->info.height;
        double resolution = msg->info.resolution;
        double x_origin = msg->info.origin.position.x;
        double y_origin = msg->info.origin.position.y;

        std::vector<int8_t> occupancy_data = msg->data;
        cv::Mat image_data = cv::Mat::zeros(height, width, CV_8UC1);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (occupancy_data[i * width + j] == -1) {
                    image_data.at<uint8_t>(height - 1 - i, j) = 127;  // Unknown
                } else if (occupancy_data[i * width + j] == 0) {
                    image_data.at<uint8_t>(height - 1 - i, j) = 0;    // Free space
                } else {
                    image_data.at<uint8_t>(height - 1 - i, j) = 255;  // Occupied space
                }
            }
        }

        cv::Mat bgr_image, grayscale_again, thresh;
        cv::cvtColor(image_data, bgr_image, cv::COLOR_GRAY2BGR);
        cv::cvtColor(bgr_image, grayscale_again, cv::COLOR_BGR2GRAY);
        cv::threshold(grayscale_again, thresh, 252, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        cv::Mat result = bgr_image.clone();
        visualization_msgs::MarkerArray marker_array;
        roboat_core::EllipseList ellipse_list;
        int len = 0;

        double half_width = 0.225;
        double safety_distance = 0.12;//0.075;
        double position_threshold = 6.0;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() >= 5) {

                cv::RotatedRect rect = cv::minAreaRect(contours[i]);
                // Get the 4 corner points of the rotated rectangle
                cv::Point2f box[4];
                rect.points(box);
                // Convert the points to integer
                std::vector<cv::Point> box_int;
                for (int i = 0; i < 4; i++) {
                    box_int.push_back(cv::Point(static_cast<int>(box[i].x), static_cast<int>(box[i].y)));
                }

                double d = sqrt((box[0].x - box[2].x)*(box[0].x - box[2].x) + (box[0].y - box[2].y)*(box[0].y - box[2].y));
                double w = std::min(rect.size.width,rect.size.height);

                double p_centx = rect.center.x;
                double p_centy = rect.center.y;
                double p_width = rect.size.width;//w;
                double p_height = rect.size.height;//d; //bigger
                double d_angle = rect.angle;

                double r_centx = x_origin + (p_centx) * resolution;
                double r_centy = y_origin + (height - p_centy) * resolution;
                double r_width = p_width * resolution;//0.52;
                double r_height = p_height * resolution;//0.52;
                double r_angle = -d_angle * CV_PI / 180.0;
                

                // Duplicate the first point to meet the 5-point requirement
                box_int.push_back(box_int[0]);

                // Draw the rotated rectangle
                // std::vector<std::vector<cv::Point>> box_contours = {box_int};
                // cv::drawContours(result, box_contours, 0, cv::Scalar(0, 255, 0), 2);
                // cv::imshow("Rotated Rectangles", result);
                // cv::waitKey(0);
                // cv::destroyAllWindows();

                // cv::RotatedRect ellipse = cv::fitEllipse(box_int);
                // cv::ellipse(result, ellipse, cv::Scalar(0, 255, 0), 2);
                // cv::imshow("Ellipse Around Rectangles", result);
                // cv::waitKey(0);
                // double p_centx = ellipse.center.x;
                // double p_centy = ellipse.center.y;
                // double p_width = ellipse.size.width;
                // double p_height = ellipse.size.height; //bigger
                // double d_angle = ellipse.angle;

                if (p_height >= p_width){
                    p_height = d;
                }
                else{
                    p_width = d;
                }

                double e_centx = x_origin + (p_centx) * resolution;
                double e_centy = y_origin + (height - p_centy) * resolution;
                // double e_width = p_width * resolution / 2 - half_width + safety_distance;//0.52;
                // double e_height = p_height * resolution / 2 - half_width + safety_distance;//0.52;
                double e_width = p_width * resolution / 2 - half_width + safety_distance;//0.52;
                double e_height = p_height * resolution / 2 - half_width + safety_distance;//0.52;
                double e_angle = -d_angle * CV_PI / 180.0;

                if ((e_width >= 2) || (e_height >= 2) || (std::abs(e_centy) >= position_threshold) || (e_centx < 0)){
                    continue;
                }

                
                visualization_msgs::Marker marker = createEllipseMarker(e_centx, e_centy, e_width, e_height, e_angle, len);
                marker_array.markers.push_back(marker);
                // marker = createRectangleMarker(r_centx, r_centy, r_width, r_height, r_angle, len+10);
                // marker_array.markers.push_back(marker);

                roboat_core::Ellipse ellipseMsg;
                ellipseMsg.x = e_centx;
                ellipseMsg.y = -e_centy;
                ellipseMsg.theta = -e_angle;
                ellipseMsg.a = e_width;
                ellipseMsg.b = e_height;
                ellipse_list.ellipse.push_back(ellipseMsg);
                len = len + 1;
            }
        }
        ellipse_list.len = len;
        ellipse_pub.publish(ellipse_list);
        marker_pub_.publish(marker_array);
    }

    visualization_msgs::Marker createEllipseMarker(double center_x, double center_y, double semi_major_axis, double semi_minor_axis, double orientation, int id, const std::string& frame_id = "map") {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        //marker.ns = std::to_string(id);
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;  // Line width
        marker.color.a = 1.0;  // Alpha
        marker.color.r = 1.0;  // Red
        marker.color.g = 0.0;  // Green
        marker.color.b = 0.0;  // Blue
        marker.lifetime = ros::Duration(0.3);

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

    visualization_msgs::Marker createRectangleMarker(double center_x, double center_y, double width, double height, double orientation, int id, const std::string& frame_id = "map") {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;  // Line width
        marker.color.a = 1.0;  // Alpha
        marker.color.r = 1.0;  // Red
        marker.color.g = 0.0;  // Green
        marker.color.b = 0.0;  // Blue
        marker.lifetime = ros::Duration(0.3);
    
        // Define the four corners of the rectangle
        std::vector<geometry_msgs::Point> corners;
        corners.resize(4);
    
        // Half-width and half-height
        double half_width = width / 2.0;
        double half_height = height / 2.0;
    
        // Define the corners relative to the center
        corners[0].x = -half_width;
        corners[0].y = -half_height;
        corners[0].z = 0;
    
        corners[1].x = half_width;
        corners[1].y = -half_height;
        corners[1].z = 0;
    
        corners[2].x = half_width;
        corners[2].y = half_height;
        corners[2].z = 0;
    
        corners[3].x = -half_width;
        corners[3].y = half_height;
        corners[3].z = 0;
    
        // Rotate the corners by the given orientation and translate to the center
        for (auto& corner : corners) {
            double x_rot = corner.x * cos(orientation) - corner.y * sin(orientation);
            double y_rot = corner.x * sin(orientation) + corner.y * cos(orientation);
    
            corner.x = center_x + x_rot;
            corner.y = center_y + y_rot;
            marker.points.push_back(corner);
        }
    
        // Close the rectangle by adding the first point again
        marker.points.push_back(marker.points[0]);
    
        return marker;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ellipse_detector");
    EllipseDetector ellipseDetector;
    ros::Rate rate(10);

    while (ros::ok()) {
        //ellipseDetector.detect();
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}