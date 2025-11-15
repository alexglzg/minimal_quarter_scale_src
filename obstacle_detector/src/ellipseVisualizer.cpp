#include <iostream>
#include <math.h>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_datatypes.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <roboat_core/Ellipse.h>
#include <roboat_core/EllipseList.h>

class EllipseVisualizer {
public:
    std::vector<double> state;
    EllipseVisualizer() {

        odometry_sub = nh_.subscribe("/odometry/filtered", 1, &EllipseVisualizer::odometryCallback, this);
        ellipse_sub = nh_.subscribe("/ellipse_msg", 1, &EllipseVisualizer::ellipseCallback, this);
        
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/inflated_ellipses", 10);
        state = std::vector<double>(6);
    }

    roboat_core::Ellipse ellipseMsg;

    /*void detect() {
        // No need to implement as it is used only for placeholder in Python
    }*/

private:
    ros::NodeHandle nh_;
    ros::Subscriber odometry_sub;
    ros::Publisher marker_pub_;
    ros::Subscriber ellipse_sub;

    void odometryCallback(const nav_msgs::Odometry::ConstPtr& o)
    {
        state[0] = o->pose.pose.position.x;
        state[1] = -o->pose.pose.position.y;

        tf::Quaternion q(
        o->pose.pose.orientation.x,
        o->pose.pose.orientation.y,
        o->pose.pose.orientation.z,
        o->pose.pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        state[2] = -yaw;

        state[3] = o->twist.twist.linear.x;
        state[4] = -o->twist.twist.linear.y;
        state[5] = -o->twist.twist.angular.z;
        
    }

    void ellipseCallback(const roboat_core::EllipseList::ConstPtr& ellipse_)
    {
        visualization_msgs::MarkerArray marker_array;
        for (int i = 0; i < ellipse_->len; i++){

            double obstacle_x = ellipse_->ellipse[i].x;
            double obstacle_y = ellipse_->ellipse[i].y;
            double obstacle_theta = ellipse_->ellipse[i].theta;

            // double co_x, co_y;
            // closest_point_on_tilted_ellipse(obstacle_x,obstacle_y,ellipse_->ellipse[i].a,ellipse_->ellipse[i].b,obstacle_theta,state[0],state[1],co_x,co_y);
            // double added_radius = choose_radius(co_x,co_y,state[0],state[1],state[2]);
            
            Rectangle rect = {{state[0], state[1]}, 0.9, 0.45, state[2]};
            Ellipse ellipse = {{obstacle_x, obstacle_y}, ellipse_->ellipse[i].a, ellipse_->ellipse[i].b, obstacle_theta};
            auto [nearestRectPoint, nearestEllipsePoint] = closestPoints(rect, ellipse);
            ROS_ERROR("points are : %f,%f,%f,%f\n",nearestRectPoint.x, nearestRectPoint.y, nearestEllipsePoint.x, nearestEllipsePoint.y);
            double added_radius = choose_radius(nearestEllipsePoint.x,nearestEllipsePoint.y,nearestRectPoint.x,nearestRectPoint.y,state[2]);
            // double dist = distance_func({nearestRectPoint.x,nearestRectPoint.y}, {nearestEllipsePoint.x,nearestEllipsePoint.y});
            double alpha_i = alpha_value(nearestEllipsePoint.x,nearestEllipsePoint.y,nearestRectPoint.x,nearestRectPoint.y,state[2]);

            double thresh_1 = 0.25;
            double thresh_2 = 0.1;
            if ((std::abs(std::abs(alpha_i) - M_PI/2) >= thresh_1) && (std::abs(alpha_i) >= thresh_2) && (std::abs(alpha_i) <= (M_PI-thresh_2))){
              added_radius = choose_radius(obstacle_x,obstacle_y,state[0],state[1],state[2]);
            }
            // double added_radius = choose_radius(obstacle_x,obstacle_y,state[0],state[1],state[2]);
            double obstacle_a = ellipse_->ellipse[i].a + added_radius;
            double obstacle_b = ellipse_->ellipse[i].b + added_radius;

            double min_a = ellipse_->ellipse[i].a + 0.225;
            double min_b = ellipse_->ellipse[i].b + 0.225;

            visualization_msgs::Marker marker = createEllipseMarker(obstacle_x, -obstacle_y, obstacle_a, obstacle_b, -obstacle_theta, i);
            marker_array.markers.push_back(marker);
            double index_2 = ellipse_->len+i;
            marker = createEllipseMarker(obstacle_x, -obstacle_y, min_a, min_b, -obstacle_theta, index_2, "map", 1.0);
            marker_array.markers.push_back(marker);

        }

         marker_pub_.publish(marker_array);
    }
    
    double choose_radius(double x_i, double y_i, double robot_x, double robot_y, double robot_angle){
        // double dx = x_i - robot_x;
        // double dy = y_i - robot_y;
        // double phi_i = std::atan2(dy,dx);
        // // ROS_ERROR("phi radius : %f\n",phi_i);
        // double alpha_i = robot_angle-phi_i;
        double alpha_i = alpha_value(x_i, y_i, robot_x, robot_y, robot_angle);
        // ROS_ERROR("alpha radius : %f\n",alpha_i);
        double r_i = (0.45*abs(sin(alpha_i)) + 0.9*abs(cos(alpha_i)))/2;
        return r_i;
      }

      double alpha_value(double x_i, double y_i, double robot_x, double robot_y, double robot_angle){
        double dx = x_i - robot_x;
        double dy = y_i - robot_y;
        double phi_i = std::atan2(dy,dx);
        // ROS_ERROR("phi radius : %f\n",phi_i);
        double alpha_i = robot_angle-phi_i;
        //Wrap to [-pi pi]
        if (std::abs(alpha_i) > M_PI){
          alpha_i = (alpha_i/std::abs(alpha_i))*(std::abs(alpha_i)-2*M_PI);
        }
        // ROS_ERROR("alpha radius : %f\n",alpha_i);
        return alpha_i;
      }


    double f(double t, double h, double k, double a, double b, double phi, double x0, double y0) {
        return (h + a * cos(t) * cos(phi) - b * sin(t) * sin(phi) - x0) * (-a * sin(t) * cos(phi) - b * cos(t) * sin(phi)) + 
            (k + a * cos(t) * sin(phi) + b * sin(t) * cos(phi) - y0) * (-a * sin(t) * sin(phi) + b * cos(t) * cos(phi));
    }

    double f_prime(double t, double h, double k, double a, double b, double phi, double x0, double y0) {
        double term1 = -a * cos(t) * cos(phi) * (h + a * cos(t) * cos(phi) - b * sin(t) * sin(phi) - x0);
        double term2 = -a * a * sin(t) * cos(phi) * cos(phi) - b * b * cos(t) * sin(phi) * sin(phi);
        double term3 = -b * sin(t) * sin(phi) * (h + a * cos(t) * cos(phi) - b * sin(t) * sin(phi) - x0);
        double term4 = -a * cos(t) * sin(phi) * (k + a * cos(t) * sin(phi) + b * sin(t) * cos(phi) - y0);
        double term5 = -a * a * sin(t) * sin(phi) * sin(phi) - b * b * cos(t) * cos(phi) * cos(phi);
        double term6 = b * cos(t) * cos(phi) * (k + a * cos(t) * sin(phi) + b * sin(t) * cos(phi) - y0);
        return term1 + term2 + term3 + term4 + term5 + term6;
    }

    void closest_point_on_tilted_ellipse(double h, double k, double a, double b, double phi, double x0, double y0, double& closest_x, double& closest_y) {
        double t = 0.0; // initial guess
        const double tol = 1e-6;
        const int max_iter = 100;

        for (int i = 0; i < max_iter; ++i) {
            double f_val = f(t, h, k, a, b, phi, x0, y0);
            double f_prime_val = f_prime(t, h, k, a, b, phi, x0, y0);
            double t_new = t - f_val / f_prime_val;
            if (std::abs(t_new - t) < tol) {
                t = t_new;
                break;
            }
            t = t_new;
        }

        closest_x = h + a * cos(t) * cos(phi) - b * sin(t) * sin(phi);
        closest_y = k + a * cos(t) * sin(phi) + b * sin(t) * cos(phi);
    }

    struct Point {
        double x, y;
    };

    struct Ellipse {
        Point center;  // center of ellipse
        double a, b;   // semi-major and semi-minor axes
        double theta;  // rotation angle in radians
    };

    struct Rectangle {
        Point center;  // center of rectangle
        double width, height; // dimensions of the rectangle
        double theta;   // rotation angle in radians
    };

    Point rotatePoint(Point p, double theta) {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        return {p.x * cosTheta - p.y * sinTheta, p.x * sinTheta + p.y * cosTheta};
    }

    std::vector<Point> rectangleVertices(const Rectangle& rect) {
        double w = rect.width / 2.0;
        double h = rect.height / 2.0;

        // Vertices relative to the center (rotated)
        std::vector<Point> vertices;
        vertices.push_back(rotatePoint({-w, -h}, rect.theta));  // Bottom-left
        vertices.push_back(rotatePoint({ w, -h}, rect.theta));  // Bottom-right
        vertices.push_back(rotatePoint({ w,  h}, rect.theta));  // Top-right
        vertices.push_back(rotatePoint({-w,  h}, rect.theta));  // Top-left

        // Translate to the rectangle's center
        for (auto& v : vertices) {
            v.x += rect.center.x;
            v.y += rect.center.y;
        }
        return vertices;
    }

    double distance_func(const Point& p1, const Point& p2) {
        return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
    }

    // Point closestPointOnEllipse(const Ellipse& ellipse, const Point& point) {
    //     const int numSamples = 200;  // Number of points to sample on the ellipse
    //     double minDist = std::numeric_limits<double>::infinity();
    //     Point closestPoint;

    //     for (int i = 0; i < numSamples; ++i) {
    //         double theta = 2 * M_PI * i / numSamples;

    //         // Parametric equation of the ellipse (before rotation)
    //         double ex = ellipse.a * cos(theta);
    //         double ey = ellipse.b * sin(theta);

    //         // Rotate the ellipse point by the ellipse's angle and translate to its center
    //         Point ellipsePoint = rotatePoint({ex, ey}, ellipse.theta);
    //         ellipsePoint.x += ellipse.center.x;
    //         ellipsePoint.y += ellipse.center.y;

    //         // Compute the distance to the given point
    //         double dist = distance_func(ellipsePoint, point);
    //         if (dist < minDist) {
    //             minDist = dist;
    //             closestPoint = ellipsePoint;
    //         }
    //     }
    //     return closestPoint;
    // }

    // std::pair<Point, Point> closestPoints(const Rectangle& rect, const Ellipse& ellipse) {
    //     std::vector<Point> vertices = rectangleVertices(rect);
    //     double minDist = std::numeric_limits<double>::infinity();
    //     Point bestRectPoint, bestEllipsePoint;

    //     // For each vertex of the rectangle
    //     for (const auto& vertex : vertices) {
    //         // Find the closest point on the ellipse to this vertex
    //         Point closestOnEllipse = closestPointOnEllipse(ellipse, vertex);

    //         // Compute the distance
    //         double dist = distance_func(vertex, closestOnEllipse);
    //         if (dist < minDist) {
    //             minDist = dist;
    //             bestRectPoint = vertex;
    //             bestEllipsePoint = closestOnEllipse;
    //         }
    //     }

    //     return {bestRectPoint, bestEllipsePoint};  // Return the closest pair of points
    // }

    // Function to find the closest point on a line segment to a given point
    // Point closestPointOnSegment(const Point& p, const Point& segStart, const Point& segEnd) {
    //     Point segDir = {segEnd.x - segStart.x, segEnd.y - segStart.y};
    //     Point pointDir = {p.x - segStart.x, p.y - segStart.y};

    //     double segLengthSquared = segDir.x * segDir.x + segDir.y * segDir.y;
    //     double t = (pointDir.x * segDir.x + pointDir.y * segDir.y) / segLengthSquared;

    //     t = std::max(0.0, std::min(1.0, t));  // Clamp t to [0, 1]

    //     Point closest;
    //     closest.x = segStart.x + t * segDir.x;
    //     closest.y = segStart.y + t * segDir.y;
    //     return closest;
    //     }

    // // Function to find the closest points between a rectangle and an ellipse
    // std::pair<Point, Point> closestPoints(const Rectangle& rect, const Ellipse& ellipse) {
    // std::vector<Point> vertices = rectangleVertices(rect);
    // double minDist = std::numeric_limits<double>::infinity();
    // Point bestRectPoint, bestEllipsePoint;

    // // Check each edge of the rectangle
    // for (size_t i = 0; i < vertices.size(); ++i) {
    //     size_t next = (i + 1) % vertices.size();  // Next vertex (wraps around)
    //     Point segStart = vertices[i];
    //     Point segEnd = vertices[next];

    //     // Find the closest point on the ellipse to this edge
    //     Point closestOnEllipse = closestPointOnEllipse(ellipse, segStart);

    //     // Find the closest point on the edge to the ellipse point
    //     Point closestOnEdge = closestPointOnSegment(closestOnEllipse, segStart, segEnd);

    //     // Compute the distance
    //     double dist = distance_func(closestOnEdge, closestOnEllipse);
    //     if (dist < minDist) {
    //         minDist = dist;
    //         bestRectPoint = closestOnEdge;
    //         bestEllipsePoint = closestOnEllipse;
    //     }
    // }

    // return {bestRectPoint, bestEllipsePoint};  // Return the closest pair of points
    // }

    // Function to sample points along a rectangle edge
    std::vector<Point> samplesEdge(const Point& start, const Point& end, int numSamples) {
        std::vector<Point> samples;
        for (int i = 0; i <= numSamples; ++i) {
            double t = static_cast<double>(i) / numSamples;
            samples.push_back({start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)});
        }
        return samples;
        }

        // Main function to find closest points on the rectangle and ellipse
        std::pair<Point, Point> closestPoints(const Rectangle& rect, const Ellipse& ellipse) {
        const int rectSamples = 400;  // Increase for higher precision
        const int ellipseSamples = 200;

        std::vector<Point> rectVertices = rectangleVertices(rect);
        std::vector<Point> rectSamplesList;
        std::vector<Point> ellipseSamplesList = samplesEllipse(ellipse, ellipseSamples);

        // Sample points along each rectangle edge
        for (size_t i = 0; i < rectVertices.size(); ++i) {
            size_t next = (i + 1) % rectVertices.size();
            std::vector<Point> edgeSamples = samplesEdge(rectVertices[i], rectVertices[next], rectSamples);
            rectSamplesList.insert(rectSamplesList.end(), edgeSamples.begin(), edgeSamples.end());
        }

        // Find closest pair of points
        double minDist = std::numeric_limits<double>::infinity();
        Point bestRectPoint, bestEllipsePoint;

        for (const auto& rPoint : rectSamplesList) {
            for (const auto& ePoint : ellipseSamplesList) {
                double dist = distance_func(rPoint, ePoint);
                if (dist < minDist) {
                    minDist = dist;
                    bestRectPoint = rPoint;
                    bestEllipsePoint = ePoint;
                }
            }
        }

        return {bestRectPoint, bestEllipsePoint};
        }

        // Function to sample points along the perimeter of an ellipse
        std::vector<Point> samplesEllipse(const Ellipse& ellipse, int numSamples) {
        std::vector<Point> samples;
        for (int i = 0; i < numSamples; ++i) {
            double theta = 2 * M_PI * i / numSamples;
            double ex = ellipse.a * std::cos(theta);
            double ey = ellipse.b * std::sin(theta);
            Point ellipsePoint = rotatePoint({ex, ey}, ellipse.theta);
            ellipsePoint.x += ellipse.center.x;
            ellipsePoint.y += ellipse.center.y;
            samples.push_back(ellipsePoint);
        }
        return samples;
        }

    visualization_msgs::Marker createEllipseMarker(double center_x, double center_y, double semi_major_axis, double semi_minor_axis, double orientation, int id, const std::string& frame_id = "map", double red=0.0) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        //marker.ns = std::to_string(id);
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.05;  // Line width
        marker.color.a = 1.0;  // Alpha
        marker.color.r = red;  // Red
        marker.color.g = 0.0;  // Green
        marker.color.b = 1.0;  // Blue
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
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ellipse_visualizer");
    EllipseVisualizer ellipseVisualizer;
    ros::Rate rate(10);

    while (ros::ok()) {
        //ellipseDetector.detect();
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}