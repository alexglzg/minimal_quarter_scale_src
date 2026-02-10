#include <ros/ros.h>
#include <vector>
#include <string>
#include <cmath>

// Custom Messages
#include <obstacle_detector/Polytope2D.h>
#include <obstacle_detector/Polytope2DArray.h>

// Visualization
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>

using namespace std;

class EnvironmentPublisher {
private:
    ros::NodeHandle nh_;
    ros::Publisher pub_obstacles_;
    ros::Publisher pub_markers_;
    
    // Internal representation of obstacles (list of vertices)
    vector<vector<geometry_msgs::Point32>> obstacles_;
    
    string env_type_;
    string frame_id_;

public:
    EnvironmentPublisher() : nh_("~") {
        // Parameters
        // nh_.param<string>("env_type", env_type_, "maze");
        nh_.param<string>("env_type", env_type_, "oblique_maze");
        nh_.param<string>("frame_id", frame_id_, "map");

        // Publishers
        pub_obstacles_ = nh_.advertise<obstacle_detector::Polytope2DArray>("/obstacles", 1, true);
        pub_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("/obstacle_markers", 1, true);

        // Build Environment
        createEnv(env_type_);
        
        ROS_INFO("Environment Publisher Initialized. Type: %s, Obstacles: %lu", 
                 env_type_.c_str(), obstacles_.size());
    }

    void spin() {
        ros::Rate rate(1.0); // 1 Hz is enough for static maps
        while (ros::ok()) {
            publishObstacles();
            publishVisuals();
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    // --- Helper to create a Rectangle ---
    void addRectangle(double x_min, double x_max, double y_min, double y_max) {
        vector<geometry_msgs::Point32> poly;
        geometry_msgs::Point32 p;
        
        // Counter-Clockwise order
        p.x = x_min; p.y = y_min; poly.push_back(p);
        p.x = x_max; p.y = y_min; poly.push_back(p);
        p.x = x_max; p.y = y_max; poly.push_back(p);
        p.x = x_min; p.y = y_max; poly.push_back(p);
        
        obstacles_.push_back(poly);
    }

    // --- Helper to add arbitrary Polygon ---
    void addPolytope(const vector<pair<double, double>>& points) {
        vector<geometry_msgs::Point32> poly;
        for(const auto& pt : points) {
            geometry_msgs::Point32 p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.0;
            poly.push_back(p);
        }
        obstacles_.push_back(poly);
    }

    // --- Logic Ported from Python ---
    void createEnv(string type) {
        obstacles_.clear();

        if (type == "s_path") {
            double s = 1.0;
            // obstacles.append(RectangleRegion(0.0 * s, 1.0 * s, 0.9 * s, 1.0 * s))
            addRectangle(0.0*s, 1.0*s, 0.9*s, 1.0*s);
            // obstacles.append(RectangleRegion(0.0 * s, 0.4 * s, 0.4 * s, 1.0 * s))
            addRectangle(0.0*s, 0.4*s, 0.4*s, 1.0*s);
            // obstacles.append(RectangleRegion(0.6 * s, 1.0 * s, 0.0 * s, 0.7 * s))
            addRectangle(0.6*s, 1.0*s, 0.0*s, 0.7*s);
        }
        else if (type == "maze") {
            double s = 0.15;
            // Note: In Python RectangleRegion(x_min, x_max, y_min, y_max)
            addRectangle(0.0*s, 3.0*s, 0.0*s, 3.0*s);
            addRectangle(1.0*s, 2.0*s, 4.0*s, 6.0*s);
            addRectangle(2.0*s, 6.0*s, 5.0*s, 6.0*s);
            addRectangle(6.0*s, 7.0*s, 4.0*s, 6.0*s);
            addRectangle(4.0*s, 5.0*s, 0.0*s, 4.0*s);
            addRectangle(5.0*s, 7.0*s, 2.0*s, 3.0*s);
            addRectangle(6.0*s, 9.0*s, 1.0*s, 2.0*s);
            addRectangle(8.0*s, 9.0*s, 2.0*s, 4.0*s);
            addRectangle(9.0*s, 12.0*s, 3.0*s, 4.0*s);
            addRectangle(11.0*s, 12.0*s, 4.0*s, 5.0*s);
            addRectangle(8.0*s, 10.0*s, 5.0*s, 6.0*s);
            addRectangle(10.0*s, 11.0*s, 0.0*s, 2.0*s);
            addRectangle(12.0*s, 13.0*s, 1.0*s, 2.0*s);
            
            // Boundaries
            addRectangle(0.0*s, 13.0*s, 6.0*s, 7.0*s);
            addRectangle(-1.0*s, 0.0*s, -1.0*s, 7.0*s);
            addRectangle(0.0*s, 13.0*s, -1.0*s, 0.0*s);
            addRectangle(13.0*s, 14.0*s, -1.0*s, 7.0*s);
        }
        else if (type == "oblique_maze") {
            double s = 0.15;
            
            // Rectangles
            addRectangle(-1.0*s, 0.0*s, 0.0*s, 8.0*s);
            addRectangle(0.0*s, 10.0*s, 0.0*s, 1.0*s);
            addRectangle(0.0*s, 8.0*s, 7.0*s, 8.0*s);
            addRectangle(10.0*s, 11.0*s, 0.0*s, 8.0*s);

            // Polytopes (Manual Vertex Entry)
            // 1
            addPolytope({{0.0*s, 2.0*s}, {1.25*s, 3.875*s}, {2.875*s, 3.125*s}, {2.5*s, 2.25*s}});
            // 2
            addPolytope({{1.0*s, 4.75*s}, {0.0*s, 5.0*s}, {0.875*s, 7.0*s}, {1.875*s, 6.375*s}});
            // 3
            addPolytope({{2.75*s, 1.0*s}, {4.2*s, 3.25*s}, {5.125*s, 3.75*s}, {6.625*s, 2.5*s}, {6.5*s, 1.0*s}});
            // 4
            addPolytope({{6.0*s, 7.0*s}, {6.0*s, 6.0*s}, {6.5*s, 7.0*s}});
            // 5
            addPolytope({{2.375*s, 4.875*s}, {2.875*s, 5.875*s}, {4.5*s, 5.875*s}, {4.75*s, 4.0*s}, {3.375*s, 4.0*s}});
            // 6
            addPolytope({{6.75*s, 1.0*s}, {7.25*s, 2.375*s}, {8.5*s, 2.0*s}, {8.5*s, 1.0*s}});
            // 7
            addPolytope({{8.625*s, 1.0*s}, {10.0*s, 2.5*s}, {10.0*s, 1.0*s}});
            // 8
            addPolytope({{10.0*s, 2.875*s}, {9.5*s, 5.75*s}, {10.0*s, 5.875*s}});
            // 9
            addPolytope({{8.875*s, 3.125*s}, {8.0*s, 5.5*s}, {6.875*s, 6.375*s}, {5.875*s, 5.875*s}, {6.25*s, 4.375*s}, {7.125*s, 3.5*s}});
        }
    }

    // --- Message Publisher ---
    void publishObstacles() {
        obstacle_detector::Polytope2DArray msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = frame_id_;

        for(const auto& poly_verts : obstacles_) {
            obstacle_detector::Polytope2D poly;
            poly.vertices = poly_verts;
            msg.obstacles.push_back(poly);
        }
        pub_obstacles_.publish(msg);
    }

    // --- Rviz Marker Publisher ---
    void publishVisuals() {
        visualization_msgs::MarkerArray ma;
        int id = 0;

        for(const auto& poly_verts : obstacles_) {
            visualization_msgs::Marker m;
            m.header.frame_id = frame_id_;
            m.header.stamp = ros::Time::now();
            m.ns = "polytopes";
            m.id = id++;
            m.type = visualization_msgs::Marker::LINE_STRIP;
            m.action = visualization_msgs::Marker::ADD;
            
            // Visual Properties
            m.scale.x = 0.02; // Line width
            m.color.a = 1.0;
            m.color.r = 0.0; m.color.g = 0.0; m.color.b = 0.0; // Black lines

            // Add points
            for(const auto& p : poly_verts) {
                geometry_msgs::Point gp;
                gp.x = p.x; gp.y = p.y; gp.z = 0;
                m.points.push_back(gp);
            }
            
            // Close the loop
            if(!poly_verts.empty()) {
                geometry_msgs::Point gp;
                gp.x = poly_verts[0].x; gp.y = poly_verts[0].y; gp.z = 0;
                m.points.push_back(gp);
            }

            m.pose.orientation.w = 1.0;
            ma.markers.push_back(m);
        }
        pub_markers_.publish(ma);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "polytope_publisher");
    EnvironmentPublisher env;
    env.spin();
    return 0;
}