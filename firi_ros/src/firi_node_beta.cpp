#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <decomp_ros_msgs/PolyhedronArray.h>
#include <visualization_msgs/MarkerArray.h> // Standard Vis
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <cmath>

using namespace Eigen;

// ==========================================
// FIRI SOLVER
// ==========================================
class FIRISolver {
public:
    FIRISolver() {}

    std::vector<std::pair<Vector2d, double>> compute(
        const std::vector<Vector2d>& obstacles,
        const std::vector<Vector2d>& seed_poly,
        const Vector4d& bbox) 
    {
        // 1. Init Ellipsoid at seed centroid
        Vector2d d = Vector2d::Zero();
        for(const auto& p : seed_poly) d += p;
        if(!seed_poly.empty()) d /= seed_poly.size();
        Matrix2d C = Matrix2d::Identity() * 0.1;

        std::vector<std::pair<Vector2d, double>> planes;
        
        // BBox planes (x >= xmin, x <= xmax, y >= ymin, y <= ymax)
        // Stored as: n, p where n^T x <= p
        std::vector<std::pair<Vector2d, double>> bbox_planes;
        bbox_planes.push_back({Vector2d(-1, 0), -bbox(0)}); 
        bbox_planes.push_back({Vector2d( 1, 0),  bbox(1)}); 
        bbox_planes.push_back({Vector2d( 0,-1), -bbox(2)}); 
        bbox_planes.push_back({Vector2d( 0, 1),  bbox(3)}); 

        int max_iter = 4; 

        for(int k=0; k<max_iter; ++k) {
            std::vector<Vector2d> O_bar = transform(obstacles, C, d);
            std::vector<Vector2d> Q_bar = transform(seed_poly, C, d);
            
            struct RawPlane { Vector2d b; Vector2d n; double p; };
            std::vector<RawPlane> candidates;

            // --- RsI Step ---
            // We removed the u.norm() check to ensure we see nearby walls
            for(const auto& u : O_bar) {
                Vector2d b;
                if(solveRsI(u, Q_bar, b)) {
                    double b_sq = b.squaredNorm();
                    if(b_sq > 1e-6) {
                        Vector2d a_bar = b / b_sq;
                        // Transform back to world
                        Vector2d n_raw = C.inverse().transpose() * a_bar;
                        double p_raw = a_bar.squaredNorm() + n_raw.dot(d);
                        double norm = n_raw.norm();
                        candidates.push_back({b, n_raw/norm, p_raw/norm});
                    }
                }
            }

            // --- Pruning ---
            std::vector<std::pair<Vector2d, double>> current_poly = bbox_planes;
            pruneSetCover(O_bar, candidates, current_poly);

            // --- Simple Expansion ---
            C *= 1.2; // Grow "Lens" to see further next time
            planes = current_poly;
        }
        return planes;
    }

private:
    std::vector<Vector2d> transform(const std::vector<Vector2d>& pts, const Matrix2d& C, const Vector2d& d) {
        Matrix2d C_inv = C.inverse();
        std::vector<Vector2d> res;
        res.reserve(pts.size());
        for(const auto& p : pts) res.push_back(C_inv * (p - d));
        return res;
    }

    bool solveRsI(const Vector2d& u, const std::vector<Vector2d>& V, Vector2d& b_out) {
        OsqpEigen::Solver solver;
        solver.settings()->setVerbosity(false);
        solver.settings()->setAlpha(1.0);

        int n_vars = 2; 
        int n_cons = 1 + V.size();

        Eigen::SparseMatrix<double> H(2, 2);
        H.insert(0,0) = 2.0; H.insert(1,1) = 2.0;
        
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(2);
        Eigen::SparseMatrix<double> A(n_cons, 2);
        Eigen::VectorXd l(n_cons), upper(n_cons);

        // Constraint 1: Obstacle (u^T b >= 1)
        A.insert(0,0) = u(0); A.insert(0,1) = u(1);
        l(0) = 1.0; upper(0) = OsqpEigen::INFTY;

        // Constraint 2: Seed (v^T b <= 0.999)
        for(size_t i=0; i<V.size(); ++i) {
            A.insert(1+i, 0) = V[i](0);
            A.insert(1+i, 1) = V[i](1);
            l(1+i) = -OsqpEigen::INFTY;
            upper(1+i) = 0.999;
        }

        solver.data()->setNumberOfVariables(n_vars);
        solver.data()->setNumberOfConstraints(n_cons);
        if(!solver.data()->setHessianMatrix(H)) return false;
        if(!solver.data()->setGradient(gradient)) return false;
        if(!solver.data()->setLinearConstraintsMatrix(A)) return false;
        if(!solver.data()->setLowerBound(l)) return false;
        if(!solver.data()->setUpperBound(upper)) return false;

        if(!solver.initSolver()) return false;
        if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return false;

        b_out = solver.getSolution();
        return true;
    }

    template <typename T>
    void pruneSetCover(const std::vector<Vector2d>& obstacles, 
                       const std::vector<T>& candidates,
                       std::vector<std::pair<Vector2d, double>>& final_planes) 
    {
        if(candidates.empty()) return;
        
        // Coverage Matrix
        std::vector<std::vector<bool>> coverage(candidates.size(), std::vector<bool>(obstacles.size(), false));
        for(size_t i=0; i<candidates.size(); ++i) {
            for(size_t j=0; j<obstacles.size(); ++j) {
                if(obstacles[j].dot(candidates[i].b) >= 0.99) {
                    coverage[i][j] = true;
                }
            }
        }

        std::vector<bool> obs_covered(obstacles.size(), false);
        int total_covered = 0;
        int target = obstacles.size();

        while(total_covered < target) {
            int best_idx = -1;
            int max_new_cover = 0;

            for(size_t i=0; i<candidates.size(); ++i) {
                int new_cover = 0;
                for(size_t j=0; j<obstacles.size(); ++j) {
                    if(!obs_covered[j] && coverage[i][j]) new_cover++;
                }
                if(new_cover > max_new_cover) {
                    max_new_cover = new_cover;
                    best_idx = i;
                }
            }

            if(best_idx == -1) break; 
            final_planes.push_back({candidates[best_idx].n, candidates[best_idx].p});
            
            for(size_t j=0; j<obstacles.size(); ++j) {
                if(!obs_covered[j] && coverage[best_idx][j]) {
                    obs_covered[j] = true;
                    total_covered++;
                }
            }
            if(final_planes.size() > 50) break; 
        }
    }
};

// ==========================================
// ROS NODE
// ==========================================
class FIRINode {
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_, sub_cloud_;
    ros::Publisher pub_poly_, pub_vis_;
    
    FIRISolver solver_;
    Vector2d robot_pos_;
    double robot_yaw_;
    bool odom_rx_ = false;

    double length_ = 0.6; 
    double width_ = 0.4;

public:
    FIRINode() : nh_("~") {
        nh_.param("robot_length", length_, 0.6);
        nh_.param("robot_width", width_, 0.4);

        sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &FIRINode::odomCb, this);
        sub_cloud_ = nh_.subscribe("/filtered_cloud", 1, &FIRINode::cloudCb, this);
        
        pub_poly_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1);
        pub_vis_  = nh_.advertise<visualization_msgs::MarkerArray>("firi_visuals", 1);
        
        ROS_INFO("FIRI Node Running.");
    }

    void odomCb(const nav_msgs::Odometry::ConstPtr& msg) {
        robot_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y;
        
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch;
        m.getRPY(roll, pitch, robot_yaw_);
        odom_rx_ = true;
    }

    void cloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if(!odom_rx_) return;

        // 1. Convert Cloud
        sensor_msgs::PointCloud cloud;
        sensor_msgs::convertPointCloud2ToPointCloud(*msg, cloud);
        std::vector<Vector2d> obstacles;
        obstacles.reserve(cloud.points.size());
        for(const auto& p : cloud.points) {
            obstacles.push_back(Vector2d(p.x, p.y));
        }

        // 2. Define Seed (Rotated Robot)
        std::vector<Vector2d> seed;
        double hl = length_ / 2.0; 
        double hw = width_ / 2.0;
        Matrix2d R;
        R << cos(robot_yaw_), -sin(robot_yaw_), sin(robot_yaw_),  cos(robot_yaw_);
        std::vector<Vector2d> corners = {
            Vector2d( hl,  hw), Vector2d( hl, -hw),
            Vector2d(-hl, -hw), Vector2d(-hl,  hw)
        };
        for(const auto& c : corners) seed.push_back(robot_pos_ + R * c);

        // 3. Compute
        Vector4d bbox(robot_pos_.x()-5, robot_pos_.x()+5, robot_pos_.y()-5, robot_pos_.y()+5);
        
        ros::Time t0 = ros::Time::now();
        auto planes = solver_.compute(obstacles, seed, bbox);
        ros::Duration dt = ros::Time::now() - t0;

        // ROS_INFO_THROTTLE(0.5, "FIRI: %lu obs -> %lu planes (%.1f ms)", 
        //     obstacles.size(), planes.size(), dt.toSec()*1000.0);

        // 4. Publish Polyhedron (Logic)
        publishPolyhedron(planes, msg->header);

        // 5. Publish Markers (Visuals)
        publishMarkers(planes, seed, msg->header);
    }

    void publishPolyhedron(const std::vector<std::pair<Vector2d, double>>& planes, const std_msgs::Header& header) {
        decomp_ros_msgs::PolyhedronArray msg;
        msg.header = header;
        decomp_ros_msgs::Polyhedron poly;
        for(const auto& p : planes) {
            geometry_msgs::Point pt, n;
            n.x = p.first.x(); n.y = p.first.y(); n.z = 0;
            Vector2d v = p.first * p.second; // point on plane
            pt.x = v.x(); pt.y = v.y(); pt.z = 0;
            poly.normals.push_back(n);
            poly.points.push_back(pt);
        }
        msg.polyhedrons.push_back(poly);
        pub_poly_.publish(msg);
    }

    void publishMarkers(const std::vector<std::pair<Vector2d, double>>& planes, 
                        const std::vector<Vector2d>& seed,
                        const std_msgs::Header& header) 
    {
        visualization_msgs::MarkerArray arr;
        int id = 0;

        // A. Draw Seed (Robot Footprint)
        visualization_msgs::Marker m_seed;
        m_seed.header = header;
        m_seed.ns = "seed";
        m_seed.id = id++;
        m_seed.type = visualization_msgs::Marker::LINE_STRIP;
        m_seed.scale.x = 0.05; 
        m_seed.color.a = 1.0; m_seed.color.b = 1.0; // Blue
        m_seed.pose.orientation.w = 1.0;
        for(const auto& p : seed) {
            geometry_msgs::Point pt; pt.x = p.x(); pt.y = p.y();
            m_seed.points.push_back(pt);
        }
        m_seed.points.push_back(m_seed.points[0]); // Close loop
        arr.markers.push_back(m_seed);

        // B. Draw Planes (Normals)
        // Since we don't have vertices easily without geometry lib, 
        // we draw arrows representing the constraints
        for(const auto& p : planes) {
            visualization_msgs::Marker m;
            m.header = header;
            m.ns = "planes";
            m.id = id++;
            m.type = visualization_msgs::Marker::ARROW;
            m.scale.x = 0.3; m.scale.y = 0.05; m.scale.z = 0.05;
            m.color.a = 0.8; m.color.g = 1.0; // Green
            m.pose.orientation.w = 1.0;

            // Arrow Start: Point on plane
            Vector2d start = p.first * p.second;
            // Arrow End: Point + Normal (pointing OUT to obstacle)
            Vector2d end = start + p.first * 0.5;

            geometry_msgs::Point p1, p2;
            p1.x = start.x(); p1.y = start.y();
            p2.x = end.x();   p2.y = end.y();
            m.points.push_back(p1);
            m.points.push_back(p2);
            arr.markers.push_back(m);
        }
        pub_vis_.publish(arr);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "firi_node");
    FIRINode node;
    ros::spin();
    return 0;
}