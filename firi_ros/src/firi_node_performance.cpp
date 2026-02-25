#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_geometry/geometric_utils.h>

#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <cmath>
#include <unordered_set>

using namespace Eigen;

// ==========================================
// HELPER: SPATIAL HASH FILTER
// ==========================================
struct PointHash {
    size_t operator()(const Vector2i& k) const {
        return std::hash<int>()(k.x()) ^ (std::hash<int>()(k.y()) << 1);
    }
};

std::vector<Vector2d> voxelFilter(const std::vector<Vector2d>& pts, double res) {
    if(pts.empty()) return {};
    std::unordered_set<Vector2i, PointHash> grid;
    std::vector<Vector2d> filtered;
    filtered.reserve(pts.size());
    
    double inv_res = 1.0 / res;
    
    for(const auto& p : pts) {
        Vector2i k(std::floor(p.x() * inv_res), std::floor(p.y() * inv_res));
        if(grid.insert(k).second) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

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
        Vector2d d = Vector2d::Zero();
        for(const auto& p : seed_poly) d += p;
        if(!seed_poly.empty()) d /= seed_poly.size();
        Matrix2d C = Matrix2d::Identity() * 0.1;

        std::vector<std::pair<Vector2d, double>> planes;
        
        std::vector<std::pair<Vector2d, double>> bbox_planes;
        bbox_planes.push_back({Vector2d(-1, 0), -bbox(0)}); 
        bbox_planes.push_back({Vector2d( 1, 0),  bbox(1)}); 
        bbox_planes.push_back({Vector2d( 0,-1), -bbox(2)}); 
        bbox_planes.push_back({Vector2d( 0, 1),  bbox(3)}); 

        int max_iter = 3; 

        for(int k=0; k<max_iter; ++k) {
            std::vector<Vector2d> O_bar = transform(obstacles, C, d);
            std::vector<Vector2d> Q_bar = transform(seed_poly, C, d);
            
            struct RawPlane { Vector2d b; Vector2d n; double p; };
            std::vector<RawPlane> candidates;
            candidates.reserve(O_bar.size());

            // 1. RsI Step
            for(const auto& u : O_bar) {
                // --- FIX: REMOVED DISTANCE CHECK ---
                // We process ALL voxel-filtered points to ensure we catch nearby walls
                // even if the initial ellipsoid is tiny.
                
                Vector2d b;
                if(solveRsI(u, Q_bar, b)) {
                    double b_sq = b.squaredNorm();
                    if(b_sq > 1e-6) {
                        Vector2d a_bar = b / b_sq;
                        Vector2d n_raw = C.inverse().transpose() * a_bar;
                        double p_raw = a_bar.squaredNorm() + n_raw.dot(d);
                        double norm = n_raw.norm();
                        candidates.push_back({b, n_raw/norm, p_raw/norm});
                    }
                }
            }

            // 2. Pruning
            std::vector<std::pair<Vector2d, double>> current_poly = bbox_planes;
            pruneSetCover(O_bar, candidates, current_poly);

            // 3. Expansion
            C *= 1.2; 
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

        // Obstacle
        A.insert(0,0) = u(0); A.insert(0,1) = u(1);
        l(0) = 1.0; upper(0) = OsqpEigen::INFTY;

        // Seed
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
    ros::Publisher pub_poly_; 
    
    FIRISolver solver_;
    Vector2d robot_pos_;
    double robot_yaw_;
    bool odom_rx_ = false;

    double length_ = 0.9; 
    double width_ = 0.45;
    double voxel_size_ = 0.1; // 10cm by default

public:
    FIRINode() : nh_("~") {
        nh_.param("robot_length", length_, 0.9);
        nh_.param("robot_width", width_, 0.45);
        nh_.param("voxel_size", voxel_size_, 0.1);

        sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &FIRINode::odomCb, this);
        sub_cloud_ = nh_.subscribe("/filtered_cloud", 1, &FIRINode::cloudCb, this);
        pub_poly_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);
        
        ROS_INFO("FIRI Node Ready. Robot: %.2f x %.2f, Voxel: %.2f", length_, width_, voxel_size_);
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

        // 1. Convert
        sensor_msgs::PointCloud cloud;
        sensor_msgs::convertPointCloud2ToPointCloud(*msg, cloud);
        std::vector<Vector2d> raw_obs;
        raw_obs.reserve(cloud.points.size());
        for(const auto& p : cloud.points) {
            raw_obs.push_back(Vector2d(p.x, p.y));
        }

        // 2. Downsample
        std::vector<Vector2d> obstacles = voxelFilter(raw_obs, voxel_size_);

        // --- DEBUG PRINT ---
        // Verify we aren't filtering everything away
        ROS_INFO_THROTTLE(1.0, "[FIRI] Cloud: %lu -> Voxel: %lu points", 
            raw_obs.size(), obstacles.size());

        // 3. Seed
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

        // 4. Compute
        Vector4d bbox(robot_pos_.x()-1, robot_pos_.x()+5, robot_pos_.y()-1, robot_pos_.y()+5);
        auto planes = solver_.compute(obstacles, seed, bbox);

        publishPolyhedron(planes, msg->header);
    }

    void publishPolyhedron(const std::vector<std::pair<Vector2d, double>>& planes, const std_msgs::Header& header) {
        Polyhedron2D poly; 
        for(const auto& p : planes) {
            Vector2d n = p.first;
            double d = p.second;
            Vector2d pt = n * d;
            poly.add(Hyperplane2D(pt, n));
        }
        vec_E<Polyhedron2D> polys;
        polys.push_back(poly);
        decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polys);
        poly_msg.header = header;
        pub_poly_.publish(poly_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "firi_node");
    FIRINode node;
    ros::spin();
    return 0;
}