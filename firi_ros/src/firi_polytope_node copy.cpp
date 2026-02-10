#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// Messages
#include <obstacle_detector/Polytope2DArray.h>
#include <decomp_ros_msgs/PolyhedronArray.h>

// Decomp Utils
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_geometry/geometric_utils.h>

// Solvers
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <vector>
#include <cmath>

using namespace Eigen;
using namespace std;

// ==========================================
// FIRI SOLVER (Polytope Version)
// ==========================================
class FIRIPolytopeSolver {
    // Persistent Solver
    std::unique_ptr<OsqpEigen::Solver> rsi_solver_;
    Eigen::SparseMatrix<double> H_, A_;
    Eigen::VectorXd gradient_, lower_, upper_;
    bool solver_initialized_ = false;
    size_t last_seed_size_ = 0;
    size_t last_obs_verts_ = 0;

    // --- FIX: Define Struct at Class Level ---
    struct RawPlane { 
        Vector2d b; 
        Vector2d n; 
        double p; 
        int obs_idx; 
    };

public:
    FIRIPolytopeSolver() {}

    // Input: List of Polytopes (each is a list of vertices)
    vector<pair<Vector2d, double>> compute(
        const vector<vector<Vector2d>>& obstacles,
        const vector<Vector2d>& seed_poly,
        const Vector4d& bbox) 
    {
        // 1. Init Ellipsoid
        Vector2d d = Vector2d::Zero();
        for(const auto& p : seed_poly) d += p;
        if(!seed_poly.empty()) d /= seed_poly.size();
        Matrix2d C = Matrix2d::Identity() * 0.1;

        vector<pair<Vector2d, double>> planes;
        
        // BBox planes
        vector<pair<Vector2d, double>> bbox_planes;
        bbox_planes.push_back({Vector2d(-1, 0), -bbox(0)}); 
        bbox_planes.push_back({Vector2d( 1, 0),  bbox(1)}); 
        bbox_planes.push_back({Vector2d( 0,-1), -bbox(2)}); 
        bbox_planes.push_back({Vector2d( 0, 1),  bbox(3)}); 

        int max_iter = 3; 

        for(int k=0; k<max_iter; ++k) {
            // Transform Seed
            vector<Vector2d> Q_bar = transform(seed_poly, C, d);
            
            vector<RawPlane> candidates;

            // --- RsI Step (One QP per Obstacle Polytope) ---
            for(size_t i=0; i<obstacles.size(); ++i) {
                // Transform Obstacle Vertices
                vector<Vector2d> O_bar = transform(obstacles[i], C, d);
                
                // Optimization: Check closest vertex distance
                double min_dist_sq = 1000.0;
                for(const auto& v : O_bar) min_dist_sq = std::min(min_dist_sq, v.squaredNorm());
                if(min_dist_sq > 16.0) continue; // > 4.0 units away

                Vector2d b;
                // Solve QP: One plane separating ALL O_bar vertices from ALL Q_bar vertices
                if(solveRsIPolytope(O_bar, Q_bar, b)) {
                    double b_sq = b.squaredNorm();
                    if(b_sq > 1e-6) {
                        Vector2d a_bar = b / b_sq;
                        Vector2d n_raw = C.inverse().transpose() * a_bar;
                        double p_raw = a_bar.squaredNorm() + n_raw.dot(d);
                        double norm = n_raw.norm();
                        candidates.push_back({b, n_raw/norm, p_raw/norm, (int)i});
                    }
                }
            }

            // --- Greedy Pruning ---
            vector<pair<Vector2d, double>> current_poly = bbox_planes;
            prunePolytopes(obstacles, C, d, candidates, current_poly);

            // --- MVIE Update (Heuristic) ---
            C *= 1.2; 
            planes = current_poly;
        }
        return planes;
    }

private:
    vector<Vector2d> transform(const vector<Vector2d>& pts, const Matrix2d& C, const Vector2d& d) {
        Matrix2d C_inv = C.inverse();
        vector<Vector2d> res;
        res.reserve(pts.size());
        for(const auto& p : pts) res.push_back(C_inv * (p - d));
        return res;
    }

    bool solveRsIPolytope(const vector<Vector2d>& O_verts, const vector<Vector2d>& Q_verts, Vector2d& b_out) {
        OsqpEigen::Solver solver;
        solver.settings()->setVerbosity(false);
        solver.settings()->setAlpha(1.0);

        int n_vars = 2; 
        int n_cons = O_verts.size() + Q_verts.size();

        Eigen::SparseMatrix<double> H(2, 2);
        H.insert(0,0) = 2.0; H.insert(1,1) = 2.0;
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(2);
        
        Eigen::SparseMatrix<double> A(n_cons, 2);
        Eigen::VectorXd l(n_cons), u(n_cons);

        int row = 0;
        // 1. Obstacle Vertices (Exclude ALL)
        for(const auto& vert : O_verts) {
            A.insert(row, 0) = vert(0);
            A.insert(row, 1) = vert(1);
            l(row) = 1.0; 
            u(row) = OsqpEigen::INFTY;
            row++;
        }

        // 2. Seed Vertices (Include ALL)
        for(const auto& vert : Q_verts) {
            A.insert(row, 0) = vert(0);
            A.insert(row, 1) = vert(1);
            l(row) = -OsqpEigen::INFTY;
            u(row) = 0.999;
            row++;
        }

        solver.data()->setNumberOfVariables(n_vars);
        solver.data()->setNumberOfConstraints(n_cons);
        if(!solver.data()->setHessianMatrix(H)) return false;
        if(!solver.data()->setGradient(gradient)) return false;
        if(!solver.data()->setLinearConstraintsMatrix(A)) return false;
        if(!solver.data()->setLowerBound(l)) return false;
        if(!solver.data()->setUpperBound(u)) return false;

        if(!solver.initSolver()) return false;
        if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return false;

        b_out = solver.getSolution();
        return true;
    }

    void prunePolytopes(const vector<vector<Vector2d>>& obstacles,
                        const Matrix2d& C, const Vector2d& d,
                        vector<RawPlane>& candidates,
                        vector<pair<Vector2d, double>>& final_planes)
    {
        if(candidates.empty()) return;

        vector<bool> obs_removed(obstacles.size(), false);
        
        // Transform ALL obstacles once for checking coverage
        vector<vector<Vector2d>> all_obs_bar;
        for(const auto& obs : obstacles) all_obs_bar.push_back(transform(obs, C, d));

        // Precompute coverage [plane_idx][obs_idx]
        vector<vector<bool>> covers(candidates.size(), vector<bool>(obstacles.size(), false));
        for(size_t i=0; i<candidates.size(); ++i) {
            Vector2d b = candidates[i].b;
            for(size_t j=0; j<obstacles.size(); ++j) {
                // Check if ALL vertices of obs[j] satisfy u^T b >= 1
                bool all_out = true;
                for(const auto& v : all_obs_bar[j]) {
                    if(v.dot(b) < 0.99) { all_out = false; break; }
                }
                if(all_out) covers[i][j] = true;
            }
        }

        // Greedy Selection
        int total_removed = 0;
        int target_to_remove = 0; 
        
        for(size_t j=0; j<obstacles.size(); ++j) {
            bool has_candidate = false;
            for(size_t i=0; i<candidates.size(); ++i) if(covers[i][j]) has_candidate=true;
            if(has_candidate) target_to_remove++;
            else obs_removed[j] = true; 
        }

        while(total_removed < target_to_remove) {
            int best_idx = -1;
            int max_cover = 0;

            for(size_t i=0; i<candidates.size(); ++i) {
                int count = 0;
                for(size_t j=0; j<obstacles.size(); ++j) {
                    if(!obs_removed[j] && covers[i][j]) count++;
                }
                if(count > max_cover) {
                    max_cover = count;
                    best_idx = i;
                }
            }

            if(best_idx == -1) break;

            final_planes.push_back({candidates[best_idx].n, candidates[best_idx].p});
            
            for(size_t j=0; j<obstacles.size(); ++j) {
                if(!obs_removed[j] && covers[best_idx][j]) {
                    obs_removed[j] = true;
                    total_removed++;
                }
            }
            if(final_planes.size() > 50) break;
        }
    }
};

// ==========================================
// ROS NODE
// ==========================================
class FIRIPolytopeNode {
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_, sub_obs_, sub_estimate_;
    ros::Publisher pub_poly_; 
    
    FIRIPolytopeSolver solver_;
    Vector2d robot_pos_;
    double robot_yaw_;
    bool odom_rx_ = false;

    double length_ = 0.1; 
    double width_ = 0.1;

public:
    FIRIPolytopeNode() : nh_("~") {
        nh_.param("robot_length", length_, 0.1);
        nh_.param("robot_width", width_, 0.1);

        sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &FIRIPolytopeNode::odomCb, this);
        // Subscribe to Custom Polytope Array
        sub_obs_ = nh_.subscribe("/obstacles", 1, &FIRIPolytopeNode::obstaclesCb, this);
        sub_estimate_ =nh_.subscribe("/initialpose", 1, &FIRIPolytopeNode::initialPoseCb, this);
        
        pub_poly_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);
        
        ROS_INFO("FIRI Polytope Node Ready.");
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

    void initialPoseCb(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
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

    void obstaclesCb(const obstacle_detector::Polytope2DArray::ConstPtr& msg) {
        if(!odom_rx_) return;

        // 1. Convert Input Message to Eigen
        vector<vector<Vector2d>> obstacles;
        obstacles.reserve(msg->obstacles.size());
        
        for(const auto& poly_msg : msg->obstacles) {
            vector<Vector2d> poly;
            for(const auto& pt : poly_msg.vertices) {
                poly.push_back(Vector2d(pt.x, pt.y));
            }
            if(!poly.empty()) obstacles.push_back(poly);
        }

        // 2. Define Seed
        vector<Vector2d> seed;
        double hl = length_ / 2.0; 
        double hw = width_ / 2.0;
        Matrix2d R;
        R << cos(robot_yaw_), -sin(robot_yaw_), sin(robot_yaw_),  cos(robot_yaw_);
        
        vector<Vector2d> corners = {
            Vector2d( hl,  hw), Vector2d( hl, -hw),
            Vector2d(-hl, -hw), Vector2d(-hl,  hw)
        };
        for(const auto& c : corners) seed.push_back(robot_pos_ + R * c);

        // 3. Compute
        Vector4d bbox(robot_pos_.x()-5, robot_pos_.x()+5, robot_pos_.y()-5, robot_pos_.y()+5);
        
        auto planes = solver_.compute(obstacles, seed, bbox);

        publishPolyhedron(planes, msg->header);
    }

    void publishPolyhedron(const vector<pair<Vector2d, double>>& planes, const std_msgs::Header& header) {
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
    ros::init(argc, argv, "firi_polytope_node");
    FIRIPolytopeNode node;
    ros::spin();
    return 0;
}