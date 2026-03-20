// ==========================================================================
// firi_polytope_node.cpp — FIRI for 2D with Polytope2DArray obstacle input
//
// Based on: Wang et al., "Fast Iterative Region Inflation for Computing
//           Large 2-D/3-D Convex Regions of Obstacle-Free Space"
//           IEEE Transactions on Robotics, Vol. 41, 2025
//
// Same FIRI solver as firi_node_sdmn.cpp (SDMN + MVIE), adapted so that
// each obstacle is a convex polytope (set of vertices) rather than a
// single point.  The RsI SDMN call adds one constraint per vertex of each
// obstacle polytope; the separation check requires ALL vertices to satisfy
// the halfplane condition.
//
// Key components:
//   1. SDMN  — Seidel's Small-Dimensional Minimum-Norm (2D)
//              [Paper Sec. IV, Algorithm 2]
//   2. MVIE  — Maximum Volume Inscribed Ellipsoid via log-barrier Newton
//              [Paper Sec. V]
//   3. FIRI  — Full outer loop: RsI → greedy halfplane select → MVIE → repeat
//              [Paper Sec. III, Algorithm 1]
//   4. ROS node subscribing to /obstacles (Polytope2DArray) + /odom
//
// No OSQP dependency. Only requires Eigen, decomp_ros, standard ROS.
// ==========================================================================

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/Header.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// Messages
#include <obstacle_detector/Polytope2DArray.h>
#include <decomp_ros_msgs/PolyhedronArray.h>

// Decomp Utils
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_geometry/geometric_utils.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>

using namespace Eigen;

// ==========================================================================
// SDMN: Small-Dimensional Minimum-Norm (2D)
// ==========================================================================
// [Paper Section IV, Algorithm 2]
// Solves:  min ||y||^2   s.t.  e_i^T y <= f_i,  i = 1..d
// Expected O(d) complexity for n=2.

class SDMN2D {
public:
    SDMN2D() : rng_(42) {}

    struct Result {
        Vector2d y;
        bool feasible;
    };

    Result solve(const std::vector<Vector2d>& e,
                 const std::vector<double>& f)
    {
        size_t d = e.size();
        if (d == 0) return {Vector2d::Zero(), true};

        std::vector<size_t> perm(d);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        Vector2d y = Vector2d::Zero();

        for (size_t ii = 0; ii < d; ++ii) {
            size_t idx = perm[ii];

            if (e[idx].dot(y) <= f[idx] + 1e-12) continue;

            const Vector2d& eh = e[idx];
            double fh = f[idx];
            double eTe = eh.squaredNorm();
            if (eTe < 1e-15) return {Vector2d::Zero(), false};

            Vector2d v = (fh / eTe) * eh;

            int j = (std::abs(v(0)) >= std::abs(v(1))) ? 0 : 1;
            int k = 1 - j;

            Vector2d m_col;
            double v_norm = v.norm();
            if (v_norm < 1e-15) {
                m_col = Vector2d(-eh(1), eh(0));
                double mn = m_col.norm();
                if (mn > 1e-15) m_col /= mn;
                else return {Vector2d::Zero(), false};
            } else {
                double sign_vj = (v(j) >= 0) ? 1.0 : -1.0;
                Vector2d u_ref = v + sign_vj * v_norm * Vector2d::Unit(j);
                double uTu = u_ref.squaredNorm();
                if (uTu < 1e-15) {
                    m_col = Vector2d(-eh(1), eh(0)).normalized();
                } else {
                    m_col = Vector2d::Unit(k) - (2.0 * u_ref(k) / uTu) * u_ref;
                }
            }

            double lo = -1e18, hi = 1e18;
            bool feasible = true;

            for (size_t pp = 0; pp < ii; ++pp) {
                size_t pidx = perm[pp];
                double a1d = e[pidx].dot(m_col);
                double b1d = f[pidx] - e[pidx].dot(v);

                if (std::abs(a1d) < 1e-15) {
                    if (b1d < -1e-10) { feasible = false; break; }
                    continue;
                }
                double bound = b1d / a1d;
                if (a1d > 0) hi = std::min(hi, bound);
                else         lo = std::max(lo, bound);
            }

            if (!feasible || lo > hi + 1e-10)
                return {Vector2d::Zero(), false};

            double t;
            if (lo <= 0.0 && 0.0 <= hi) t = 0.0;
            else if (lo > 0.0)           t = lo;
            else                          t = hi;

            y = m_col * t + v;
        }

        return {y, true};
    }

private:
    mutable std::mt19937 rng_;
};

// ==========================================================================
// MVIE: Maximum Volume Inscribed Ellipsoid (2D)
// ==========================================================================
// [Paper Section V concept, simplified for 2D]
// Log-barrier interior-point, 5 decision variables.

class MVIE2D {
public:
    struct Ellipsoid {
        Matrix2d L;
        Vector2d d;
        double volume() const { return M_PI * std::abs(L(0, 0) * L(1, 1)); }
    };

    Ellipsoid solve(const MatrixXd& A, const VectorXd& b,
                    const Vector2d& center_hint)
    {
        int m = A.rows();

        Vector2d c = center_hint;
        double r = 1e18;
        for (int i = 0; i < m; ++i) {
            double norm_ai = A.row(i).norm();
            if (norm_ai > 1e-10) {
                double gap = b(i) - A.row(i).dot(c);
                r = std::min(r, gap / norm_ai);
            }
        }
        if (r <= 0) r = 1e-4;
        r *= 0.9;
        r = std::max(r, 1e-6);

        VectorXd x(5);
        x << r, 0.0, r, c(0), c(1);

        double t = 1.0;
        const double mu = 4.0;

        for (int outer = 0; outer < 20; ++outer) {
            for (int inner = 0; inner < 40; ++inner) {
                VectorXd grad = gradient(A, b, x, t);
                MatrixXd H = hessian(A, b, x, t);
                H += 1e-8 * MatrixXd::Identity(5, 5);

                VectorXd dx = H.ldlt().solve(-grad);
                double lambda_sq = -grad.dot(dx);
                if (lambda_sq < 1e-6) break;

                double alpha = 1.0;
                double f0 = objective(A, b, x, t);
                for (int ls = 0; ls < 32; ++ls) {
                    VectorXd xn = x + alpha * dx;
                    if (xn(0) > 1e-10 && xn(2) > 1e-10) {
                        double fn = objective(A, b, xn, t);
                        if (std::isfinite(fn) && fn < f0 + 0.3 * alpha * grad.dot(dx)) {
                            x = xn;
                            break;
                        }
                    }
                    alpha *= 0.5;
                    if (alpha < 1e-12) break;
                }
            }
            if ((double)m / t < 1e-3) break;
            t *= mu;
        }

        Ellipsoid E;
        E.L << x(0), 0.0,
               x(1), x(2);
        E.d << x(3), x(4);
        return E;
    }

private:
    double objective(const MatrixXd& A, const VectorXd& b,
                     const VectorXd& x, double t)
    {
        double L11 = x(0), L21 = x(1), L22 = x(2), d1 = x(3), d2 = x(4);
        if (L11 <= 0 || L22 <= 0) return 1e18;
        double val = -t * (std::log(L11) + std::log(L22));
        for (int i = 0; i < A.rows(); ++i) {
            double a1 = A(i, 0), a2 = A(i, 1);
            double r1 = L11 * a1 + L21 * a2;
            double r2 = L22 * a2;
            double gap = b(i) - a1 * d1 - a2 * d2 - std::sqrt(r1 * r1 + r2 * r2);
            if (gap <= 0) return 1e18;
            val -= std::log(gap);
        }
        return val;
    }

    VectorXd gradient(const MatrixXd& A, const VectorXd& b,
                      const VectorXd& x, double t)
    {
        double L11 = x(0), L21 = x(1), L22 = x(2), d1 = x(3), d2 = x(4);
        VectorXd g = VectorXd::Zero(5);
        g(0) = -t / L11;
        g(2) = -t / L22;

        for (int i = 0; i < A.rows(); ++i) {
            double a1 = A(i, 0), a2 = A(i, 1);
            double r1 = L11 * a1 + L21 * a2;
            double r2 = L22 * a2;
            double nr = std::sqrt(r1 * r1 + r2 * r2);
            double gap = b(i) - a1 * d1 - a2 * d2 - nr;
            if (gap < 1e-15) gap = 1e-15;
            double ig = 1.0 / gap;

            if (nr > 1e-15) {
                double inr = 1.0 / nr;
                g(0) += ig * r1 * a1 * inr;
                g(1) += ig * r1 * a2 * inr;
                g(2) += ig * r2 * a2 * inr;
            }
            g(3) += ig * a1;
            g(4) += ig * a2;
        }
        return g;
    }

    MatrixXd hessian(const MatrixXd& A, const VectorXd& b,
                     const VectorXd& x, double t)
    {
        const double eps = 1e-6;
        MatrixXd H(5, 5);
        for (int j = 0; j < 5; ++j) {
            VectorXd xp = x, xm = x;
            xp(j) += eps;
            xm(j) -= eps;
            H.col(j) = (gradient(A, b, xp, t) - gradient(A, b, xm, t)) / (2.0 * eps);
        }
        return 0.5 * (H + H.transpose());
    }
};

// ==========================================================================
// FIRI SOLVER (Polytope Obstacle Variant)
// ==========================================================================
// [Paper Algorithm 1] — adapted for polytope obstacles.
//
// Each obstacle is a convex polytope (list of vertices).
// The SDMN call for obstacle i adds one constraint per vertex:
//   -u_bar_j^T b <= -1   for all j in obstacle i
// A polytope is "separated" when ALL its vertices satisfy b_sol^T u_bar_j >= 1.

class FIRISolver {
public:
    struct HalfPlane {
        Vector2d normal;  // unit normal
        double offset;    // n^T x <= offset
    };

    struct Result {
        std::vector<HalfPlane> planes;
        int iterations;
        double solve_time_ms;
    };

    FIRISolver() {}

    // obstacles: each element is a list of vertices forming a convex polytope
    Result compute(const std::vector<std::vector<Vector2d>>& obstacles,
                   const std::vector<Vector2d>& seed_vertices,
                   const std::vector<HalfPlane>& bbox_planes,
                   int max_iter = 10,
                   double rho = 0.02)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        if (obstacles.empty() || seed_vertices.empty()) {
            return {bbox_planes, 0, 0.0};
        }

        // Initialize ellipsoid strictly inside seed [Paper Sec. III-B]
        Vector2d d = Vector2d::Zero();
        for (const auto& v : seed_vertices) d += v;
        d /= seed_vertices.size();

        double r_init = inscribedRadius(seed_vertices, d);
        r_init = std::max(r_init * 0.8, 1e-4);
        Matrix2d L = r_init * Matrix2d::Identity();

        double prev_vol = r_init * r_init * M_PI;
        std::vector<HalfPlane> best_planes = bbox_planes;
        int iters = 0;

        for (int k = 0; k < max_iter; ++k) {
            iters = k + 1;

            auto planes = runRsI(obstacles, seed_vertices, L, d, bbox_planes);
            best_planes = planes;

            int m = planes.size();
            MatrixXd A(m, 2);
            VectorXd b(m);
            for (int i = 0; i < m; ++i) {
                A.row(i) = planes[i].normal.transpose();
                b(i) = planes[i].offset;
            }

            auto mvie = mvie_solver_.solve(A, b, d);

            double new_vol = mvie.volume();
            if (k > 0 && (new_vol - prev_vol) / (prev_vol + 1e-15) < rho) {
                break;
            }
            prev_vol = new_vol;
            L = mvie.L;
            d = mvie.d;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        return {best_planes, iters, ms};
    }

private:
    SDMN2D sdmn_;
    MVIE2D mvie_solver_;

    double inscribedRadius(const std::vector<Vector2d>& verts, const Vector2d& c) {
        double r = 1e18;
        int n = verts.size();
        for (int i = 0; i < n; ++i) {
            Vector2d a = verts[i];
            Vector2d b = verts[(i + 1) % n];
            Vector2d edge = b - a;
            double len = edge.norm();
            if (len < 1e-15) continue;
            Vector2d normal(-edge.y(), edge.x());
            normal /= len;
            double dist = std::abs(normal.dot(c - a));
            r = std::min(r, dist);
        }
        return r;
    }

    std::vector<HalfPlane> runRsI(
        const std::vector<std::vector<Vector2d>>& obstacles,
        const std::vector<Vector2d>& seed_vertices,
        const Matrix2d& L,
        const Vector2d& d,
        const std::vector<HalfPlane>& bbox_planes)
    {
        Matrix2d L_inv = L.inverse();
        Matrix2d L_inv_T = L_inv.transpose();

        // Transform seed vertices to normalized space [Paper Eq. 5]
        std::vector<Vector2d> seed_bar;
        seed_bar.reserve(seed_vertices.size());
        for (const auto& v : seed_vertices)
            seed_bar.push_back(L_inv * (v - d));

        // Transform obstacle polytopes to normalized space [Paper Eq. 6]
        std::vector<std::vector<Vector2d>> obs_bar;
        obs_bar.reserve(obstacles.size());
        for (const auto& poly : obstacles) {
            std::vector<Vector2d> poly_bar;
            poly_bar.reserve(poly.size());
            for (const auto& u : poly)
                poly_bar.push_back(L_inv * (u - d));
            obs_bar.push_back(std::move(poly_bar));
        }

        // Pre-build seed containment constraints (shared across all SDMN calls)
        // seed_bar[i]^T b <= 1  for each seed vertex
        int n_seed = seed_bar.size();
        std::vector<Vector2d> base_normals(n_seed);
        std::vector<double>   base_bounds(n_seed);
        for (int i = 0; i < n_seed; ++i) {
            base_normals[i] = seed_bar[i];
            base_bounds[i]  = 1.0;
        }

        struct ObsHalfPlane {
            Vector2d b_sol;   // SDMN solution in transformed space
            Vector2d a;       // a = b / ||b||^2
            double a_norm;    // ||a|| = 1/||b||
            int obs_idx;
        };
        std::vector<ObsHalfPlane> candidates;
        candidates.reserve(obs_bar.size());

        for (size_t i = 0; i < obs_bar.size(); ++i) {
            // Build constraints: seed containment + ALL vertices of obstacle i excluded
            std::vector<Vector2d> normals = base_normals;
            std::vector<double>   bounds  = base_bounds;

            for (const auto& u : obs_bar[i]) {
                normals.push_back(-u);
                bounds.push_back(-1.0);
            }

            auto result = sdmn_.solve(normals, bounds);

            if (result.feasible) {
                double b_sq = result.y.squaredNorm();
                if (b_sq > 1e-10) {
                    Vector2d a = result.y / b_sq;
                    candidates.push_back({result.y, a, a.norm(), (int)i});
                }
            }
        }

        // Greedy halfplane selection: sort by ||a|| ascending (closest first)
        std::sort(candidates.begin(), candidates.end(),
                  [](const ObsHalfPlane& x, const ObsHalfPlane& y) {
                      return x.a_norm < y.a_norm;
                  });

        std::vector<bool> separated(obs_bar.size(), false);
        std::vector<HalfPlane> result_planes = bbox_planes;

        for (const auto& hp : candidates) {
            if (separated[hp.obs_idx]) continue;

            // Transform halfplane back to original space
            Vector2d n_orig = L_inv_T * hp.a;
            double d_orig = hp.a.squaredNorm() + n_orig.dot(d);

            double n_len = n_orig.norm();
            if (n_len < 1e-15) continue;

            result_planes.push_back({n_orig / n_len, d_orig / n_len});

            // Polytope j is separated when ALL its transformed vertices satisfy
            // b_sol^T u_bar >= 1
            for (size_t j = 0; j < obs_bar.size(); ++j) {
                if (separated[j]) continue;
                bool all_sep = true;
                for (const auto& u : obs_bar[j]) {
                    if (hp.b_sol.dot(u) < 1.0 - 1e-8) { all_sep = false; break; }
                }
                if (all_sep) separated[j] = true;
            }

            if (result_planes.size() > 50) break;
        }

        return result_planes;
    }
};

// ==========================================================================
// ROS NODE
// ==========================================================================
class FIRIPolytopeNode {
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_, sub_obs_, sub_estimate_;
    ros::Publisher pub_poly_;

    FIRISolver solver_;
    Vector2d robot_pos_;
    double robot_yaw_ = 0.0;
    bool odom_rx_ = false;

    // Parameters
    double robot_length_ = 0.15;
    double robot_width_  = 0.06;
    double rear_axle_offset_ = 0.05;
    int    max_firi_iter_    = 10;
    double convergence_rho_  = 0.02;
    double bbox_behind_      = 1.0;
    double bbox_ahead_       = 3.0;
    double bbox_side_        = 1.0;

public:
    FIRIPolytopeNode() : nh_("~") {
        nh_.param("robot_length",     robot_length_,     0.15);
        nh_.param("robot_width",      robot_width_,      0.06);
        nh_.param("rear_axle_offset", rear_axle_offset_, 0.05);
        nh_.param("max_firi_iter",    max_firi_iter_,    10);
        nh_.param("convergence_rho",  convergence_rho_,  0.02);
        nh_.param("bbox_behind",      bbox_behind_,      1.0);
        nh_.param("bbox_ahead",       bbox_ahead_,       3.0);
        nh_.param("bbox_side",        bbox_side_,        1.0);

        sub_odom_     = nh_.subscribe("/odom",         1, &FIRIPolytopeNode::odomCb,        this);
        sub_obs_      = nh_.subscribe("/obstacles",    1, &FIRIPolytopeNode::obstaclesCb,   this);
        sub_estimate_ = nh_.subscribe("/initialpose",  1, &FIRIPolytopeNode::initialPoseCb, this);

        pub_poly_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);

        ROS_INFO("[FIRI Polytope] Robot: L=%.2f W=%.2f Offset=%.2f | MaxIter: %d | Rho: %.3f",
                 robot_length_, robot_width_, rear_axle_offset_, max_firi_iter_, convergence_rho_);
        ROS_INFO("[FIRI Polytope] BBox: behind=%.1f ahead=%.1f side=%.1f",
                 bbox_behind_, bbox_ahead_, bbox_side_);
    }

    void odomCb(const nav_msgs::Odometry::ConstPtr& msg) {
        robot_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y;
        tf2::Quaternion q(
            msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        double roll, pitch;
        tf2::Matrix3x3(q).getRPY(roll, pitch, robot_yaw_);
        odom_rx_ = true;
    }

    void initialPoseCb(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
        robot_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y;
        tf2::Quaternion q(
            msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        double roll, pitch;
        tf2::Matrix3x3(q).getRPY(roll, pitch, robot_yaw_);
        odom_rx_ = true;
    }

    void obstaclesCb(const obstacle_detector::Polytope2DArray::ConstPtr& msg) {
        if (!odom_rx_) return;

        // 1. Convert message to eigen polytopes
        std::vector<std::vector<Vector2d>> obstacles;
        obstacles.reserve(msg->obstacles.size());
        for (const auto& poly_msg : msg->obstacles) {
            std::vector<Vector2d> poly;
            poly.reserve(poly_msg.vertices.size());
            for (const auto& pt : poly_msg.vertices)
                poly.push_back(Vector2d(pt.x, pt.y));
            if (!poly.empty()) obstacles.push_back(std::move(poly));
        }

        // 2. Build robot footprint seed (kinematic car, rear-axle frame)
        double front_dist = robot_length_ - rear_axle_offset_;
        double rear_dist  = rear_axle_offset_;
        double hw = robot_width_ / 2.0;

        Matrix2d R;
        R << cos(robot_yaw_), -sin(robot_yaw_),
             sin(robot_yaw_),  cos(robot_yaw_);

        std::vector<Vector2d> seed = {
            robot_pos_ + R * Vector2d( front_dist,  hw),  // front left
            robot_pos_ + R * Vector2d( front_dist, -hw),  // front right
            robot_pos_ + R * Vector2d(-rear_dist,  -hw),  // rear right
            robot_pos_ + R * Vector2d(-rear_dist,   hw)   // rear left
        };

        // 3. Heading-aligned bounding box as 4 halfplanes (n^T x <= offset)
        Vector2d fwd = R.col(0);
        Vector2d lft = R.col(1);

        std::vector<FIRISolver::HalfPlane> bbox_planes = {
            { fwd,  fwd.dot(robot_pos_) + bbox_ahead_},
            {-fwd, -fwd.dot(robot_pos_) + bbox_behind_},
            { lft,  lft.dot(robot_pos_) + bbox_side_},
            {-lft, -lft.dot(robot_pos_) + bbox_side_}
        };

        // 4. Run FIRI
        auto result = solver_.compute(obstacles, seed, bbox_planes,
                                      max_firi_iter_, convergence_rho_);

        ROS_INFO_THROTTLE(1.0, "[FIRI Polytope] %zu obstacles | %d iters | %lu planes | %.2f ms",
                          obstacles.size(), result.iterations,
                          result.planes.size(), result.solve_time_ms);

        // 5. Publish
        publishPolyhedron(result.planes, msg->header);
    }

private:
    void publishPolyhedron(const std::vector<FIRISolver::HalfPlane>& planes,
                           const std_msgs::Header& header)
    {
        Polyhedron2D poly;
        for (const auto& hp : planes) {
            Vector2d pt = hp.normal * hp.offset;
            poly.add(Hyperplane2D(pt, hp.normal));
        }
        vec_E<Polyhedron2D> polys;
        polys.push_back(poly);
        decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polys);
        poly_msg.header = header;
        pub_poly_.publish(poly_msg);
    }
};

// ==========================================================================
// MAIN
// ==========================================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "firi_polytope_node");
    ROS_INFO("FIRI Polytope Node (SDMN + MVIE) starting...");
    FIRIPolytopeNode node;
    ros::spin();
    return 0;
}
