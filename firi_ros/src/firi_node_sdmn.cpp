// ==========================================================================
// firi_node.cpp — FIRI (Fast Iterative Region Inflation) for 2D
//
// Based on: Wang et al., "Fast Iterative Region Inflation for Computing
//           Large 2-D/3-D Convex Regions of Obstacle-Free Space"
//           IEEE Transactions on Robotics, Vol. 41, 2025
//
// Key components:
//   1. SDMN  — Seidel's Small-Dimensional Minimum-Norm (replaces OSQP)
//              Expected O(d) for 2D. [Paper Sec. IV, Algorithm 2]
//   2. MVIE  — Maximum Volume Inscribed Ellipsoid via log-barrier Newton
//              [Paper Sec. V, SOCP reformulation concept]
//   3. FIRI  — Full outer loop: RsI → greedy halfplane select → MVIE → repeat
//              With convergence criterion. [Paper Sec. III, Algorithm 1]
//   4. ROS node subscribing to LaserScan + odometry
//
// No OSQP dependency. Only requires Eigen, decomp_ros, standard ROS.
// ==========================================================================

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_geometry/geometric_utils.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_set>
#include <chrono>

using namespace Eigen;

// ==========================================================================
// SPATIAL HASH VOXEL FILTER (unchanged)
// ==========================================================================
struct PointHash {
    size_t operator()(const Vector2i& k) const {
        return std::hash<int>()(k.x()) ^ (std::hash<int>()(k.y()) << 1);
    }
};

std::vector<Vector2d> voxelFilter(const std::vector<Vector2d>& pts, double res) {
    if (pts.empty()) return {};
    std::unordered_set<Vector2i, PointHash> grid;
    std::vector<Vector2d> filtered;
    filtered.reserve(pts.size());
    double inv_res = 1.0 / res;
    for (const auto& p : pts) {
        Vector2i k(std::floor(p.x() * inv_res), std::floor(p.y() * inv_res));
        if (grid.insert(k).second) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

// ==========================================================================
// SDMN: Small-Dimensional Minimum-Norm (2D specialization)
// ==========================================================================
// Solves:  min ||y||^2   s.t.  e_i^T y <= f_i,  i = 1..d
//
// Generalizes Seidel's randomized LP algorithm to minimum-norm QP.
// For n=2: expected complexity O(2! * d) = O(d), linear in constraints.
// Replaces OSQP for the RsI halfplane computation.
// [Paper Section IV, Algorithm 2]

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

        // Random permutation [Paper Sec. IV-B3: ensures expected linear time]
        std::vector<size_t> perm(d);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        Vector2d y = Vector2d::Zero(); // unconstrained minimum

        for (size_t ii = 0; ii < d; ++ii) {
            size_t idx = perm[ii];

            // Violation check [Paper Fig. 3(a)→(b): not violated, keep y]
            if (e[idx].dot(y) <= f[idx] + 1e-12) continue;

            // Violated → constraint is active at the optimum
            // Must solve on the constraint plane: e_h^T y = f_h
            // [Paper Fig. 3(a)→(c): project to 1D subproblem]
            const Vector2d& eh = e[idx];
            double fh = f[idx];
            double eTe = eh.squaredNorm();
            if (eTe < 1e-15) return {Vector2d::Zero(), false};

            // Minimum-norm point on the constraint plane [Paper Eq. 18]
            Vector2d v = (fh / eTe) * eh;

            // Householder reflection for dimensionality reduction [Paper Eq. 24-26]
            // Maps v to be parallel to e_j, giving basis for 1D subspace
            int j = (std::abs(v(0)) >= std::abs(v(1))) ? 0 : 1;
            int k = 1 - j;

            Vector2d m_col; // basis vector for constraint plane
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
                    // Column k of H^T [Paper: M = H^T without column j]
                    m_col = Vector2d::Unit(k) - (2.0 * u_ref(k) / uTu) * u_ref;
                }
            }

            // 1D subproblem: min t^2 s.t. a_i*t <= b_i [Paper Eq. 23, then base case]
            // Transform previous constraints: e_p^T(m*t + v) <= f_p
            //   => (e_p^T m)*t <= f_p - e_p^T v
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

            // 1D minimum-norm: closest to 0 in [lo, hi] [Paper Alg. 2, Line 4]
            double t;
            if (lo <= 0.0 && 0.0 <= hi) t = 0.0;
            else if (lo > 0.0)           t = lo;
            else                          t = hi;

            y = m_col * t + v; // [Paper Eq. 21]
        }

        return {y, true};
    }

private:
    mutable std::mt19937 rng_;
};

// ==========================================================================
// MVIE: Maximum Volume Inscribed Ellipsoid (2D)
// ==========================================================================
// Solves:  max det(L)  s.t.  ||L^T a_i|| + a_i^T d <= b_i,  i=1..m
//          L lower-triangular with positive diagonal
//          Ellipsoid = { L*x + d : ||x|| <= 1 }
//
// Uses log-barrier interior-point method with Newton steps.
// 5 decision variables (L11, L21, L22, d1, d2), typically 10-30 constraints.
// [Paper Section V concept, simplified for 2D]

class MVIE2D {
public:
    struct Ellipsoid {
        Matrix2d L;  // lower triangular
        Vector2d d;  // center
        double volume() const { return M_PI * std::abs(L(0, 0) * L(1, 1)); }
    };

    // Solve MVIE for polytope {x : A*x <= b}
    // center_hint should be strictly inside the polytope
    Ellipsoid solve(const MatrixXd& A, const VectorXd& b,
                    const Vector2d& center_hint)
    {
        int m = A.rows();

        // --- Initialization: Chebyshev ball at center_hint ---
        Vector2d c = center_hint;
        double r = 1e18;
        for (int i = 0; i < m; ++i) {
            double norm_ai = A.row(i).norm();
            if (norm_ai > 1e-10) {
                double gap = b(i) - A.row(i).dot(c);
                r = std::min(r, gap / norm_ai);
            }
        }
        if (r <= 0) r = 1e-4; // fallback
        r *= 0.9; // strictly feasible
        r = std::max(r, 1e-6);

        // State vector: x = [L11, L21, L22, d1, d2]
        VectorXd x(5);
        x << r, 0.0, r, c(0), c(1);

        // --- Log-barrier method ---
        // Minimize: -t*(log L11 + log L22) + sum_i -log(gap_i)
        double t = 1.0;
        const double mu = 4.0;

        for (int outer = 0; outer < 20; ++outer) {
            // Newton centering steps
            for (int inner = 0; inner < 40; ++inner) {
                VectorXd grad = gradient(A, b, x, t);
                MatrixXd H = hessian(A, b, x, t);
                H += 1e-8 * MatrixXd::Identity(5, 5); // regularize

                VectorXd dx = H.ldlt().solve(-grad);
                double lambda_sq = -grad.dot(dx);
                if (lambda_sq < 1e-6) break; // converged

                // Backtracking line search
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
    // Barrier objective value
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

    // Analytical gradient of barrier objective
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
                g(0) += ig * r1 * a1 * inr;    // ∂/∂L11
                g(1) += ig * r1 * a2 * inr;    // ∂/∂L21
                g(2) += ig * r2 * a2 * inr;    // ∂/∂L22
            }
            g(3) += ig * a1;                    // ∂/∂d1
            g(4) += ig * a2;                    // ∂/∂d2
        }
        return g;
    }

    // Numerical Hessian via central differences on gradient
    // 5 variables → 10 gradient evaluations (symmetric), each O(m). Negligible.
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
// FIRI SOLVER: Full Algorithm
// ==========================================================================
// [Paper Algorithm 1]
//
// Outer loop:
//   1. Transform obstacles & seed into normalized space (ellipsoid → unit ball)
//   2. RsI: for each obstacle, solve SDMN for separating halfplane
//   3. Greedy halfplane selection (closest first, remove separated obstacles)
//   4. Transform polytope back to original space
//   5. Compute MVIE of the polytope
//   6. Check convergence (MVIE volume improvement < rho)

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

    Result compute(const std::vector<Vector2d>& obstacles,
                   const std::vector<Vector2d>& seed_vertices,
                   const std::vector<HalfPlane>& bbox_planes,
                   int max_iter = 10,
                   double rho = 0.02)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        if (obstacles.empty() || seed_vertices.empty()) {
            return {bbox_planes, 0, 0.0};
        }

        // --- Initialize ellipsoid strictly inside seed [Paper Sec. III-B] ---
        Vector2d d = Vector2d::Zero();
        for (const auto& v : seed_vertices) d += v;
        d /= seed_vertices.size();

        // Inscribed ball of seed polygon at centroid
        double r_init = inscribedRadius(seed_vertices, d);
        r_init = std::max(r_init * 0.8, 1e-4);
        Matrix2d L = r_init * Matrix2d::Identity();

        // --- Outer FIRI loop ---
        double prev_vol = r_init * r_init * M_PI;
        std::vector<HalfPlane> best_planes = bbox_planes;
        int iters = 0;

        for (int k = 0; k < max_iter; ++k) {
            iters = k + 1;

            // --- RsI step [Paper Sec. III-B1, Lines 7-18] ---
            auto planes = runRsI(obstacles, seed_vertices, L, d, bbox_planes);
            best_planes = planes;

            // --- MVIE [Paper Sec. III-B2, Line 21] ---
            // Build H-representation matrix for MVIE
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
                break; // Converged [Paper Sec. III-B3, Line 23]
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

    // (bbox planes are now constructed by the caller and passed directly)

    // Compute inscribed ball radius of a convex polygon at a given center
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

    // --- Full RsI step ---
    // For each obstacle: transform, solve SDMN, get halfplane.
    // Then greedy selection. Return planes in original space.
    std::vector<HalfPlane> runRsI(
        const std::vector<Vector2d>& obstacles,
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
        for (const auto& v : seed_vertices) {
            seed_bar.push_back(L_inv * (v - d));
        }

        // Transform obstacles to normalized space [Paper Eq. 6]
        std::vector<Vector2d> obs_bar;
        obs_bar.reserve(obstacles.size());
        for (const auto& u : obstacles) {
            obs_bar.push_back(L_inv * (u - d));
        }

        // --- Solve SDMN for each obstacle [Paper Alg. 1, Lines 10-11] ---
        // Constraints for obstacle u_bar:
        //   Seed containment:    v_bar^T b <= 1   for each seed vertex
        //   Obstacle exclusion: -u_bar^T b <= -1   (i.e., u_bar^T b >= 1)
        //
        // [Paper Eq. 15: min ||b||^2 s.t. v^T b <= 1, u^T b >= 1]

        int n_seed = seed_bar.size();
        int n_cons = n_seed + 1;

        // Pre-build seed constraint normals (shared across all obstacle SDMN calls)
        std::vector<Vector2d> base_normals(n_cons);
        std::vector<double> base_bounds(n_cons);
        for (int i = 0; i < n_seed; ++i) {
            base_normals[i] = seed_bar[i];
            base_bounds[i] = 1.0;
        }
        // Last slot reserved for obstacle (filled per-obstacle)

        struct ObsHalfPlane {
            Vector2d b_sol;    // SDMN solution in transformed space
            Vector2d a;        // contact point: a = b/||b||^2
            double a_norm;     // ||a|| = 1/||b|| (smaller = closer = more constraining)
            int obs_idx;
        };
        std::vector<ObsHalfPlane> candidates;
        candidates.reserve(obs_bar.size());

        for (size_t i = 0; i < obs_bar.size(); ++i) {
            // Set obstacle constraint: -u^T b <= -1
            base_normals[n_seed] = -obs_bar[i];
            base_bounds[n_seed] = -1.0;

            auto result = sdmn_.solve(base_normals, base_bounds);

            if (result.feasible) {
                double b_sq = result.y.squaredNorm();
                if (b_sq > 1e-10) {
                    Vector2d a = result.y / b_sq;
                    candidates.push_back({result.y, a, a.norm(), (int)i});
                }
            }
        }

        // --- Greedy halfplane selection [Paper Alg. 1, Lines 12-16] ---
        // Sort by ||a|| ascending (closest halfplane to unit ball first)
        std::sort(candidates.begin(), candidates.end(),
                  [](const ObsHalfPlane& a, const ObsHalfPlane& b) {
                      return a.a_norm < b.a_norm;
                  });

        std::vector<bool> separated(obs_bar.size(), false);
        std::vector<HalfPlane> result_planes = bbox_planes;

        for (const auto& hp : candidates) {
            if (separated[hp.obs_idx]) continue;

            // Transform halfplane to original space
            // Transformed space: a^T x_bar <= ||a||^2
            // Original space: (L^{-T} a)^T x <= ||a||^2 + (L^{-T} a)^T d
            Vector2d n_orig = L_inv_T * hp.a;
            double d_orig = hp.a.squaredNorm() + n_orig.dot(d);

            // Normalize to unit normal
            double n_len = n_orig.norm();
            if (n_len < 1e-15) continue;

            result_planes.push_back({n_orig / n_len, d_orig / n_len});

            // Remove obstacles separated by this halfplane [Paper Alg. 1, Line 16]
            // Obstacle j is separated if b_selected^T u_bar_j >= 1
            for (size_t j = 0; j < obs_bar.size(); ++j) {
                if (!separated[j] && hp.b_sol.dot(obs_bar[j]) >= 1.0 - 1e-8) {
                    separated[j] = true;
                }
            }

            if (result_planes.size() > 50) break; // safety cap
        }

        return result_planes;
    }
};

// ==========================================================================
// ROS NODE
// ==========================================================================
class FIRINode {
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_, sub_scan_;
    ros::Publisher pub_poly_;

    FIRISolver solver_;
    Vector2d robot_pos_;
    double robot_yaw_;
    bool odom_rx_ = false;

    // Parameters
    double length_ = 0.9;
    double width_ = 0.45;
    double voxel_size_ = 0.1;
    int max_firi_iter_ = 10;
    double convergence_rho_ = 0.02;
    double bbox_behind_ = 2.0;
    double bbox_ahead_ = 6.0;
    double bbox_side_ = 4.0;

public:
    FIRINode() : nh_("~") {
        nh_.param("robot_length", length_, 0.9);
        nh_.param("robot_width", width_, 0.45);
        nh_.param("voxel_size", voxel_size_, 0.1);
        nh_.param("max_firi_iter", max_firi_iter_, 10);
        nh_.param("convergence_rho", convergence_rho_, 0.02);
        nh_.param("bbox_behind", bbox_behind_, 1.0);
        nh_.param("bbox_ahead", bbox_ahead_, 7.0);
        nh_.param("bbox_side", bbox_side_, 3.0);

        sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &FIRINode::odomCb, this);
        sub_scan_ = nh_.subscribe("/filtered_scan", 1, &FIRINode::scanCb, this);
        pub_poly_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);

        ROS_INFO("[FIRI] Ready. Robot: %.2f x %.2f | Voxel: %.2f | MaxIter: %d | Rho: %.3f",
                 length_, width_, voxel_size_, max_firi_iter_, convergence_rho_);
        ROS_INFO("[FIRI] BBox: behind=%.1f ahead=%.1f side=%.1f",
                 bbox_behind_, bbox_ahead_, bbox_side_);
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

    void scanCb(const sensor_msgs::LaserScan::ConstPtr& msg) {
        if (!odom_rx_) return;

        // 1. Convert LaserScan → 2D obstacle points in map frame
        std::vector<Vector2d> raw_obs;
        raw_obs.reserve(msg->ranges.size());
        double angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); ++i, angle += msg->angle_increment) {
            float r = msg->ranges[i];
            if (r < msg->range_min || r > msg->range_max || !std::isfinite(r)) continue;
            double map_angle = angle + robot_yaw_;
            raw_obs.push_back(Vector2d(
                robot_pos_.x() + r * cos(map_angle),
                robot_pos_.y() + r * sin(map_angle)));
        }

        // 2. Downsample
        std::vector<Vector2d> obstacles = voxelFilter(raw_obs, voxel_size_);

        ROS_INFO_THROTTLE(1.0, "[FIRI] Scan rays: %lu -> Obstacles: %lu",
                          raw_obs.size(), obstacles.size());

        // 3. Robot footprint as seed polygon [Paper Eq. 1: seed = conv{v1..vs}]
        double hl = length_ / 2.0 + 0.025; // add small margin
        double hw = width_ / 2.0 + 0.025;
        Matrix2d R;
        R << cos(robot_yaw_), -sin(robot_yaw_),
             sin(robot_yaw_),  cos(robot_yaw_);

        std::vector<Vector2d> seed = {
            robot_pos_ + R * Vector2d( hl,  hw),
            robot_pos_ + R * Vector2d( hl, -hw),
            robot_pos_ + R * Vector2d(-hl, -hw),
            robot_pos_ + R * Vector2d(-hl,  hw)
        };

        // 4. Heading-aligned bounding box as 4 halfplanes (n^T x <= d)
        //    Constructed in body frame then expressed in map frame
        Vector2d fwd = R.col(0);  // forward unit vector
        Vector2d lft = R.col(1);  // left unit vector

        std::vector<FIRISolver::HalfPlane> bbox_planes = {
            { fwd,  fwd.dot(robot_pos_) + bbox_ahead_},   // front wall
            {-fwd, -fwd.dot(robot_pos_) + bbox_behind_},  // rear wall
            { lft,  lft.dot(robot_pos_) + bbox_side_},     // left wall
            {-lft, -lft.dot(robot_pos_) + bbox_side_}      // right wall
        };

        // 5. Run FIRI
        auto result = solver_.compute(obstacles, seed, bbox_planes, max_firi_iter_, convergence_rho_);

        ROS_INFO_THROTTLE(1.0, "[FIRI] %d iters, %lu planes, %.2f ms",
                          result.iterations, result.planes.size(), result.solve_time_ms);

        // 6. Publish
        std_msgs::Header header;
        header.stamp = msg->header.stamp;
        header.frame_id = "map";
        publishPolyhedron(result.planes, header);
    }

    void publishPolyhedron(const std::vector<FIRISolver::HalfPlane>& planes,
                           const std_msgs::Header& header)
    {
        Polyhedron2D poly;
        for (const auto& hp : planes) {
            // Hyperplane2D convention: point on plane + normal
            // n^T x <= d  →  point = n * d (works for unit normal)
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
    ros::init(argc, argv, "firi_node");
    ROS_INFO("FIRI Node (SDMN + MVIE) starting...");
    FIRINode node;
    ros::spin();
    return 0;
}