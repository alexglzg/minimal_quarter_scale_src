// ==========================================================================
// firi_grid_node.cpp — FIRI for 2D with OccupancyGrid input
//
// Based on: Wang et al., "Fast Iterative Region Inflation for Computing
//           Large 2-D/3-D Convex Regions of Obstacle-Free Space"
//           IEEE Transactions on Robotics, Vol. 41, 2025
//
// Same FIRI solver as firi_node.cpp (SDMN + MVIE), but subscribes to
// nav_msgs::OccupancyGrid instead of LaserScan.
//
// Key difference from grid_decomp_node.cpp (which uses decomp_util/RILS):
//   - Boundary extraction: only occupied cells adjacent to free space
//   - FIRI solver with seed polytope manageability guarantee
//   - Proper MVIE-based iterative inflation
//
// Topics (matching grid_decomp_node):
//   Subscribed:  /map (OccupancyGrid), /odometry/filtered (Odometry)
//   Published:   /polyhedron_array (PolyhedronArray)
//
// No OSQP dependency. Only requires Eigen, decomp_ros, standard ROS.
// ==========================================================================

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
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
// SPATIAL HASH VOXEL FILTER
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
// FIRI SOLVER
// ==========================================================================
// [Paper Algorithm 1] — identical to firi_node.cpp

class FIRISolver {
public:
    struct HalfPlane {
        Vector2d normal;
        double offset;
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

    // (bbox planes are now constructed by the caller and passed directly)

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
        const std::vector<Vector2d>& obstacles,
        const std::vector<Vector2d>& seed_vertices,
        const Matrix2d& L,
        const Vector2d& d,
        const std::vector<HalfPlane>& bbox_planes)
    {
        Matrix2d L_inv = L.inverse();
        Matrix2d L_inv_T = L_inv.transpose();

        std::vector<Vector2d> seed_bar;
        seed_bar.reserve(seed_vertices.size());
        for (const auto& v : seed_vertices) {
            seed_bar.push_back(L_inv * (v - d));
        }

        std::vector<Vector2d> obs_bar;
        obs_bar.reserve(obstacles.size());
        for (const auto& u : obstacles) {
            obs_bar.push_back(L_inv * (u - d));
        }

        int n_seed = seed_bar.size();
        int n_cons = n_seed + 1;

        std::vector<Vector2d> base_normals(n_cons);
        std::vector<double> base_bounds(n_cons);
        for (int i = 0; i < n_seed; ++i) {
            base_normals[i] = seed_bar[i];
            base_bounds[i] = 1.0;
        }

        struct ObsHalfPlane {
            Vector2d b_sol;
            Vector2d a;
            double a_norm;
            int obs_idx;
        };
        std::vector<ObsHalfPlane> candidates;
        candidates.reserve(obs_bar.size());

        for (size_t i = 0; i < obs_bar.size(); ++i) {
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

        std::sort(candidates.begin(), candidates.end(),
                  [](const ObsHalfPlane& a, const ObsHalfPlane& b) {
                      return a.a_norm < b.a_norm;
                  });

        std::vector<bool> separated(obs_bar.size(), false);
        std::vector<HalfPlane> result_planes = bbox_planes;

        for (const auto& hp : candidates) {
            if (separated[hp.obs_idx]) continue;

            Vector2d n_orig = L_inv_T * hp.a;
            double d_orig = hp.a.squaredNorm() + n_orig.dot(d);

            double n_len = n_orig.norm();
            if (n_len < 1e-15) continue;

            result_planes.push_back({n_orig / n_len, d_orig / n_len});

            for (size_t j = 0; j < obs_bar.size(); ++j) {
                if (!separated[j] && hp.b_sol.dot(obs_bar[j]) >= 1.0 - 1e-8) {
                    separated[j] = true;
                }
            }

            if (result_planes.size() > 50) break;
        }

        return result_planes;
    }
};

// ==========================================================================
// ROS NODE: OccupancyGrid input
// ==========================================================================
// Follows same architecture as grid_decomp_node:
//   - Cache obstacle points on grid receipt
//   - Filter to local region each cycle
//   - Rate-based main loop
//
// Improvement over grid_decomp_node:
//   - Boundary extraction instead of brute-force downsampling
//   - FIRI solver with robot footprint manageability

class FIRIGridNode {
    ros::NodeHandle nh_;
    ros::Subscriber grid_sub_, odom_sub_;
    ros::Publisher poly_pub_;

    FIRISolver solver_;

    // Robot state
    Vector2d robot_pos_;
    double robot_yaw_ = 0.0;
    bool odom_received_ = false;

    // Cached obstacle data
    std::vector<Vector2d> all_boundary_obs_;  // boundary corner points
    bool grid_cached_ = false;
    std::string frame_id_;
    double grid_resolution_ = 0.05;  // updated from received grid

    // Parameters
    double robot_length_ = 0.9;
    double robot_width_ = 0.45;
    double voxel_size_ = 0.1;
    int max_firi_iter_ = 10;
    double convergence_rho_ = 0.02;
    double bbox_behind_ = 2.0;
    double bbox_ahead_ = 6.0;
    double bbox_side_ = 4.0;
    int occupancy_threshold_ = 50;
    bool use_boundary_extraction_ = true;

public:
    FIRIGridNode() : nh_("~") {
        nh_.param("robot_length", robot_length_, 0.9);
        nh_.param("robot_width", robot_width_, 0.45);
        nh_.param("voxel_size", voxel_size_, 0.1);
        nh_.param("max_firi_iter", max_firi_iter_, 10);
        nh_.param("convergence_rho", convergence_rho_, 0.02);
        nh_.param("bbox_behind", bbox_behind_, 1.0);
        nh_.param("bbox_ahead", bbox_ahead_, 3.0);
        nh_.param("bbox_side", bbox_side_, 0.75);
        nh_.param("occupancy_threshold", occupancy_threshold_, 50);
        nh_.param("use_boundary_extraction", use_boundary_extraction_, true);

        poly_pub_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);

        grid_sub_ = nh_.subscribe("/map", 1, &FIRIGridNode::gridCallback, this);
        odom_sub_ = nh_.subscribe("/odometry/filtered", 1, &FIRIGridNode::odomCallback, this);

        ROS_INFO("[FIRI Grid] Robot: %.2f x %.2f | Voxel: %.2f", robot_length_, robot_width_, voxel_size_);
        ROS_INFO("[FIRI Grid] MaxIter: %d | Rho: %.3f | BBox: behind=%.1f ahead=%.1f side=%.1f",
                 max_firi_iter_, convergence_rho_, bbox_behind_, bbox_ahead_, bbox_side_);
        ROS_INFO("[FIRI Grid] Occupancy threshold: %d | Boundary extraction: %s",
                 occupancy_threshold_, use_boundary_extraction_ ? "ON" : "OFF");
    }

    // ----------------------------------------------------------------
    // Grid callback: convert to obstacle points and cache
    // ----------------------------------------------------------------
    void gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        frame_id_ = msg->header.frame_id;
        grid_resolution_ = msg->info.resolution;
        auto t_start = std::chrono::high_resolution_clock::now();

        if (use_boundary_extraction_) {
            extractBoundaryObstacles(msg);
        } else {
            extractAllObstacles(msg);
        }

        grid_cached_ = true;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ROS_INFO("[FIRI Grid] Grid %dx%d (%.3fm) -> %zu boundary corner points in %.1f ms",
                 msg->info.width, msg->info.height, msg->info.resolution,
                 all_boundary_obs_.size(), ms);
    }

    // ----------------------------------------------------------------
    // Odometry callback
    // ----------------------------------------------------------------
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        robot_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y;

        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch;
        m.getRPY(roll, pitch, robot_yaw_);
        odom_received_ = true;
    }

    // ----------------------------------------------------------------
    // Main loop: called at fixed rate
    // ----------------------------------------------------------------
    void run() {
        if (!grid_cached_) {
            ROS_WARN_THROTTLE(2.0, "[FIRI Grid] Waiting for occupancy grid...");
            return;
        }
        if (!odom_received_) {
            ROS_WARN_THROTTLE(2.0, "[FIRI Grid] Waiting for odometry...");
            return;
        }

        // 1. Filter cached obstacles to local bounding box
        std::vector<Vector2d> local_obs = filterLocalObstacles();

        if (local_obs.empty()) {
            ROS_WARN_THROTTLE(2.0, "[FIRI Grid] No obstacles in local region");
            return;
        }

        // 2. Optional additional voxel filter
        //    With corner-point boundary extraction, points are already
        //    deduplicated on the grid-vertex lattice.  Applying a voxel
        //    filter with bin size > grid_resolution would merge distinct
        //    corners and reintroduce the safety gap.  So: skip the filter
        //    when using boundary extraction; apply only in fallback mode.
        if (!use_boundary_extraction_) {
            local_obs = voxelFilter(local_obs, voxel_size_);
        }

        // 3. Build robot footprint seed
        double hl = robot_length_ / 2.0 + 0.025;  // add small margin to ensure manageability
        double hw = robot_width_ / 2.0 + 0.025;
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
        Vector2d fwd = R.col(0);  // forward unit vector
        Vector2d lft = R.col(1);  // left unit vector

        std::vector<FIRISolver::HalfPlane> bbox_planes = {
            { fwd,  fwd.dot(robot_pos_) + bbox_ahead_},   // front wall
            {-fwd, -fwd.dot(robot_pos_) + bbox_behind_},  // rear wall
            { lft,  lft.dot(robot_pos_) + bbox_side_},     // left wall
            {-lft, -lft.dot(robot_pos_) + bbox_side_}      // right wall
        };

        // 5. Run FIRI
        auto result = solver_.compute(local_obs, seed, bbox_planes, max_firi_iter_, convergence_rho_);

        ROS_INFO_THROTTLE(1.0, "[FIRI Grid] %zu local obs | %d iters | %lu planes | %.2f ms",
                          local_obs.size(), result.iterations, result.planes.size(),
                          result.solve_time_ms);

        // 6. Publish
        publishPolyhedron(result.planes);
    }

private:
    // ================================================================
    // BOUNDARY EXTRACTION  (corner-point variant)
    // ================================================================
    // Safety issue with cell-center representation:
    //   Each occupied cell is a res×res square, but a single center
    //   point lets FIRI's halfplane cut up to res√2/2 into the cell.
    //
    // Fix: emit the 4 corner vertices of every boundary cell instead
    // of the center.  Adjacent cells share corners, so we deduplicate
    // via an integer-grid hash set.  Along a connected boundary of N
    // cells the unique corner count is ~2N, not 4N.
    //
    // Corner indexing:  cell (col,row) has corners at grid vertices
    //   (col, row), (col+1, row), (col, row+1), (col+1, row+1)
    // which map to metric coordinates (col*res, row*res), etc.
    // (no +0.5 offset — these are actual cell edges, not centers).

    void extractBoundaryObstacles(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        all_boundary_obs_.clear();
        const auto& info = msg->info;
        int w = info.width;
        int h = info.height;

        // Map origin transform
        tf2::Quaternion q_map(
            info.origin.orientation.x, info.origin.orientation.y,
            info.origin.orientation.z, info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_map).getRPY(roll, pitch, yaw);
        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);
        double ox = info.origin.position.x;
        double oy = info.origin.position.y;
        double res = info.resolution;

        // 4-connected neighbor offsets
        const int dx[] = {1, -1, 0, 0};
        const int dy[] = {0, 0, 1, -1};

        // Deduplicate corners via integer grid-vertex indices.
        // Corner vertices live on a (w+1)×(h+1) grid.
        std::unordered_set<Vector2i, PointHash> corner_set;
        corner_set.reserve(w * h / 4); // rough estimate

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                int idx = row * w + col;
                int8_t val = msg->data[idx];

                // Must be occupied
                if (val < occupancy_threshold_) continue;

                // Check if any 4-connected neighbor is free
                bool is_boundary = false;
                for (int n = 0; n < 4; ++n) {
                    int nr = row + dy[n];
                    int nc = col + dx[n];

                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                        is_boundary = true;
                        break;
                    }

                    int8_t nval = msg->data[nr * w + nc];
                    if (nval >= 0 && nval < occupancy_threshold_) {
                        is_boundary = true;
                        break;
                    }
                }

                if (!is_boundary) continue;

                // Insert the 4 corner vertices of this cell
                corner_set.insert(Vector2i(col,     row    ));
                corner_set.insert(Vector2i(col + 1, row    ));
                corner_set.insert(Vector2i(col,     row + 1));
                corner_set.insert(Vector2i(col + 1, row + 1));
            }
        }

        // Convert unique corner vertices to world coordinates
        all_boundary_obs_.reserve(corner_set.size());
        for (const auto& cv : corner_set) {
            double x_grid = cv.x() * res;
            double y_grid = cv.y() * res;
            double x = ox + x_grid * cos_yaw - y_grid * sin_yaw;
            double y = oy + x_grid * sin_yaw + y_grid * cos_yaw;
            all_boundary_obs_.push_back(Vector2d(x, y));
        }
    }

    // Fallback: extract all occupied cells (like original grid_decomp_node)
    void extractAllObstacles(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        all_boundary_obs_.clear();
        const auto& info = msg->info;
        int w = info.width;
        int h = info.height;

        tf2::Quaternion q_map(
            info.origin.orientation.x, info.origin.orientation.y,
            info.origin.orientation.z, info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q_map).getRPY(roll, pitch, yaw);
        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);
        double ox = info.origin.position.x;
        double oy = info.origin.position.y;
        double res = info.resolution;

        all_boundary_obs_.reserve(w * h / 4);

        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                int idx = row * w + col;
                if (msg->data[idx] >= occupancy_threshold_) {
                    double x_grid = (col + 0.5) * res;
                    double y_grid = (row + 0.5) * res;
                    double x = ox + x_grid * cos_yaw - y_grid * sin_yaw;
                    double y = oy + x_grid * sin_yaw + y_grid * cos_yaw;
                    all_boundary_obs_.push_back(Vector2d(x, y));
                }
            }
        }
    }

    // ================================================================
    // LOCAL OBSTACLE FILTERING
    // ================================================================
    // Radius filter around robot position, matching grid_decomp_node pattern.
    // For static maps this is cheap since the full set is cached.

    std::vector<Vector2d> filterLocalObstacles() {
        double search_radius = std::max({bbox_ahead_, bbox_behind_, bbox_side_}) * 1.2;
        double search_radius_sq = search_radius * search_radius;

        std::vector<Vector2d> local;
        local.reserve(all_boundary_obs_.size() / 4);

        for (const auto& obs : all_boundary_obs_) {
            double dx = obs.x() - robot_pos_.x();
            double dy = obs.y() - robot_pos_.y();
            if (dx * dx + dy * dy <= search_radius_sq) {
                local.push_back(obs);
            }
        }
        return local;
    }

    // ================================================================
    // PUBLISHING
    // ================================================================
    void publishPolyhedron(const std::vector<FIRISolver::HalfPlane>& planes) {
        Polyhedron2D poly;
        for (const auto& hp : planes) {
            Vector2d pt = hp.normal * hp.offset;
            poly.add(Hyperplane2D(pt, hp.normal));
        }
        vec_E<Polyhedron2D> polys;
        polys.push_back(poly);
        decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polys);
        poly_msg.header.frame_id = frame_id_;
        poly_msg.header.stamp = ros::Time::now();
        poly_pub_.publish(poly_msg);
    }
};

// ==========================================================================
// MAIN
// ==========================================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "firi_grid_node");
    ROS_INFO("FIRI Grid Node (SDMN + MVIE) starting...");

    FIRIGridNode node;

    ros::Rate rate(10); // 10 Hz, matching grid_decomp_node

    while (ros::ok()) {
        ros::spinOnce();
        node.run();
        rate.sleep();
    }

    return 0;
}