import casadi as ca
from casadi import sin as ca_sin, cos as ca_cos, exp as ca_exp, atan2 as ca_atan2
import numpy as np
import matplotlib.pyplot as plt
from diffmpccbf.utils.path import Path


DEFAULT_MODEL_PARAMS = {
    "m11": 12.0,
    "m22": 24.0,
    "m33": 3.0,
    "d11": 6.0,
    "d22": 8.0,
    "d33": 0.6,
    "aa": 0.45,
    "bb": 0.9,
    "ego_length": 0.9,
    "ego_width": 0.45,
    "max_force": 6.0,
    "u_ref": 0.0,
}


class vessel_mpc_opti():
    """ASV vessel MPC using CasADi Opti with RK4 multiple shooting.

    State x (6):  [ned_x, ned_y, psi, surge, sway, yaw_rate]
    Control u (4): [u1, u2, u3, u4]  (thruster forces)

    Dynamics (continuous):
        ned_x_dot  = surge*cos(psi) - sway*sin(psi)
        ned_y_dot  = surge*sin(psi) + sway*cos(psi)
        psi_dot    = yaw_rate
        surge_dot  = -d11/m11 * surge + (u1 + u2) / m11
        sway_dot   = -d22/m22 * sway  + (u3 + u4) / m22
        yaw_dot    = -d33/m33 * yaw_rate + aa/(2*m33)*(u1-u2) + bb/(2*m33)*(u3-u4)
    """

    VALID_CBF_TYPES = ('c_hocbf', 'dcbf', 'combo_chocbf_dcbf', None)

    def __init__(self, T, N, model_params=None):
        self.T = T
        self.N = N
        self.dt = T / N
        p = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.m11 = p['m11']
        self.m22 = p['m22']
        self.m33 = p['m33']
        self.d11 = p['d11']
        self.d22 = p['d22']
        self.d33 = p['d33']
        self.aa = p['aa']
        self.bb = p['bb']
        self.max_force = p['max_force']
        self.u_ref_default = p['u_ref']

    def define_problem(self, n_obs,
                       obstacle_radius=0.5, vessel_radius=0.45,
                       cbf_type=None, adaptive=False,
                       mode='waypoint', path=None,
                       Qpos=20.0, Qu=200.0, Qr=0.1, Qpsi=10.0):
        """Build the NLP structure. Call once; use solve() repeatedly for MPC.

        Parameters
        ----------
        n_obs : int   number of obstacles (fixes NLP structure; positions are set at solve time)
        obstacle_radius, vessel_radius : floats  define safety clearance r_safe = sum
        cbf_type : one of VALID_CBF_TYPES
        adaptive : bool   make alpha values decision variables (penalised toward nominal)
        mode : 'waypoint' or 'path'
            'waypoint' : minimise distance to a goal point + heading error to a desired
                         heading (6-state, goal=(x, y, psi_desired) set at solve time)
            'path'     : minimise cross-track error to a parametric path (7-state, s is carried
                         as a state with s_dot = surge). Requires path.
        path : diffmpccbf.utils.path.Path, required for mode='path'
            Provides x(s), y(s), x_dot(s), y_dot(s) defining the reference path in NED.
        Qpos : cost weight — waypoint position error OR cross-track error (Qye)
        Qu, Qr : surge tracking, yaw-rate damping weights
        Qpsi : heading alignment weight — to psi_desired in waypoint mode, to the
            path tangent in path mode
        """
        if cbf_type not in self.VALID_CBF_TYPES:
            raise ValueError(f"cbf_type must be one of {self.VALID_CBF_TYPES}, got {cbf_type!r}")
        if mode not in ('waypoint', 'path'):
            raise ValueError(f"mode must be 'waypoint' or 'path', got {mode!r}")
        if mode == 'path' and path is None:
            raise ValueError("path is required for mode='path'")

        self.cbf_type = cbf_type
        self.adaptive = adaptive
        self.obstacle_radius = obstacle_radius
        self.vessel_radius = vessel_radius
        self.mode = mode

        N = self.N
        dt = self.dt
        r_safe = obstacle_radius + vessel_radius

        m11, m22, m33 = self.m11, self.m22, self.m33
        d11, d22, d33 = self.d11, self.d22, self.d33
        aa, bb = self.aa, self.bb
        max_force = self.max_force

        n_states = 7 if mode == 'path' else 6   # path mode carries s as a state
        self.n_states = n_states

        opti = ca.Opti()
        self.opti = opti

        # Decision variables
        # waypoint: [ned_x, ned_y, psi, surge, sway, yaw_rate]
        # path:     [ned_x, ned_y, psi, surge, sway, yaw_rate, s]
        X = opti.variable(n_states, N + 1)
        U = opti.variable(4, N)        # [u1, u2, u3, u4]
        self.X = X
        self.U = U

        # Adaptive alpha decision variables
        if adaptive:
            n_steps = N + 1
            if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                alpha1_bar = opti.variable(n_obs, n_steps)
                alpha2_bar = opti.variable(n_obs, n_steps)
                virtual_ctrl_alpha1 = opti.variable(n_obs, n_steps)
                self.alpha1_bar = alpha1_bar
                self.alpha2_bar = alpha2_bar
                self.virtual_ctrl_alpha1 = virtual_ctrl_alpha1
            elif cbf_type == 'dcbf':
                alpha_bar = opti.variable(n_obs, n_steps)
                self.alpha_bar = alpha_bar

        # Parameters — values are set at solve time, not here
        x0_param = opti.parameter(n_states)
        self.x0_param = x0_param

        if mode == 'waypoint':
            goal_param = opti.parameter(3)   # current waypoint target (ned_x, ned_y, psi_desired)
            self.goal_param = goal_param

        uref_param = opti.parameter()        # desired surge speed
        self.uref_param = uref_param

        obs_x_param = opti.parameter(n_obs)  # obstacle x positions
        obs_y_param = opti.parameter(n_obs)  # obstacle y positions
        self.obs_x_param = obs_x_param
        self.obs_y_param = obs_y_param

        # Nominal CBF alpha parameters
        if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            alpha1_param = opti.parameter(n_obs)
            alpha2_param = opti.parameter(n_obs)
            self.alpha1_param = alpha1_param
            self.alpha2_param = alpha2_param
        elif cbf_type == 'dcbf':
            alpha_param = opti.parameter(n_obs)
            self.alpha_param = alpha_param

        def _heading_error(psi, psi_d):
            """Squared sin/cos heading error — avoids angle-wrap issues vs. (psi - psi_d)**2."""
            return (ca.sin(psi) - ca.sin(psi_d)) ** 2 + (ca.cos(psi) - ca.cos(psi_d)) ** 2

        # --- Path tracking (mode='path' only) ---
        if mode == 'path':
            self.path = path

            def _cross_track(s_val, ned_x, ned_y, psi):
                """Cross-track error and path-tangent heading at arc-length s_val."""
                x_d = path.x(s_val)
                y_d = path.y(s_val)
                gamma_p = ca_atan2(path.y_dot(s_val), path.x_dot(s_val))
                ye = -(ned_x - x_d) * ca.sin(gamma_p) + (ned_y - y_d) * ca.cos(gamma_p)
                return ye, _heading_error(psi, gamma_p)

        # --- Dynamics ---
        def f(x, ctrl):
            ned_x, ned_y, psi = x[0], x[1], x[2]
            surge, sway, yr = x[3], x[4], x[5]
            u1, u2, u3, u4 = ctrl[0], ctrl[1], ctrl[2], ctrl[3]
            xdot = ca.vertcat(
                surge * ca.cos(psi) - sway * ca.sin(psi),
                surge * ca.sin(psi) + sway * ca.cos(psi),
                yr,
                -d11 / m11 * surge + (u1 + u2) / m11,
                -d22 / m22 * sway  + (u3 + u4) / m22,
                -d33 / m33 * yr    + aa / (2 * m33) * (u1 - u2) + bb / (2 * m33) * (u3 - u4),
            )
            if mode == 'path':
                xdot = ca.vertcat(xdot, surge)   # s_dot = surge
            return xdot

        def rk4(x, ctrl):
            k1 = f(x, ctrl)
            k2 = f(x + dt / 2 * k1, ctrl)
            k3 = f(x + dt / 2 * k2, ctrl)
            k4 = f(x + dt * k3, ctrl)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Initial condition + multiple-shooting continuity
        opti.subject_to(X[:, 0] == x0_param)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == rk4(X[:, k], U[:, k]))

        # Auxiliary alpha dynamics (for adaptive c_hocbf / combo)
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            for k in range(N - 1):
                for i in range(n_obs):
                    opti.subject_to(
                        alpha1_bar[i, k + 1] == alpha1_bar[i, k] + dt * virtual_ctrl_alpha1[i, k]
                    )

        # Control bounds
        for j in range(4):
            opti.subject_to(-max_force <= U[j, :])
            opti.subject_to(U[j, :] <= max_force)

        # --- Objective ---
        obj = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            if mode == 'waypoint':
                # Position tracking + heading alignment to the desired heading
                obj += Qpos * ((xk[0] - goal_param[0]) ** 2 + (xk[1] - goal_param[1]) ** 2)
                obj += Qpsi * _heading_error(xk[2], goal_param[2])
            else:
                # Cross-track error + heading alignment to the path tangent
                ye, heading_err = _cross_track(xk[6], xk[0], xk[1], xk[2])
                obj += Qpos * ye ** 2
                obj += Qpsi * heading_err
            # Surge speed tracking
            obj += Qu * (xk[3] - uref_param) ** 2
            # Yaw rate damping
            obj += Qr * xk[5] ** 2
            # Control effort
            obj += uk[0] ** 2 + uk[1] ** 2 + uk[2] ** 2 + uk[3] ** 2

            if adaptive:
                if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                    for i in range(n_obs):
                        obj += 1e4 * (alpha1_bar[i, k] - alpha1_param[i]) ** 2
                        obj += 1e4 * (alpha2_bar[i, k] - alpha2_param[i]) ** 2
                elif cbf_type == 'dcbf':
                    for i in range(n_obs):
                        obj += 1e4 * (alpha_bar[i, k] - alpha_param[i]) ** 2

        # Terminal cost
        xf = X[:, N]
        if mode == 'waypoint':
            obj += 1000.0 * ((xf[0] - goal_param[0]) ** 2 + (xf[1] - goal_param[1]) ** 2)
            obj += 1000.0 * _heading_error(xf[2], goal_param[2])
        else:
            ye_f, heading_err_f = _cross_track(xf[6], xf[0], xf[1], xf[2])
            obj += 1000.0 * ye_f ** 2
            obj += 1000.0 * heading_err_f
        obj += 1000.0 * xf[5] ** 2  # terminal yaw rate damping
        obj += 1000.0 * (xf[3] - uref_param) ** 2  # terminal surge tracking

        if adaptive:
            if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                for i in range(n_obs):
                    obj += 1e4 * (alpha1_bar[i, N] - alpha1_param[i]) ** 2
                    obj += 1e4 * (alpha2_bar[i, N] - alpha2_param[i]) ** 2
            elif cbf_type == 'dcbf':
                for i in range(n_obs):
                    obj += 1e4 * (alpha_bar[i, N] - alpha_param[i]) ** 2

        opti.minimize(obj)

        # --- CBF derivatives helper ---
        def _cbf_derivs(xk, uk, ox, oy):
            """Compute h, ḣ, ḧ for a circular obstacle at (ox, oy)."""
            ned_x, ned_y, psi = xk[0], xk[1], xk[2]
            surge, sway, yr = xk[3], xk[4], xk[5]
            u1, u2, u3, u4 = uk[0], uk[1], uk[2], uk[3]

            h = (ned_x - ox) ** 2 + (ned_y - oy) ** 2 - r_safe ** 2

            px_dot = surge * ca.cos(psi) - sway * ca.sin(psi)
            py_dot = surge * ca.sin(psi) + sway * ca.cos(psi)
            h_dot = 2 * (ned_x - ox) * px_dot + 2 * (ned_y - oy) * py_dot

            surge_dot = -d11 / m11 * surge + (u1 + u2) / m11
            sway_dot  = -d22 / m22 * sway  + (u3 + u4) / m22
            px_ddot = (surge_dot * ca.cos(psi) - surge * yr * ca.sin(psi)
                       - sway_dot * ca.sin(psi) - sway * yr * ca.cos(psi))
            py_ddot = (surge_dot * ca.sin(psi) + surge * yr * ca.cos(psi)
                       + sway_dot * ca.cos(psi) - sway * yr * ca.sin(psi))
            h_ddot = (2 * (px_dot ** 2 + py_dot ** 2)
                      + 2 * (ned_x - ox) * px_ddot
                      + 2 * (ned_y - oy) * py_ddot)

            return h, h_dot, h_ddot

        # --- Obstacle constraints ---
        for i in range(n_obs):
            ox = obs_x_param[i]
            oy = obs_y_param[i]

            if cbf_type is None:
                # Direct distance constraint at every node except k=0 (fixed initial state)
                for k in range(1, N + 1):
                    h = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2 - r_safe ** 2
                    opti.subject_to(h >= 0)

            elif cbf_type == 'c_hocbf':
                for k in range(N):
                    h, h_dot, h_ddot = _cbf_derivs(X[:, k], U[:, k], ox, oy)
                    if not adaptive:
                        a1, a2 = alpha1_param[i], alpha2_param[i]
                        hocbf = h_ddot + (a1 + a2) * h_dot + a1 * a2 * h
                    else:
                        a1, a2 = alpha1_bar[i, k], alpha2_bar[i, k]
                        opti.subject_to(alpha1_bar[i, k] > 0)
                        opti.subject_to(alpha2_bar[i, k] > 0)
                        opti.subject_to(virtual_ctrl_alpha1[i, k] > -1000)
                        opti.subject_to(virtual_ctrl_alpha1[i, k] < 1000)
                        hocbf = h_ddot + (a1 + a2) * h_dot + (a1 * a2 + virtual_ctrl_alpha1[i, k]) * h
                    opti.subject_to(hocbf >= 0)
                # Terminal safety
                h_N = (X[0, N] - ox) ** 2 + (X[1, N] - oy) ** 2 - r_safe ** 2
                opti.subject_to(h_N >= 0)

            elif cbf_type == 'dcbf':
                for k in range(N):
                    h_k    = (X[0, k]     - ox) ** 2 + (X[1, k]     - oy) ** 2 - r_safe ** 2
                    h_next = (X[0, k + 1] - ox) ** 2 + (X[1, k + 1] - oy) ** 2 - r_safe ** 2
                    if not adaptive:
                        a = alpha_param[i]
                    else:
                        a = alpha_bar[i, k]
                        opti.subject_to(alpha_bar[i, k] > 0)
                        opti.subject_to(alpha_bar[i, k] < 1)
                    opti.subject_to(h_next - h_k + a * h_k >= 0)

            elif cbf_type == 'combo_chocbf_dcbf':
                for k in range(N):
                    if not adaptive:
                        a1, a2 = alpha1_param[i], alpha2_param[i]
                    else:
                        a1, a2 = alpha1_bar[i, k], alpha2_bar[i, k]
                        opti.subject_to(alpha1_bar[i, k] > 0)
                        opti.subject_to(alpha2_bar[i, k] > 0)
                        opti.subject_to(virtual_ctrl_alpha1[i, k] > -1000)
                        opti.subject_to(virtual_ctrl_alpha1[i, k] < 1000)

                    h_k    = (X[0, k]     - ox) ** 2 + (X[1, k]     - oy) ** 2 - r_safe ** 2
                    h_next = (X[0, k + 1] - ox) ** 2 + (X[1, k + 1] - oy) ** 2 - r_safe ** 2
                    # dcbf: h(k+1) >= exp(-a1*dt) * h(k)
                    opti.subject_to(h_next - ca_exp(-a1 * dt) * h_k >= 0)

                    # c_hocbf
                    h, h_dot, h_ddot = _cbf_derivs(X[:, k], U[:, k], ox, oy)
                    if adaptive:
                        hocbf = h_ddot + (a1 + a2) * h_dot + (a1 * a2 + virtual_ctrl_alpha1[i, k]) * h
                    else:
                        hocbf = h_ddot + (a1 + a2) * h_dot + a1 * a2 * h
                    opti.subject_to(hocbf >= 0)

        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False})

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def cbf_h(self, state, obs):
        """h(x) = (ned_x - ox)^2 + (ned_y - oy)^2 - r_safe^2"""
        r_safe = self.vessel_radius + self.obstacle_radius
        return (state[0] - obs[0]) ** 2 + (state[1] - obs[1]) ** 2 - r_safe ** 2
    
    def cbf_h_dot(self, state, obs):
        """ḣ(x) = 2(x-ox)vx + 2(y-oy)vy"""
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        ox, oy = obs[0], obs[1]
        return 2*(x - ox)*vx + 2*(y - oy)*vy

    def cbf_psi1(self, state, obs, alpha1):
        """ψ₁(x) = ḣ + α₁·h  (first HOCBF intermediate condition, must be ≥ 0)"""
        result = self.cbf_h_dot(state, obs) + alpha1 * self.cbf_h(state, obs)

        if result < 0:
            print(f"Warning: ψ₁ is negative ({result}) for state {state} and obs {obs} with alpha1 {alpha1}. This may indicate a violation of the HOCBF condition.")
        
        return result

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, initial_state, waypoint=None, obstacles=(), u_ref=0.0,
              alpha1=None, alpha2=None,
              return_full_solution=False, return_alpha_bars=False):
        """Run one MPC step.

        Parameters
        ----------
        initial_state : array-like
            (6,) for mode='waypoint'; (7,) [..., s] for mode='path'.
        waypoint   : array-like (3,)           current target (ned_x, ned_y, psi_desired); mode='waypoint' only
        obstacles  : list of (ox, oy)         current obstacle positions
        u_ref      : float                    desired surge speed
        alpha1 : float or array (n_obs,)
            For c_hocbf / combo: alpha1 (nominal). For dcbf: used as alpha.
        alpha2 : float or array (n_obs,), optional
            For c_hocbf / combo: alpha2 (nominal). Unused for dcbf.
        return_full_solution : bool           return full trajectory arrays
        return_alpha_bars : bool              return solved alpha bars (adaptive only)

        Returns
        -------
        (u1, u2, u3, u4) first control step, or full solution tuple if requested.
        """
        if self.mode == 'waypoint' and waypoint is None:
            raise ValueError("waypoint is required for mode='waypoint'")

        self.opti.set_value(self.x0_param, np.asarray(initial_state, dtype=float))
        if self.mode == 'waypoint':
            self.opti.set_value(self.goal_param, np.asarray(waypoint[:3], dtype=float))
        self.opti.set_value(self.obs_x_param, np.array([o[0] for o in obstacles], dtype=float))
        self.opti.set_value(self.obs_y_param, np.array([o[1] for o in obstacles], dtype=float))
        self.opti.set_value(self.uref_param, float(u_ref))

        if self.cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            self.opti.set_value(self.alpha1_param, np.atleast_1d(alpha1).astype(float))
            self.opti.set_value(self.alpha2_param, np.atleast_1d(alpha2).astype(float))
        elif self.cbf_type == 'dcbf':
            self.opti.set_value(self.alpha_param, np.atleast_1d(alpha1).astype(float))

        sol = self.opti.solve()

        X_sol = sol.value(self.X)   # (n_states, N+1)
        U_sol = sol.value(self.U)   # (4, N)

        # Cache for external inspection
        self._last_X_sol = X_sol
        self._last_U_sol = U_sol

        # Warm-start next solve
        self.opti.set_initial(self.X, X_sol)
        self.opti.set_initial(self.U, U_sol)

        if self.adaptive:
            if self.cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                a1_sol = sol.value(self.alpha1_bar)
                a2_sol = sol.value(self.alpha2_bar)
                vc_sol = sol.value(self.virtual_ctrl_alpha1)
                self.opti.set_initial(self.alpha1_bar, a1_sol)
                self.opti.set_initial(self.alpha2_bar, a2_sol)
                self.opti.set_initial(self.virtual_ctrl_alpha1, vc_sol)
            elif self.cbf_type == 'dcbf':
                a_sol = sol.value(self.alpha_bar)
                self.opti.set_initial(self.alpha_bar, a_sol)

        if return_full_solution:
            t = np.linspace(0, self.T, self.N + 1)
            if self.adaptive:
                if self.cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                    return sol, t, X_sol, U_sol, a1_sol, a2_sol
                elif self.cbf_type == 'dcbf':
                    return sol, t, X_sol, U_sol, a_sol
            return sol, t, X_sol, U_sol

        if return_alpha_bars and self.adaptive:
            if self.cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                return tuple(U_sol[:, 0]), a1_sol, a2_sol
            elif self.cbf_type == 'dcbf':
                return tuple(U_sol[:, 0]), a_sol

        return float(U_sol[0, 0]), float(U_sol[1, 0]), float(U_sol[2, 0]), float(U_sol[3, 0])

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------

    def visualize(self, X_sol, waypoints=None, obstacles=None,
                  xlim=None, ylim=None, name_save=None, s_range=None):
        """Quick trajectory plot.

        Parameters
        ----------
        X_sol : (n_states, N+1) array
        waypoints : list of (x, y) waypoint coordinates — mode='waypoint' only
        obstacles : list of (ox, oy) — uses obstacle_radius from define_problem
        s_range : (s_min, s_max) — mode='path' only; draws the reference path
            over this arc-length range using self.path from define_problem
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        x_traj = X_sol[0, :]
        y_traj = X_sol[1, :]
        ax.plot(x_traj, y_traj, 'b-', label='trajectory')

        if self.mode == 'path' and s_range is not None:
            s_vals = np.linspace(s_range[0], s_range[1], 200)
            x_path = self.path.x(s_vals)
            y_path = self.path.y(s_vals)
            ax.plot(x_path, y_path, 'r--', label='reference path')

        for j, wp in enumerate(waypoints or []):
            ax.scatter(wp[0], wp[1], marker='*', s=120, zorder=5,
                       label='waypoint' if j == 0 else None)

        if obstacles is not None:
            r = self.obstacle_radius
            for ox, oy in obstacles:
                circle = plt.Circle((ox, oy), r, color='gray', alpha=0.4)
                ax.add_artist(circle)
                safe_ring = plt.Circle((ox, oy), r + self.vessel_radius,
                                       color='gray', alpha=0.15, linestyle='--', fill=False)
                ax.add_artist(safe_ring)

        ax.set_aspect('equal')
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel('NED x [m]')
        ax.set_ylabel('NED y [m]')
        ax.legend()
        plt.tight_layout()

        if name_save is not None:
            import os
            fig.savefig(name_save)

        plt.show()
        return fig, ax
