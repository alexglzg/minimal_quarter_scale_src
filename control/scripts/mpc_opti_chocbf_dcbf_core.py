import casadi as ca
import numpy as np


class MPCController:
    """ASV path-following MPC with obstacle avoidance, built with CasADi Opti
    (RK4 multiple shooting) instead of rockit. Same constructor/solve() interface as
    mpc_rockit_core.MPCController, so it is a drop-in replacement for mpc_node.py.

    State X (7):   [nedx, nedy, psi, u, v, r, s]   (s = path arc-length parameter)
    Control U (4): [u1, u2, u3, u4]                (thruster forces)

    Obstacle constraint formulation is selected via parameters_mpc["cbf_type"]
    (defaults to "c_hocbf", matching mpc_rockit_core), one of VALID_CBF_TYPES,
    mirroring the formulations in vessel.py:
        "c_hocbf"            high-order CBF using alpha1, alpha2
        "dcbf"               discrete-time CBF decrease constraint using alpha1
        "combo_chocbf_dcbf"  both dcbf and c_hocbf constraints, using alpha1, alpha2
        "none" / None        plain distance constraint, no alphas
    alpha1/alpha2 are always supplied by the caller (per mpc_rockit_core's solve()
    signature); types that need fewer alphas simply ignore the unused one.

    parameters_mpc["adaptive"] (default False) mirrors vessel.py's adaptive mode:
    the alpha(s) become per-obstacle, per-stage decision variables (alpha1_bar /
    alpha2_bar, or alpha_bar for "dcbf") penalised toward the nominal alpha1/alpha2
    passed into solve(), instead of being used directly as fixed gains. This is
    internal to the compiled NLP — solve()'s signature is unchanged.
    """

    VALID_CBF_TYPES = ('c_hocbf', 'dcbf', 'combo_chocbf_dcbf', None)

    def __init__(self, parameters_model, parameters_mpc, path):
        self.Nhor = parameters_mpc["Nhor"]
        self.dt = parameters_mpc["dt"]
        self.Tf = parameters_mpc["Tf"]
        self.nx = parameters_mpc["nx"]
        self.nu = parameters_mpc["nu"]

        self.parameters_model = parameters_model
        self.num_obs = parameters_mpc["num_obstacles"]
        self.path = path

        if self.Tf != self.Nhor * self.dt:
            raise ValueError("Tf must be equal to N * dt")

        cbf_type = parameters_mpc.get("cbf_type", "c_hocbf")
        if isinstance(cbf_type, str) and cbf_type.lower() == "none":
            cbf_type = None
        if cbf_type not in self.VALID_CBF_TYPES:
            raise ValueError(f"cbf_type must be one of {self.VALID_CBF_TYPES}, got {cbf_type!r}")
        self.cbf_type = cbf_type

        adaptive = bool(parameters_mpc.get("adaptive", False))
        self.adaptive = adaptive

        N = self.Nhor
        dt = self.dt
        n_obs = self.num_obs

        m11, m22, m33 = parameters_model["m11"], parameters_model["m22"], parameters_model["m33"]
        d11, d22, d33 = parameters_model["d11"], parameters_model["d22"], parameters_model["d33"]
        aa, bb = parameters_model["aa"], parameters_model["bb"]
        u_ref = parameters_model["u_ref"]
        max_force_limit = parameters_model["max_force_limit"]

        ego_length = parameters_model.get("ego_length", 0.9)
        ego_width = parameters_model.get("ego_width", 0.45)
        radius_ego = float(np.hypot(ego_length, ego_width) / 2)

        Qye = parameters_mpc["Qye"]
        Qr = parameters_mpc["Qr"]
        Qpsi = parameters_mpc["Qpsi"]
        Qu = parameters_mpc["Qu"]

        opti = ca.Opti()
        self.opti = opti

        # Define variables for FATROP as (x0, u0, x1, u1, ..., xN, uN) per stage.
        X = []
        U = []
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            alpha1_bar = []
            alpha2_bar = []
            virtual_ctrl_alpha1 = []
        elif adaptive and cbf_type == 'dcbf':
            alpha_bar = []
        for k in range(N):
            # State-like (recursively-defined) variables for stage k must be declared
            # contiguously, followed by the control-like (free) variables for stage k
            # -- FATROP's structure_detection="auto" partitions each stage into an
            # x-block then a u-block strictly by declaration order, so alpha1_bar
            # (which has its own recursion below, like X) has to sit next to X, not
            # after U alongside virtual_ctrl_alpha1/alpha2_bar (which are free inputs,
            # like U). Interleaving as X, U, alpha1_bar, ... silently breaks detection.
            Xk = opti.variable(self.nx)
            X.append(Xk)
            if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                alpha1_bar.append(opti.variable(n_obs))

            Uk = opti.variable(self.nu)
            U.append(Uk)
            if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                virtual_ctrl_alpha1.append(opti.variable(n_obs))
                alpha2_bar.append(opti.variable(n_obs))
            elif adaptive and cbf_type == 'dcbf':
                alpha_bar.append(opti.variable(n_obs))

        X.append(opti.variable(self.nx))  # terminal state
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            alpha1_bar.append(opti.variable(n_obs))  # terminal alpha1_bar
            alpha2_bar.append(opti.variable(n_obs))  # terminal alpha2_bar
        elif adaptive and cbf_type == 'dcbf':
            alpha_bar.append(opti.variable(n_obs))  # terminal alpha_bar

        X = ca.horzcat(*X)
        U = ca.horzcat(*U)
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            alpha1_bar = ca.horzcat(*alpha1_bar)
            alpha2_bar = ca.horzcat(*alpha2_bar)
            virtual_ctrl_alpha1 = ca.horzcat(*virtual_ctrl_alpha1)
        elif adaptive and cbf_type == 'dcbf':
            alpha_bar = ca.horzcat(*alpha_bar)

        self.X = X
        self.U = U
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            self.alpha1_bar = alpha1_bar
            self.alpha2_bar = alpha2_bar
            self.virtual_ctrl_alpha1 = virtual_ctrl_alpha1
        elif adaptive and cbf_type == 'dcbf':
            self.alpha_bar = alpha_bar

        X_0 = opti.parameter(self.nx)
        obstacle_x = opti.parameter(n_obs)
        obstacle_y = opti.parameter(n_obs)
        obstacle_radius = opti.parameter(n_obs)
        alpha1 = opti.parameter(n_obs)
        alpha2 = opti.parameter(n_obs)

        # --- Dynamics ---
        def f(x, u):
            psi, surge, sway, r = x[2], x[3], x[4], x[5]
            u1, u2, u3, u4 = u[0], u[1], u[2], u[3]
            return ca.vertcat(
                surge * ca.cos(psi) - sway * ca.sin(psi),
                surge * ca.sin(psi) + sway * ca.cos(psi),
                r,
                -d11 / m11 * surge + u1 / m11 + u2 / m11,
                -d22 / m22 * sway + u3 / m22 + u4 / m22,
                -d33 / m33 * r + aa / (2 * m33) * u1 - aa / (2 * m33) * u2
                    + bb / (2 * m33) * u3 - bb / (2 * m33) * u4,
                surge,  # s_dot
            )

        def rk4(x, u):
            k1 = f(x, u)
            k2 = f(x + dt / 2 * k1, u)
            k3 = f(x + dt / 2 * k2, u)
            k4 = f(x + dt * k3, u)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # --- Path-tracking error: cross-track distance + heading-to-tangent ---
        def path_errors(s, nedx, nedy, psi):
            x_d = self.path.x(s)
            y_d = self.path.y(s)
            gamma_p = ca.atan2(self.path.y_dot(s), self.path.x_dot(s))
            ye = -(nedx - x_d) * ca.sin(gamma_p) + (nedy - y_d) * ca.cos(gamma_p)
            heading_err = (ca.sin(psi) - ca.sin(gamma_p)) ** 2 + (ca.cos(psi) - ca.cos(gamma_p)) ** 2
            return ye, heading_err

        # --- Obstacle constraint helpers (formulation selected by cbf_type) ---
        def h_plain(xk, ox, oy, r_safe):
            x_diff, y_diff = xk[0] - ox, xk[1] - oy
            return x_diff ** 2 + y_diff ** 2 - r_safe ** 2

        def hocbf_terms(xk, uk, ox, oy, r_safe):
            nedx, nedy, psi, surge, sway, r = xk[0], xk[1], xk[2], xk[3], xk[4], xk[5]
            u1, u2, u3, u4 = uk[0], uk[1], uk[2], uk[3]

            x_diff, y_diff = nedx - ox, nedy - oy
            bsafe = x_diff ** 2 + y_diff ** 2 - r_safe ** 2

            x_dot = surge * ca.cos(psi) - sway * ca.sin(psi)
            y_dot = surge * ca.sin(psi) + sway * ca.cos(psi)
            bsafe_dot = 2 * (x_diff * x_dot + y_diff * y_dot)

            surge_dot = -d11 / m11 * surge + u1 / m11 + u2 / m11
            sway_dot = -d22 / m22 * sway + u3 / m22 + u4 / m22
            x_ddot = (surge_dot * ca.cos(psi) - surge * r * ca.sin(psi)
                      - sway_dot * ca.sin(psi) - sway * r * ca.cos(psi))
            y_ddot = (surge_dot * ca.sin(psi) + surge * r * ca.cos(psi)
                      + sway_dot * ca.cos(psi) - sway * r * ca.sin(psi))
            bsafe_ddot = 2 * (x_dot ** 2 + x_diff * x_ddot + y_dot ** 2 + y_diff * y_ddot)

            return bsafe, bsafe_dot, bsafe_ddot

        # --- Stage-wise constraints ---
        # FATROP's structure_detection="auto" recovers the banded OCP structure by
        # scanning constraints in declaration order: every constraint touching stage
        # k must be added before moving on to stage k+1. So dynamics, control bounds
        # and obstacle-avoidance constraints are interleaved in one loop over k here,
        # instead of being grouped into separate per-type / per-obstacle loops. Also,
        # any "next-state" term in a path constraint must be the explicit dynamics
        # expression (xnext) rather than the X[:, k+1] variable itself -- referencing
        # that variable from within stage k's block (instead of only via the gap-
        # closing equality) breaks FATROP's structure detection, even though xnext
        # and X[:, k+1] are equal at the solution.
        for k in range(N):
            xnext = rk4(X[:, k], U[:, k])

            if k == 0:
                # Initial constraint
                opti.subject_to(X[:, 0] == X_0)

            # Multiple-shooting gap-closing constraint
            opti.subject_to(X[:, k + 1] == xnext)

            if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                alpha1_bar_next = alpha1_bar[:, k] + dt * virtual_ctrl_alpha1[:, k]
                opti.subject_to(
                    alpha1_bar[:, k + 1] == alpha1_bar_next
                )

            # Control bounds
            opti.subject_to(-max_force_limit <= (U[:, k] <= max_force_limit))

            # Obstacle avoidance for stage k
            for i in range(n_obs):
                ox, oy = obstacle_x[i], obstacle_y[i]
                r_safe = obstacle_radius[i] + radius_ego
                a1, a2 = alpha1[i], alpha2[i]

                if cbf_type is None:
                    # Plain distance constraint on the freshly-closed stage k+1
                    opti.subject_to(h_plain(xnext, ox, oy, r_safe) >= 0)

                elif cbf_type == 'c_hocbf':
                    bsafe, bsafe_dot, bsafe_ddot = hocbf_terms(X[:, k], U[:, k], ox, oy, r_safe)
                    if not adaptive:
                        hocbf = bsafe_ddot + (a1 + a2) * bsafe_dot + a1 * a2 * bsafe
                    else:
                        a1k, a2k = alpha1_bar[i, k], alpha2_bar[i, k]
                        opti.subject_to(alpha1_bar[i, k] > 0)
                        opti.subject_to(alpha2_bar[i, k] > 0)
                        opti.subject_to(-1000 <= (virtual_ctrl_alpha1[i, k] <= 1000))
                        hocbf = bsafe_ddot + (a1k + a2k) * bsafe_dot + (a1k * a2k + virtual_ctrl_alpha1[i, k]) * bsafe
                    opti.subject_to(hocbf >= 0)
                    if k == N - 1:
                        # Terminal node has no control to form bsafe_ddot -> plain distance fallback
                        opti.subject_to(h_plain(xnext, ox, oy, r_safe) >= 0)

                elif cbf_type == 'dcbf':
                    h_k = h_plain(X[:, k], ox, oy, r_safe)
                    h_next = h_plain(xnext, ox, oy, r_safe)
                    if not adaptive:
                        a = a1
                    else:
                        a = alpha_bar[i, k]
                        opti.subject_to(0 <= (alpha_bar[i, k] <= 1))
                    opti.subject_to(h_next - h_k + a * h_k >= 0)

                elif cbf_type == 'combo_chocbf_dcbf':
                    if not adaptive:
                        a1k, a2k = a1, a2
                    else:
                        a1k, a2k = alpha1_bar[i, k], alpha2_bar[i, k]
                        opti.subject_to(alpha1_bar[i, k] > 0)
                        opti.subject_to(alpha2_bar[i, k] > 0)
                        # opti.subject_to(-1000 <= (virtual_ctrl_alpha1[i, k] <= 1000))

                    h_k = h_plain(X[:, k], ox, oy, r_safe)
                    h_next = h_plain(xnext, ox, oy, r_safe)
                    opti.subject_to(h_next - ca.exp(-a1k * dt) * h_k >= 0)

                    bsafe, bsafe_dot, bsafe_ddot = hocbf_terms(X[:, k], U[:, k], ox, oy, r_safe)
                    if adaptive:
                        hocbf = bsafe_ddot + (a1k + a2k) * bsafe_dot + (a1k * a2k + virtual_ctrl_alpha1[i, k]) * bsafe
                    else:
                        hocbf = bsafe_ddot + (a1k + a2k) * bsafe_dot + a1k * a2k * bsafe
                    opti.subject_to(hocbf >= 0)

        # --- Objective ---
        obj = 0
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            ye, heading_err = path_errors(xk[6], xk[0], xk[1], xk[2])
            obj += Qye * ye ** 2 + Qr * xk[5] ** 2 + Qpsi * heading_err + Qu * (xk[3] - u_ref) ** 2
            obj += uk[0] ** 2 + uk[1] ** 2 + uk[2] ** 2 + uk[3] ** 2

            if adaptive:
                if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                    for i in range(n_obs):
                        obj += 1e4 * (alpha1_bar[i, k] - alpha1[i]) ** 2
                        obj += 1e4 * (alpha2_bar[i, k] - alpha2[i]) ** 2
                elif cbf_type == 'dcbf':
                    for i in range(n_obs):
                        obj += 1e4 * (alpha_bar[i, k] - alpha1[i]) ** 2

        xN = X[:, N]
        yeN, heading_errN = path_errors(xN[6], xN[0], xN[1], xN[2])
        obj += Qye * yeN ** 2 + Qr * xN[5] ** 2 + Qpsi * heading_errN + Qu * (xN[3] - u_ref) ** 2

        if adaptive:
            if cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
                for i in range(n_obs):
                    obj += 1e4 * (alpha1_bar[i, N] - alpha1[i]) ** 2
                    obj += 1e4 * (alpha2_bar[i, N] - alpha2[i]) ** 2
            elif cbf_type == 'dcbf':
                for i in range(n_obs):
                    obj += 1e4 * (alpha_bar[i, N] - alpha1[i]) ** 2

        opti.minimize(obj)

        # Codegen of helper functions: JIT-compiles the generated C code for the
        # NLP callbacks instead of interpreting them, at the cost of a one-time
        # compile on construction.
        jit_opts = {
            "jit": True,
            "jit_temp_suffix": False,
            "jit_options": {"flags": ["-O3"], "compiler": "gcc"},
        }

        if parameters_mpc.get("solver", "ipopt") == "ipopt":
            opti.solver('ipopt', {
                'ipopt.print_level': 0,
                'print_time': False,
                'expand': True,
                'error_on_fail': True,
                # **jit_opts,
            })
        elif parameters_mpc.get("solver", "ipopt") == "fatrop":
            # Solver options
            opts = {
                "fatrop.print_level": 0,
                "print_time": 0,
                "fatrop.max_iter": 100,
                "fatrop.tol": 1e-4,
                "fatrop.mu_init": 1e-1,
                "structure_detection": "auto",
                "expand": True,
                "debug": False,
                # **jit_opts,
                'error_on_fail': True
            }
            self.opti.solver('fatrop', opts)
        else:
            raise ValueError(f"Unknown solver {parameters_mpc.get('solver')}, must be 'ipopt' or 'fatrop'")

        # --- Compile to a CasADi Function for fast repeated MPC calls ---
        opti.set_value(X_0, np.zeros(self.nx))
        opti.set_value(obstacle_x, np.zeros(n_obs))
        opti.set_value(obstacle_y, np.zeros(n_obs))
        opti.set_value(obstacle_radius, np.zeros(n_obs))
        opti.set_value(alpha1, np.zeros(n_obs))
        opti.set_value(alpha2, np.zeros(n_obs))
        if adaptive and cbf_type in ('c_hocbf', 'combo_chocbf_dcbf'):
            opti.set_initial(alpha1_bar, np.zeros((n_obs, N + 1)))
            opti.set_initial(alpha2_bar, np.zeros((n_obs, N + 1)))
            opti.set_initial(virtual_ctrl_alpha1, np.zeros((n_obs, N)))
        elif adaptive and cbf_type == 'dcbf':
            opti.set_initial(alpha_bar, np.zeros((n_obs, N + 1)))

        self.ocp_func = opti.to_function(
            'ocp_func',
            [X_0, obstacle_x, obstacle_y, obstacle_radius, alpha1, alpha2, X, U],
            [U, X],
            ['X_0', 'obstacle_x', 'obstacle_y', 'obstacle_radius', 'alpha1', 'alpha2',
             'initial_guess_state', 'initial_guess_control'],
            ['U', 'X'],
        )

    def solve(self, current_state, obstacle_x, obstacle_y, obstacle_radius, alpha1, alpha2,
              initial_guess_state, initial_guess_control):
        U, X = self.ocp_func(current_state, obstacle_x, obstacle_y, obstacle_radius,
                              alpha1, alpha2, initial_guess_state, initial_guess_control)
        u = np.array(U[:, 0]).flatten()
        return u, U, X
