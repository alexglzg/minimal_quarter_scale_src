from rockit import *
from casadi import *
import numpy as np


class MPCController:
    def __init__(self, parameters_model, 
                    parameters_mpc, path, cbf = False):
        self.Nhor = parameters_mpc["Nhor"]
        self.dt = parameters_mpc["dt"]
        self.Tf = parameters_mpc["Tf"]
        self.nx = parameters_mpc["nx"]
        self.nu = parameters_mpc["nu"]

        self.parameters_model = parameters_model

        self.num_obs = parameters_mpc["num_obstacles"]

        if self.Tf != self.Nhor * self.dt:
            raise ValueError("Tf must be equal to N * dt")
        
        # Define OCP
        ocp = Ocp(T=self.Tf)

        # Define states
        nedx = ocp.state()
        nedy = ocp.state()
        psi = ocp.state()
        u = ocp.state()
        v = ocp.state()
        r = ocp.state()
        s = ocp.state()

        # Defince controls
        u1 = ocp.control()
        u2 = ocp.control()
        u3 = ocp.control()
        u4 = ocp.control()

        # Define parameter
        X_0 = ocp.parameter(self.nx)
        obstacle_x = ocp.parameter(self.num_obs)
        obstacle_y = ocp.parameter(self.num_obs)
        obstacle_radius = ocp.parameter(self.num_obs)

        alpha1 = ocp.parameter(self.num_obs)
        alpha2 = ocp.parameter(self.num_obs)

        self.path = path

        # ODEs
        m11 = self.parameters_model["m11"]
        m22 = self.parameters_model["m22"]
        m33 = self.parameters_model["m33"]
        d11 = self.parameters_model["d11"]
        d22 = self.parameters_model["d22"]
        d33 = self.parameters_model["d33"]
        aa = self.parameters_model["aa"]
        bb = self.parameters_model["bb"]
        u_ref = self.parameters_model["u_ref"]
        max_force_limit = self.parameters_model["max_force_limit"]

        ocp.set_der(nedx, (u*cos(psi) - v*sin(psi)))
        ocp.set_der(nedy, (u*sin(psi) + v*cos(psi)))
        ocp.set_der(psi, r)
        ocp.set_der(u, (-d11/m11*u+u1/(m11)+u2/(m11)))
        ocp.set_der(v, (-d22/m22*v+u3/(m22)+u4/(m22)))
        ocp.set_der(r, (-d33/m33*r+aa/(2*(m33))*u1-aa/(2*(m33))*u2+bb/(2*(m33))*u3-bb/(2*(m33))*u4))
        ocp.set_der(s, u)

        # Lagrange objective
        Qye = parameters_mpc["Qye"]
        Qr = parameters_mpc["Qr"]
        Qpsi = parameters_mpc["Qpsi"]
        Qu = parameters_mpc["Qu"]

        x_d = self.path.x(s)
        y_d = self.path.y(s)
        x_dot_d = self.path.x_dot(s)
        y_dot_d = self.path.y_dot(s)
        gamma_p = atan2(y_dot_d, x_dot_d)
        ye = -(nedx-x_d)*sin(gamma_p)+(nedy-y_d)*cos(gamma_p) # Cross-track error = projection of the position error onto the direction perpendicular to the path. Positive cross-track error = vehicle is to the left of the pat

        ocp.add_objective(ocp.sum(Qye*(ye**2) + Qr*(r**2) + Qpsi*(sin(psi)-sin(gamma_p))**2 + Qpsi*(cos(psi)-cos(gamma_p))**2 + Qu*(u-u_ref)**2 + u1**2 + u2**2 + u3**2 + u4**2))
        ocp.add_objective(ocp.at_tf(Qye*(ye**2) + Qr*(r**2) + Qpsi*(sin(psi)-sin(gamma_p))**2 + Qpsi*(cos(psi)-cos(gamma_p))**2 + Qu*(u-u_ref)**2))


        # Path constraints
        ocp.subject_to( (-max_force_limit <= u1) <= max_force_limit )
        ocp.subject_to( (-max_force_limit <= u2) <= max_force_limit )
        ocp.subject_to( (-max_force_limit <= u3) <= max_force_limit )
        ocp.subject_to( (-max_force_limit <= u4) <= max_force_limit )

        # Initial state constraint
        X = vertcat(nedx,nedy,psi,u,v,r,s)
        U = vertcat(u1,u2,u3,u4)
        ocp.subject_to(ocp.at_t0(X)==X_0)

        # # Useful for CBF constraints
        x_dot = (u*cos(psi) - v*sin(psi))
        y_dot = (u*sin(psi) + v*cos(psi))
        u_dot = (-d11/m11*u+u1/(m11)+u2/(m11))
        v_dot = (-d22/m22*v+u3/(m22)+u4/(m22))
        x_ddot = u_dot*cos(psi) - u*r*sin(psi) - v_dot*sin(psi) - v*r*cos(psi)
        y_ddot = u_dot*sin(psi) + u*r*cos(psi) + v_dot*cos(psi) - v*r*sin(psi)

        length_ego = 0.9
        width_ego = 0.45
        radius_ego = np.hypot(length_ego, width_ego)/2

        # CBF constraints for collision avoidance with obstacles
        for i in range(self.num_obs):
            x_diff = nedx - obstacle_x[i]
            y_diff = nedy - obstacle_y[i]
            bsafe = x_diff*x_diff + y_diff*y_diff - (obstacle_radius[i] + radius_ego)**2
            bsafe_dot = 2*(x_diff*x_dot + y_diff*y_dot)
            bsafe_ddot = 2*(x_dot*x_dot + x_diff*x_ddot + y_dot*y_dot + y_diff*y_ddot)
            cbf = bsafe_ddot + (alpha1[i]+alpha2[i])*bsafe_dot + alpha1[i]*alpha2[i]*bsafe
            ocp.subject_to(cbf >= 0, include_last=False)

        # Pick a solution method
        # options = {"ipopt": {"print_level": 0}, "expand": True, "print_time": False}
        options = {
            "expand": True,
            "structure_detection": "auto",
            "print_time": False,
            "fatrop.print_level": 0,
            "error_on_fail": True,
        }
        # ocp.solver('ipopt',options)
        ocp.solver('fatrop', options)

        # Make it concrete for this ocp
        ocp.method(MultipleShooting(N=self.Nhor,M=1,intg='rk'))

        # Get discretisd dynamics as CasADi function
        # Sim_asv_dyn = ocp._method.discrete_system(ocp)

        # Set initial value to make sure you can make a function
        ocp.set_value(X_0, np.zeros(self.nx))
        ocp.set_value(obstacle_x,  np.zeros(self.num_obs))
        ocp.set_value(obstacle_y, np.zeros(self.num_obs))
        ocp.set_value(obstacle_radius, np.zeros(self.num_obs))
        ocp.set_value(alpha1, np.zeros(self.num_obs))
        ocp.set_value(alpha2, np.zeros(self.num_obs))

        # Make a function

        # self.ocp_func = ocp.to_function('ocp_func', 
        #                         [ocp.value(X_0)], 
        #                         [ ocp.sample(u1,grid='control')[1], 
        #                         ocp.sample(u2,grid='control')[1], 
        #                         ocp.sample(u3,grid='control')[1], 
        #                         ocp.sample(u4,grid='control')[1] ],
        #                         ['X_0'], ['u1', 'u2', 'u3', 'u4'])
        

        self.ocp_func = ocp.to_function('ocp_func', 
                                [ocp.value(X_0), ocp.value(obstacle_x), 
                                ocp.value(obstacle_y), ocp.value(obstacle_radius), 
                                ocp.value(alpha1), ocp.value(alpha2), 
                                ocp.sample(X, grid='control')[1], 
                                ocp.sample(U, grid='control-')[1]], 
                                [ocp.sample(U, grid='control-')[1], ocp.sample(X, grid='control')[1]],
                                ['X_0', 'obstacle_x', 'obstacle_y', 'obstacle_radius', 'alpha1', 'alpha2', 'initial_guess_state', 'initial_guess_control'], 
                                ['U', 'X'])


    def solve(self, current_state, obstacle_x, obstacle_y, obstacle_radius, alpha1, alpha2, initial_guess_state, initial_guess_control):
        U, X = self.ocp_func(current_state, obstacle_x, obstacle_y, obstacle_radius, alpha1, alpha2, initial_guess_state, initial_guess_control)
        # print("MPC control outputs:", f1, f2, f3, f4)
        u = np.array([U[0][0], U[1][0], U[2][0], U[3][0]])
        return u, U, X