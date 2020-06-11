import warnings
from math import isnan

import numpy as np
from scipy import integrate

from ross.fluid_flow.fluid_flow_geometry import move_rotor_center


def calculate_oil_film_force(fluid_flow_object, force_type=None):
    """This function calculates the forces of the oil film in the N and T directions, ie in the
    opposite direction to the eccentricity and in the tangential direction.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    force_type: str
        If set, calculates the oil film force matrix analytically considering the chosen type: 'short' or 'long'.
        If set to 'numerical', calculates the oil film force numerically.
    Returns
    -------
    radial_force: float
        Force of the oil film in the opposite direction to the eccentricity direction.
    tangential_force: float
        Force of the oil film in the tangential direction
    f_x: float
        Components of forces in the x direction
    f_y: float
        Components of forces in the y direction
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_oil_film_force(my_fluid_flow) # doctest: +ELLIPSIS
    (...
    """
    if force_type != "numerical" and (
        force_type == "short" or fluid_flow_object.bearing_type == "short_bearing"
    ):
        radial_force = (
            0.5
            * fluid_flow_object.viscosity
            * (
                fluid_flow_object.radius_rotor
                / fluid_flow_object.difference_between_radius
            )
            ** 2
            * (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor)
            * (
                (
                    2
                    * fluid_flow_object.eccentricity_ratio ** 2
                    * fluid_flow_object.omega
                )
                / (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
            )
        )

        tangential_force = (
            0.5
            * fluid_flow_object.viscosity
            * (
                fluid_flow_object.radius_rotor
                / fluid_flow_object.difference_between_radius
            )
            ** 2
            * (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor)
            * (
                (np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega)
                / (2 * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** (3.0 / 2))
            )
        )
    elif force_type != "numerical" and (
        force_type == "long" or fluid_flow_object.bearing_type == "long_bearing"
    ):
        radial_force = (
            6
            * fluid_flow_object.viscosity
            * (
                fluid_flow_object.radius_rotor
                / fluid_flow_object.difference_between_radius
            )
            ** 2
            * fluid_flow_object.radius_rotor
            * fluid_flow_object.length
            * (
                (
                    2
                    * fluid_flow_object.eccentricity_ratio ** 2
                    * fluid_flow_object.omega
                )
                / (
                    (2 + fluid_flow_object.eccentricity_ratio ** 2)
                    * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                )
            )
        )
        tangential_force = (
            6
            * fluid_flow_object.viscosity
            * (
                fluid_flow_object.radius_rotor
                / fluid_flow_object.difference_between_radius
            )
            ** 2
            * fluid_flow_object.radius_rotor
            * fluid_flow_object.length
            * (
                (np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega)
                / (
                    (2 + fluid_flow_object.eccentricity_ratio ** 2)
                    * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 0.5
                )
            )
        )
    else:
        p_mat = fluid_flow_object.p_mat_numerical
        a = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        b = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        g1 = np.zeros(fluid_flow_object.nz)
        g2 = np.zeros(fluid_flow_object.nz)
        base_vector = np.array(
            [
                fluid_flow_object.xre[0][0] - fluid_flow_object.xi,
                fluid_flow_object.yre[0][0] - fluid_flow_object.yi,
            ]
        )
        for i in range(fluid_flow_object.nz):
            for j in range(int(fluid_flow_object.ntheta / 2)):
                vector_from_rotor = np.array(
                    [
                        fluid_flow_object.xre[i][j] - fluid_flow_object.xi,
                        fluid_flow_object.yre[i][j] - fluid_flow_object.yi,
                    ]
                )
                angle_between_vectors = np.arccos(
                    np.dot(base_vector, vector_from_rotor)
                    / (np.linalg.norm(base_vector) * np.linalg.norm(vector_from_rotor))
                )
                if isnan(angle_between_vectors):
                    angle_between_vectors = 0
                if angle_between_vectors != 0 and j * fluid_flow_object.dtheta > np.pi:
                    angle_between_vectors += np.pi
                a[i][j] = p_mat[i][j] * np.cos(angle_between_vectors)
                b[i][j] = p_mat[i][j] * np.sin(angle_between_vectors)

        for i in range(fluid_flow_object.nz):
            g1[i] = integrate.simps(a[i][:], fluid_flow_object.gama[0])
            g2[i] = integrate.simps(b[i][:], fluid_flow_object.gama[0])

        integral1 = integrate.simps(g1, fluid_flow_object.z_list)
        integral2 = integrate.simps(g2, fluid_flow_object.z_list)

        radial_force = -fluid_flow_object.radius_rotor * integral1
        tangential_force = fluid_flow_object.radius_rotor * integral2
    force_x = -radial_force * np.sin(
        fluid_flow_object.attitude_angle
    ) + tangential_force * np.cos(fluid_flow_object.attitude_angle)
    force_y = radial_force * np.cos(
        fluid_flow_object.attitude_angle
    ) + tangential_force * np.sin(fluid_flow_object.attitude_angle)
    return radial_force, tangential_force, force_x, force_y


def calculate_coefficients_matrix(fluid_flow_object):
    N = 6  # Number of time steps
    t = np.linspace(0, 2* np.pi / fluid_flow_object.omegap, N)  # Time vector for 1 period
    fluid_flow_object.xp = fluid_flow_object.difference_between_radius * 0.0005  # Perturbation along x
    fluid_flow_object.yp = fluid_flow_object.difference_between_radius * 0.0005  # Perturbation along y
    xi0 = fluid_flow_object.xi  # Eq. pos. along x
    yi0 = fluid_flow_object.yi  # Eq. pos. along y
    dx = np.zeros(N) # Displ. vetor from eq. pos. along x
    dy = np.zeros(N) # Displ. vetor from eq. pos. along x
    xdot = np.zeros(N)  # Vector with displ. perturbations along x
    xddot = np.zeros(N)  # Vector with accel. perturbations along x
    ydot = np.zeros(N)  # Vector with vel. perturbations along x
    yddot = np.zeros(N)  # Vector with accel. perturbations along x
    radial_force = np.zeros(N)
    tangential_force = np.zeros(N)
    force_xx = np.zeros(N) # Force along x for a perturbation along x
    force_yx = np.zeros(N) # Force along y for a perturbation along x
    force_xy = np.zeros(N) # Force along x for a perturbation along y
    force_yy = np.zeros(N) # Force along y for a perturbation along y
    X = np.zeros([N,3]) # Displ. and vel. vector
    X2 = np.zeros([N, 3])  # Displ. and vel. vector
    F = np.zeros(N) # Forces vector
    F2 = np.zeros(N)  # Forces vector

    # Compute the coefficients of the continuity equation for eq. position
    fluid_flow_object.calculate_coefficients()
    p_mat = fluid_flow_object.calculate_pressure_matrix_numerical()
    [radial_force_eq, tangential_force_eq, feqx, feqy] = \
        calculate_oil_film_force(fluid_flow_object, force_type='numerical')
    print("feqx: ",feqx)
    print("feqy: ", feqy)
    # This loop computes the hor. and vert. forces for perturbations along x and y dir with a freq. omegap
    # The time vector t ranges from 0-T (one period), with N timesteps
    # For each t, 2 set of equations are obtained:
    #   One for a perturbation along x:
    #       Kxx*x(t) + Cxx*xdot(t) = Fxx(t) - Feqx
    #       Kxy*x(t) + Cxy*xdot(t) = Fxy(t) - Feqy
    #   One for a perturbation along y
    #       Kyx*y(t) + Cyx*ydot(t) = Fyx(t) - Feqx
    #       Kyy*y(t) + Cyy*ydot(t) = Fyy(t) - Feqy
    # Thus, for all the timesteps, the following equations are obtained for the x direction:
    #       Kxx*x(t[0]) + Cxx*xdot(t[0]) = Fxx(t[0]) - Feqx
    #       Kxy*x(t[0]) + Cxy*xdot(t[0]) = Fxy(t[0]) - Feqy
    #       Kyx*y(t[0]) + Cyx*ydot(t[0]) = Fyx(t[0]) - Feqx
    #       Kyy*y(t[0]) + Cyy*ydot(t[0]) = Fyy(t[0]) - Feqy
    #       Kxx*x(t[1]) + Cxx*xdot(t[1]) = Fxx(t[1]) - Feqx
    #       Kxy*x(t[1]) + Cxy*xdot(t[1]) = Fxy(t[1]) - Feqy
    #       Kyx*y(t[1]) + Cyx*ydot(t[1]) = Fyx(t[1]) - Feqx
    #       Kyy*y(t[1]) + Cyy*ydot(t[1]) = Fyy(t[1]) - Feqy
    #           ...     +       ...      =      ...
    #       Kxx*x(t[N-1]) + Cxx*xdot(t[N-1]) = Fxx(t[N-1]) - Feqx
    #       Kxy*x(t[N-1]) + Cxy*xdot(t[N-1]) = Fxy(t[N-1]) - Feqy
    #       Kyx*y(t[N-1]) + Cyx*ydot(t[N-1]) = Fyx(t[N-1]) - Feqx
    #       Kyy*y(t[N-1]) + Cyy*ydot(t[N-1]) = Fyy(t[N-1]) - Feqy
    # The equations can be expressed in matrix-form as:
    #       X*P = F
    #  where
    # X is 4Nx8: [[x(t[i]),xdot(t[i]),  0    ,   0      ,   0   ,   0      ,   0   ,    0      ]
    #             [   0   ,   0      ,x(t[i]),xdot(t[i]),   0   ,   0      ,   0   ,    0      ]
    #             [   0   ,   0      ,   0   ,    0     ,y(t[i]),ydot(t[i]),   0   ,    0      ]
    #             [   0   ,   0      ,   0   ,    0     ,   0   ,   0      ,y(t[i]), ydot(t[i])] ]
    # P is 8x1: [Kxx,Cxx,Kxy,Cxy,Kyx,Cyx,Kyy,Cyy]
    # P is 8x1: [Kxx,Cxx,Mxx,Kxy,Cxy,Mxy,Kyx,Cyx,Myx,Kyy,Cyy,Myy]
    # F is 4Nx1: [Fxx(t[i]) - Feqx,Fxy(t[i]) - Feqx,Fyx(t[i]) - Feqx,Fyy(t[i]) - Feqx]
    # Thus, the parameter vector P can be obtain by using the pseudo-inverse of X ('cause it cannot be inverted):
    # P = (x^T*X)^(-1)*(x^T)*F
    for i in range(N):
        print("Progress: {:2.1%}".format(i / N), end="\r")
        fluid_flow_object.t = t[i]

        # X DIRECTION
        # Compute the variation of the displ. along x for t[i]
        delta_x = fluid_flow_object.xp * np.sin(fluid_flow_object.omegap * fluid_flow_object.t)
        # Compute parameters (xi,yi,ecc,eps,beta) for the new state
        fluid_flow_object.xi = xi0 + delta_x
        fluid_flow_object.yi = yi0
        fluid_flow_object.eccentricity = np.sqrt(fluid_flow_object.xi ** 2 + fluid_flow_object.yi ** 2)
        fluid_flow_object.eccentricity_ratio = fluid_flow_object.eccentricity / fluid_flow_object.difference_between_radius
        fluid_flow_object.attitude_angle = -np.arctan(fluid_flow_object.xi / fluid_flow_object.yi)
        # Store the displ. and velocity perturbations in vectors
        dx[i] = delta_x
        xdot[i] = fluid_flow_object.omegap * fluid_flow_object.xp * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        xddot[i] = -fluid_flow_object.omegap**2 * fluid_flow_object.xp * np.sin(fluid_flow_object.omegap * fluid_flow_object.t)
        # Compute the coefficients of the continuity equation along x
        fluid_flow_object.calculate_coefficients(direction="x")
        p_mat = fluid_flow_object.calculate_pressure_matrix_numerical()
        # Compute the forces for a perturbation along x
        [radial_force[i], tangential_force[i], force_xx[i], force_yx[i]] = \
            calculate_oil_film_force(fluid_flow_object, force_type='numerical')

        # Y DIRECTION
        # Compute the variation of the displ. along Y for t[i]
        delta_y = fluid_flow_object.yp * np.sin(fluid_flow_object.omegap * fluid_flow_object.t)
        # Compute parameters (xi,yi,ecc,eps,beta) for the new state
        fluid_flow_object.xi = xi0
        fluid_flow_object.yi = yi0 + delta_y
        fluid_flow_object.eccentricity = np.sqrt(fluid_flow_object.xi ** 2 + fluid_flow_object.yi ** 2)
        fluid_flow_object.eccentricity_ratio = fluid_flow_object.eccentricity / fluid_flow_object.difference_between_radius
        fluid_flow_object.attitude_angle = -np.arctan(fluid_flow_object.xi / fluid_flow_object.yi)
        # Store the displ. and velocity perturbations in vectors
        dy[i] = delta_y
        ydot[i] = fluid_flow_object.omegap * fluid_flow_object.yp * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        yddot[i] = -fluid_flow_object.omegap ** 2 * fluid_flow_object.yp * np.sin(fluid_flow_object.omegap * fluid_flow_object.t)
        # Compute the coefficients of the continuity equation along Y
        fluid_flow_object.calculate_coefficients(direction="y")
        p_mat = fluid_flow_object.calculate_pressure_matrix_numerical()
        # Compute the forces for a perturbation along Y
        [radial_force[i], tangential_force[i], force_xy[i], force_yy[i]] = \
            calculate_oil_film_force(fluid_flow_object, force_type='numerical')

        #  P is 8x1: [Kxx,Cxx,Mxx,Kxy,Cxy,Mxy,Kyx,Cyx,Myx,Kyy,Cyy,Myy]
        # Assemble X and F matrices
        X[i]  =   [1, dx[i],xdot[i]]
        X2[i] =   [1, dy[i],ydot[i]]

        F[i]   = -force_xx[i]
        F2[i] = -force_yy[i]


    # Compute the parameters vector according to # P = (x^T*X)^(-1)*(x^T)*F
    P = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), F)
    P2 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), F2)
    print("fxx0", P[0])
    print("fyy0", P2[0])
   # print("fxy", P[1])
   # print("fyx", P[2])
   # print("fyy", P[3])
    print("Kxx,Cxx: ", P[1],P[2])
    print("Kyy,Cyy: ", P2[1], P2[2])
  #  print("Cxx,Cxy,Cyx,Cyy: ", P[5], P[8], P[11], P[14])
   # print("Mxx,Mxy,Myx,Myy: ", P[6], P[9], P[12], P[15])
    return P


def calculate_stiffness_matrix(
    fluid_flow_object, force_type=None, oil_film_force="numerical"
):
    """This function calculates the bearing stiffness matrix numerically.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    oil_film_force: str
        If set, calculates the oil film force analytically considering the chosen type: 'short' or 'long'.
        If set to 'numerical', calculates the oil film force numerically.
    force_type: str
        If set, calculates the stiffness matrix analytically considering the chosen type: 'short'.
        If set to 'numerical', calculates the stiffness matrix numerically.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_stiffness_matrix(my_fluid_flow)  # doctest: +ELLIPSIS
    [417...
    """
    if force_type != "numerical" and (
        force_type == "short" or fluid_flow_object.bearing_type == "short_bearing"
    ):
        h0 = 1.0 / (
            (
                (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                + 16 * fluid_flow_object.eccentricity_ratio ** 2
            )
            ** 1.5
        )
        a = fluid_flow_object.load / fluid_flow_object.radial_clearance
        kxx = (
            a
            * h0
            * 4
            * (
                (np.pi ** 2) * (2 - fluid_flow_object.eccentricity_ratio ** 2)
                + 16 * fluid_flow_object.eccentricity_ratio ** 2
            )
        )
        kxy = (
            a
            * h0
            * np.pi
            * (
                (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
                - 16 * fluid_flow_object.eccentricity_ratio ** 4
            )
            / (
                fluid_flow_object.eccentricity_ratio
                * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)
            )
        )
        kyx = (
            -a
            * h0
            * np.pi
            * (
                (np.pi ** 2)
                * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
                + (32 * fluid_flow_object.eccentricity_ratio ** 2)
                * (1 + fluid_flow_object.eccentricity_ratio ** 2)
            )
            / (
                fluid_flow_object.eccentricity_ratio
                * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)
            )
        )
        kyy = (
            a
            * h0
            * 4
            * (
                (np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
                + (
                    (32 * fluid_flow_object.eccentricity_ratio ** 2)
                    * (1 + fluid_flow_object.eccentricity_ratio ** 2)
                )
                / (1 - fluid_flow_object.eccentricity_ratio ** 2)
            )
        )
    else:

        [radial_force, tangential_force, force_x, force_y] = calculate_oil_film_force(
            fluid_flow_object, force_type=oil_film_force
        )
        delta = fluid_flow_object.difference_between_radius / 100

        move_rotor_center(fluid_flow_object, delta, 0)
        fluid_flow_object.calculate_coefficients()
        fluid_flow_object.calculate_pressure_matrix_numerical()
        [
            radial_force_x,
            tangential_force_x,
            force_x_x,
            force_y_x,
        ] = calculate_oil_film_force(fluid_flow_object, force_type=oil_film_force)

        move_rotor_center(fluid_flow_object, -delta, 0)
        move_rotor_center(fluid_flow_object, 0, delta)
        fluid_flow_object.calculate_coefficients()
        fluid_flow_object.calculate_pressure_matrix_numerical()
        [
            radial_force_y,
            tangential_force_y,
            force_x_y,
            force_y_y,
        ] = calculate_oil_film_force(fluid_flow_object, force_type=oil_film_force)
        move_rotor_center(fluid_flow_object, 0, -delta)

        kxx = (force_x - force_x_x) / delta
        kyx = (force_y - force_y_x) / delta
        kxy = (force_x - force_x_y) / delta
        kyy = (force_y - force_y_y) / delta

    return [kxx, kxy, kyx, kyy]


def calculate_damping_matrix(fluid_flow_object, force_type=None):
    """Returns the damping matrix calculated analytically.
    Suitable only for short bearings.
    Parameters
    -------
    fluid_flow_object: A FluidFlow object.
    force_type: str
        If set, calculates the stiffness matrix analytically considering the chosen type: 'short'.
        If set to 'numerical', calculates the stiffness matrix numerically.
    Returns
    -------
    list of floats
        A list of length four including damping floats in this order: cxx, cxy, cyx, cyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_damping_matrix(my_fluid_flow) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    if force_type != 'numerical' and (force_type == 'short' or fluid_flow_object.bearing_type == 'short_bearing'):
        h0 = 1.0 / (((np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                     + 16 * fluid_flow_object.eccentricity_ratio ** 2) ** 1.5)
        a = fluid_flow_object.load / (fluid_flow_object.radial_clearance * fluid_flow_object.omega)
        cxx = (a * h0 * 2 * np.pi * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2) *
               ((np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
                - 16 * fluid_flow_object.eccentricity_ratio ** 2) / fluid_flow_object.eccentricity_ratio)
        cxy = (-a * h0 * 8 * ((np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
                              - 16 * fluid_flow_object.eccentricity_ratio ** 2))
        cyx = cxy
        cyy = (a * h0 * (2 * np.pi * (
                (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
                + 48 * fluid_flow_object.eccentricity_ratio ** 2)) /
               (fluid_flow_object.eccentricity_ratio * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)))
    else:
        cxx, cxy, cyx, cyy = warnings.warn(
            "Function calculate_damping_matrix cannot yet be calculated numerically. "
            "The force_type should be  'short' or 'short_bearing"
        )
    # fmt: on
    return [cxx, cxy, cyx, cyy]


def find_equilibrium_position(
    fluid_flow_object,
    print_along=True,
    tolerance=1e-05,
    increment_factor=1e-03,
    max_iterations=10,
    increment_reduction_limit=1e-04,
    return_iteration_map=False,
):
    """This function returns an eccentricity value with calculated forces matching the load applied,
    meaning an equilibrium position of the rotor.
    It first moves the rotor center on x-axis, aiming for the minimum error in the force on x (zero), then
    moves on y-axis, aiming for the minimum error in the force on y (meaning load minus force on y equals zero).
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    print_along: bool, optional
        If True, prints the iteration process.
    tolerance: float, optional
    increment_factor: float, optional
        This number will multiply the first eccentricity found to reach an increment number.
    max_iterations: int, optional
    increment_reduction_limit: float, optional
        The error should always be approximating zero. If it passes zeros (for instance, from a positive error
        to a negative one), the iteration goes back one step and the increment is reduced. This reduction must
        have a limit to avoid long iterations.
    return_iteration_map: bool, optional
        If True, along with the eccentricity found, the function will return a map of position and errors in
        each step of the iteration.
    Returns
    -------
    None, or
    Matrix of floats
        A matrix [4, n], being n the number of iterations. In each line, it contains the x and y of the rotor
        center, followed by the error in force x and force y.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> find_equilibrium_position(my_fluid_flow, print_along=False,
    ...                           tolerance=0.1, increment_factor=0.01,
    ...                           max_iterations=5, increment_reduction_limit=1e-03)
    """
    fluid_flow_object.calculate_coefficients()
    fluid_flow_object.calculate_pressure_matrix_numerical()
    r_force, t_force, force_x, force_y = calculate_oil_film_force(
        fluid_flow_object, force_type="numerical"
    )
    increment = increment_factor * fluid_flow_object.eccentricity
    error_x = abs(force_x)
    error_y = abs(force_y - fluid_flow_object.load)
    error = max(error_x, error_y)
    k = 1
    map_vector = []
    while error > tolerance and k <= max_iterations:
        increment_x = increment
        increment_y = increment
        iter_x = 0
        iter_y = 0
        previous_x = fluid_flow_object.xi
        previous_y = fluid_flow_object.yi
        infinite_loop_x_check = False
        infinite_loop_y_check = False
        if print_along:
            print("\nIteration " + str(k) + "\n")
        while error_x > tolerance:
            iter_x += 1
            move_rotor_center(fluid_flow_object, increment_x, 0)
            fluid_flow_object.calculate_coefficients()
            fluid_flow_object.calculate_pressure_matrix_numerical()
            (
                new_r_force,
                new_t_force,
                new_force_x,
                new_force_y,
            ) = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
            new_error_x = abs(new_force_x)
            move_rotor_center(fluid_flow_object, -increment_x, 0)
            if print_along:
                print("Iteration in x axis " + str(iter_x))
                print("Force x: " + str(new_force_x))
                print("Previous force x: " + str(force_x))
                print("Increment x: ", str(increment_x))
                print("Error x: " + str(new_error_x))
                print("Previous error x: " + str(error_x) + "\n")
            if new_force_x * force_x < 0:
                infinite_loop_x_check = False
                increment_x = increment_x / 10
                if print_along:
                    print("Went beyond error 0. Reducing increment. \n")
                if abs(increment_x) < abs(increment * increment_reduction_limit):
                    if print_along:
                        print("Increment too low. Breaking x iteration. \n")
                    break
            elif new_error_x > error_x:
                if print_along:
                    print("Error increased. Changing sign of increment. \n")
                increment_x = -increment_x
                if infinite_loop_x_check:
                    break
                else:
                    infinite_loop_x_check = True
            else:
                infinite_loop_x_check = False
                move_rotor_center(fluid_flow_object, increment_x, 0)
                error_x = new_error_x
                force_x = new_force_x
                force_y = new_force_y
                error_y = abs(new_force_y - fluid_flow_object.load)
                error = max(error_x, error_y)

        while error_y > tolerance:
            iter_y += 1
            move_rotor_center(fluid_flow_object, 0, increment_y)
            fluid_flow_object.calculate_coefficients()
            fluid_flow_object.calculate_pressure_matrix_numerical()
            (
                new_r_force,
                new_t_force,
                new_force_x,
                new_force_y,
            ) = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
            new_error_y = abs(new_force_y - fluid_flow_object.load)
            move_rotor_center(fluid_flow_object, 0, -increment_y)
            if print_along:
                print("Iteration in y axis " + str(iter_y))
                print("Force y: " + str(new_force_y))
                print("Previous force y: " + str(force_y))
                print("Increment y: ", str(increment_y))
                print(
                    "Force y minus load: " + str(new_force_y - fluid_flow_object.load)
                )
                print(
                    "Previous force y minus load: "
                    + str(force_y - fluid_flow_object.load)
                )
                print("Error y: " + str(new_error_y))
                print("Previous error y: " + str(error_y) + "\n")
            if (new_force_y - fluid_flow_object.load) * (
                force_y - fluid_flow_object.load
            ) < 0:
                infinite_loop_y_check = False
                increment_y = increment_y / 10
                if print_along:
                    print("Went beyond error 0. Reducing increment. \n")
                if abs(increment_y) < abs(increment * increment_reduction_limit):
                    if print_along:
                        print("Increment too low. Breaking y iteration. \n")
                    break
            elif new_error_y > error_y:
                if print_along:
                    print("Error increased. Changing sign of increment. \n")
                increment_y = -increment_y
                if infinite_loop_y_check:
                    break
                else:
                    infinite_loop_y_check = True
            else:
                infinite_loop_y_check = False
                move_rotor_center(fluid_flow_object, 0, increment_y)
                error_y = new_error_y
                force_y = new_force_y
                force_x = new_force_x
                error_x = abs(new_force_x)
                error = max(error_x, error_y)
        if print_along:
            print("Iteration " + str(k))
            print("Error x: " + str(error_x))
            print("Error y: " + str(error_y))
            print(
                "Current x, y: ("
                + str(fluid_flow_object.xi)
                + ", "
                + str(fluid_flow_object.yi)
                + ")"
            )
        k += 1
        map_vector.append(
            [fluid_flow_object.xi, fluid_flow_object.yi, error_x, error_y]
        )
        if previous_x == fluid_flow_object.xi and previous_y == fluid_flow_object.yi:
            if print_along:
                print("Rotor center did not move during iteration. Breaking.")
            break

    if print_along:
        print(map_vector)
    if return_iteration_map:
        return map_vector
