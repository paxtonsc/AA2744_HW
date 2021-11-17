import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

    x = xvec[0]
    y = xvec[1]
    th = xvec[2]
    V = u[0]
    om = u[1]

    g = np.zeros(3)
    Gx = np.eye(3)
    Gu = np.zeros((3, 2))

    if abs(om) < EPSILON_OMEGA:
        g[0] = x + V*dt*np.cos(th)
        g[1] = y + V*dt*np.sin(th)
        g[2] = th

        Gx[0,2] = -V*dt*np.sin(th)
        Gx[1,2] = V*dt*np.cos(th)

        Gu[0,0] = dt*np.cos(th)
        Gu[1,0] = dt*np.sin(th)
        Gu[0,1] = -1/2*V*dt**2*np.sin(th)
        Gu[1,1] = 1/2*V*dt**2*np.cos(th)
        Gu[2,1] = dt
    else:
        g[0] = x + V/om * (np.sin(th + om*dt) - np.sin(th))
        g[1] = y + V/om * (-np.cos(th + om*dt) + np.cos(th))
        g[2] = th + om*dt

        Gx[0,2] = V/om * (np.cos(th + om*dt) - np.cos(th))
        Gx[1,2] = V/om * (np.sin(th + om*dt) - np.sin(th))

        Gu[0,0] = 1/om * (np.sin(th + om*dt) - np.sin(th))
        Gu[1,0] = 1/om * (-np.cos(th + om*dt) + np.cos(th))
        Gu[0,1] = -V/om**2 * (np.sin(th + om*dt) - np.sin(th)) + V*dt/om * np.cos(th + om*dt)
        Gu[1,1] = -V/om**2 * (-np.cos(th + om*dt) + np.cos(th)) + V*dt/om * np.sin(th + om*dt)
        Gu[2,1] = dt

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    a_w = line[0]
    r_w = line[1]

    x_b = x[0]
    y_b = x[1]
    th_b = x[2]

    x_c = tf_base_to_camera[0]
    y_c = tf_base_to_camera[1]
    th_c = tf_base_to_camera[2]

    r_b = x_b * np.cos(a_w) + y_b * np.sin(a_w) - r_w
    a_b = -np.pi + a_w - th_b

    r_c = x_c * np.cos(a_b) + y_c * np.sin(a_b) - r_b
    a_c = np.pi + a_b - th_c

    h = np.array([a_c, r_c])

    Hx = np.zeros((2,3))
    Hx[0, 2] = -1
    Hx[1, 0] = -np.cos(a_w)
    Hx[1, 1] = -np.sin(a_w)
    Hx[1, 2] = -x_c * np.sin(a_w - th_b) + y_c * np.cos(a_w - th_b)

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
