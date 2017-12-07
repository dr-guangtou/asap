""" Functions implementing a Gaussian selection kernel on a set of points in the xy-plane
"""
import numpy as np


def counter_clockwise_2d_rotation_matrix(rot_angle_in_degrees):
    """
    """
    rot_angle_in_radians = rot_angle_in_degrees*np.pi/180.
    c, s = np.cos(rot_angle_in_radians), np.sin(rot_angle_in_radians)
    return np.array(((c, s), (-s, c))).T


def rotate_2d(v, rot_angle_in_degrees):
    """ Rotate a 2-d vector ``v`` in the counter-clockwise direction by some angle

    Parameters
    ----------
    v : ndarray
        Array of shape (2, )

    rot_angle_in_degrees : float
        Counter-clockwise rotation angle in units of degrees

    Returns
    -------
    rotated_v : ndarray
        Array of shape (2, )
    """
    return np.dot(counter_clockwise_2d_rotation_matrix(rot_angle_in_degrees), v)


def ellipse_distance(xpts, ypts, xc, yc, major_axis_angle, b_by_a):
    """ Calculate the distance between (xc, yc) and each point in (xpts, ypts)
    using an elliptical distance metric, with the ellipse defined by the
    axis ``major_axis_angle`` (in degrees) and ``b_by_a`` major-to-minor axis length
    """
    major_axis = rotate_2d((1, 0), major_axis_angle)
    minor_axis = rotate_2d(major_axis, 90)
    pts = np.vstack((xpts-xc, ypts-yc)).T
    a = np.dot(pts, major_axis)
    b = np.dot(pts, minor_axis)
    return np.sqrt(a**2 + b_by_a*b**2)


def ellipse_selector(xpts, ypts, xc, yc, major_axis_angle, b_by_a, nkeep):
    """ From an input set of points in the xy plane, select a sample of ``nkeep`` points
    defined by the ellipse defined by the input ``major_axis_angle`` and ``b_by_a``.

    Parameters
    ----------
    xpts, ypts : ndarrays
        Numpy arrays of shape (ngals, ) storing, for example, the values of M10 and M100

    xc, yc : floats
        Centroid of the ellipse

    major_axis_angle : float
        Angle defining the major axis direction of the ellipse.
        Angle should be in degrees in the counter-clockwise direction off of the positive x-axis

    b_by_a : float
        Major-to-minor axis ratio

    nkeep : int
        Number of galaxies in the output sample

    Returns
    -------
    idx : ndarray
        Numpy array of shape (nkeep, ) that can be used as a mask to select the
        galaxies passing the elliptical selection

    Notes
    -----
    See ellipse_selection_demo.ipynb for usage demo

    """
    d = ellipse_distance(xpts, ypts, xc, yc, major_axis_angle, b_by_a)
    idx_sorted_d = np.argsort(d)
    return idx_sorted_d[:nkeep]


def split_sample_along_axis_angle(xpts, ypts, xc, yc, axis_angle):
    """ From an input set of points in the xy plane, split the sample into
    points on one side or the other of the line passing through (xc, yc) with
    angle ``axis_angle``.

    Parameters
    ----------
    xpts, ypts : ndarrays
        Numpy arrays of shape (ngals, ) storing, for example, the values of M10 and M100

    xc, yc : floats
        Centroid of the ellipse

    axis_angle : float
        Angle defining the axis direction along which the sample will be split.
        Angle should be in degrees in the counter-clockwise direction off of the positive x-axis

    Returns
    -------
    idx : ndarray
        Numpy array of shape (nkeep, ) that can be used as a mask to divide the input sample

    Notes
    -----
    The output mask just directly splits the sample according to which side of the
    line the points fall on.
    This does *nothing* to guarantee that there will be an equal number of points
    in the resulting subsamples.

    See ellipse_selection_demo.ipynb for usage demo
    """
    splitting_axis = rotate_2d((1, 0), axis_angle)
    if splitting_axis[0] != 0:
        slope = splitting_axis[1]/splitting_axis[0]
        b = yc - slope*xc
        ycut = slope*xpts + b
        return ypts > ycut
    else:
        return xpts > xc
