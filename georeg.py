"""Geographical Regression Module"""
import numpy as np
from matplotlib import path
from scipy import stats


def geo_regression(coordinates, x, y, radius):
    """Georaphical univariate linear regression.

    Pearson's r value is calculated for each coordinate using the input data
    within a given radius. If there are less than three data points an r value
    of zero is used.

    Parameters
    ----------
    coordinates : (N, 2) array_like
        UTM coordinates (meters)
        [x-coord, y-coord]

    x : (N,) array_like
        independant variable

    y : (N,) array_like
        dependent variable

    radius : float
        radius (meters)

    Returns
    -------
    (N, 2) ndarray
        [Pearson's r value, number of data points included in the regression]

    """
    coordinates = np.asarray(coordinates)
    x = np.asarray(x)
    y = np.asarray(y)

    r = []
    num_points = []

    for coord in coordinates:
        p = path.Path.circle(center=(coord[0], coord[1]), radius=radius)
        filter_mask = p.contains_points(coordinates)

        coords_selected = coordinates[filter_mask]
        x_selected = x[filter_mask]
        y_selected = y[filter_mask]

        num_points.append(len(coords_selected))

        if len(coords_selected) < 3:
            r.append(0)
        else:
            r.append(stats.pearsonr(x_selected, y_selected)[0])

    return np.array([r, num_points]).T
