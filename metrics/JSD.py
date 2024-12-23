import numpy as np

from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import warnings

def grid_coordinate(w_space, h_space):
    grid = np.ndarray((int(330//h_space), int(300//w_space), 2), np.float32)
    
    for i in range(int(300//w_space)):
        for j in range(int(330//h_space)):
            grid[j, i, 0] = (i + 0.5) * w_space + 25
            grid[j, i, 1] = (j + 0.5) * h_space + 150

    return grid


def jsd_between_list_of_points(sample_points, ref_points, space):
    sample_grid_var = entropy_of_occupancy_grid(sample_points, space)
    ref_grid_var = entropy_of_occupancy_grid(ref_points, space)
    
    space3x3 = {'w_space': 100, 'h_space': 110}
    sample_grid_var_3x3 = entropy_of_occupancy_grid(sample_points, space3x3)
    ref_grid_var_3x3 = entropy_of_occupancy_grid(ref_points, space3x3)
    
    return jensen_shannon_divergence(sample_grid_var_3x3, ref_grid_var_3x3), jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(points, space):
    grid_coordinates = grid_coordinate(space['w_space'], space['h_space'])
    gw, gh, _ = grid_coordinates.shape
    grid_coordinates = grid_coordinates.reshape(-1, 2)
    grid_counters = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    _, indices = nn.kneighbors(points)
    indices = np.squeeze(indices)
    for i in indices:
        grid_counters[i] += 1

    s = 1.5 if space['w_space'] < 30 else 1
    grid_counters = np.transpose(grid_counters.reshape(gw, gh))
    grid_counters = ndimage.filters.gaussian_filter(grid_counters, sigma=s, mode='constant')
    grid_counters = grid_counters.reshape(-1, 1)

    return grid_counters

def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


