"""Boost-factor emulator for the cubic Galileon gravity model.

One pre-trained SEPIA emulator (PCA basis + Gaussian process on the PCA weights) is
bundled per training redshift snapshot. SEPIA rebuilds the PCA basis and conditions the
GP on the training set, so the (smoothed) training boosts are bundled alongside the
saved GP hyperparameter samples. Predictions between snapshots are linearly
interpolated in redshift.
"""

__all__ = ['emulate', 'PARAM_NAMES', 'PARAM_MIN', 'PARAM_MAX', 'Z_MIN', 'Z_MAX']

import contextlib
import functools
import io
import os
import warnings

import numpy as np
from scipy.signal import savgol_filter
from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaPredict import SepiaEmulatorPrediction

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_PACKAGE_DIR, 'data')
_MODEL_DIR = os.path.join(_PACKAGE_DIR, 'model')

_EXP_VARIANCE = 0.999   # PCA truncation used when the bundled models were trained
_SMOOTH_WINDOW = 17     # Savitzky-Golay smoothing applied to the training boosts
_SMOOTH_ORDER = 3

PARAM_NAMES = ['Omega_m', 'n_s', '1e9 A_s', 'h', 'f_phi']


def _load_training_set():
    p_train = np.loadtxt(os.path.join(_DATA_DIR, 'cosmo_newdesign.txt'))
    p_train[:, 2] = p_train[:, 2] / 1e-9  # A_s rescaling

    zk = np.loadtxt(os.path.join(_DATA_DIR, 'z_k.txt'))
    z_grid = zk[:, 0][np.isfinite(zk[:, 0])]
    k_grid = zk[:, 1]

    Bk = np.load(os.path.join(_DATA_DIR, 'Boost.npy'))
    Bk_smooth = savgol_filter(Bk, _SMOOTH_WINDOW, _SMOOTH_ORDER, axis=-1)
    return p_train, Bk_smooth, k_grid, z_grid


_P_TRAIN, _BK_TRAIN, _K_GRID, _Z_GRID = _load_training_set()
_N_SNAPSHOTS = len([f for f in os.listdir(_MODEL_DIR) if f.endswith('.pkl')])
_Z_GRID = _Z_GRID[:_N_SNAPSHOTS]  # redshifts that have a trained model

PARAM_MIN = _P_TRAIN.min(axis=0)
PARAM_MAX = _P_TRAIN.max(axis=0)
Z_MIN, Z_MAX = float(_Z_GRID[0]), float(_Z_GRID[-1])


@functools.lru_cache(maxsize=None)
def _snapshot_model(z_index):
    """Rebuild the PCA basis for one snapshot and restore its trained GP."""
    with contextlib.redirect_stdout(io.StringIO()):  # SEPIA is chatty
        data = SepiaData(t_sim=_P_TRAIN, y_sim=_BK_TRAIN[:, z_index, :], y_ind_sim=_K_GRID)
        data.transform_xt()
        data.standardize_y()
        data.create_K_basis(n_pc=_EXP_VARIANCE)
        model = SepiaModel(data)
        model.restore_model_info(os.path.join(_MODEL_DIR, 'multivariate_model_z_index%d' % z_index))
    return model, data


def _emulate_snapshot(z_index, params):
    model, data = _snapshot_model(z_index)
    K = data.sim_data.K  # (n_pc, n_k)
    y_sd, y_mean = data.sim_data.orig_y_sd, data.sim_data.orig_y_mean
    samples = model.get_samples(numsamples=1)

    means, stds = [], []
    for p in params:
        pred = SepiaEmulatorPrediction(t_pred=p[None, :], samples=samples, model=model, storeMuSigma=True)
        mu, Sigma = pred.mu[0], pred.sigma[0]
        y_mu = K.T @ mu
        y_std = np.sqrt(np.clip(np.diag(K.T @ Sigma @ K), 0, None))
        means.append(y_sd * y_mu + y_mean)
        stds.append(y_sd * y_std)
    return np.array(means), np.array(stds)


def emulate(params, z):
    """Emulate the boost factor B(k) at the given cosmologies and redshift.

    Parameters
    ----------
    params : array_like, shape (5,) or (n, 5)
        Columns are [Omega_m, n_s, 1e9 A_s, h, f_phi] (see `PARAM_NAMES`).
    z : float
        Redshift, within [`Z_MIN`, `Z_MAX`].

    Returns
    -------
    k : ndarray, shape (n_k,)
        Wavenumbers in h/Mpc.
    Bk_mean : ndarray, shape (n, n_k)
        Emulated mean boost.
    Bk_std : ndarray, shape (n, n_k)
        Emulator (GP) standard deviation.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if params.shape[1] != len(PARAM_NAMES):
        raise ValueError('params must have %d columns %s, got shape %s'
                         % (len(PARAM_NAMES), PARAM_NAMES, params.shape))
    z = float(z)
    if not Z_MIN <= z <= Z_MAX:
        raise ValueError('z=%g is outside the emulated range [%g, %g]' % (z, Z_MIN, Z_MAX))
    if np.any(params < PARAM_MIN) or np.any(params > PARAM_MAX):
        warnings.warn('Some parameters are outside the training range %s - %s; the emulator '
                      'is extrapolating.' % (PARAM_MIN, PARAM_MAX), stacklevel=2)

    # Bracketing snapshots z1 <= z <= z2
    i1 = int(np.clip(np.searchsorted(_Z_GRID, z, side='right') - 1, 0, _N_SNAPSHOTS - 2))
    i2 = i1 + 1
    w = (z - _Z_GRID[i1]) / (_Z_GRID[i2] - _Z_GRID[i1])

    mean1, std1 = _emulate_snapshot(i1, params)
    mean2, std2 = _emulate_snapshot(i2, params)
    return _K_GRID.copy(), (1 - w) * mean1 + w * mean2, (1 - w) * std1 + w * std2
