import os

import numpy as np
import pytest

import CubicGalileonEmu
from CubicGalileonEmu import emulate, PARAM_MIN, PARAM_MAX, Z_MIN, Z_MAX

DATA_DIR = os.path.join(os.path.dirname(CubicGalileonEmu.__file__), 'data')


def _training_set():
    p = np.loadtxt(os.path.join(DATA_DIR, 'cosmo_newdesign.txt'))
    p[:, 2] /= 1e-9
    zk = np.loadtxt(os.path.join(DATA_DIR, 'z_k.txt'))
    z = zk[:, 0][np.isfinite(zk[:, 0])]
    Bk = np.load(os.path.join(DATA_DIR, 'Boost.npy'))
    return p, z, Bk


def test_output_shapes():
    p_mid = 0.5 * (PARAM_MIN + PARAM_MAX)
    k, mean, std = emulate(p_mid, 0.5)
    assert mean.shape == std.shape == (1, len(k))
    k, mean, std = emulate(np.stack([p_mid, p_mid]), 0.5)
    assert mean.shape == (2, len(k))
    assert np.all(np.isfinite(mean)) and np.all(std >= 0)


def test_reproduces_training_simulation():
    p, z, Bk = _training_set()
    k, mean, _ = emulate(p[0], z[8])
    assert np.max(np.abs(mean[0] / Bk[0, 8] - 1)) < 0.05


def test_redshift_bounds():
    p_mid = 0.5 * (PARAM_MIN + PARAM_MAX)
    emulate(p_mid, Z_MIN)
    emulate(p_mid, Z_MAX)
    with pytest.raises(ValueError):
        emulate(p_mid, Z_MAX + 1)
    with pytest.raises(ValueError):
        emulate(p_mid, -0.1)
