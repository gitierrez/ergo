import pytest
import numpy as np
from numpy.random import default_rng

from ergo.monitor.shift import BaseShiftDetector


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_detect_shift_on_different_distributions_with_same_size(shift_detector):
    size = (10000, 30)
    rng = default_rng(42)
    d1 = rng.normal(0, 1, size)
    d2 = rng.normal(10, 5, size)
    assert shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_does_not_detect_shift_on_identical_distributions_with_same_size(shift_detector):
    size = (10000, 30)
    rng = default_rng(42)
    d1 = rng.normal(0, 1, size)
    d2 = rng.normal(0, 1, size)
    assert not shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_does_not_detect_shift_on_eps_perturbed_distributions_with_same_size(shift_detector):
    size = (10000, 30)
    rng = default_rng(42)
    eps = 5e-10
    d1 = rng.normal(0, 1, size)
    d2 = rng.normal(0, 1 + eps, size)
    assert not shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_detects_shift_on_different_distributions_with_different_size(shift_detector):
    n1 = (20000, 30)
    n2 = (10000, 30)
    rng = default_rng(42)
    d1 = rng.normal(0, 1, n1)
    d2 = rng.normal(10, 5, n2)
    assert shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_does_not_detect_shift_on_identical_distributions_with_different_size(shift_detector):
    n1 = (20000, 30)
    n2 = (10000, 30)
    rng = default_rng(42)
    d1 = rng.normal(0, 1, n1)
    d2 = rng.normal(0, 1, n2)
    assert not shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_does_not_detect_shift_on_eps_perturbed_distributions_with_different_size(shift_detector):
    n1 = (20000, 30)
    n2 = (10000, 30)
    rng = default_rng(42)
    eps = 0.05
    d1 = rng.normal(0, 1, n1)
    d2 = rng.normal(0, 1 + eps, n2)
    assert not shift_detector.is_drift(d1, d2)


@pytest.mark.parametrize("shift_detector", [subclass() for subclass in BaseShiftDetector.__subclasses__()])
def test_does_not_detect_shift_on_non_random_sample_of_same_distribution(shift_detector):
    n1 = (20000, 30)
    n2 = (10000, 30)
    rng = default_rng(42)
    d1 = rng.normal(0, 1, n1)
    d2 = rng.normal(10, 5, n2)
    indices_for_d3 = np.random.choice(len(d2), size=len(d2) // 2, replace=False)
    d3 = d2[indices_for_d3]
    assert not shift_detector.is_drift(np.vstack((d1, d2)), d3)
