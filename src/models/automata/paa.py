import numpy as np


def paa_transform(series: np.ndarray, window_size: int) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA).

    Divides the time series into `window_size` equal-length segments
    and computes the mean of each segment.

    Args:
        series      : 1D numpy array (PC1 values)
        window_size : number of PAA segments

    Returns:
        paa_out : 1D numpy array of length window_size
    """
    series = np.array(series, dtype=float)
    n = len(series)

    if n < window_size:
        raise ValueError(
            f"Series length ({n}) must be >= window_size ({window_size})."
        )

    # Split into window_size equal segments, take mean of each
    segment_size = n / window_size
    paa_out = np.array([
        series[int(i * segment_size): int((i + 1) * segment_size)].mean()
        for i in range(window_size)
    ])

    return paa_out


def paa_sliding_window(series: np.ndarray,
                        window_size: int) -> np.ndarray:
    """
    Applies PAA over a sliding window across the entire series.
    Each window of length `window_size` is reduced to a single PAA vector.

    Args:
        series      : 1D numpy array (full PC1 time series)
        window_size : size of each sliding window

    Returns:
        paa_windows : 2D numpy array of shape (n_windows, window_size)
                      where n_windows = len(series) - window_size + 1
    """
    series = np.array(series, dtype=float)
    n = len(series)

    if n < window_size:
        raise ValueError(
            f"Series length ({n}) is shorter than window_size ({window_size})."
        )

    paa_windows = []
    for start in range(n - window_size + 1):
        window = series[start: start + window_size]
        # Each element in window is already 1 point, so PAA here
        # means the window IS the PAA representation (1 point per segment)
        paa_windows.append(window)

    return np.array(paa_windows)  # shape: (n_windows, window_size)


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    test_series = np.random.randn(100)

    # Single PAA
    paa_out = paa_transform(test_series, window_size=4)
    print(f"PAA output (window=4): {paa_out}")
    print(f"PAA length: {len(paa_out)}")

    # Sliding window PAA
    windows = paa_sliding_window(test_series, window_size=4)
    print(f"\nSliding window PAA shape: {windows.shape}")
    print(f"First window: {windows[0]}")
    print(f"Second window: {windows[1]}")