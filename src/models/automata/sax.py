import numpy as np
from scipy.stats import norm


# Gaussian breakpoints for alphabet sizes 2–10
# Each entry: list of (alphabet_size - 1) breakpoints
_BREAKPOINTS = {
    2: [-0.0],
    3: [-0.4307, 0.4307],
    4: [-0.6745, 0.0,    0.6745],
    5: [-0.8416, -0.2533, 0.2533, 0.8416],
    6: [-0.9674, -0.4307, 0.0,    0.4307, 0.9674],
    7: [-1.0676, -0.5659, -0.1800, 0.1800, 0.5659, 1.0676],
    8: [-1.1503, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.1503],
    9: [-1.2206, -0.7647, -0.4307, -0.1397, 0.1397, 0.4307, 0.7647, 1.2206],
   10: [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816],
}

# Alphabet letters
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def get_breakpoints(alphabet_size: int) -> np.ndarray:
    """Returns Gaussian breakpoints for the given alphabet size."""
    if alphabet_size not in _BREAKPOINTS:
        raise ValueError(f"alphabet_size must be between 2 and 10, got {alphabet_size}")
    return np.array(_BREAKPOINTS[alphabet_size])


def value_to_symbol(value: float, breakpoints: np.ndarray) -> str:
    """Maps a single float value to a SAX symbol using breakpoints."""
    idx = np.searchsorted(breakpoints, value, side="right")
    return _ALPHABET[idx]


def paa_to_sax(paa_window: np.ndarray, alphabet_size: int) -> str:
    """
    Converts a PAA window (1D array) to a SAX string.

    Args:
        paa_window    : 1D array of PAA values (already normalized)
        alphabet_size : number of SAX symbols

    Returns:
        sax_str : string of length len(paa_window)
    """
    breakpoints = get_breakpoints(alphabet_size)
    symbols = [value_to_symbol(v, breakpoints) for v in paa_window]
    return "".join(symbols)


def series_to_sax_patterns(series: np.ndarray,
                             window_size: int,
                             alphabet_size: int) -> list:
    """
    Converts a full time series to a list of SAX patterns using
    a sliding window approach.

    Args:
        series        : 1D normalized numpy array (PC1)
        window_size   : sliding window size
        alphabet_size : SAX alphabet size

    Returns:
        patterns : list of SAX strings, length = len(series) - window_size + 1
    """
    series = np.array(series, dtype=float)
    n = len(series)
    breakpoints = get_breakpoints(alphabet_size)
    patterns = []

    for start in range(n - window_size + 1):
        window = series[start: start + window_size]
        symbols = [value_to_symbol(v, breakpoints) for v in window]
        patterns.append("".join(symbols))

    return patterns


def build_sax_vocabulary(patterns: list) -> set:
    """Returns the set of unique SAX patterns seen in training data."""
    return set(patterns)


if __name__ == "__main__":
    np.random.seed(42)

    # Simulate normalized PC1 values
    series = np.random.randn(50)

    print("=" * 45)
    print(f"Series length: {len(series)}")
    print(f"Series sample: {series[:5].round(3)}")

    # Test with window_size=4, alphabet_size=3
    patterns = series_to_sax_patterns(series, window_size=4, alphabet_size=3)
    vocab    = build_sax_vocabulary(patterns)

    print(f"\nwindow_size=4, alphabet_size=3")
    print(f"Total patterns  : {len(patterns)}")
    print(f"Unique patterns : {len(vocab)}")
    print(f"First 5 patterns: {patterns[:5]}")
    print(f"Vocabulary      : {sorted(vocab)}")

    # Test with window_size=4, alphabet_size=4
    patterns2 = series_to_sax_patterns(series, window_size=4, alphabet_size=4)
    vocab2    = build_sax_vocabulary(patterns2)
    print(f"\nwindow_size=4, alphabet_size=4")
    print(f"Unique patterns : {len(vocab2)}")
    print(f"First 5 patterns: {patterns2[:5]}")