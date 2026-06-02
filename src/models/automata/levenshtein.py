import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein (edit) distance between two strings.
    Classic dynamic programming implementation.

    Args:
        s1, s2 : two strings to compare

    Returns:
        int : minimum edit distance
    """
    m, n = len(s1), len(s2)

    # dp[i][j] = edit distance between s1[:i] and s2[:j]
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1]  # substitution
                )

    return int(dp[m][n])


def find_nearest_pattern(unseen: str, vocabulary: set) -> tuple:
    """
    Finds the closest known pattern to an unseen pattern
    using Levenshtein distance.

    Args:
        unseen     : SAX string not in training vocabulary
        vocabulary : set of known SAX patterns from training

    Returns:
        (nearest_pattern, distance) : tuple of best match and its distance
    """
    if not vocabulary:
        raise ValueError("Vocabulary is empty — cannot find nearest pattern.")

    best_pattern  = None
    best_distance = float("inf")

    for known in vocabulary:
        dist = levenshtein_distance(unseen, known)
        if dist < best_distance:
            best_distance = dist
            best_pattern  = known

    return best_pattern, best_distance


def resolve_pattern(pattern: str, vocabulary: set) -> tuple:
    """
    Returns the pattern to use for state transition lookup.

    If pattern is in vocabulary → (pattern, "seen", 0)
    If pattern is unseen       → (nearest, "unseen", distance)

    Args:
        pattern    : incoming SAX pattern
        vocabulary : set of known training patterns

    Returns:
        (resolved_pattern, status, distance)
    """
    if pattern in vocabulary:
        return pattern, "seen", 0

    nearest, distance = find_nearest_pattern(pattern, vocabulary)
    return nearest, "unseen", distance


if __name__ == "__main__":
    # ── Unit tests (mandatory per project spec) ──────────────
    print("=" * 45)
    print("Levenshtein distance unit tests")
    print("=" * 45)

    test_cases = [
        ("abc",  "abc",  0),   # identical
        ("abc",  "abd",  1),   # one substitution
        ("abc",  "ab",   1),   # one deletion
        ("ab",   "abc",  1),   # one insertion
        ("aab",  "abc",  2),   # two edits
        ("adc",  "abc",  1),   # example from project spec
        ("",     "abc",  3),   # empty string
        ("abc",  "",     3),   # empty string
    ]

    all_passed = True
    for s1, s2, expected in test_cases:
        result = levenshtein_distance(s1, s2)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}  levenshtein('{s1}', '{s2}') = {result}  (expected {expected})")

    print(f"\nAll tests passed: {all_passed}")

    # ── Nearest pattern test ──────────────────────────────────
    print("\n" + "=" * 45)
    print("Nearest pattern lookup test")
    vocabulary = {"aab", "abc", "bcc", "bba", "ccc"}
    unseen_pattern = "adc"

    nearest, dist = find_nearest_pattern(unseen_pattern, vocabulary)
    print(f"Unseen pattern : '{unseen_pattern}'")
    print(f"Vocabulary     : {sorted(vocabulary)}")
    print(f"Nearest match  : '{nearest}' (distance={dist})")

    # resolve_pattern test
    resolved, status, distance = resolve_pattern("abc", vocabulary)
    print(f"\nresolve('abc') → '{resolved}', status='{status}', dist={distance}")

    resolved, status, distance = resolve_pattern("adc", vocabulary)
    print(f"resolve('adc') → '{resolved}', status='{status}', dist={distance}")