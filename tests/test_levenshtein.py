"""
Pytest unit tests for Levenshtein distance and nearest-pattern lookup.
Mandatory per project rubric — covers the unseen-pattern handler in
src/models/automata/levenshtein.py.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.automata.levenshtein import (
    levenshtein_distance,
    find_nearest_pattern,
    resolve_pattern,
)


# ── levenshtein_distance ──────────────────────────────────────────────────────

class TestLevenshteinDistance:

    def test_identical_strings(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_identical_single_char(self):
        assert levenshtein_distance("a", "a") == 0

    def test_one_substitution(self):
        assert levenshtein_distance("abc", "abd") == 1

    def test_one_insertion(self):
        # "ab" → "abc" requires inserting 'c'
        assert levenshtein_distance("ab", "abc") == 1

    def test_one_deletion(self):
        # "abc" → "ab" requires deleting 'c'
        assert levenshtein_distance("abc", "ab") == 1

    def test_empty_vs_nonempty(self):
        assert levenshtein_distance("", "abc") == 3

    def test_nonempty_vs_empty(self):
        assert levenshtein_distance("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_two_edits(self):
        assert levenshtein_distance("aab", "abc") == 2

    def test_spec_example(self):
        # From project spec: "adc" vs "abc" = 1 substitution
        assert levenshtein_distance("adc", "abc") == 1

    def test_completely_different(self):
        # "aaa" → "bbb" requires 3 substitutions
        assert levenshtein_distance("aaa", "bbb") == 3

    def test_single_char_vs_empty(self):
        assert levenshtein_distance("a", "") == 1
        assert levenshtein_distance("", "a") == 1

    def test_symmetry(self):
        # distance(s1, s2) == distance(s2, s1)
        assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")

    def test_classic_kitten_sitting(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_longer_sax_strings(self):
        # Typical SAX window strings from the automata model
        assert levenshtein_distance("aabc", "abbc") == 1
        assert levenshtein_distance("aabc", "aabc") == 0


# ── find_nearest_pattern ──────────────────────────────────────────────────────

class TestFindNearestPattern:

    def setup_method(self):
        self.vocab = {"aab", "abc", "bcc", "bba", "ccc"}

    def test_exact_match_in_vocabulary(self):
        # Even when the pattern is in vocab, find_nearest should still work
        nearest, dist = find_nearest_pattern("abc", self.vocab)
        assert nearest == "abc"
        assert dist == 0

    def test_unseen_one_edit_away(self):
        # "adc" is 1 substitution from "abc"
        nearest, dist = find_nearest_pattern("adc", self.vocab)
        assert dist == 1
        assert nearest == "abc"

    def test_returns_minimum_distance(self):
        _, dist = find_nearest_pattern("bbc", self.vocab)
        # "bbc" is 1 from "bcc" (substitution b→c at position 1)
        # and 1 from "bba" (substitution c→a at position 2)
        assert dist == 1

    def test_empty_vocabulary_raises(self):
        with pytest.raises(ValueError, match="Vocabulary is empty"):
            find_nearest_pattern("abc", set())

    def test_single_item_vocabulary(self):
        nearest, dist = find_nearest_pattern("xyz", {"abc"})
        assert nearest == "abc"
        assert dist == 3

    def test_result_is_in_vocabulary(self):
        nearest, _ = find_nearest_pattern("adc", self.vocab)
        assert nearest in self.vocab

    def test_distance_is_non_negative(self):
        _, dist = find_nearest_pattern("zzz", self.vocab)
        assert dist >= 0

    def test_unseen_all_same_distance_returns_one(self):
        # When multiple patterns tie, one is returned (determinism not required, just valid)
        nearest, dist = find_nearest_pattern("xyz", {"aaa", "bbb", "ccc"})
        assert nearest in {"aaa", "bbb", "ccc"}
        assert dist == levenshtein_distance("xyz", nearest)


# ── resolve_pattern ───────────────────────────────────────────────────────────

class TestResolvePattern:

    def setup_method(self):
        self.vocab = {"aab", "abc", "bcc", "bba", "ccc"}

    def test_seen_pattern_returns_seen_status(self):
        resolved, status, distance = resolve_pattern("abc", self.vocab)
        assert resolved == "abc"
        assert status == "seen"
        assert distance == 0

    def test_unseen_pattern_returns_unseen_status(self):
        resolved, status, distance = resolve_pattern("adc", self.vocab)
        assert status == "unseen"
        assert distance > 0
        assert resolved in self.vocab

    def test_unseen_distance_matches_nearest(self):
        resolved, status, distance = resolve_pattern("adc", self.vocab)
        expected_nearest, expected_dist = find_nearest_pattern("adc", self.vocab)
        assert resolved == expected_nearest
        assert distance == expected_dist

    def test_seen_distance_is_zero(self):
        _, _, distance = resolve_pattern("bba", self.vocab)
        assert distance == 0

    def test_unseen_resolved_is_valid_vocab_member(self):
        resolved, _, _ = resolve_pattern("zzz", self.vocab)
        assert resolved in self.vocab

    def test_all_vocab_members_resolve_as_seen(self):
        for pattern in self.vocab:
            _, status, dist = resolve_pattern(pattern, self.vocab)
            assert status == "seen"
            assert dist == 0
