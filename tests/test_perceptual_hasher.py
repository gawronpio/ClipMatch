"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.

Perceptual Hasher unit tests for ClipMatch class.
"""

import unittest

import cv2
import numpy as np

from clipmatch.clipmatch import PerceptualHasher


class TestPerceptualHasher(unittest.TestCase):
    """Test cases for PerceptualHasher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.identical_image = self.test_image.copy()
        self.similar_image = cv2.GaussianBlur(self.test_image, (3, 3), 0)
        self.different_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.mirrored_image = cv2.flip(self.test_image, 1)

    def test_dhash_consistency(self):
        """Test that dHash produces consistent results for the same image."""
        hash1 = PerceptualHasher.dhash(self.test_image)
        hash2 = PerceptualHasher.dhash(self.identical_image)
        self.assertEqual(hash1, hash2)

    def test_phash_consistency(self):
        """Test that pHash produces consistent results for the same image."""
        hash1 = PerceptualHasher.phash(self.test_image)
        hash2 = PerceptualHasher.phash(self.identical_image)
        self.assertEqual(hash1, hash2)

    def test_dhash_length(self):
        """Test that dHash produces correct length output."""
        hash_result = PerceptualHasher.dhash(self.test_image)
        # Default hash_size=8 should produce 8*8=64 bit hash
        self.assertEqual(len(hash_result), 64)

    def test_phash_length(self):
        """Test that pHash produces correct length output."""
        hash_result = PerceptualHasher.phash(self.test_image)
        # Default hash_size=8 should produce 8*8=64 bit hash
        self.assertEqual(len(hash_result), 64)

    def test_dhash_different_sizes(self):
        """Test dHash with different hash sizes."""
        for size in [4, 8, 16]:
            hash_result = PerceptualHasher.dhash(self.test_image, hash_size=size)
            expected_length = size * size
            self.assertEqual(len(hash_result), expected_length)

    def test_phash_different_sizes(self):
        """Test pHash with different hash sizes."""
        for size in [4, 8, 16]:
            hash_result = PerceptualHasher.phash(self.test_image, hash_size=size)
            expected_length = size * size
            self.assertEqual(len(hash_result), expected_length)

    def test_hamming_distance_identical(self):
        """Test Hamming distance for identical hashes."""
        hash1 = PerceptualHasher.dhash(self.test_image)
        hash2 = PerceptualHasher.dhash(self.identical_image)
        distance = PerceptualHasher.hamming_distance(hash1, hash2)
        self.assertEqual(distance, 0)

    def test_hamming_distance_different_lengths(self):
        """Test Hamming distance for hashes of different lengths."""
        hash1 = "1010"
        hash2 = "101010"
        distance = PerceptualHasher.hamming_distance(hash1, hash2)
        self.assertEqual(distance, 6)  # Should return max length

    def test_similarity_score_identical(self):
        """Test similarity score for identical hashes."""
        hash1 = PerceptualHasher.dhash(self.test_image)
        hash2 = PerceptualHasher.dhash(self.identical_image)
        score = PerceptualHasher.similarity_score(hash1, hash2)
        self.assertEqual(score, 1.0)

    def test_similarity_score_similar_images(self):
        """Test similarity score for similar images."""
        hash1 = PerceptualHasher.dhash(self.test_image)
        hash2 = PerceptualHasher.dhash(self.similar_image)
        score = PerceptualHasher.similarity_score(hash1, hash2)
        # Similar images should have high similarity
        self.assertGreater(score, 0.7)

    def test_similarity_score_different_images(self):
        """Test similarity score for different images."""
        hash1 = PerceptualHasher.dhash(self.test_image)
        hash2 = PerceptualHasher.dhash(self.different_image)
        score = PerceptualHasher.similarity_score(hash1, hash2)
        # Different images should have low similarity
        self.assertLess(score, 0.6)

    def test_hash_binary_format(self):
        """Test that hashes contain only binary digits."""
        dhash = PerceptualHasher.dhash(self.test_image)
        phash = PerceptualHasher.phash(self.test_image)

        # Check that all characters are 0 or 1
        self.assertTrue(all(c in '01' for c in dhash))
        self.assertTrue(all(c in '01' for c in phash))

    def test_robustness_to_noise(self):
        """Test hash robustness to small amounts of noise."""
        # Add small amount of noise
        noise = np.random.normal(0, 5, self.test_image.shape).astype(np.int16)
        noisy_image = np.clip(self.test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        hash1 = PerceptualHasher.phash(self.test_image)
        hash2 = PerceptualHasher.phash(noisy_image)

        score = PerceptualHasher.similarity_score(hash1, hash2)
        # Should still be quite similar despite noise
        self.assertGreater(score, 0.8)

    def test_different_hash_algorithms(self):
        """Test that dHash and pHash produce different results."""
        dhash = PerceptualHasher.dhash(self.test_image)
        phash = PerceptualHasher.phash(self.test_image)

        # They should be different (very unlikely to be identical)
        self.assertNotEqual(dhash, phash)
