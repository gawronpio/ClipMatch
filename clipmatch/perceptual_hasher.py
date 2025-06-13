"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.

PerceptualHasher class.
"""

import cv2
import numpy as np


class PerceptualHasher:
    """
    Perceptual hashing implementation for video frames.
    Uses DCT-based hashing similar to pHash algorithm.
    """

    @staticmethod
    def dhash(image: np.ndarray, hash_size: int = 8) -> str:
        """
        Calculate difference hash (dHash) for an image.

        :param image: Input image as numpy array
        :param hash_size: Size of the hash (default 8 for 64-bit hash)
        :return: Hash as binary string
        """
        resized = cv2.resize(image, (hash_size + 1, hash_size))

        diff = resized[:, 1:] > resized[:, :-1]

        return ''.join(['1' if x else '0' for x in diff.flatten()])

    @staticmethod
    def phash(image: np.ndarray, hash_size: int = 8) -> str:
        """
        Calculate perceptual hash (pHash) using DCT.

        :param image: Input image as numpy array
        :param hash_size: Size of the hash (default 8 for 64-bit hash)
        :return: Hash as binary string
        """
        img_size = hash_size * 4
        resized = cv2.resize(image, (img_size, img_size))
        resized = np.float32(resized)
        dct = cv2.dct(resized)
        dct_low = dct[0:hash_size, 0:hash_size]
        median = np.median(dct_low)
        diff = dct_low > median
        return ''.join(['1' if x else '0' for x in diff.flatten()])

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two binary hash strings."""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @staticmethod
    def similarity_score(hash1: str, hash2: str) -> float:
        """
        Calculate similarity score between two hashes (0.0 to 1.0).
        1.0 means identical, 0.0 means completely different.
        """
        distance = PerceptualHasher.hamming_distance(hash1, hash2)
        max_distance = max(len(hash1), len(hash2))
        return 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
