"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.

ClipMatch main class.
"""

import hashlib
import os
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import cv2

from .perceptual_hasher import PerceptualHasher


@dataclass
class VideoHash:
    """Data class to store video hash information."""
    file_path: str
    duration: float
    frame_count: int
    fps: float
    resolution: Tuple[int, int]
    perceptual_hashes: List[str]
    temporal_hash: str


@dataclass
class SimilarityResult:
    """Data class to store similarity comparison results."""
    file1: str
    file2: str
    similarity_score: float
    temporal_similarity: float
    frame_matches: int
    is_similar: bool


class ClipMatch:  # pylint: disable=too-few-public-methods
    """
    A class for finding similar video clips using perceptual hashing.

    This class analyzes video files to detect similarities between them using
    perceptual hashing techniques. It can handle videos with different formats,
    resolutions, bitrates, and even mirrored versions while identifying their
    similarity relationships.

    :param directory: Path to the directory containing video files
    :type directory: str
    :param recursive: Whether to search for videos recursively in subdirectories
    :type recursive: bool
    :param videos_extensions: Tuple of supported video file extensions
    :type videos_extensions: tuple
    :param n_processes: Number of processes to use for parallel processing (0 = auto)
    :type n_processes: int
    :param similarity_threshold: Minimum similarity score to consider videos as similar (0.0-1.0)
    :type similarity_threshold: float
    :param sample_rate: How often to sample frames (every N seconds)
    :type sample_rate: float
    """

    def __init__(
            self,
            directory: str,
            recursive: bool = False,
            videos_extensions: tuple = ('.avi', '.mp4', '.mov', '.wmv', '.flv', '.mkv', '.webm'),
            n_processes: int = 0,
            similarity_threshold: float = 0.85,
            sample_rate: float = 1.0,
    ):
        self.directory = directory
        self.recursive = recursive
        self.videos_extensions = videos_extensions
        self.n_processes = n_processes
        if n_processes == 0 or n_processes > cpu_count():
            self.n_processes = cpu_count() - 1 if cpu_count() > 1 else 1
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate

        self.files = None
        self.video_hashes: List[VideoHash] = []

    def run(self) -> List[SimilarityResult]:
        """
        Execute the complete video similarity analysis workflow.

        This method orchestrates the entire process:
        1. Finds all video files in the specified directory
        2. Processes each video to extract perceptual hashes
        3. Compares all video pairs to find similarities
        4. Returns similarity results

        The method prints progress information during execution and outputs
        the final similarity results as a JSON string to stdout.

        :returns: List of similarity results
        :rtype: List[SimilarityResult]
        """
        self._find_videos(self.directory)

        if not self.files:
            print("No video files found.")
            return []

        files_count = len(self.files)

        print(f'Found {files_count} video files')
        print('Processing video files:')

        # Process videos to extract hashes
        if self.n_processes > 1:
            self._process_videos_parallel()
        else:
            self._process_videos_sequential()

        print(f'Processed {len(self.video_hashes)} videos successfully')

        print('Comparing videos for similarities...')
        similarities = self._compare_videos()

        similar_videos = [s for s in similarities if s.is_similar]
        similar_videos.sort(key=lambda x: x.similarity_score, reverse=True)

        print(f'Found {len(similar_videos)} similar video pairs')

        return similarities

    def _find_videos(self, directory: str):
        """
        Find all video files in the specified directory.

        Searches for video files with supported extensions either in the specified
        directory only or recursively through all subdirectories, depending on
        the recursive setting. Updates the self.files attribute with the list
        of found video file paths.

        :param directory: Path to the directory to search for video files
        :type directory: str
        :returns: None (updates self.files attribute)
        :rtype: None
        """
        if self.recursive:
            self.files = []
            for root, _, fs in os.walk(directory, followlinks=True):
                for file in fs:
                    file = os.path.join(root, file)
                    if os.path.isfile(file) and os.path.splitext(file)[1] in self.videos_extensions:
                        self.files.append(file)
        else:
            self.files = [
                os.path.abspath(entry.path)
                for entry in os.scandir(directory)
                if entry.is_file() and os.path.splitext(entry.name)[1] in self.videos_extensions
            ]

    def _process_videos_sequential(self):
        """Process videos sequentially to extract perceptual hashes."""
        for i, video_path in enumerate(self.files):
            print(f'Processing {i+1}/{len(self.files)}: {os.path.basename(video_path)}')
            try:
                video_hash = self._process_video_worker(video_path)
                if video_hash:
                    self.video_hashes.append(video_hash)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f'Error processing {video_path}: {str(e)}')

    def _process_videos_parallel(self):
        """Process videos in parallel using multiprocessing."""
        print(f'Using {self.n_processes} processes for parallel processing')

        results = []
        total = len(self.files)
        with Pool(processes=self.n_processes) as pool:
            for i, result in enumerate(pool.imap_unordered(self._process_video_worker, self.files), 1):
                if result is not None:
                    results.append(result)
                    print(f'\rProcessed {i} / {total} files.', end='')
        print()
        self.video_hashes = results

    def _compare_videos(self) -> List[SimilarityResult]:
        """
        Compare all video pairs to find similarities.

        :return: List of similarity results for all video pairs
        """
        all_pairs = list(combinations(self.video_hashes, 2))
        if len(all_pairs) == 0:
            return []

        chunk_size = max(1, len(all_pairs) // (self.n_processes * 4))  # 4x processes for better load balancing

        pair_chunks = []
        for i in range(0, len(all_pairs), chunk_size):
            chunk = all_pairs[i:i + chunk_size]
            pair_chunks.append(chunk)

        print(f'Comparing {len(all_pairs)} pairs using {self.n_processes} processes...')

        similarity = []
        total = len(pair_chunks) * chunk_size
        with Pool(self.n_processes) as pool:
            for i, chunk_result in enumerate(pool.imap_unordered(self._compare_videos_chunk, pair_chunks), 1):
                similarity.extend(chunk_result)
                print(f'\rCompared {i * chunk_size} / {total} hashes', end='')

        return similarity

    def _compare_videos_chunk(self, chunk: List[Tuple[VideoHash, VideoHash]]) -> List[SimilarityResult]:
        results = []
        for video1, video2 in chunk:
            similarity = self._compare_video_pair(video1, video2)
            results.append(similarity)

        return results

    def _compare_video_pair(self, video1: VideoHash, video2: VideoHash) -> SimilarityResult:
        """
        Compare two videos and calculate their similarity.

        :param video1: First video hash
        :param video2: Second video hash
        :return: Similarity result
        """
        temporal_similarity = PerceptualHasher.similarity_score(
            video1.temporal_hash, video2.temporal_hash
        )

        frame_matches = 0
        total_comparisons = 0
        similarity_scores = []

        # Compare each frame hash from video1 with all frame hashes from video2
        for hash1 in video1.perceptual_hashes:
            best_match = 0.0
            for hash2 in video2.perceptual_hashes:
                score = PerceptualHasher.similarity_score(hash1, hash2)
                best_match = max(best_match, score)
                total_comparisons += 1

            similarity_scores.append(best_match)
            if best_match > self.similarity_threshold:
                frame_matches += 1

        # Calculate overall similarity score
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
        else:
            avg_similarity = 0.0

        # Combine temporal and frame similarities
        overall_similarity = temporal_similarity * 0.3 + avg_similarity * 0.7

        # Determine if videos are similar
        is_similar = (
            overall_similarity > self.similarity_threshold or
            temporal_similarity > self.similarity_threshold or
            (frame_matches / len(video1.perceptual_hashes) > 0.5 if video1.perceptual_hashes else False)
        )

        return SimilarityResult(
            file1=video1.file_path,
            file2=video2.file_path,
            similarity_score=overall_similarity,
            temporal_similarity=temporal_similarity,
            frame_matches=frame_matches,
            is_similar=is_similar
        )

    def _process_video_worker(self, video_path: str) -> Optional[VideoHash]:
        """
        Worker function to process a single video file.

        :param video_path: Path to the video file
        :return: VideoHash object or None if processing failed
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f'Could not open video: {video_path}')
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if duration == 0:
                cap.release()
                return None

            # Calculate frame sampling interval
            frame_interval = max(1, int(fps * self.sample_rate))

            perceptual_hashes = []
            frame_hashes_for_temporal = []
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    dhash = PerceptualHasher.dhash(gray)
                    phash = PerceptualHasher.phash(gray)
                    combined_hash = dhash + phash
                    perceptual_hashes.append(combined_hash)
                    frame_hashes_for_temporal.append(dhash[:16])  # Use the first 16 bits of dhash

                frame_number += 1

            cap.release()

            if not perceptual_hashes:
                return None

            temporal_data = ''.join(frame_hashes_for_temporal)
            temporal_hash = hashlib.md5(temporal_data.encode()).hexdigest()

            return VideoHash(
                file_path=video_path,
                duration=duration,
                frame_count=frame_count,
                fps=fps,
                resolution=(width, height),
                perceptual_hashes=perceptual_hashes,
                temporal_hash=temporal_hash
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f'Error processing video {video_path}: {str(e)}')
            return None
