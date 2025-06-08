import json
import os
from itertools import combinations
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import cv2
import numpy as np
import psutil
import xxhash


class ClipMatch:
    """
    A class for finding similar video clips by comparing frame hashes.

    This class processes video files to generate perceptual hashes for each frame,
    then compares these hashes between videos to identify similar content using
    the Longest Common Subsequence (LCS) algorithm.

    :param directory: Path to the directory containing video files
    :type directory: str
    :param recursive: Whether to search for videos recursively in subdirectories
    :type recursive: bool
    :param resolution: Target resolution for frame resizing (width and height)
    :type resolution: int
    :param color_bits: Number of bits to use for color quantization (reduces color space)
    :type color_bits: int
    :param videos_extensions: Tuple of supported video file extensions
    :type videos_extensions: tuple
    :param n_processes: Number of processes to use for parallel processing (0 = auto)
    :type n_processes: int
    :param percent_mem_to_use: Percentage of available memory to use for batch processing
    :type percent_mem_to_use: float
    """

    def __init__(
            self,
            directory: str,
            recursive: bool = False,
            resolution: int = 256,
            color_bits: int = 8,
            videos_extensions: tuple = ('.avi.', '.mp4', '.mov', '.wmv', '.flv', '.mkv', '.webm'),
            n_processes: int = 0,
            percent_mem_to_use: float = 0.5,
    ):
        self.directory = directory
        self.recursive = recursive
        self.resolution = resolution
        self.color_bits = color_bits
        self.videos_extensions = videos_extensions
        self.n_processes = n_processes
        if n_processes is None or n_processes == 0 or n_processes > cpu_count():
            self.n_processes = cpu_count() - 1 if cpu_count() > 1 else 1
        self.percent_mem_to_use = percent_mem_to_use

        self.files = None

    def run(self):
        """
        Execute the complete video similarity analysis workflow.

        This method orchestrates the entire process:
        1. Finds all video files in the specified directory
        2. Processes each video to generate frame hashes
        3. Compares all video pairs to calculate similarity scores
        4. Outputs the results as JSON

        The method prints progress information during execution and outputs
        the final similarity results as a JSON string to stdout.

        :returns: None (prints results to stdout)
        :rtype: None
        """
        self._find_videos(self.directory)
        files_count = len(self.files)
        digits = len(str(len(self.files)))

        hashes = {}
        print('Processing video files:')
        for i, video_path in enumerate(self.files):
            line_text = f"{i + 1:>{digits}}/{files_count} - {video_path}"
            print(line_text, end='')
            hash_vals = self._process_video(video_path)
            hashes[video_path] = hash_vals
            print(' ' * len(line_text), end='\r')
        print('Videos processing complete.')
        similarity = self._compare_dictionary_lists(hashes)
        similarity_json = json.dumps(similarity)
        print(similarity_json)

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

    def _get_optimal_batch_size(
            self,
            width: int,
            height: int,
    ) -> int:
        """
        Calculate the optimal batch size for video frame processing.

        Determines the maximum number of frames that can be processed in parallel
        based on available system memory, frame dimensions, and the configured
        memory usage percentage. This helps prevent memory overflow during
        batch processing.

        :param width: Width of video frames in pixels
        :type width: int
        :param height: Height of video frames in pixels
        :type height: int
        :returns: Optimal number of frames to process in a single batch
        :rtype: int
        """
        frame_size = width * height * 3
        frame_batch_size = frame_size * self.n_processes
        available_memory = psutil.virtual_memory().available
        safe_memory = available_memory * self.percent_mem_to_use
        optimal_batch_size = int(safe_memory / frame_batch_size)
        return optimal_batch_size

    def _load_video_batch(
            self,
            video_path: str,
            start_frame: int,
            batch_size: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Load a batch of consecutive frames from a video file.

        Opens the specified video file and reads a batch of frames starting
        from the given frame index. Each frame is paired with its index
        for tracking purposes.

        :param video_path: Path to the video file to read from
        :type video_path: str
        :param start_frame: Index of the first frame to read
        :type start_frame: int
        :param batch_size: Maximum number of frames to read in this batch
        :type batch_size: int
        :returns: List of tuples containing (frame_index, frame_data)
        :rtype: List[Tuple[int, np.ndarray]]
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        batch_data = []
        frame_idx = start_frame

        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            batch_data.append((frame_idx, frame))
            frame_idx += 1

        cap.release()
        return batch_data

    def _process_single_frame_with_index(
            self,
            frame_data: Tuple[int, np.ndarray],
    ) -> Tuple[int, int]:
        """
        Process a single video frame to generate its perceptual hash.

        Takes a frame and its index, resizes it to the target resolution,
        applies color quantization to reduce the color space, and generates
        a hash using xxHash algorithm. This creates a perceptual fingerprint
        of the frame that can be used for similarity comparison.

        :param frame_data: Tuple containing (frame_index, frame_array)
        :type frame_data: Tuple[int, np.ndarray]
        :returns: Tuple containing (frame_index, frame_hash)
        :rtype: Tuple[int, int]
        """
        frame_idx, frame = frame_data

        frame_resized = cv2.resize(frame, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        frame_reduced = (frame_resized // self.color_bits) * self.color_bits

        frame_bytes = frame_reduced.tobytes()
        frame_hash = xxhash.xxh64(frame_bytes).intdigest()

        return frame_idx, frame_hash

    def _process_single_batch(self, batch_info_with_video: Tuple[str, int, int]) -> List[Tuple[int, int]]:
        """
        Process a single batch of video frames to generate hashes.

        Loads a batch of frames from the specified video and processes each
        frame to generate its hash. This method is designed to be used in
        parallel processing scenarios.

        :param batch_info_with_video: Tuple containing (video_path, start_frame, batch_size)
        :type batch_info_with_video: Tuple[str, int, int]
        :returns: List of tuples containing (frame_index, frame_hash) sorted by frame index
        :rtype: List[Tuple[int, int]]
        """
        video_path, start_frame, batch_size = batch_info_with_video
        batch_data = self._load_video_batch(video_path, start_frame, batch_size)
        results = []
        for frame_data in batch_data:
            results.append(self._process_single_frame_with_index(frame_data))
        results.sort(key=lambda x: x[0])
        return results

    def _process_multiple_batches_parallel(
            self,
            video_path: str,
            batches_info: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Process multiple batches of video frames in parallel.

        Coordinates parallel processing of multiple frame batches from a video
        using multiprocessing. Each batch is processed independently and the
        results are combined and sorted by frame index.

        :param video_path: Path to the video file being processed
        :type video_path: str
        :param batches_info: List of tuples containing (start_frame, batch_size) for each batch
        :type batches_info: List[Tuple[int, int]]
        :returns: List of tuples containing (frame_index, frame_hash) sorted by frame index
        :rtype: List[Tuple[int, int]]
        """
        batches_with_path = [(video_path, start_frame, batch_size) for start_frame, batch_size in batches_info]

        with Pool(self.n_processes) as pool:
            batch_results = pool.map(self._process_single_batch, batches_with_path)

        all_results = []
        for batch in batch_results:
            all_results.extend(batch)

        return sorted(all_results, key=lambda x: x[0])

    def _process_video(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Process an entire video file to generate frame hashes.

        Opens the video file, determines its properties (frame count, dimensions),
        calculates optimal batch size, divides the video into batches, and
        processes all frames in parallel to generate perceptual hashes.

        :param video_path: Path to the video file to process
        :type video_path: str
        :returns: List of tuples containing (frame_index, frame_hash) for all frames
        :rtype: List[Tuple[int, int]]
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        batch_size = self._get_optimal_batch_size(width, height)
        additional_text = f' - batch size: {batch_size}'
        print(additional_text, end='')

        batches_info = []
        for start_frame in range(0, total_frames, batch_size):
            actual_batch_size = min(batch_size, total_frames - start_frame)
            batches_info.append((start_frame, actual_batch_size))

        hashes = self._process_multiple_batches_parallel(video_path, batches_info)

        print(' ' * len(additional_text), end='\r')

        return hashes

    @staticmethod
    def _sequence_similarity(seq1: List, seq2: List) -> float:
        """
        Calculate similarity between two sequences using Longest Common Subsequence (LCS).

        Compares two sequences of frame hashes to determine their similarity using
        the LCS algorithm. The similarity score is normalized by the length of the
        longer sequence and includes a bonus for sequences with similar lengths.

        :param seq1: First sequence of frame hashes (can be tuples or plain values)
        :type seq1: List
        :param seq2: Second sequence of frame hashes (can be tuples or plain values)
        :type seq2: List
        :returns: Similarity score between 0.0 and 1.0 (1.0 = identical)
        :rtype: float
        """
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        hashes1 = [item[1] if isinstance(item, tuple) else item for item in seq1]
        hashes2 = [item[1] if isinstance(item, tuple) else item for item in seq2]

        # LCS (Longest Common Subsequence) calculation
        m, n = len(hashes1), len(hashes2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hashes1[i - 1] == hashes2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_length = dp[m][n]

        max_length = max(len(hashes1), len(hashes2))
        similarity = lcs_length / max_length if max_length > 0 else 0.0

        # Additional bonus for sequences with similar lengths
        length_ratio = min(len(hashes1), len(hashes2)) / max(len(hashes1), len(hashes2))
        similarity = similarity * (0.5 + 0.5 * length_ratio)

        return similarity

    @staticmethod
    def _compare_pairs_chunk(args: Tuple[List[Tuple[str, str]], dict]) -> List[Tuple[str, str, float]]:
        """
        Compare similarity for a chunk of video pairs.

        Processes a subset of video pairs to calculate their similarity scores.
        This method is designed for parallel processing to distribute the
        computational load across multiple processes.

        :param args: Tuple containing (pairs_chunk, data_dict) where pairs_chunk
                    is a list of video path pairs and data_dict contains frame hashes
        :type args: Tuple[List[Tuple[str, str]], dict]
        :returns: List of tuples containing (video1_path, video2_path, similarity_score)
        :rtype: List[Tuple[str, str, float]]
        """
        pairs_chunk, data_dict = args
        results = []

        for key1, key2 in pairs_chunk:
            list1 = data_dict[key1]
            list2 = data_dict[key2]
            similarity = ClipMatch._sequence_similarity(list1, list2)
            results.append((key1, key2, similarity))

        return results

    def _compare_dictionary_lists(self, data_dict: dict) -> List[Tuple[str, str, float]]:
        """
        Compare all video pairs to calculate similarity scores.

        Generates all possible pairs of videos and compares their frame hash
        sequences to calculate similarity scores. Uses parallel processing
        to distribute the computational load across multiple processes.

        :param data_dict: Dictionary mapping video paths to their frame hash sequences
        :type data_dict: dict
        :returns: List of tuples containing (video1_path, video2_path, similarity_score)
                 sorted by similarity score in descending order
        :rtype: List[Tuple[str, str, float]]
        """
        all_pairs = list(combinations(data_dict.keys(), 2))
        total_pairs = len(all_pairs)

        if total_pairs == 0:
            return []

        chunk_size = max(1, total_pairs // (self.n_processes * 4))  # 4x processes for better load balancing

        pair_chunks = []
        for i in range(0, total_pairs, chunk_size):
            chunk = all_pairs[i:i + chunk_size]
            pair_chunks.append((chunk, data_dict))

        print(f'Comparing {total_pairs} pairs using {self.n_processes} processes...')

        with Pool(self.n_processes) as pool:
            chunk_results = pool.map(self._compare_pairs_chunk, pair_chunks)

        similarities = []
        for chunk_result in chunk_results:
            similarities.extend(chunk_result)

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)

        return similarities
