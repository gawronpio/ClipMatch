"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.

Unit tests for ClipMatch class.
"""
# pylint: disable=protected-access

import os
import shutil
import tempfile
from unittest.mock import patch

from clipmatch import ClipMatch, SimilarityResult, VideoHash


class TestClipMatchUnit:
    """Unit test cases for ClipMatch class."""

    def test_init_default_parameters(self, test_setup):
        """Test ClipMatch initialization with default parameters."""
        clip_match = ClipMatch(test_setup['test_dir'])

        assert clip_match.directory == test_setup['test_dir']
        assert clip_match.recursive is False
        assert clip_match.similarity_threshold == 0.85
        assert clip_match.sample_rate == 1.0
        assert clip_match.files is None
        assert len(clip_match.video_hashes) == 0

    def test_init_custom_parameters(self, test_setup):
        """Test ClipMatch initialization with custom parameters."""
        clip_match = ClipMatch(
            directory=test_setup['test_dir'],
            recursive=True,
            similarity_threshold=0.9,
            sample_rate=2.0,
            n_processes=2
        )

        assert clip_match.directory == test_setup['test_dir']
        assert clip_match.recursive is True
        assert clip_match.similarity_threshold == 0.9
        assert clip_match.sample_rate == 2.0
        assert clip_match.n_processes == 2

    def test_find_videos_non_recursive(self, test_setup):
        """Test finding videos in non-recursive mode."""
        clip_match = ClipMatch(test_setup['test_dir'], recursive=False)
        clip_match._find_videos(test_setup['test_dir'])

        # Should find 4 video files (excluding .txt file)
        expected_videos = {'test1.mp4', 'test2.avi', 'test3.mov', 'test4.mkv'}
        found_videos = {os.path.basename(f) for f in clip_match.files}

        assert found_videos == expected_videos

    def test_find_videos_recursive(self, test_setup):
        """Test finding videos in recursive mode."""
        clip_match = ClipMatch(test_setup['test_dir'], recursive=True)
        clip_match._find_videos(test_setup['test_dir'])

        # Should find 6 video files (4 in main dir + 2 in subdir)
        expected_videos = {
            'test1.mp4', 'test2.avi', 'test3.mov', 'test4.mkv',
            'sub1.mp4', 'sub2.webm'
        }
        found_videos = {os.path.basename(f) for f in clip_match.files}

        assert found_videos == expected_videos

    def test_find_videos_empty_directory(self):
        """Test finding videos in empty directory."""
        empty_dir = tempfile.mkdtemp()

        try:
            clip_match = ClipMatch(empty_dir)
            clip_match._find_videos(empty_dir)

            assert len(clip_match.files) == 0
        finally:
            shutil.rmtree(empty_dir)

    def test_compare_video_pair(self, test_setup):
        """Test comparing two video hashes."""
        video1 = VideoHash(
            'test1.mp4', 10.0, 300, 30.0, (1920, 1080),
            ['1010101010101010', '1111000011110000'],
            'temporal1'
        )
        video2 = VideoHash(
            'test2.avi', 15.0, 450, 30.0, (1280, 720),
            ['1010101010101010', '1111000011110000'],  # Same hashes for high similarity
            'temporal1'  # Same temporal hash
        )

        clip_match = ClipMatch(test_setup['test_dir'], similarity_threshold=0.8)
        result = clip_match._compare_video_pair(video1, video2)

        assert isinstance(result, SimilarityResult)
        assert result.file1 == 'test1.mp4'
        assert result.file2 == 'test2.avi'
        assert result.similarity_score > 0.99
        assert result.is_similar is True

    def test_compare_video_pair_different(self, test_setup):
        """Test comparing two different video hashes."""
        video1 = VideoHash(
            'test1.mp4', 10.0, 300, 30.0, (1920, 1080),
            ['1010101010101010', '1111000011110000'],
            'temporal1'
        )
        video2 = VideoHash(
            'test2.avi', 15.0, 450, 30.0, (1280, 720),
            ['0101010101010101', '0000111100001111'],  # Different hashes
            'completely_different_temporal_hash'  # Very different temporal hash
        )

        clip_match = ClipMatch(test_setup['test_dir'], similarity_threshold=0.9)  # Higher threshold
        result = clip_match._compare_video_pair(video1, video2)

        assert isinstance(result, SimilarityResult)
        assert result.similarity_score < 0.5
        assert result.is_similar is False

    def test_compare_video_without_hashes(self, test_setup):
        """Test comparing videos without hashes."""
        clip_match = ClipMatch(test_setup['test_dir'])
        clip_match.video_hashes = []
        result = clip_match._compare_videos()

        assert isinstance(result, list)
        assert len(result) == 0

    def test_compare_videos(self, test_setup):
        """Test comparing multiple videos."""
        clip_match = ClipMatch(test_setup['test_dir'])
        clip_match.video_hashes = [
            VideoHash('test1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1'),
            VideoHash('test2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2'),
            VideoHash('test3.mov', 20.0, 600, 30.0, (1920, 1080), ['hash3'], 'temporal3')
        ]

        results = clip_match._compare_videos()

        # Should have 3 comparisons for 3 videos: (1,2), (1,3), (2,3)
        assert len(results) == 3
        assert all(isinstance(r, SimilarityResult) for r in results)

    def test_compare_videos_chunk_empty_list(self, test_setup):
        """Test comparing empty chunk list."""
        chunk = []

        clip_match = ClipMatch(test_setup['test_dir'])
        results = clip_match._compare_videos_chunk(chunk)

        assert isinstance(results, list)
        assert len(results) == 0

    @patch('clipmatch.ClipMatch._compare_video_pair')
    def test_compare_videos_chunk_single_pair(self, mock_compare, test_setup):
        """Test comparing chunk list with one hash pair."""
        result_expected = SimilarityResult('video1.mp4', 'video2.avi', 0.9, 0.8, 5, True)
        mock_compare.return_value = result_expected
        hash1 = VideoHash('video1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1')
        hash2 = VideoHash('video2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2')
        chunk = [(hash1, hash2)]

        clip_match = ClipMatch(test_setup['test_dir'])
        results = clip_match._compare_videos_chunk(chunk)

        mock_compare.assert_called_once_with(hash1, hash2)
        assert isinstance(results, list)
        assert isinstance(results[0], SimilarityResult)
        assert len(results) == 1
        assert results[0] == result_expected

    @patch('clipmatch.ClipMatch._compare_video_pair')
    def test_compare_videos_chunk_multiple_pairs(self, mock_compare, test_setup):
        """Test comparing chunk list with multiple hash pairs."""
        result_expected = [
            SimilarityResult('test1.mp4', 'test2.avi', 0.9, 0.8, 5, True),
            SimilarityResult('test1.mp4', 'test3.mov', 0.8, 0.7, 4, True),
            SimilarityResult('test2.avi', 'test3.mov', 0.7, 0.6, 3, True)
        ]
        hash1 = VideoHash('test1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1')
        hash2 = VideoHash('test2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2')
        hash3 = VideoHash('test3.mov', 20.0, 600, 30.0, (1920, 1080), ['hash3'], 'temporal3')
        chunk = [
            (hash1, hash2),
            (hash1, hash3),
            (hash2, hash3),
        ]
        mock_compare.side_effect = result_expected

        clip_match = ClipMatch(test_setup['test_dir'])
        results = clip_match._compare_videos_chunk(chunk)

        mock_compare.assert_called()
        assert isinstance(results, list)
        assert len(results) == 3
        for i, res in enumerate(results):
            assert isinstance(res, SimilarityResult)
            assert res == result_expected[i]

    @patch('clipmatch.ClipMatch._process_videos_sequential')
    @patch('clipmatch.ClipMatch._compare_videos')
    def test_run_no_videos(self, mock_compare, mock_process, test_setup):
        """Test run method with no videos found."""
        clip_match = ClipMatch(test_setup['test_dir'])

        # Mock empty file list
        with patch.object(clip_match, '_find_videos') as mock_find:
            mock_find.return_value = None
            clip_match.files = []

            results = clip_match.run()

            assert len(results) == 0
            mock_process.assert_not_called()
            mock_compare.assert_not_called()

    @patch('clipmatch.ClipMatch._process_videos_sequential')
    @patch('clipmatch.ClipMatch._compare_videos')
    def test_run_with_videos(self, mock_compare, mock_process, test_setup):
        """Test run method with videos."""
        mock_compare.return_value = [
            SimilarityResult('test1.mp4', 'test2.avi', 0.9, 0.8, 5, True)
        ]

        # Force sequential processing by setting n_processes to 1
        clip_match = ClipMatch(test_setup['test_dir'], n_processes=1)
        clip_match.files = ['test1.mp4', 'test2.avi']
        clip_match.video_hashes = [
            VideoHash('test1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1'),
            VideoHash('test2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2')
        ]

        with patch('builtins.print'):  # Suppress print output
            results = clip_match.run()

        assert len(results) == 1
        mock_process.assert_called_once()
        mock_compare.assert_called_once()

    def test_process_video_worker_invalid_file(self):
        """Test video processing with invalid file."""
        clip_match = ClipMatch('/tmp')

        result = clip_match._process_video_worker('nonexistent_file.mp4')

        assert result is None

    def test_process_video_worker_empty_file(self):
        """Test video processing with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            clip_match = ClipMatch('/tmp')
            result = clip_match._process_video_worker(video_path)
            assert result is None
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch('clipmatch.ClipMatch._process_video_worker')
    @patch('builtins.ValueError')
    def test_process_videos_sequential_with_error(self, mock_value_error, mock_worker, test_setup):
        """Test sequential video processing with an error."""""
        mock_worker.side_effect = ValueError('Test error')

        clip_match = ClipMatch(test_setup['test_dir'], n_processes=1)
        clip_match.files = ['test1.mp4', 'test2.avi']

        with patch('builtins.print'):  # Suppress print output
            clip_match._process_videos_sequential()

        assert mock_worker.call_count == 2
        mock_value_error.assert_called()
