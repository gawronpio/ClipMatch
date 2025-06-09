"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.

Integration tests for ClipMatch class.
"""
# pylint: disable=protected-access

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from clipmatch import ClipMatch, VideoHash


class TestClipMatchIntegration:
    """Integration test cases for ClipMatch class."""

    def test_process_videos_sequential(self, test_setup):
        """Test sequential video processing."""
        clip_match = ClipMatch(test_setup['test_dir'])
        clip_match._find_videos(test_setup['test_dir'])

        with patch.object(clip_match, '_process_video_worker') as mock_worker:
            mock_worker.side_effect = [
                VideoHash('test1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1'),
                VideoHash('test2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2'),
                None,  # Simulate failed processing
                VideoHash('test4.mkv', 20.0, 600, 30.0, (1920, 1080), ['hash4'], 'temporal4')
            ]

            clip_match._process_videos_sequential()

            # Should have 3 successful video hashes (one failed)
            assert len(clip_match.video_hashes) == 3
            assert mock_worker.call_count == 4

    @patch('clipmatch.clipmatch.Pool')
    def test_process_videos_parallel(self, mock_pool_class, test_setup):
        """Test parallel video processing."""
        # Mock the pool and its methods
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool

        mock_pool.map.return_value = [
            VideoHash('test1.mp4', 10.0, 300, 30.0, (1920, 1080), ['hash1'], 'temporal1'),
            VideoHash('test2.avi', 15.0, 450, 30.0, (1280, 720), ['hash2'], 'temporal2'),
            None,  # Simulate failed processing
        ]

        clip_match = ClipMatch(test_setup['test_dir'], n_processes=2)
        clip_match._find_videos(test_setup['test_dir'])
        clip_match._process_videos_parallel()

        # Should have 2 successful video hashes
        assert len(clip_match.video_hashes) == 2
        mock_pool.map.assert_called_once()


class TestVideoProcessingIntegration:
    """Integration test cases for video processing functions."""

    def create_test_video(self, filename, duration=2, fps=30, width=640, height=480):
        """Create a test video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        total_frames = int(duration * fps)
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (i * 10 % width, i * 5 % height),
                          ((i * 10 + 50) % width, (i * 5 + 50) % height),
                          (255, 255, 255), -1)
            out.write(frame)

        out.release()

    def test_process_video_worker_success(self):
        """Test successful video processing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            self.create_test_video(video_path, duration=1, fps=10)

            clip_match = ClipMatch('/tmp', sample_rate=0.5)
            result = clip_match._process_video_worker(video_path)

            assert result is not None
            assert isinstance(result, VideoHash)
            assert result.file_path == video_path
            assert result.duration > 0
            assert result.frame_count > 0
            assert result.fps > 0
            assert len(result.perceptual_hashes) > 0
            assert result.temporal_hash is not None

        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
