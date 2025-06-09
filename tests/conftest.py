"""
Copyright (c) 2025 Piotr Gawron (dev@gawron.biz)
This file is licensed under the MIT License.
For details, see the LICENSE file in the project root.
"""

import os
import shutil
import tempfile

import pytest


@pytest.fixture
def test_setup():
    """Set up test fixtures."""
    test_dir = tempfile.mkdtemp()
    video_files = [
        'test1.mp4',
        'test2.avi',
        'test3.mov',
        'not_video.txt',
        'test4.mkv'
    ]

    for filename in video_files:
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write('dummy content')

    sub_dir = os.path.join(test_dir, 'subdir')
    os.makedirs(sub_dir)

    sub_video_files = ['sub1.mp4', 'sub2.webm']
    for filename in sub_video_files:
        filepath = os.path.join(sub_dir, filename)
        with open(filepath, 'w') as f:
            f.write('dummy content')

    yield {
        'test_dir': test_dir,
        'sub_dir': sub_dir,
        'video_files': video_files,
        'sub_video_files': sub_video_files
    }

    shutil.rmtree(test_dir)
