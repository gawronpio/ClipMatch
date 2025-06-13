# ClipMatch

ClipMatch is a Python library for finding similar video clips using perceptual hashing techniques. It can detect 
similarities between videos even when they have different formats, resolutions, bitrates, or are mirrored versions of 
each other.

## Features

- **Perceptual Hashing**: Uses advanced perceptual hashing algorithms (dHash and pHash) to identify similar video content
- **Format Agnostic**: Works with various video formats (.mp4, .avi, .mov, .mkv, .webm, etc.)
- **Resolution Independent**: Detects similarities regardless of video resolution
- **Fragment Detection**: Can identify when one video is a fragment of another
- **Multiprocessing Support**: Utilizes multiple CPU cores for faster processing
- **Configurable Thresholds**: Adjustable similarity thresholds for different use cases
- **Comprehensive Output**: Provides detailed similarity metrics

## Installation

```bash
pip install git+https://github.com/gawronpio/ClipMatch.git
```

## Usage

```python
from clipmatch import ClipMatch

# Initialize ClipMatch
clip_match = ClipMatch(
    directory='/path/to/videos',
    recursive=True,
    similarity_threshold=0.85,
    sample_rate=1.0,
    n_processes=4
)

# Run analysis
results = clip_match.run()

# Process results
for result in results:
    if result.is_similar:
        print(f"Similar videos found:")
        print(f"  File 1: {result.file1}")
        print(f"  File 2: {result.file2}")
        print(f"  Similarity Score: {result.similarity_score:.3f}")
        print(f"  Frame Matches: {result.frame_matches}")
```

## Algorithm Details

### Perceptual Hashing

ClipMatch uses two complementary perceptual hashing algorithms:

1. **dHash (Difference Hash)**: Compares adjacent pixels to create a hash based on gradients
2. **pHash (Perceptual Hash)**: Uses Discrete Cosine Transform (DCT) to focus on low-frequency components

### Video Analysis Process

1. **Frame Sampling**: Extracts frames at specified intervals (configurable sample rate)
2. **Hash Generation**: Computes perceptual hashes for each sampled frame
3. **Temporal Analysis**: Creates a temporal hash representing the video's overall structure
4. **Similarity Comparison**: Compares videos using both frame-level and temporal similarities

### Similarity Metrics

- **Frame Similarity**: Average similarity of best-matching frames between videos
- **Temporal Similarity**: Similarity of overall video structure and timing
- **Combined Score**: Weighted combination of frame and temporal similarities (70% frame, 30% temporal)

## Limitations

- **Processing Time**: Large video files and collections require significant processing time
- **Codec Support**: Limited by OpenCV's codec support
- **Accuracy**: Very short videos (< 1 second) may not provide reliable similarity detection
- **Memory**: Very large video collections may require batch processing

## License

MIT License - see LICENSE file for details.
