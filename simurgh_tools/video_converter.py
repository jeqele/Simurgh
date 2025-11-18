"""
Video to Audio Converter Tool

Converts MP4 video files to MP3 audio files.
"""

import os
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip


def convert_video_to_audio(video_path: str, output_path: str = None) -> str:
    """
    Convert a video file to MP3 audio.

    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path for the output audio file.
                                     If not provided, uses the same name as video with .mp3 extension

    Returns:
        str: Path to the output audio file, or error message if conversion fails
    """
    # Validate input file
    if not os.path.exists(video_path):
        return f"Error: Video file not found: {video_path}"

    # Generate output path if not provided
    if output_path is None:
        video_file = Path(video_path)
        output_path = str(video_file.with_suffix('.mp3'))

    try:
        # Load video file
        video = VideoFileClip(video_path)

        # Extract audio and write to MP3
        audio = video.audio
        audio.write_audiofile(output_path, bitrate='192k', logger=None)

        # Clean up
        audio.close()
        video.close()

        return f"Successfully converted {video_path} to {output_path}"
    except Exception as e:
        return f"Error during conversion: {str(e)}"

