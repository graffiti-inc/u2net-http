
import io
import os
import subprocess
import sys
import random

# for debugging
from PIL import Image

# decord for fast frame extraction
from decord import VideoReader
from decord import cpu, gpu

# cv2 to draw on frames and putting them together
import cv2
import numpy as np

# imageio_ffmpeg for handling audio
import imageio_ffmpeg

# import u2net
import u2net


def create_keying_background(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# processes frames an saves them into a mp4 video


def process_frames(original_filename, output_filename, mask_filename, thumbnail_filename):
    # change to gpu(0) for faster processing
    vr = VideoReader(original_filename, ctx=cpu(0))

    height, width, layers = vr[0].shape
    print(f'\u001b[33mInput frame {height}x{width}x{layers}\u001b[0m')

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    video = cv2.VideoWriter(output_filename + '.lossless.mkv', fourcc,
                            vr.get_avg_fps(), (width, height))
    video_mask = cv2.VideoWriter(mask_filename + '.lossless.mkv', fourcc,
                            vr.get_avg_fps(), (320, 320))

    # solid color image
    keying_bg = create_keying_background(width, height, (0, 255, 0))

    for frame in vr:

        # convert to numpy format
        frame_np = frame.asnumpy()

        # run u2net
        mask_np = u2net.run(frame_np)

        # write frame to mask video
        mask_np_uint8 = (mask_np * 255).astype(np.uint8)
        mask_np_bgr = np.stack([mask_np_uint8]*3, axis=-1) # https://stackoverflow.com/a/40119878
        video_mask.write(mask_np_bgr)

        # resize u2net output (320x320) to original frame resolution
        mask_cv2 = cv2.resize(mask_np, (width, height))

        # scale mask values from the range [0, 1] to [0, 255]
        mask_cv2_uint8 = (mask_cv2 * 255).astype(np.uint8)

        # thresholding the mask to have clear outlines
        ret, mask_cv2_uint8 = cv2.threshold(
            mask_cv2_uint8, 10, 255, cv2.THRESH_BINARY)

        # compute inverse mask
        mask_cv2_uint8_inv = cv2.bitwise_not(mask_cv2_uint8)

        # apply mask to image and merge with keying background
        frame_fg = cv2.bitwise_and(frame_np, frame_np, mask=mask_cv2_uint8)
        frame_bg = cv2.bitwise_and(
            keying_bg, keying_bg, mask=mask_cv2_uint8_inv)
        output_cv2 = frame_fg + frame_bg

        # convert the color space back to BGR
        output_cv2 = cv2.cvtColor(output_cv2, cv2.COLOR_RGB2BGR)

        video.write(output_cv2)

    cv2.destroyAllWindows()
    video.release()
    video_mask.release()

    # encode videos to h264
    thumbnail_proc = start_thumbnail(output_filename + '.lossless.mkv', thumbnail_filename)
    video_enc_proc = start_encode_video(output_filename + '.lossless.mkv', output_filename)
    video_mask_enc_proc = start_encode_video(mask_filename + '.lossless.mkv', mask_filename)
    assert thumbnail_proc.wait() == 0, 'Thumbnail encoding failed'
    assert video_enc_proc.wait() == 0, 'Video encoding failed'
    assert video_mask_enc_proc.wait() == 0, 'Mask video encoding failed'

def start_encode_video(src, dst):
    return subprocess.Popen([
        imageio_ffmpeg._utils.get_ffmpeg_exe(),
        '-y', # overwrite output
        '-i', src, # input
        '-codec', 'h264', # encode using x264
        '-pix_fmt', 'yuv420p', # 4:2:0 colour format for compatibility with unity
        '-crf', '20', # aims for consistent video quality of 20, using as much bitrate as needed
        dst
    ])

def start_thumbnail(src, dst):
    return subprocess.Popen([
        imageio_ffmpeg._utils.get_ffmpeg_exe(),
        '-y', # overwrite output
        '-i', src, # input
        '-vframes', '1', # only process 1 frame
        dst
    ])

# creates a video with src_video as a video source and src_audio as an audio source
def insert_audio(src_audio, src_video, dst):
    try:
        subprocess.check_call([
            imageio_ffmpeg._utils.get_ffmpeg_exe(),
            '-y', # overwrite output
            '-i', src_video, # video as input 0
            '-i', src_audio, # video with audio as input 1
            '-map', '0:v', # use video from input 0
            '-map', '1:a', # use audio from input 1
            '-codec', 'copy', # don't re-encode video
            '-acodec', 'copy', # don't re-encode audio
            dst
        ])
    except subprocess.CalledProcessError:
        # just make a video only file
        subprocess.check_call([
            imageio_ffmpeg._utils.get_ffmpeg_exe(),
            '-y', # overwrite output
            '-i', src_video, # video as input 0
            '-codec', 'copy', # don't re-encode video
            dst
        ])


if __name__ == '__main__':
    original_filename = 'video3.mp4'
    no_sound_filename = 'no_sound.mp4'
    mask_filename = 'mask.mp4'
    output_filename = 'output3.mp4'

    print(
        '\u001b[33mProcessing frames and putting frames back together...\u001b[0m')
    process_frames(original_filename, no_sound_filename, mask_filename)

    print('\u001b[33mInserting audio...\u001b[0m')
    insert_audio(original_filename, no_sound_filename, output_filename)
