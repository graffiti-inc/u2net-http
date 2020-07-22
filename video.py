
import io
import os
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

# moviepy for handling audio
from moviepy.editor import *

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


def process_frames(original_filename, output_filename):
    # change to gpu(0) for faster processing
    vr = VideoReader(original_filename, ctx=cpu(0))

    height, width, layers = vr[0].shape
    print(f'\u001b[33mInput frame {height}x{width}x{layers}\u001b[0m')

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video = cv2.VideoWriter(output_filename, fourcc,
                            vr.get_avg_fps(), (width, height))

    # solid color image
    keying_bg = create_keying_background(width, height, (0, 255, 0))

    for frame in vr:

        # convert to numpy format
        frame_np = frame.asnumpy()

        # run u2net
        mask_np = u2net.run(frame_np)

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

# creates a video with src_video as a video source and src_audio as an audio source


def insert_audio(src_audio, src_video, dst):
    src_video = VideoFileClip(src_video)
    src_audio = VideoFileClip(src_audio)
    audio = src_audio.audio
    final_video = src_video.set_audio(audio)
    final_video.write_videofile(dst)


if __name__ == '__main__':
    original_filename = 'video3.mp4'
    no_sound_filename = 'no_sound.mp4'
    output_filename = 'output3.mp4'

    print(
        '\u001b[33mProcessing frames and putting frames back together...\u001b[0m')
    process_frames(original_filename, no_sound_filename)

    print('\u001b[33mInserting audio...\u001b[0m')
    insert_audio(original_filename, no_sound_filename, output_filename)
