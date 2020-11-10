from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import decord.bridge
import io
import logging
import numpy as np
import os
import queue
import shutil
import sys
import threading
import traceback
import time
import urllib3

import u2net
import video

logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)


# Simple probe.
@app.route('/', methods=['GET'])
def hello():
    print('Hello U^2-Net! from print')
    return 'Hello U^2-Net!'


# Route http posts to this method
@app.route('/image', methods=['POST'])
def run():
    start = time.time()

    # Convert string of image data to uint8
    if 'data' not in request.files:
        return jsonify({'error': 'missing file param `data`'}), 400
    data = request.files['data'].read()
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    # Convert string data to PIL Image
    print('Convert string data to PIL Image')
    img = Image.open(io.BytesIO(data))

    # Ensure i,qge size is under 1024
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024, 1024))

    # Process Image
    print('Process Image')
    res = u2net.run(np.array(img))

    # Convert to PIL Image
    print('Convert to PIL Image')
    im = Image.fromarray(res * 255).convert('RGB')

    # Save to buffer
    print('Save to buffer')
    buff = io.BytesIO()
    res.save(buff, 'PNG')
    buff.seek(0)

    # Print stats
    logging.info(f'Completed in {time.time() - start:.2f}s')

    # Return data
    return send_file(buff, mimetype='image/png')

http = None
video_queue = queue.Queue(32)

@app.route('/startSegment', methods=['POST'])
def start_segment():
    data = request.get_json()
    logging.info('start_segment {}'.format(data))
    if not data:
        return jsonify({'error': 'invalid request'}), 400
    video_queue.put(data)
    return jsonify({}), 200

def video_task(data):
    logging.info('video_task started')
    global http, video_queue

    # temporary files only. TODO allow downloading, uploading, decoding, encoding, etc
    # to be parallelised and only serialise the GPU task
    # that will need randomised filenames and for the files to be deleted after processing
    original_filename = 'video3.mp4'
    no_sound_filename = 'no_sound.mp4'
    mask_filename = 'mask.mp4'
    output_filename = 'output3.mp4'
    thumbnail_filename = 'chromakey_thumbnail.jpg'

    src_signed_url = data['srcSignedUrl'][0]
    logging.info('video_task src_signed_url {}'.format(src_signed_url))
    dst_signed_url = data['dstSignedUrl'][0]
    logging.info('video_task dst_signed_url {}'.format(dst_signed_url))
    dst_mask_signed_url = data['dstMaskSignedUrl'][0]
    logging.info('video_task dst_mask_signed_url {}'.format(dst_mask_signed_url))
    dst_thumbnail_signed_url = data['dstThumbnailSignedUrl'][0]
    logging.info('video_task dst_thumbnail_signed_url {}'.format(dst_thumbnail_signed_url))

    with http.request('GET', src_signed_url, preload_content=False) as r, open(original_filename, 'wb') as f:
        logging.info('video_task get status {}'.format(r.status))
        shutil.copyfileobj(r, f)

    logging.info('video_task process_frames')
    video.process_frames(original_filename, no_sound_filename, mask_filename, thumbnail_filename)
    logging.info('video_task insert_audio')
    video.insert_audio(original_filename, no_sound_filename, output_filename)

    with open(output_filename, 'rb') as f:
        logging.info('video_task put')
        r = http.request('PUT', dst_signed_url, headers={'Content-Type': 'video/mp4'}, body=f)
        logging.info('video_task put status {}'.format(r.status))
        assert r.status == 200

    with open(mask_filename, 'rb') as f:
        logging.info('video_task put mask')
        r = http.request('PUT', dst_mask_signed_url, headers={'Content-Type': 'video/mp4'}, body=f)
        logging.info('video_task put mask status {}'.format(r.status))
        assert r.status == 200

    with open(thumbnail_filename, 'rb') as f:
        logging.info('video_task put thumbnail')
        r = http.request('PUT', dst_thumbnail_signed_url, headers={'Content-Type': 'video/mp4'}, body=f)
        logging.info('video_task put thumbnail status {}'.format(r.status))
        assert r.status == 200

    logging.info('video_task finished')

def video_loop():
    global http, video_queue
    http = urllib3.PoolManager()
    # set up thread local variable
    # https://github.com/dmlc/decord/blob/e5ab942cf30fb44e03d048dced415c76a2c6b220/python/decord/bridge/__init__.py#L19
    # `_CURRENT_BRIDGE.type = 'native'` doesn't get run except for the main thread
    # this has been fixed on master, but not yet released. TODO in the future we can upgrade decord to avoid this
    decord.bridge.reset_bridge()
    while True:
        i = video_queue.get()
        try:
            video_task(i)
        except:
            # just print the error without terminating the thread
            traceback.print_exc()
        finally:
            video_queue.task_done()

threading.Thread(target=video_loop, daemon=True).start()

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
