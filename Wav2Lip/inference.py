import os
import sys
import subprocess
import platform
import argparse
import numpy as np
import cv2
import audio
import torch
from tqdm import tqdm
import face_detection
from models import Wav2Lip

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--face', type=str, required=True)
parser.add_argument('--audio', type=str, required=True)
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, default=False)
parser.add_argument('--fps', type=float, default=25.)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
parser.add_argument('--face_det_batch_size', type=int, default=16)
parser.add_argument('--wav2lip_batch_size', type=int, default=128)
parser.add_argument('--resize_factor', default=1, type=int)
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
parser.add_argument('--rotate', default=False, action='store_true')
parser.add_argument('--nosmooth', default=False, action='store_true')

args = parser.parse_args()
args.img_size = 96
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
    args.static = True

mel_step_size = 16


def ensure_temp_dir():
    if not os.path.exists("temp"):
        os.makedirs("temp")


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i:i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    batch_size = args.face_det_batch_size

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('OOM error during face detection.')
            batch_size //= 2
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected!')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        face_det_results = face_detect(frames if not args.static else [frames[0]])
    else:
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            yield _prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        yield _prepare_batches(img_batch, mel_batch, frame_batch, coords_batch)


def _prepare_batches(img_batch, mel_batch, frame_batch, coords_batch):
    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
    img_masked = img_batch.copy()
    img_masked[:, args.img_size // 2:] = 0
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    return img_batch, mel_batch, frame_batch, coords_batch


def load_model(path):
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main():
    ensure_temp_dir()

    if not os.path.isfile(args.face):
        raise ValueError('--face must be a valid path to a video/image file')

    if args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            y2 = y2 if y2 != -1 else frame.shape[0]
            x2 = x2 if x2 != -1 else frame.shape[1]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    if not args.audio.endswith('.wav'):
        print("Converting audio to WAV...")
        args.audio = 'temp/temp.wav'
        command = f'ffmpeg -y -i "{parser.parse_args().audio}" -strict -2 "{args.audio}"'
        subprocess.call(command, shell=True)

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    mel_chunks = []

    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[:len(mel_chunks)]
    model = load_model(args.checkpoint_path)

    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps,
                          (full_frames[0].shape[1], full_frames[0].shape[0]))

    for img_batch, mel_batch, frames, coords in tqdm(datagen(full_frames.copy(), mel_chunks),
                                                      total=int(np.ceil(len(mel_chunks) / args.wav2lip_batch_size))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch).cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            f[y1:y2, x1:x2] = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            out.write(f)

    out.release()

    command = f'ffmpeg -y -i "{args.audio}" -i "temp/result.avi" -strict -2 -q:v 1 "{args.outfile}"'
    subprocess.call(command, shell=True)
    print("Finished! Output saved to:", args.outfile)


if __name__ == '__main__':
    main()
