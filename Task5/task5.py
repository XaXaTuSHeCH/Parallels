import argparse
import time
from queue import Empty, Queue
from threading import Thread

import cv2
from ultralytics import YOLO


class PoseModel:
    def __init__(self):
        self.model = YOLO("yolov8s-pose.pt")

    def __del__(self):
        del self.model

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        return results[0].plot()


def worker(inputs: Queue, outputs: Queue):
    model = PoseModel()
    while True:
        item = inputs.get()
        if item is None:
            inputs.task_done()
            break
        idx, frame = item
        processed = model.predict(frame)
        outputs.put((idx, processed))
        inputs.task_done()


def multithread_mode(frames, workers=4):
    inputs = Queue()
    outputs = Queue()

    threads = []
    for i in range(workers):
        t = Thread(target=worker, args=(inputs, outputs), daemon=True)
        t.start()
        threads.append(t)

    for item in frames:
        inputs.put(item)
    for _ in threads:
        inputs.put(None)

    inputs.join()

    result = {}
    for _ in frames:
        idx, processed = outputs.get()
        result[idx] = processed

    for t in threads:
        t.join()

    return [(i, result[i]) for i in sorted(result)]


def single_mode(frames):
    model = PoseModel()
    output = []
    for idx, frame in frames:
        output.append((idx, model.predict(frame)))
    return output


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames, fps


def write_video(frames, output_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    for _, frame in frames:
        out.write(frame)
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--video", type=str, required=True, help="Путь к входному видео"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.mp4", help="Путь к выходному видео"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4, help="Число потоков, 1 для single mode"
    )
    args = parser.parse_args()

    frames, fps = read_video(args.video)
    start = time.time()

    if args.threads == 1:
        processed = single_mode(frames)
    else:
        processed = multithread_mode(frames, args.threads)

    end = time.time()
    print(f"Processing time: {end - start:.3f} sec")
    write_video(processed, args.output, fps)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
