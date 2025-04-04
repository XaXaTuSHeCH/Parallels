import argparse
import time
from queue import Queue
from threading import Thread

import cv2
from ultralytics import YOLO


class PoseModel:
    def __init__(self):
        self.model = YOLO("yolov8s-pose.pt")
        print("model loaded")

    def __del__(self):
        print("model deleted")

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        return results[0].plot()


def worker(input_q, output_q, model):
    while True:
        item = input_q.get()
        if item is None:
            break
        idx, frame = item
        processed = model.predict(frame)
        output_q.put((idx, processed))


def process_video_single(frames, model):
    output = []
    for idx, frame in frames:
        processed = model.predict(frame)
        output.append((idx, processed))
    return output


def process_video_threaded(frames, model, num_workers=4):
    input_q = Queue()
    output_q = Queue()
    for item in frames:
        input_q.put(item)
    threads = []
    for _ in range(num_workers):
        t = Thread(target=worker, args=(input_q, output_q, model))
        t.start()
        threads.append(t)
    for _ in range(num_workers):
        input_q.put(None)
    result = {}
    for _ in range(len(frames)):
        idx, processed = output_q.get()
        result[idx] = processed
    for t in threads:
        t.join()
    return [(i, result[i]) for i in sorted(result)]


def read_video_frames(video_path, resize=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames, fps


def write_video(frames, output_path, fps, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for _, frame in frames:
        out.write(frame)
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, required=True, help="Путь к входному видео (640x480)"
    )
    parser.add_argument(
        "--mode", choices=["single", "thread"], required=True, help="Режим обработки"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Путь к выходному видео"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Количество потоков (только для 'thread')",
    )
    args = parser.parse_args()
    frames, fps = read_video_frames(args.video)
    model = PoseModel()
    start = time.time()
    if args.mode == "single":
        processed = process_video_single(frames, model)
    elif args.mode == "thread":
        processed = process_video_threaded(frames, model, args.num_workers)
    else:
        raise ValueError("incorrect mode")
    end = time.time()
    duration = end - start
    print(f"{duration:.2f} sec")
    print(f"saved to {args.output}")
    write_video(processed, args.output, fps)


if __name__ == "__main__":
    main()
