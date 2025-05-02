import argparse
import logging
import os
import threading
import time
from queue import Queue

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam:
    def __init__(self, camera_name: str, resolution: tuple):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = None

        try:
            if camera_name.isdigit():
                self.cap = cv2.VideoCapture(int(camera_name))
            else:
                self.cap = cv2.VideoCapture(camera_name)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {camera_name}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        except Exception as e:
            logging.error(f"Camera error: {str(e)}")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            raise

    def get(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                raise RuntimeError("Failed to grab frame")
            return frame
        except Exception as e:
            logging.error(f"Frame error: {str(e)}")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            raise

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


class WindowImage:
    def __init__(self, display_fps: float):
        self.display_freq = display_fps
        self.window_name = "SensorX"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, img, sensor1_val, sensor2_val, sensor3_val):
        try:
            if img is not None:
                display_img = img.copy()
                cv2.putText(
                    display_img,
                    f"Sensor 1 (100Hz): {sensor1_val}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display_img,
                    f"Sensor 2 (10Hz): {sensor2_val}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_img,
                    f"Sensor 3 (1Hz): {sensor3_val}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
                cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(int(1000 / self.display_freq)) & 0xFF
            return key != ord("q")
        except Exception as e:
            logging.error(f"Display error: {str(e)}")
            raise

    def __del__(self):
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass


os.makedirs("log", exist_ok=True)
logging.basicConfig(filename="log/app.log", level=logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=str, default="0")
parser.add_argument("--res", type=str, default="1920x1080")
parser.add_argument("--fps", type=float, default=60.0)
args = parser.parse_args()
try:
    width, height = map(int, args.res.split("x"))
    resolution = (width, height)
except:
    print("Invalid resolution. Using standard")
    resolution = (1920, 1080)

sensor1 = SensorX(0.01)
sensor2 = SensorX(0.1)
sensor3 = SensorX(1.0)

try:
    cam = SensorCam(args.cam, resolution)
except Exception as e:
    print(f"Camera init failed: {str(e)}")
    exit()

queues = [Queue() for _ in range(4)]


class SensorThread(threading.Thread):
    def __init__(self, sensor, queue, sensor_name):
        super().__init__()
        self.sensor = sensor
        self.queue = queue
        self.sensor_name = sensor_name
        self._stop = threading.Event()
        self.daemon = True

    def run(self):
        while not self._stop.is_set():
            try:
                data = self.sensor.get()
                self.queue.put((self.sensor_name, data))
            except Exception as e:
                logging.error(f"Sensor error: {str(e)}")
                break

    def stop(self):
        self._stop.set()


threads = []
for sensor, queue, name in zip(
        [sensor1, sensor2, sensor3, cam],
        queues,
        ["Sensor1", "Sensor2", "Sensor3", "Camera"],
):
    thread = SensorThread(sensor, queue, name)
    thread.start()
    threads.append(thread)

window = WindowImage(args.fps)

sensor1_val = 0
sensor2_val = 0
sensor3_val = 0
frame = None
running = True

while running:
    for queue in queues:
        try:
            while not queue.empty():
                name, data = queue.get_nowait()
                if name == "Sensor1":
                    sensor1_val = data
                elif name == "Sensor2":
                    sensor2_val = data
                elif name == "Sensor3":
                    sensor3_val = data
                else:
                    frame = data
        except:
            pass

    running = window.show(frame, sensor1_val, sensor2_val, sensor3_val)

for thread in threads:
    thread.stop()

for thread in threads:
    thread.join()
