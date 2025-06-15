# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the DepthAICamera class for capturing frames from Luxonis OAK cameras using DepthAI.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional
import numpy as np
import cv2

try:
    import depthai as dai
except Exception as e:
    logging.info(f"Could not import depthai: {e}")

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_depthai import DepthAICameraConfig
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

# Helper for resolution string/tuple to DepthAI enum
_RESOLUTION_COLOR_MAP = {
    "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P,
    (1920, 1080): dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    (3840, 2160): dai.ColorCameraProperties.SensorResolution.THE_4_K,
    (4056, 3040): dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    (1280, 720): dai.ColorCameraProperties.SensorResolution.THE_720_P,
}
_RESOLUTION_MONO_MAP = {
    "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    (640, 400): dai.MonoCameraProperties.SensorResolution.THE_400_P,
    (640, 480): dai.MonoCameraProperties.SensorResolution.THE_480_P,
    (1280, 720): dai.MonoCameraProperties.SensorResolution.THE_720_P,
    (1280, 800): dai.MonoCameraProperties.SensorResolution.THE_800_P,
}

class DepthAICamera(Camera):
    """
    Manages interactions with Luxonis DepthAI (OAK) cameras for frame and depth (disparity) recording.

    This class provides an interface similar to RealSenseCamera but tailored for DepthAI devices.
    It uses the camera's unique serial number or name for identification. It supports capturing
    color frames (always enabled) and disparity maps (if use_depth=True).
    """
    def __init__(self, config: DepthAICameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.rotation = get_cv2_rotation(config.rotation)

        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None
        self.queues: Dict[str, dai.DataOutputQueue] = {}

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number_or_name})"

    @property
    def is_connected(self) -> bool:
        return self.device is not None and bool(self.queues)

    def _get_color_resolution(self):
        res = self.config.resolution_color
        return _RESOLUTION_COLOR_MAP.get(res, dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    def _get_mono_resolution(self):
        res = self.config.resolution_mono
        return _RESOLUTION_MONO_MAP.get(res, dai.MonoCameraProperties.SensorResolution.THE_400_P)

    def connect(self, warmup: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.pipeline = dai.Pipeline()
        self.queues = {}

        if self.use_depth:
            # Strictly supported color resolutions for IMX378
            supported_color_res = ["1080p", "4k", "12mp"]
            requested_color_res = self.config.resolution_color
            if requested_color_res in supported_color_res:
                chosen_color_res = requested_color_res
            else:
                logger.warning(f"[DepthAICamera] Requested color resolution '{requested_color_res}' is not supported by IMX378. Forcing to 1080p.")
                chosen_color_res = "1080p"
            cam_rgb = self.pipeline.createColorCamera()
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setFps(self.config.fps_color)
            cam_rgb.setInterleaved(False)
            cam_rgb.setResolution(_RESOLUTION_COLOR_MAP[chosen_color_res])
            xout_rgb = self.pipeline.createXLinkOut()
            xout_rgb.setStreamName("color")
            cam_rgb.isp.link(xout_rgb.input)

            # Strictly supported mono resolutions for OV9282/OV7251: default to 400p
            mono_res = "400p"
            cam_left = self.pipeline.createMonoCamera()
            cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            cam_left.setResolution(_RESOLUTION_MONO_MAP[mono_res])
            cam_left.setFps(self.config.fps_color)
            cam_right = self.pipeline.createMonoCamera()
            cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            cam_right.setResolution(_RESOLUTION_MONO_MAP[mono_res])
            cam_right.setFps(self.config.fps_color)

            stereo = self.pipeline.createStereoDepth()
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            # Do NOT align depth to color; output at mono native resolution
            cam_left.out.link(stereo.left)
            cam_right.out.link(stereo.right)
            xout_disp = self.pipeline.createXLinkOut()
            xout_disp.setStreamName("disparity")
            stereo.disparity.link(xout_disp.input)
        else:
            # RGB-only mode: strictly supported color resolutions for IMX378
            supported_color_res = ["1080p", "4k", "12mp"]
            requested_color_res = self.config.resolution_color
            if requested_color_res in supported_color_res:
                chosen_res = requested_color_res
            else:
                logger.warning(f"[DepthAICamera] Requested color resolution '{requested_color_res}' is not supported by IMX378. Forcing to 1080p.")
                chosen_res = "1080p"
            cam_rgb = self.pipeline.createColorCamera()
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setFps(self.config.fps_color)
            cam_rgb.setInterleaved(False)
            cam_rgb.setResolution(_RESOLUTION_COLOR_MAP[chosen_res])
            xout_rgb = self.pipeline.createXLinkOut()
            xout_rgb.setStreamName("color")
            cam_rgb.video.link(xout_rgb.input)

        # Only now create the device
        # Accept 0, "0", None, or "" as 'use first available device'
        if self.serial_number_or_name and str(self.serial_number_or_name) not in ("0", "", "None"):
            device_info = dai.DeviceInfo(str(self.serial_number_or_name))
            self.device = dai.Device(self.pipeline, deviceInfo=device_info)
        else:
            self.device = dai.Device(self.pipeline)

        # Output queues
        if self.use_depth:
            self.queues["color"] = self.device.getOutputQueue(name="color", maxSize=4, blocking=False)
            self.queues["disparity"] = self.device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        else:
            self.queues["color"] = self.device.getOutputQueue(name="color", maxSize=4, blocking=False)

        # Warmup
        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                if self.use_depth:
                    self.read(timeout_ms=1000)
                else:
                    self.read(timeout_ms=200)
                time.sleep(0.1)
        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        try:
            devices = dai.Device.getAllAvailableDevices()
            return [{
                "id": d.getMxId(),
                "name": getattr(d, 'name', None),
                "type": "DepthAI"
            } for d in devices]
        except Exception as e:
            logger.error(f"Error finding DepthAI devices: {e}")
            return []

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        # DepthAI getCvFrame() returns RGB by default
        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        if self.rotation is not None:
            processed_image = cv2.rotate(processed_image, self.rotation)
        return processed_image

    def read(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame synchronously from the camera.
        If use_depth is True, returns a tuple (color, depth) with both frames.
        If use_depth is False, returns the color frame.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.use_depth:
            # Read both color and depth frames
            queue_color = self.queues["color"]
            queue_disp = self.queues["disparity"]
            in_data_color = None
            in_data_disp = None
            start = time.time()
            while (time.time() - start) < (timeout_ms / 1000.0):
                if in_data_color is None:
                    in_data_color = queue_color.tryGet()
                if in_data_disp is None:
                    in_data_disp = queue_disp.tryGet()
                if in_data_color is not None and in_data_disp is not None:
                    break
                time.sleep(0.01)
            if in_data_color is None:
                raise RuntimeError(f"No color frame received from {self} within timeout.")
            if in_data_disp is None:
                raise RuntimeError(f"No disparity frame received from {self} within timeout.")
            color = self._postprocess_image(in_data_color.getCvFrame())
            depth = in_data_disp.getFrame()
            return color, depth
        queue = self.queues["color"]
        in_data = None
        start = time.time()
        while (time.time() - start) < (timeout_ms / 1000.0):
            in_data = queue.tryGet()
            if in_data is not None:
                break
            time.sleep(0.01)
        if in_data is None:
            raise RuntimeError(f"No color frame received from {self} within timeout.")
        image = in_data.getCvFrame()
        return self._postprocess_image(image)

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single disparity frame synchronously from the camera.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth or "disparity" not in self.queues:
            raise RuntimeError(f"Disparity stream is not enabled for {self}.")
        queue = self.queues["disparity"]
        in_data = None
        start = time.time()
        while (time.time() - start) < (timeout_ms / 1000.0):
            in_data = queue.tryGet()
            if in_data is not None:
                break
            time.sleep(0.01)
        if in_data is None:
            raise RuntimeError(f"No disparity frame received from {self} within timeout.")
        image = in_data.getFrame()
        return image

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.use_depth:
                    color, depth = self.read(timeout_ms=1000)
                    with self.frame_lock:
                        self.latest_frame = (color, depth)
                else:
                    frame = self.read()
                    with self.frame_lock:
                        self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        if self.use_depth:
            # Always return (color, depth) tuple
            return frame
        else:
            return frame

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def disconnect(self):
        if not self.is_connected and self.thread is None:
            return  # Already disconnected, do nothing
        if self.thread is not None:
            self._stop_read_thread()
        if self.device is not None:
            self.device.close()
            self.device = None
            self.pipeline = None
            self.queues = {}
        logger.info(f"{self} disconnected.") 