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

from dataclasses import dataclass, field
from typing import Any
from ..configs import CameraConfig, ColorMode, Cv2Rotation

@CameraConfig.register_subclass("depthai")
@dataclass
class DepthAICameraConfig(CameraConfig):
    """
    Configuration class for Luxonis DepthAI (OAK) cameras.

    This class provides configuration options for DepthAI cameras, including support for depth (disparity) and device identification via serial_number_or_name (serial number or name).

    Example configurations:
    ```python
    # Basic configuration
    DepthAICameraConfig(serial_number_or_name="184430106189041300")
    # With depth (disparity)
    DepthAICameraConfig(serial_number_or_name="184430106189041300", use_depth=True)
    # With BGR output
    DepthAICameraConfig(serial_number_or_name="184430106189041300", color_mode=ColorMode.BGR)
    ```

    Attributes:
        serial_number_or_name: Serial number or name to identify the DepthAI device.
        color_mode: Color mode for image output (RGB or BGR). Defaults to BGR.
        use_depth: Whether to enable disparity (stereo/depth) stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        resolution_color: Resolution for color images
        fps_color: Frames per second for color images
        resolution_mono: Resolution for mono (depth) images
        device_id: Accept device_id for backward compatibility
    """
    serial_number_or_name: str = ""
    color_mode: ColorMode = ColorMode.BGR
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    resolution_color: str = "1080p"
    fps_color: int = 30
    resolution_mono: str = "400p"  # Default for mono (depth) cameras
    # Accept device_id for backward compatibility
    device_id: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Backward compatibility: if device_id is provided, use it as serial_number_or_name
        if self.device_id is not None and not self.serial_number_or_name:
            self.serial_number_or_name = self.device_id
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )
        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )
        # Optionally, add more validation for resolutions/fps if needed 