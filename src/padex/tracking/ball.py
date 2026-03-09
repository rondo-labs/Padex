"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: ball.py
Description:
    Ball detection and trajectory tracking.
    Provides two detection strategies:
    - TrackNetBallDetectionStrategy: lightweight CNN that processes 3
      consecutive frames as a 9-channel input and outputs a heatmap.
      Fast, purpose-built for small high-speed balls.
    - SahiYoloBallDetectionStrategy: SAHI sliced YOLO. Slower but can
      be used as a fallback or for single-frame detection.
    Both feed into the same KalmanBallTracker for gap-filling.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    Position3D,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RawBallDetection:
    """Single ball detection from a frame."""

    bbox: BoundingBox
    confidence: float
    frame_id: int
    timestamp_ms: float


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class BallDetectionStrategy(abc.ABC):
    """Abstract interface for ball detection backends."""

    @abc.abstractmethod
    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        """Detect ball in a single frame. Returns None if not found."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        ...


class BallTracker(abc.ABC):
    """Abstract interface for ball tracking across frames."""

    @abc.abstractmethod
    def update(
        self, detection: RawBallDetection | None, timestamp_ms: float
    ) -> tuple[Position3D | None, BallVisibility]:
        """Update tracker with (possibly absent) detection.

        Returns (position_in_meters, visibility_state).
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        ...

    @abc.abstractmethod
    def set_homography(self, H: np.ndarray | None) -> None:
        """Set or update the pixel-to-court homography matrix."""
        ...


# ---------------------------------------------------------------------------
# TrackNet model definition
# ---------------------------------------------------------------------------


class _ConvBlock(object):
    """Placeholder — actual implementation uses nn.Sequential inside model."""


def _build_tracknet() -> "torch.nn.Module":
    """Build the BallTrackerNet architecture matching the pretrained weights.

    Architecture (VGG encoder-decoder, no skip connections):
      Encoder:
        conv1(9→64)  conv2(64→64)   → MaxPool
        conv3(64→128) conv4(128→128) → MaxPool
        conv5(128→256) conv6(256→256) conv7(256→256) → MaxPool
        conv8(256→512) conv9(512→512) conv10(512→512)
      Decoder (bilinear 2× upsample between groups):
        conv11(512→256) conv12(256→256) conv13(256→256) → Upsample
        conv14(256→128) conv15(128→128) → Upsample
        conv16(128→64) conv17(64→64) → Upsample
        conv18(64→256) → reshape + softmax

    Input:  (B, 9, 360, 640)   — 3 frames × RGB stacked
    Output: (B, 256, 360×640) after softmax → argmax → (B, 360, 640) heatmap
    """
    import torch.nn as nn

    class _ConvBNReLU(nn.Module):
        """Wraps Conv+ReLU+BN as a sub-module named 'block' to match checkpoint keys."""
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    def conv_block(in_ch: int, out_ch: int) -> "_ConvBNReLU":
        return _ConvBNReLU(in_ch, out_ch)

    class BallTrackerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Encoder
            self.conv1 = conv_block(9, 64)
            self.conv2 = conv_block(64, 64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = conv_block(64, 128)
            self.conv4 = conv_block(128, 128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv5 = conv_block(128, 256)
            self.conv6 = conv_block(256, 256)
            self.conv7 = conv_block(256, 256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv8 = conv_block(256, 512)
            self.conv9 = conv_block(512, 512)
            self.conv10 = conv_block(512, 512)
            # Decoder
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv11 = conv_block(512, 256)
            self.conv12 = conv_block(256, 256)
            self.conv13 = conv_block(256, 256)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv14 = conv_block(256, 128)
            self.conv15 = conv_block(128, 128)
            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv16 = conv_block(128, 64)
            self.conv17 = conv_block(64, 64)
            self.conv18 = conv_block(64, 256)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            batch = x.shape[0]
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.up1(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.up2(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = self.up3(x)
            x = self.conv16(x)
            x = self.conv17(x)
            x = self.conv18(x)
            # Reshape to (batch, 256, H*W) then softmax
            x = x.reshape(batch, 256, -1)
            x = self.softmax(x)
            return x

    return BallTrackerNet()


# ---------------------------------------------------------------------------
# TrackNet ball detection strategy
# ---------------------------------------------------------------------------


class TrackNetBallDetectionStrategy(BallDetectionStrategy):
    """Ball detection using TrackNet: 3-frame heatmap CNN.

    Processes 3 consecutive frames stacked as a 9-channel input (640×360).
    Outputs a heatmap; extracts ball center via argmax + Hough circles.

    Significantly faster than SAHI+YOLO: one forward pass per frame instead
    of ~20 sliced YOLO inferences.
    """

    # TrackNet was trained on 640×360
    INFER_W: int = 640
    INFER_H: int = 360
    HEATMAP_THRESHOLD: int = 127

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is None:
            from padex.weights import get_weight_path

            model_path = str(get_weight_path("ball_detection_TrackNet.pt"))
        self.model_path = model_path
        self._device_str = device
        self._model = None
        self._frame_buffer: list[np.ndarray] = []  # last 3 resized frames

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device_str or self._auto_device()
        self._torch_device = torch.device(device)

        model = _build_tracknet()
        state = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(state)
        model.eval()
        model.to(self._torch_device)
        self._model = model
        logger.info("TrackNet loaded on %s", device)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        self._ensure_model()
        import torch

        # Resize and buffer this frame
        resized = cv2.resize(frame, (self.INFER_W, self.INFER_H))
        self._frame_buffer.append(resized)
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)

        # Need 3 frames to run inference
        if len(self._frame_buffer) < 3:
            return None

        f0, f1, f2 = self._frame_buffer
        # Stack RGB channels: newest frame first (matches training convention)
        imgs = np.concatenate([f2, f1, f0], axis=2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(imgs.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            out = self._model(tensor)  # (1, 256, H*W)

        # Argmax over 256 channels → (1, H*W) → (H, W)
        heatmap = out.argmax(dim=1).squeeze().cpu().numpy()
        heatmap = heatmap.reshape(self.INFER_H, self.INFER_W).astype(np.uint8)

        x_pred, y_pred, confidence = self._postprocess(heatmap, frame.shape)
        if x_pred is None:
            return None

        h_orig, w_orig = frame.shape[:2]
        r = 4  # approximate ball radius in original coords
        scale_x = w_orig / self.INFER_W
        scale_y = h_orig / self.INFER_H
        bx = x_pred  # already scaled in _postprocess
        by = y_pred

        return RawBallDetection(
            bbox=BoundingBox(
                x1=float(bx - r), y1=float(by - r),
                x2=float(bx + r), y2=float(by + r),
            ),
            confidence=float(confidence),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )

    def _postprocess(
        self,
        heatmap: np.ndarray,
        orig_shape: tuple,
    ) -> tuple[float | None, float | None, float]:
        """Extract ball center from heatmap. Returns (x, y, confidence) in original pixel coords."""
        h_orig, w_orig = orig_shape[:2]
        scale_x = w_orig / self.INFER_W
        scale_y = h_orig / self.INFER_H

        if heatmap.max() == 0:
            return None, None, 0.0

        # Normalize to 0-255 and threshold
        feature = (heatmap.astype(np.float32) / heatmap.max() * 255).astype(np.uint8)
        _, binary = cv2.threshold(feature, self.HEATMAP_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Find connected components and use the largest blob's centroid
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            # No foreground blob found
            return None, None, 0.0

        # Skip label 0 (background), find largest foreground component
        fg_areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(fg_areas)) + 1  # +1 to skip background

        # Filter out blobs that are too large (likely noise)
        if fg_areas[best_label - 1] > 200:
            return None, None, 0.0

        cx, cy = centroids[best_label]

        # Confidence from peak heatmap value in the blob region
        blob_mask = labels == best_label
        confidence = float(heatmap[blob_mask].max()) / 255.0

        return float(cx * scale_x), float(cy * scale_y), confidence

    def reset(self) -> None:
        self._frame_buffer.clear()


# ---------------------------------------------------------------------------
# TrackNet V3 model definition
# ---------------------------------------------------------------------------


def _build_tracknet_v3(in_dim: int = 9, out_dim: int = 3) -> "torch.nn.Module":
    """Build the TrackNet V3 architecture (U-Net with skip connections).

    Architecture (from qaz812345/TrackNetV3):
      Encoder:
        down_block_1: Double2DConv(in_dim → 64)  + MaxPool
        down_block_2: Double2DConv(64 → 128)     + MaxPool
        down_block_3: Triple2DConv(128 → 256)    + MaxPool
        bottleneck:   Triple2DConv(256 → 512)
      Decoder (Upsample + concat skip connections):
        up_block_1: Triple2DConv(512+256=768 → 256) + Upsample
        up_block_2: Double2DConv(256+128=384 → 128) + Upsample
        up_block_3: Double2DConv(128+64=192 → 64)
        predictor: Conv2d(64 → out_dim) + Sigmoid

    Input:  (B, 9, 288, 512)   — 3 frames × RGB stacked
    Output: (B, 3, 288, 512)   — per-frame ball heatmap (sigmoid)
    """
    import torch
    import torch.nn as nn

    class _Conv2DBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.relu(self.bn(self.conv(x)))

    class _Double2DConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv_1 = _Conv2DBlock(in_ch, out_ch)
            self.conv_2 = _Conv2DBlock(out_ch, out_ch)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv_2(self.conv_1(x))

    class _Triple2DConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv_1 = _Conv2DBlock(in_ch, out_ch)
            self.conv_2 = _Conv2DBlock(out_ch, out_ch)
            self.conv_3 = _Conv2DBlock(out_ch, out_ch)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNetV3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_block_1 = _Double2DConv(in_dim, 64)
            self.down_block_2 = _Double2DConv(64, 128)
            self.down_block_3 = _Triple2DConv(128, 256)
            self.bottleneck = _Triple2DConv(256, 512)
            self.up_block_1 = _Triple2DConv(768, 256)
            self.up_block_2 = _Double2DConv(384, 128)
            self.up_block_3 = _Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, kernel_size=(1, 1))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x1 = self.down_block_1(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
            x2 = self.down_block_2(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
            x3 = self.down_block_3(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
            x = self.bottleneck(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
            x = self.up_block_1(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
            x = self.up_block_2(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
            x = self.up_block_3(x)
            x = self.predictor(x)
            return self.sigmoid(x)

    return TrackNetV3()


# ---------------------------------------------------------------------------
# TrackNet V3 ball detection strategy
# ---------------------------------------------------------------------------


class TrackNetV3BallDetectionStrategy(BallDetectionStrategy):
    """Ball detection using TrackNet V3: U-Net with skip connections.

    V3 improvements over V2:
    - Skip connections in decoder (better small-target localization)
    - 3-channel sigmoid output (simpler than 256-class softmax)
    - Weighted BCE + Focal Loss training (better class imbalance handling)

    Input:  9-channel (3 RGB frames), resized to 512×288
    Output: 3-channel sigmoid heatmap; channel 2 = current frame detection
    """

    INFER_W: int = 512
    INFER_H: int = 288
    HEATMAP_THRESHOLD: float = 0.5  # sigmoid output in [0, 1]
    MAX_BLOB_AREA: int = 200
    SEQ_LEN: int = 8  # number of target frames (checkpoint param_dict: seq_len=8)
    # bg_mode='concat': input = [bg_frame, frame_1, ..., frame_8] = 9 frames = 27 channels

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is None:
            from padex.weights import get_weight_path

            model_path = str(get_weight_path("ball_detection_TrackNetV3.pt"))
        self.model_path = model_path
        self._device_str = device
        self._model = None
        self._frame_buffer: list[np.ndarray] = []  # stores up to SEQ_LEN+1 frames
        self._bg_frame: np.ndarray | None = None   # background = first seen frame

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device_str or self._auto_device()
        self._torch_device = torch.device(device)

        # in_dim=27: (SEQ_LEN+1)*3 channels; out_dim=SEQ_LEN: one heatmap per target frame
        model = _build_tracknet_v3(in_dim=(self.SEQ_LEN + 1) * 3, out_dim=self.SEQ_LEN)
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)
        # Support both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        model.to(self._torch_device)
        self._model = model
        logger.info("TrackNet V3 loaded on %s", device)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        self._ensure_model()
        import torch

        resized = cv2.resize(frame, (self.INFER_W, self.INFER_H))

        # Store first frame as static background approximation
        if self._bg_frame is None:
            self._bg_frame = resized

        self._frame_buffer.append(resized)
        if len(self._frame_buffer) > self.SEQ_LEN:
            self._frame_buffer.pop(0)

        # Need SEQ_LEN frames to run inference
        if len(self._frame_buffer) < self.SEQ_LEN:
            return None

        # Input: [bg_frame, frame_1, ..., frame_SEQ_LEN] stacked as 27 channels
        all_frames = [self._bg_frame] + self._frame_buffer
        imgs = np.concatenate(all_frames, axis=2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(imgs.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            out = self._model(tensor)  # (1, SEQ_LEN, 288, 512)

        # Last channel = current (newest) frame heatmap
        heatmap = out[0, -1].cpu().numpy()  # (288, 512), values in [0, 1]

        x_pred, y_pred, confidence = self._postprocess(heatmap, frame.shape)
        if x_pred is None:
            return None

        h_orig, w_orig = frame.shape[:2]
        r = 4
        return RawBallDetection(
            bbox=BoundingBox(
                x1=float(x_pred - r), y1=float(y_pred - r),
                x2=float(x_pred + r), y2=float(y_pred + r),
            ),
            confidence=float(confidence),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )

    def _postprocess(
        self,
        heatmap: np.ndarray,
        orig_shape: tuple,
    ) -> tuple[float | None, float | None, float]:
        """Extract ball center from sigmoid heatmap. Returns (x, y, conf) in original pixels."""
        h_orig, w_orig = orig_shape[:2]
        scale_x = w_orig / self.INFER_W
        scale_y = h_orig / self.INFER_H

        if heatmap.max() < self.HEATMAP_THRESHOLD:
            return None, None, 0.0

        binary = (heatmap >= self.HEATMAP_THRESHOLD).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            return None, None, 0.0

        fg_areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(fg_areas)) + 1

        if fg_areas[best_label - 1] > self.MAX_BLOB_AREA:
            return None, None, 0.0

        cx, cy = centroids[best_label]
        blob_mask = labels == best_label
        confidence = float(heatmap[blob_mask].max())

        return float(cx * scale_x), float(cy * scale_y), confidence

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._bg_frame = None


# ---------------------------------------------------------------------------
# SAHI + YOLO ball detection strategy
# ---------------------------------------------------------------------------


class SahiYoloBallDetectionStrategy(BallDetectionStrategy):
    """Ball detection using SAHI sliced inference with YOLO.

    SAHI slices the image into overlapping patches to detect small objects
    that would be missed by standard YOLO inference on downscaled frames.
    """

    BALL_CLASS_ID = 32  # COCO 'sports ball'

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.25,
        slice_size: int = 512,
        overlap_ratio: float = 0.2,
        device: str | None = None,
    ) -> None:
        if model_path is None:
            from padex.weights import get_weight_path

            model_path = str(get_weight_path("yolo26m.pt"))
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.device = device
        self._model = None

    def _load_model(self):
        from sahi import AutoDetectionModel

        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device or "",
        )

    def _ensure_model(self):
        if self._model is None:
            self._model = self._load_model()

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        from sahi.predict import get_sliced_prediction

        self._ensure_model()

        # SAHI expects RGB, OpenCV provides BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            image=frame_rgb,
            detection_model=self._model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False,
            verbose=0,
        )

        # Filter to ball class only
        ball_preds = [
            p
            for p in result.object_prediction_list
            if p.category.id == self.BALL_CLASS_ID
        ]

        if not ball_preds:
            return None

        # Pick highest confidence detection
        best = max(ball_preds, key=lambda p: p.score.value)
        x1, y1, x2, y2 = best.bbox.to_xyxy()

        return RawBallDetection(
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            confidence=float(best.score.value),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )

    def reset(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Kalman ball tracker
# ---------------------------------------------------------------------------


class KalmanBallTracker(BallTracker):
    """4D Kalman filter tracking ball in court coordinates.

    State vector: [x, y, vx, vy] in meters.
    Observation:  [x, y] from YOLO detection projected to court coords.
    """

    MAX_OCCLUDED_FRAMES: int = 10
    GATE_DISTANCE_M: float = 3.0

    def __init__(
        self,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.3,
        homography_matrix: np.ndarray | None = None,
    ) -> None:
        self._H_matrix = homography_matrix
        self._process_noise_std = process_noise_std
        self._measurement_noise_std = measurement_noise_std
        self._kf: cv2.KalmanFilter | None = None
        self._initialized = False
        self._occluded_count = 0

    def _build_kalman(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        # Constant velocity model
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * self._process_noise_std**2
        )
        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self._measurement_noise_std**2
        )
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def set_homography(self, H: np.ndarray | None) -> None:
        self._H_matrix = H

    def update(
        self, detection: RawBallDetection | None, timestamp_ms: float
    ) -> tuple[Position3D | None, BallVisibility]:
        if detection is None:
            return self._handle_missing()

        # Project bbox center to court coordinates
        court_xy = self._bbox_center_to_court(detection.bbox)
        if court_xy is None:
            return self._handle_missing()

        x_m, y_m = court_xy

        # Initialize on first detection
        if not self._initialized or self._kf is None:
            self._kf = self._build_kalman()
            self._kf.statePost = np.array(
                [x_m, y_m, 0, 0], dtype=np.float32
            ).reshape(4, 1)
            self._initialized = True
            self._occluded_count = 0
            return self._clamp_to_court(x_m, y_m), BallVisibility.VISIBLE

        # Gate: reject detections far from prediction
        prediction = self._kf.predict()
        pred_x, pred_y = float(prediction[0, 0]), float(prediction[1, 0])
        dist = np.sqrt((pred_x - x_m) ** 2 + (pred_y - y_m) ** 2)
        if dist > self.GATE_DISTANCE_M:
            # Ghost detection (glass reflection) — use prediction instead
            self._occluded_count += 1
            if self._occluded_count > self.MAX_OCCLUDED_FRAMES:
                self.reset()
                return None, BallVisibility.OCCLUDED
            return self._clamp_to_court(pred_x, pred_y), BallVisibility.INFERRED

        # Normal update: predict + correct
        measurement = np.array([[x_m], [y_m]], dtype=np.float32)
        corrected = self._kf.correct(measurement)
        self._occluded_count = 0
        cx, cy = float(corrected[0, 0]), float(corrected[1, 0])
        return self._clamp_to_court(cx, cy), BallVisibility.VISIBLE

    def reset(self) -> None:
        self._kf = None
        self._initialized = False
        self._occluded_count = 0

    def _handle_missing(self) -> tuple[Position3D | None, BallVisibility]:
        if not self._initialized or self._kf is None:
            return None, BallVisibility.OCCLUDED

        prediction = self._kf.predict()
        self._occluded_count += 1

        if self._occluded_count > self.MAX_OCCLUDED_FRAMES:
            self.reset()
            return None, BallVisibility.OCCLUDED

        x, y = float(prediction[0, 0]), float(prediction[1, 0])
        return self._clamp_to_court(x, y), BallVisibility.INFERRED

    def _bbox_center_to_court(
        self, bbox: BoundingBox
    ) -> tuple[float, float] | None:
        """Project bbox center from pixel to court coordinates."""
        if self._H_matrix is None:
            return None

        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        pt = np.array([[[cx, cy]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self._H_matrix)
        x, y = float(result[0, 0, 0]), float(result[0, 0, 1])
        return (x, y)

    @staticmethod
    def _clamp_to_court(x: float, y: float) -> Position3D:
        """Clamp position to valid court bounds and return Position3D."""
        x_c = max(0.0, min(10.0, x))
        y_c = max(0.0, min(20.0, y))
        return Position3D(x=x_c, y=y_c, z=0.0)


# ---------------------------------------------------------------------------
# BallDetector facade
# ---------------------------------------------------------------------------


class BallDetector:
    """Main ball detection + tracking facade.

    Combines SAHI-based detection with Kalman filter tracking.
    """

    def __init__(
        self,
        detection_strategy: BallDetectionStrategy | None = None,
        tracker: BallTracker | None = None,
        model_path: str | None = None,
        confidence_threshold: float = 0.25,
        use_tracknet: bool = True,
        use_tracknet_v3: bool = False,
    ) -> None:
        if detection_strategy is not None:
            self.detection_strategy = detection_strategy
        elif use_tracknet_v3:
            self.detection_strategy = TrackNetV3BallDetectionStrategy(
                model_path=model_path,
            )
        elif use_tracknet:
            self.detection_strategy = TrackNetBallDetectionStrategy(
                model_path=model_path,
            )
        else:
            self.detection_strategy = SahiYoloBallDetectionStrategy(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
            )
        self.tracker = tracker or KalmanBallTracker()

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> BallFrame:
        """Detect ball in a single frame (no cross-frame tracking)."""
        raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
        return BallFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=raw.bbox if raw else None,
            position=None,
            confidence=raw.confidence if raw else 0.0,
            visibility=BallVisibility.VISIBLE if raw else BallVisibility.OCCLUDED,
        )

    def detect_and_track_single(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> BallFrame:
        """Detect and track ball in a single frame (streaming mode).

        Uses the Kalman tracker for cross-frame state. Call frames in order.
        """
        raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
        position, visibility = self.tracker.update(raw, timestamp_ms)
        return BallFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=raw.bbox if raw else None,
            position=position,
            confidence=raw.confidence if raw else 0.0,
            visibility=visibility,
        )

    def track(
        self,
        frames: list[tuple[int, float, np.ndarray]],
        homography_matrix: np.ndarray | None = None,
    ) -> list[BallFrame]:
        """Track ball across multiple frames with Kalman gap-filling.

        Args:
            frames: list of (frame_id, timestamp_ms, frame_array).
            homography_matrix: pixel-to-court transform for Position3D.

        Returns:
            list[BallFrame] with visibility states and interpolated positions.
        """
        if homography_matrix is not None:
            self.tracker.set_homography(homography_matrix)

        ball_frames: list[BallFrame] = []
        for frame_id, timestamp_ms, frame in frames:
            raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
            position, visibility = self.tracker.update(raw, timestamp_ms)

            ball_frames.append(
                BallFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    bbox=raw.bbox if raw else None,
                    position=position,
                    confidence=raw.confidence if raw else 0.0,
                    visibility=visibility,
                )
            )

        return ball_frames

    def reset(self) -> None:
        """Reset detector and tracker state."""
        self.detection_strategy.reset()
        self.tracker.reset()
