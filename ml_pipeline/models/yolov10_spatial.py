# ml_pipeline/models/yolov10_spatial.py
import os
import numpy as np
import onnxruntime as ort


class YOLOv10SpatialExtractor:
    def __init__(self, model_path="weights/yolov10n.onnx"):
        """
        Loads the raw compiled ONNX graph directly.
        Bypasses heavy wrapper libraries to ensure maximum determinism and execution speed.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f" Could not find ONNX weights file at: {model_path}"
            )

        # Initialize the Inference Session
        # On your Mac, this can fall back to CPU or use the CoreML Execution Provider if configured
        self.session = ort.InferenceSession(
            model_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )

        # Get metadata about input layer requirements
        self.input_name = self.session.get_inputs()[0].name
        print(
            f"🎯 ONNX Backbone Initialized. Input layer name expected: '{self.input_name}'"
        )

    def forward(self, dummy_numpy_batch):
        """
        Args:
            dummy_numpy_batch (np.ndarray): Preprocessed image matrix of shape (Batch, Channels, Height, Width)
        Returns:
            list: Raw tensor arrays from the end-to-end network
        """
        # Run inference through the compiled graph
        outputs = self.session.run(None, {self.input_name: dummy_numpy_batch})
        return outputs


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("🎯 ONNX SPATIAL CORE VERIFICATION")
    print("=" * 40)

    # Instantiate from local path
    extractor = YOLOv10SpatialExtractor(model_path="weights/yolov10n.onnx")

    # Simulate an input frame batch: 1 Batch, 3 RGB channels, 640x640 resolution
    # ONNX Runtime natively utilizes optimized NumPy arrays for evaluation pass steps
    mock_frame = np.random.randn(1, 3, 640, 640).astype(np.float32)

    outputs = extractor.forward(mock_frame)

    print("\n✅ Verification Successful!")
    print(f"Number of output heads returned by model: {len(outputs)}")
    print(f"Primary Detections Shape Output: {outputs[0].shape}")
    print(" -> Shape Mapping: (Batch, Num_Anchors, BoundingBox_Plus_Confidence)")
    print("=" * 40 + "\n")
