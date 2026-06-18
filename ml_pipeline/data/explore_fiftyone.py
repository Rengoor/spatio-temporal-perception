import os
import fiftyone as fo


def create_and_explore_dataset(frame_dir):
    """
    Creates a FiftyOne dataset instance to visually profile robotic perception data,
    allowing engineering teams to slice errors caused by low lighting or motion blur.
    """
    print(f"Initializing FiftyOne visual session over: {frame_dir}")

    # 1. Initialize or load a named dataset instance
    dataset_name = "OpenLORIS_Robotic_Perception"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    # 2. Parse chronological frames to register them in the tracking dashboard
    frame_files = sorted(
        [
            f
            for f in os.listdir(frame_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    samples = []
    for idx, filename in enumerate(frame_files):
        filepath = os.path.join(frame_dir, filename)

        # Instantiate a standard data sample unit
        sample = fo.Sample(filepath=filepath)

        # Programmatically tag frames with specific robotic edge case constraints
        # Simulated scenario: alternating sequences represent dynamic camera motion blurs
        if idx % 3 == 0:
            sample.tags.append("motion_blur")
        if idx % 5 == 0:
            sample.tags.append("low_light")

        samples.append(sample)

    # Batch add samples to avoid I/O blocking bottlenecks
    dataset.add_samples(samples)

    print(f"Successfully ingested {len(dataset)} media nodes into FiftyOne database.")

    # 3. Launch the browser dashboard locally on Mac
    # This fires up a local webserver running on http://localhost:5151
    session = fo.launch_app(dataset, remote=False)
    session.wait()


if __name__ == "__main__":
    # Create temporary frames to demonstrate the dashboard engine capability safely
    import cv2
    import numpy as np

    demo_dir = "ml_pipeline/data/fiftyone_demo_frames"
    os.makedirs(demo_dir, exist_ok=True)

    for i in range(15):
        # Write clean synthetic frames
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"Frame {i}",
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(os.path.join(demo_dir, f"frame_{i:04d}.jpg"), img)

    try:
        create_and_explore_dataset(demo_dir)
    finally:
        # Clean up files upon exit
        for f in os.listdir(demo_dir):
            os.remove(os.path.join(demo_dir, f))
        os.rmdir(demo_dir)
