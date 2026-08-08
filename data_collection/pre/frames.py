import os
import av
import cv2
import numpy as np

def compute_laplacian_variance(frame, roi_box=None):
    """
    Computes Laplacian variance on a grayscale representation or central ROI.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    if roi_box:
        x, y, w, h = roi_box
        gray = gray[y:y+h, x:x+w]
        
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_iphone_mov_1s_stride(
    video_path, 
    output_dir="processed_fabric_frames", 
    blur_threshold=35.0, 
    sample_interval_sec=1.0,
    roi_box=None
):
    print("=" * 80)
    print(f"STARTING 1.0s STRIDE PYAV INGESTION ON: {video_path}")
    print("=" * 80)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"[FATAL] Failed to open video with PyAV: {e}")
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # Multi-threaded decoding

    fps = float(stream.average_rate)
    total_frames = stream.frames
    time_base = float(stream.time_base)
    
    print(f"[CONTAINER METADATA]")
    print(f"  - Codec: {stream.codec_context.name} | Format: {stream.pix_fmt}")
    print(f"  - FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"  - Sampling Rate: STRICT {sample_interval_sec} SECOND INTERVALS")
    print("-" * 80)

    saved_sharp_count = 0
    saved_blur_count = 0

    next_target_time_sec = 0.0
    
    for frame_idx, frame in enumerate(container.decode(video=0)):
        timestamp_sec = float(frame.pts * time_base) if frame.pts is not None else frame_idx / fps

        # Skip intermediate frames until the target timestamp (0.0s, 1.0s, 2.0s...) is reached
        if timestamp_sec < next_target_time_sec:
            continue

        # Convert 10-bit HDR frame to 8-bit RGB ndarray
        img_rgb = frame.to_ndarray(format="rgb24")
        score = compute_laplacian_variance(img_rgb, roi_box=roi_box)

        # Convert to BGR for OpenCV disk write
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if score >= blur_threshold:
            filename = f"frame_sec_{timestamp_sec:.2f}_sharp_score_{score:.1f}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            saved_sharp_count += 1
            print(f"[SHARP] Frame #{frame_idx} ({timestamp_sec:.2f}s) | Score: {score:.2f} -> Saved: {filename}")

        else:
            filename = f"frame_sec_{timestamp_sec:.2f}_BLUR_score_{score:.1f}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            saved_blur_count += 1
            print(f"[BLUR ] Frame #{frame_idx} ({timestamp_sec:.2f}s) | Score: {score:.2f} -> Saved: {filename}")

        # ALWAYS advance the target timer by 1.0 second regardless of sharp/blur status
        next_target_time_sec = timestamp_sec + sample_interval_sec

    container.close()
    print("=" * 80)
    print(f"PROCESSING COMPLETE. Processed ~1 frame/sec.")
    print(f"Saved {saved_sharp_count} sharp and {saved_blur_count} blur frames to '{output_dir}'.")
    print("=" * 80)


if __name__ == "__main__":
    VIDEO_FILE = "asset/cuts/c5.MOV"
    OUTPUT_FOLDER = "cuts"
    
    # Based on your log scores: Scores > 35 are sharp, scores < 30 are blurry
    BLUR_THRESHOLD = 35.0 

    process_iphone_mov_1s_stride(
        video_path=VIDEO_FILE,
        output_dir=OUTPUT_FOLDER,
        blur_threshold=BLUR_THRESHOLD,
        sample_interval_sec=1.0,  # 1.0 Second Strict Stride
        roi_box=None
    )