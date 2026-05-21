"""
calibrate.py — Camera-specific threshold calibration tool.

Runs the detection + recognition pipeline for a configurable duration with all
quality gates disabled, collects raw statistics on every detected face, then
recommends optimal thresholds based on the actual distributions observed on
this camera.

Usage:
    python -m scripts.calibrate [--duration 120] [--camera-id camera_01]

Output:
    - Console: formatted recommendation table
    - logs/calibration_<camera_id>_<timestamp>.json: raw sample data
"""

import argparse
import json
import logging
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config import load_config, load_merged_configs
from core.detection import SCRFDDetector
from core.recognition import AdaFaceRecognizer
from core.database import FaceDatabase
from core.tracking import ByteTracker
from core.quality import calculate_blur_score
from core.utils.pose_estimator import PoseEstimator

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    level=logging.INFO,
)
log = logging.getLogger("calibrate")


def percentile_label(p: float) -> str:
    return f"p{int(p)}"


def run_calibration(camera_id: str, camera_url: str, duration: int,
                    detector, recognizer, face_db, config: dict):
    """Open stream, collect face samples for `duration` seconds, return stats."""

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        log.error(f"Cannot open stream: {camera_url}")
        return None

    resolution = config.get('cameras', {})  # not needed here

    tracker = ByteTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )
    pose_estimator = PoseEstimator()

    # Stat buckets
    face_widths = []
    yaw_angles = []
    pitch_angles = []
    blur_scores = []
    similarity_scores = []
    total_frames = 0
    total_detections = 0

    log.info(f"[{camera_id}] Calibration started — collecting for {duration}s …")
    deadline = time.time() + duration

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        total_frames += 1
        faces = detector.detect(frame)
        tracked = tracker.update(faces)
        total_detections += len(tracked)

        face_imgs = []
        face_refs = []

        for face in tracked:
            # Pose (always compute — no gate)
            if face.kps is not None and len(face.kps) == 5:
                face.pitch, face.yaw, face.roll = pose_estimator.estimate_pose(
                    face.kps, frame.shape
                )
            else:
                face.yaw = face.pitch = 0.0

            x1, y1, x2, y2 = face.bbox[:4].astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            blur = calculate_blur_score(crop)

            face_widths.append(float(face.width))
            yaw_angles.append(abs(float(face.yaw)))
            pitch_angles.append(abs(float(face.pitch)))
            blur_scores.append(float(blur))

            # Only extract embeddings for faces that have usable keypoints
            if face.kps is not None:
                from insightface.utils import face_align
                try:
                    aligned = face_align.norm_crop(frame, face.kps)
                    face_imgs.append(aligned)
                    face_refs.append(face)
                except Exception:
                    pass

        # Batch recognition
        if face_imgs:
            try:
                embeddings, norms = recognizer.extract_embeddings_batch(face_imgs)
                threshold = config['recognition']['similarity_threshold']
                for emb in embeddings:
                    _, score = face_db.match(emb, threshold,
                                             match_margin=0.0, match_top_k=5)
                    if score > 0:
                        similarity_scores.append(float(score))
            except Exception as e:
                log.debug(f"Recognition error: {e}")

        remaining = deadline - time.time()
        if total_frames % 100 == 0:
            log.info(
                f"[{camera_id}] {remaining:.0f}s left — "
                f"faces collected: widths={len(face_widths)} "
                f"blur={len(blur_scores)} sim={len(similarity_scores)}"
            )

    cap.release()
    log.info(f"[{camera_id}] Collection complete.")

    return {
        "camera_id": camera_id,
        "duration_s": duration,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "face_widths": face_widths,
        "yaw_angles": yaw_angles,
        "pitch_angles": pitch_angles,
        "blur_scores": blur_scores,
        "similarity_scores": similarity_scores,
    }


def compute_stats(samples: list, label: str) -> dict:
    if not samples:
        return {"count": 0}
    arr = np.array(samples)
    return {
        "count": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def recommend(data: dict) -> dict:
    """Derive threshold recommendations from collected samples."""
    recs = {}
    
    widths = np.array(data["face_widths"]) if data["face_widths"] else np.array([])
    yaws = np.array(data["yaw_angles"]) if data["yaw_angles"] else np.array([])
    pitches = np.array(data["pitch_angles"]) if data["pitch_angles"] else np.array([])
    blurs = np.array(data["blur_scores"]) if data["blur_scores"] else np.array([])
    sims = np.array(data["similarity_scores"]) if data["similarity_scores"] else np.array([])

    # min_face_size: p10 of face widths (reject only the smallest 10%)
    if len(widths) >= 10:
        recs["min_face_size"] = int(np.percentile(widths, 10))
        recs["min_face_size_note"] = (
            f"p10={recs['min_face_size']}px of {len(widths)} samples "
            f"(mean={widths.mean():.0f}px, range={widths.min():.0f}–{widths.max():.0f}px)"
        )
    else:
        recs["min_face_size"] = 50
        recs["min_face_size_note"] = "Insufficient samples — using safe fallback"

    # max_yaw: p85 of observed yaw angles (pass 85% of real detections)
    if len(yaws) >= 10:
        recs["max_yaw"] = round(float(np.percentile(yaws, 85)), 1)
        recs["max_yaw_note"] = (
            f"p85={recs['max_yaw']}° of {len(yaws)} samples "
            f"(mean={yaws.mean():.1f}°, p50={np.percentile(yaws,50):.1f}°, max={yaws.max():.1f}°)"
        )
    else:
        recs["max_yaw"] = 60.0
        recs["max_yaw_note"] = "Insufficient samples — using safe fallback"

    # max_pitch: p85 of observed pitch angles
    if len(pitches) >= 10:
        recs["max_pitch"] = round(float(np.percentile(pitches, 85)), 1)
        recs["max_pitch_note"] = (
            f"p85={recs['max_pitch']}° of {len(pitches)} samples "
            f"(mean={pitches.mean():.1f}°, p50={np.percentile(pitches,50):.1f}°)"
        )
    else:
        recs["max_pitch"] = 60.0
        recs["max_pitch_note"] = "Insufficient samples — using safe fallback"

    # blur_threshold: p20 of blur scores — adaptive blur uses this as fallback
    # Filter out near-zero scores (bad crops)
    good_blurs = blurs[blurs > 5.0] if len(blurs) > 0 else blurs
    if len(good_blurs) >= 10:
        recs["blur_threshold"] = round(float(np.percentile(good_blurs, 20)), 1)
        recs["blur_threshold_note"] = (
            f"p20={recs['blur_threshold']} of {len(good_blurs)} usable samples "
            f"(mean={good_blurs.mean():.1f}, p50={np.percentile(good_blurs,50):.1f}, "
            f"p10={np.percentile(good_blurs,10):.1f})"
        )
    else:
        recs["blur_threshold"] = 30.0
        recs["blur_threshold_note"] = "Insufficient samples — using safe fallback"

    # similarity_threshold: recommend based on observed top-match scores
    # For an entry camera, you want low false negatives → use p25 of top scores as threshold
    if len(sims) >= 10:
        p25 = float(np.percentile(sims, 25))
        p50 = float(np.percentile(sims, 50))
        # Recommend slightly above the median gap between UNKNOWN (low score) and KNOWN (high score)
        # Can't perfectly separate without ground-truth, so note the distribution
        recs["similarity_threshold"] = round(p50, 2)
        recs["similarity_threshold_note"] = (
            f"p50={p50:.3f} of {len(sims)} top-match scores "
            f"(p25={p25:.3f}, p75={float(np.percentile(sims,75)):.3f}, "
            f"max={sims.max():.3f}). Adjust manually based on false positives."
        )
    else:
        recs["similarity_threshold"] = 0.55
        recs["similarity_threshold_note"] = "Insufficient matches — keep current value"

    return recs


def print_report(data: dict, recs: dict):
    cam = data["camera_id"]
    dur = data["duration_s"]
    total_f = data["total_frames"]
    total_d = data["total_detections"]

    print("\n" + "=" * 72)
    print(f"  CALIBRATION REPORT -- {cam}  ({dur}s, {total_f} frames, {total_d} raw detections)")
    print("=" * 72)

    sections = [
        ("Face Size (px)", data["face_widths"]),
        ("Yaw Angle (°)", data["yaw_angles"]),
        ("Pitch Angle (°)", data["pitch_angles"]),
        ("Blur Score (Laplacian var)", data["blur_scores"]),
        ("Similarity Score (top-match)", data["similarity_scores"]),
    ]
    for label, samples in sections:
        s = compute_stats(samples, label)
        if s["count"] == 0:
            print(f"\n  {label}: no samples collected")
            continue
        print(f"\n  {label} (n={s['count']}):")
        print(f"    min={s['min']:.1f}  p10={s['p10']:.1f}  p25={s['p25']:.1f}  "
              f"p50={s['p50']:.1f}  p75={s['p75']:.1f}  p90={s['p90']:.1f}  "
              f"p95={s['p95']:.1f}  max={s['max']:.1f}")

    print("\n" + "-" * 72)
    print("  RECOMMENDED thresholds.yaml values:")
    print("-" * 72)
    print(f"""
recognition:
  min_face_size: {recs['min_face_size']}
  # {recs['min_face_size_note']}
  max_yaw: {recs['max_yaw']}
  # {recs['max_yaw_note']}
  max_pitch: {recs['max_pitch']}
  # {recs['max_pitch_note']}
  blur_threshold: {recs['blur_threshold']}
  # {recs['blur_threshold_note']}
  similarity_threshold: {recs['similarity_threshold']}
  # {recs['similarity_threshold_note']}
""")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calibrate pipeline thresholds")
    parser.add_argument("--duration", type=int, default=120,
                        help="Seconds to collect samples (default: 120)")
    parser.add_argument("--camera-id", type=str, default=None,
                        help="Specific camera ID to calibrate (default: first enabled)")
    args = parser.parse_args()

    config = load_merged_configs([
        'configs/default.yaml',
        'configs/thresholds.yaml',
        'configs/tensorrt.yaml',
    ])
    camera_cfg = load_config('configs/cameras.yaml')
    cameras = [c for c in camera_cfg.get('cameras', []) if c.get('enabled', True)]

    if not cameras:
        log.error("No enabled cameras in configs/cameras.yaml")
        return

    if args.camera_id:
        cameras = [c for c in cameras if c['id'] == args.camera_id]
        if not cameras:
            log.error(f"Camera '{args.camera_id}' not found")
            return

    cam = cameras[0]

    log.info("Loading AI models …")
    gallery = np.load(config['dataset']['gallery_embeddings'], allow_pickle=True).item()
    detector = SCRFDDetector(config, config['models']['scrfd_onnx'])
    recognizer = AdaFaceRecognizer(config, config['models']['adaface_onnx'])
    face_db = FaceDatabase(gallery)

    data = run_calibration(
        camera_id=cam['id'],
        camera_url=cam['url'],
        duration=args.duration,
        detector=detector,
        recognizer=recognizer,
        face_db=face_db,
        config=config,
    )

    if data is None:
        return

    recs = recommend(data)
    # Save raw data FIRST (before any print that might fail)
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"calibration_{cam['id']}_{ts}.json"
    save_data = {
        "camera_id": data["camera_id"],
        "duration_s": data["duration_s"],
        "total_frames": data["total_frames"],
        "total_detections": data["total_detections"],
        "stats": {
            "face_widths": compute_stats(data["face_widths"], "widths"),
            "yaw_angles": compute_stats(data["yaw_angles"], "yaw"),
            "pitch_angles": compute_stats(data["pitch_angles"], "pitch"),
            "blur_scores": compute_stats(data["blur_scores"], "blur"),
            "similarity_scores": compute_stats(data["similarity_scores"], "sim"),
        },
        "recommendations": recs,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    log.info(f"Calibration data saved to {out_path}")

    print_report(data, recs)


if __name__ == "__main__":
    main()