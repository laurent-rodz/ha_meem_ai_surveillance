import json
import os
import sys
import hashlib
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict

# Force UTF-8 on stdout so non-ASCII chars don't crash when redirecting on Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def _log(msg):
    """Write diagnostic messages to stderr — never pollutes the markdown output."""
    print(msg, file=sys.stderr)


import contextlib

@contextlib.contextmanager
def _silence_fd1():
    """
    Redirect stdout at the C file-descriptor level (fd 1) to devnull.
    This silences C++ library prints (e.g. ORT's 'Applied providers') that
    bypass Python's sys.stdout and write straight to the underlying fd.
    Safe to use even when stdout is already redirected to a file.
    """
    sys.stdout.flush()
    try:
        fd1 = sys.stdout.fileno()  # usually 1, but may differ when redirected
    except Exception:
        # If stdout has no real fd (e.g. in some IDEs), just yield and do nothing.
        yield
        return

    saved_fd = os.dup(fd1)          # keep a copy of the real destination
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, fd1)        # point fd1 -> /dev/null
    os.close(devnull_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, fd1)      # restore original destination
        os.close(saved_fd)


# ---------------------------------------------------------------------------
# Face Crop Extraction
# ---------------------------------------------------------------------------

def crop_with_bbox(frame, bbox, output_path, pad_pct=0.20):
    """
    Crop a face from `frame` using a pre-known [x1,y1,x2,y2] bounding box,
    add padding, resize to 100×100 and save to output_path.
    Returns True on success.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    w, h = x2 - x1, y2 - y1

    pad_x = max(10, int(w * pad_pct))
    pad_y = max(10, int(h * pad_pct))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(fw, x2 + pad_x)
    y2 = min(fh, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    face_100 = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, face_100, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def extract_face_crops(snapshots_needed: dict, project_root: str, output_dir: str) -> dict:
    """
    Produce 100×100 face crops for a collection of snapshots.

    `snapshots_needed` is a dict:
        { snapshot_path -> bbox_or_None }

    Strategy (priority order):
      1. bbox present in log  →  direct crop, zero inference cost.
      2. bbox absent (legacy) →  re-detect with SCRFD (CPU), pick largest face.

    Returns: { snapshot_path -> cropped_image_path | None }
    """
    if not snapshots_needed:
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Split into fast (bbox) and slow (SCRFD) buckets
    bbox_snaps  = {p: b for p, b in snapshots_needed.items() if p and b is not None}
    scrfd_snaps = [p for p, b in snapshots_needed.items() if p and b is None]

    crop_map = {}

    # ------------------------------------------------------------------
    # Fast path — bbox already known
    # ------------------------------------------------------------------
    for snap_path, bbox in bbox_snaps.items():
        abs_path = snap_path if os.path.isabs(snap_path) else os.path.join(project_root, snap_path)
        if not os.path.exists(abs_path):
            _log(f"[report] Snapshot not found: {abs_path}")
            crop_map[snap_path] = None
            continue

        frame = cv2.imread(abs_path)
        if frame is None:
            crop_map[snap_path] = None
            continue

        snap_hash = hashlib.md5(snap_path.encode()).hexdigest()[:10]
        out_path  = os.path.join(output_dir, f"face_{snap_hash}.jpg")

        if crop_with_bbox(frame, bbox, out_path):
            crop_map[snap_path] = out_path
            _log(f"[report] (bbox) Cropped -> {out_path}")
        else:
            crop_map[snap_path] = None

    # ------------------------------------------------------------------
    # Slow path — re-detect with SCRFD (legacy logs without bbox)
    # ------------------------------------------------------------------
    if scrfd_snaps:
        detector = None
        model_path = os.path.join(project_root, 'models', 'scrfd_10g_bnkps.onnx')

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                from insightface.model_zoo import get_model

                # Silence ORT's "Applied providers" noise from stdout
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3  # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR

                with _silence_fd1():
                    detector = get_model(model_path)
                    session  = ort.InferenceSession(model_path, sess_options=sess_opts,
                                                    providers=['CPUExecutionProvider'])
                    detector.session = session
                    detector.prepare(ctx_id=-1, input_size=(640, 640))
                _log(f"[report] SCRFD loaded (CPU) for {len(scrfd_snaps)} legacy snapshot(s)")
            except Exception as e:
                _log(f"[report] Could not load SCRFD: {e}. Legacy crops will be skipped.")
        else:
            _log(f"[report] SCRFD model not found at {model_path}. Legacy crops skipped.")

        for snap_path in scrfd_snaps:
            abs_path = snap_path if os.path.isabs(snap_path) else os.path.join(project_root, snap_path)
            if not os.path.exists(abs_path):
                _log(f"[report] Snapshot not found: {abs_path}")
                crop_map[snap_path] = None
                continue

            frame = cv2.imread(abs_path)
            if frame is None:
                crop_map[snap_path] = None
                continue

            if detector is None:
                crop_map[snap_path] = None
                continue

            try:
                bboxes, _ = detector.detect(frame)
            except Exception as e:
                _log(f"[report] Detection failed on {snap_path}: {e}")
                crop_map[snap_path] = None
                continue

            if bboxes is None or len(bboxes) == 0:
                _log(f"[report] No face found in: {snap_path}")
                crop_map[snap_path] = None
                continue

            # Pick largest face by area
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            best  = bboxes[int(np.argmax(areas))]

            snap_hash = hashlib.md5(snap_path.encode()).hexdigest()[:10]
            out_path  = os.path.join(output_dir, f"face_{snap_hash}.jpg")

            if crop_with_bbox(frame, best[:4], out_path):
                crop_map[snap_path] = out_path
                _log(f"[report] (SCRFD) Cropped -> {out_path}")
            else:
                crop_map[snap_path] = None

    return crop_map



# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_markdown_report(log_file):
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

    # 1. Read and parse the logs
    with open(log_file, 'r') as f:
        logs = [json.loads(line.strip()) for line in f if line.strip()]

    if not logs:
        print("No logs found.")
        return

    # 2. Initialize tracking variables
    total_events = len(logs)
    auth_count = 0
    unknown_count = 0
    track_ids = set()
    identities = {}
    auth_scores = []
    unknown_scores = []
    unknowns = []
    hourly_activity = defaultdict(lambda: {'auth': 0, 'unknown': 0, 'total': 0})
    timestamps = []

    # 3. Process each log entry
    for log in logs:
        dt = datetime.fromisoformat(log['timestamp'])
        timestamps.append(dt)
        hour_key = dt.strftime('%H:00')

        track_ids.add(log['track_id'])
        hourly_activity[hour_key]['total'] += 1

        if log['event'] == 'AUTHORIZED':
            auth_count += 1
            auth_scores.append(log['score'])
            hourly_activity[hour_key]['auth'] += 1

            # Clean up identity name (e.g., "58_towhid" -> "Towhid")
            raw_id = log['identity']
            if raw_id:
                name = raw_id.split('_')[-1].capitalize()
                if name not in identities:
                    identities[name] = {
                        'count': 0, 'first': dt, 'last': dt,
                        'min_score': float('inf'),  'min_snapshot': None,
                        'max_score': float('-inf'), 'max_snapshot': None,
                        # bbox from log (populated by newer pipeline versions)
                        'min_bbox': log.get('bbox'),
                        'max_bbox': log.get('bbox'),
                    }

                identities[name]['count'] += 1
                identities[name]['first'] = min(identities[name]['first'], dt)
                identities[name]['last']  = max(identities[name]['last'],  dt)

                score    = log.get('score', 0)
                snapshot = log.get('snapshot')
                bbox     = log.get('bbox')   # [x1,y1,x2,y2] if available

                if score < identities[name]['min_score']:
                    identities[name]['min_score']    = score
                    identities[name]['min_snapshot'] = snapshot
                    identities[name]['min_bbox']     = bbox

                if score > identities[name]['max_score']:
                    identities[name]['max_score']    = score
                    identities[name]['max_snapshot'] = snapshot
                    identities[name]['max_bbox']     = bbox

        elif log['event'] == 'UNKNOWN':
            unknown_count += 1
            unknown_scores.append(log['score'])
            hourly_activity[hour_key]['unknown'] += 1
            unknowns.append({
                'timestamp': dt,
                'score': log.get('score', 0),
                'snapshot': log.get('snapshot'),
                'bbox': log.get('bbox'),
                'track_id': log['track_id']
            })

    # Process top 10 unknowns by highest score (unique track_ids)
    unknown_tracks = {}
    for u in unknowns:
        tid = u['track_id']
        if tid not in unknown_tracks or u['score'] > unknown_tracks[tid]['score']:
            unknown_tracks[tid] = u
    top_unknowns = sorted(unknown_tracks.values(), key=lambda x: x['score'], reverse=True)[:10]

    # 4. Extract face crops for all required snapshots
    #    Build a dict { snapshot_path -> bbox_or_None } so the extractor
    #    can use the fast bbox-crop path when available.
    snapshots_needed = {}
    for data in identities.values():
        if data['min_snapshot']:
            # Only set bbox if not already stored with a bbox for this path
            if data['min_snapshot'] not in snapshots_needed:
                snapshots_needed[data['min_snapshot']] = data.get('min_bbox')
        if data['max_snapshot']:
            if data['max_snapshot'] not in snapshots_needed:
                snapshots_needed[data['max_snapshot']] = data.get('max_bbox')

    for data in top_unknowns:
        if data['snapshot']:
            if data['snapshot'] not in snapshots_needed:
                snapshots_needed[data['snapshot']] = data.get('bbox')

    face_output_dir = os.path.join(project_root, 'report_assets', 'faces')
    crop_map = extract_face_crops(snapshots_needed, project_root, face_output_dir)

    # 5. Calculate Summary Statistics
    start_time = min(timestamps)
    end_time   = max(timestamps)
    date_str = start_time.strftime('%B %d, %Y')
    duration_mins = int((end_time - start_time).total_seconds() / 60)

    unique_persons_count = len(identities)
    auth_pct    = (auth_count    / total_events) * 100 if total_events else 0
    unknown_pct = (unknown_count / total_events) * 100 if total_events else 0

    # ------------------------------------------------------------------
    # Helper: render an <img> tag using the cropped face path
    # ------------------------------------------------------------------
    def format_img(snapshot_path):
        if not snapshot_path:
            return "N/A"
        cropped = crop_map.get(snapshot_path)
        if not cropped:
            return "N/A"
        # Use a relative path from the project root for portability
        rel = os.path.relpath(cropped, project_root).replace("\\", "/")
        return f'<img src="{rel}" width="100" height="100" alt="face" class="face-img">'

    # 6. Generate Markdown Output
    print("<style>")
    print("body { font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; }")
    print("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }")
    print("th { background-color: #f8f9fa; color: #2c3e50; font-weight: 600; text-align: left; padding: 12px; border-bottom: 2px solid #e9ecef; }")
    print("td { padding: 12px; border-bottom: 1px solid #e9ecef; vertical-align: middle; }")
    print("h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 30px; }")
    print(".badge-auth { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }")
    print(".badge-unknown { background-color: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }")
    print(".face-img { border-radius: 12px; object-fit: cover; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 2px solid #fff; }")
    print("</style>\n")

    print(f"### **Surveillance Log Report - {date_str}**\n")
    print(f"**Camera:** {logs[0]['camera_id']}")
    print(f"**Active window:** {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} (~{duration_mins}min)")
    print(f"**Total detection events:** {total_events}\n")
    print("---\n")

    print("### **Overall Summary**\n")
    print("| Metric | Value |")
    print("| :--- | :--- |")
    print(f"| **Total Events** | {total_events} |")
    print(f"| <span class=\"badge-auth\">AUTHORIZED</span> | {auth_count} ({auth_pct:.1f}%) |")
    print(f"| <span class=\"badge-unknown\">UNKNOWN</span> | {unknown_count} ({unknown_pct:.1f}%) |")
    print(f"| **Unique Identified Persons** | {unique_persons_count} |")
    print(f"| **Unique Track IDs** | {len(track_ids)} |\n")
    print("---\n")

    print("### **Hourly Activity Breakdown**\n")
    print("| Hour | <span class=\"badge-auth\">AUTHORIZED</span> | <span class=\"badge-unknown\">UNKNOWN</span> | Total |")
    print("| :--- | :--- | :--- | :--- |")

    peak_hour = max(hourly_activity, key=lambda k: hourly_activity[k]['total'])
    for hour in sorted(hourly_activity.keys()):
        data = hourly_activity[hour]
        peak_str = " Peak" if hour == peak_hour else ""
        print(f"| **{hour}** | {data['auth']} | {data['unknown']} | {data['total']}{peak_str} |")
    print("\n---\n")

    print("### **Recognition Score Statistics**\n")
    print("| Type | Min | Max | Avg |")
    print("| :--- | :--- | :--- | :--- |")

    if auth_scores:
        print(f"| <span class=\"badge-auth\">AUTHORIZED</span> match scores | {min(auth_scores):.4f} | {max(auth_scores):.4f} | {sum(auth_scores)/len(auth_scores):.4f} |")
    if unknown_scores:
        print(f"| <span class=\"badge-unknown\">UNKNOWN</span> detection scores | {min(unknown_scores):.4f} | {max(unknown_scores):.4f} | {sum(unknown_scores)/len(unknown_scores):.4f} |")
    print("\n---\n")

    sorted_identities = sorted(identities.items(), key=lambda x: x[1]['count'], reverse=True)

    print("### **Top 10 Most Detected Identities**\n")
    print("| Rank | Identity | Detections | Lowest Conf. (face) | Highest Conf. (face) | First Seen | Last Seen |")
    print("| :--- | :--- | :--- | :---: | :---: | :--- | :--- |")

    for i in range(10):
        if i < len(sorted_identities):
            name, data = sorted_identities[i]
            first_str = data['first'].strftime('%H:%M')
            last_str  = data['last'].strftime('%H:%M')
            img_min   = format_img(data['min_snapshot'])
            img_max   = format_img(data['max_snapshot'])
            score_min = data['min_score'] if data['min_score'] != float('inf')  else 0
            score_max = data['max_score'] if data['max_score'] != float('-inf') else 0
            print(
                f"| **{i+1}** | {name} | {data['count']} "
                f"| {img_min}<br>({score_min:.3f}) "
                f"| {img_max}<br>({score_max:.3f}) "
                f"| {first_str} | {last_str} |"
            )
        else:
            print(f"| **{i+1}** | - | - | - | - | - | - |")

    print(f"\n*(Note: Only {unique_persons_count} unique individuals were detected during this window.)*\n")
    print("---\n")

    print(f"### **Full Identified Persons** - {unique_persons_count} individuals\n")
    print("| Identity | Detections | Lowest Conf. (face) | Highest Conf. (face) | Window |")
    print("| :--- | :--- | :---: | :---: | :--- |")
    for name, data in sorted_identities:
        first_str = data['first'].strftime('%H:%M')
        last_str  = data['last'].strftime('%H:%M')
        window    = f"{first_str} - {last_str}" if first_str != last_str else first_str
        img_min   = format_img(data['min_snapshot'])
        img_max   = format_img(data['max_snapshot'])
        score_min = data['min_score'] if data['min_score'] != float('inf')  else 0
        score_max = data['max_score'] if data['max_score'] != float('-inf') else 0
        print(
            f"| {name} | {data['count']} "
            f"| {img_min}<br>({score_min:.3f}) "
            f"| {img_max}<br>({score_max:.3f}) "
            f"| {window} |"
        )
    print("\n---\n")

    print("### **Top 10 Unauthorized Persons**\n")
    print("| Rank | Time | Detection Score | Face |")
    print("| :--- | :--- | :---: | :---: |")
    for i in range(10):
        if i < len(top_unknowns):
            data = top_unknowns[i]
            time_str = data['timestamp'].strftime('%H:%M:%S')
            score = data['score']
            img = format_img(data['snapshot'])
            print(f"| **{i+1}** | {time_str} | {score:.3f} | {img} |")
        else:
            print(f"| **{i+1}** | - | - | - |")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'events_11_may.jsonl')
    generate_markdown_report(log_file_path)