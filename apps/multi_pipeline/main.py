import cv2
import yaml
import time
import numpy as np
import math
import queue

from core.detection import SCRFDDetector
from core.recognition import AdaFaceRecognizer
from core.database import FaceDatabase
from core.camera.worker import CameraWorker

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():
    # Load configs
    default_cfg = load_config('configs/default.yaml')
    camera_cfg = load_config('configs/cameras.yaml')
    threshold_cfg = load_config('configs/thresholds.yaml')
    dataset_cfg = load_config('configs/dataset.yaml')
    
    # Merge configs
    config = {**default_cfg, **threshold_cfg}
    
    gallery_path = dataset_cfg['dataset']['gallery_embeddings']
    gallery_embeddings = np.load(gallery_path, allow_pickle=True).item()
    
    # Initialize shared components (thread-safe for inference)
    print("Initializing shared AI models...")
    detector = SCRFDDetector(config, config['models']['scrfd_onnx'])
    recognizer = AdaFaceRecognizer(config, config['models']['adaface_onnx'])
    face_db = FaceDatabase(gallery_embeddings)
    
    workers = []
    
    # Parse cameras from config
    cameras = [cam for cam in camera_cfg.get('cameras', []) if cam.get('enabled', True)]
    
    if not cameras:
        print("No enabled cameras found in configs/cameras.yaml")
        return
        
    print(f"Starting pipeline with {len(cameras)} cameras...")
    
    for cam_info in cameras:
        worker = CameraWorker(
            camera_id=cam_info['id'],
            camera_url=cam_info['url'],
            detector=detector,
            recognizer=recognizer,
            face_db=face_db,
            config=config
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    print("Pipeline running. Press 'q' to exit.")
    
    # Display config
    grid_cols = math.ceil(math.sqrt(len(workers)))
    grid_rows = math.ceil(len(workers) / grid_cols)
    cell_width = 640
    cell_height = 360
    
    # Pre-allocate frames dict
    latest_frames = {worker.camera_id: np.zeros((cell_height, cell_width, 3), dtype=np.uint8) for worker in workers}
    
    try:
        while True:
            # Poll frames from all workers
            for worker in workers:
                try:
                    frame = worker.frame_queue.get_nowait()
                    # Resize for grid
                    latest_frames[worker.camera_id] = cv2.resize(frame, (cell_width, cell_height))
                except queue.Empty:
                    pass
            
            # Construct grid
            grid_frame = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)
            
            for idx, worker in enumerate(workers):
                r = idx // grid_cols
                c = idx % grid_cols
                y1 = r * cell_height
                y2 = (r + 1) * cell_height
                x1 = c * cell_width
                x2 = (c + 1) * cell_width
                
                frame = latest_frames[worker.camera_id]
                grid_frame[y1:y2, x1:x2] = frame
                
                # Add camera label
                cv2.putText(grid_frame, worker.camera_id, (x1 + 10, y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imshow('Ha-Meem AI Surveillance (Multi-Camera)', grid_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.01) # Small sleep, not too long
            
    finally:
        print("Shutting down pipeline...")
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()