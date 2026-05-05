import os
import cv2
import sys
import numpy as np
from pathlib import Path

# Ensure we can import from the core module if run as a script
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.recognition import AdaFaceRecognizer
from core.config import load_merged_configs


def main():
    # 1. Load config
    config = load_merged_configs([
        'configs/default.yaml',
        'configs/thresholds.yaml',
        'configs/tensorrt.yaml'
    ])
    
    # 2. Get dataset paths
    dataset_cfg = config.get('dataset', {})
    aligned_dir = Path(dataset_cfg.get('aligned_faces', 'dataset/aligned_faces'))
    
    if not aligned_dir.exists():
        print(f"Error: Aligned faces directory '{aligned_dir}' does not exist.")
        print("Please run apps/dataset_tools/extract_faces.py first.")
        return

    # 3. Initialize AdaFaceRecognizer
    if 'models' not in config or 'adaface_onnx' not in config['models']:
        print("Error: AdaFace model path not found in config.")
        return
        
    model_path = config['models']['adaface_onnx']
    if not os.path.exists(model_path):
        print(f"Error: AdaFace model not found at {model_path}")
        print("Ensure the model file exists before running this script.")
        return

    print("Initializing AdaFace Recognizer...")
    recognizer = AdaFaceRecognizer(config, model_path)
    
    gallery = {}
    persons_processed = 0
    total_images_used = 0

    print(f"Starting gallery building...")
    print(f"Input: {aligned_dir}")
    print("-" * 40)

    # 4. Iterate through all subfolders inside aligned_faces (Dynamically discover identities)
    # The folder name is the identity key.
    subdirs = sorted([d for d in aligned_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        print(f"No identity subfolders found in {aligned_dir}")
        return

    for person_path in subdirs:
        person_id = person_path.name
        person_imgs = []
        
        # 5. Process each image in the person folder
        for img_path in person_path.iterdir():
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            person_imgs.append(img)
            total_images_used += 1

        if person_imgs:
            # Extract and normalize embeddings using batch processing
            person_embeddings, _ = recognizer.extract_embeddings_batch(person_imgs)
            
            # person_embeddings is already a numpy array of normalized vectors
            normalized_embeddings = list(person_embeddings)

            # limit to max 10 embeddings per person (for efficiency)
            MAX_EMB = 10
            normalized_embeddings = normalized_embeddings[:MAX_EMB]

            gallery[person_id] = normalized_embeddings
            persons_processed += 1
            print(f"Processed: {person_id:20} | Images: {len(person_embeddings):3d}")

    # 6. Save results
    output_path = Path(dataset_cfg.get('gallery_embeddings', 'dataset/gallery_embeddings.npy'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as dictionary
    np.save(str(output_path), gallery)

    # 7. Console Summary
    print("-" * 40)
    print("Gallery Build Summary:")
    print(f"Persons processed: {persons_processed}")
    print(f"Images used:      {total_images_used}")
    print(f"Gallery size:     {len(gallery)} identities")
    print(f"Saved to:         {output_path}")
    print("-" * 40)

if __name__ == "__main__":
    main()