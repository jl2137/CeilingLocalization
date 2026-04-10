import cv2
import numpy as np
import json
from pathlib import Path

def build_database():
    print("=" * 60)
    print("Multi-Building Ceiling Image Database Builder")
    print("=" * 60)

    # Initialize ORB feature detector (increase number of keypoints for better matching)
    orb = cv2.ORB_create(nfeatures=1500)

    database = {}
    database_dir = Path("database")

    if not database_dir.exists():
        print("Error: database folder not found!")
        print("Please create a 'database' folder first and organize images by building inside it.")
        return

    # Get all building subdirectories
    building_dirs = [d for d in database_dir.iterdir() if d.is_dir()]
    if not building_dirs:
        print("Error: No building subdirectories found under the database folder!")
        print("Please create subdirectories like Mary_Burton, Colin_Maclaurin, etc.")
        return

    print(f"Found {len(building_dirs)} buildings: {[d.name for d in building_dirs]}")

    # Iterate through each building
    for building_dir in building_dirs:
        building_name = building_dir.name
        print(f"\n📁 Processing building: {building_name}")
        database[building_name] = {}

        # Recursively find all image files
        image_files = list(building_dir.rglob("*.jpg")) + list(building_dir.rglob("*.png"))
        if not image_files:
            print(f"   No images found in this building")
            continue

        print(f"   Found {len(image_files)} images")

        # Iterate through all images in the building
        for img_path in image_files:
            # Read as grayscale image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   Cannot read: {img_path.name}")
                continue


            height, width = img.shape
            if height > 800 or width > 800:
                scale = 800 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            img = cv2.equalizeHist(img)

            # Extract ORB features
            keypoints, descriptors = orb.detectAndCompute(img, None)

            if descriptors is None or len(keypoints) < 10:
                print(f"   Too few keypoints: {img_path.name}")
                continue

            # Generate location ID: relative path (relative to database folder) without extension, joined by underscores
            # Example: Mary_Burton/floor_1/pos_001 -> Mary_Burton_floor_1_pos_001
            relative_path = img_path.relative_to(database_dir)
            location_id = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
            print(f"   Location ID: {location_id} Keypoints: {len(keypoints)}")

            # Store keypoint coordinates (used for RANSAC)
            keypoints_data = []
            for kp in keypoints:
                keypoints_data.append({
                    "pt": (float(kp.pt[0]), float(kp.pt[1])),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response)
                })

            # Store in database
            database[building_name][location_id] = {
                "filename": str(img_path),
                "keypoints": keypoints_data,
                "descriptors": descriptors.tolist(),
                "keypoints_count": len(keypoints)
            }

    # Save database to JSON file
    output_file = "database.json"
    with open(output_file, "w") as f:
        json.dump(database, f, indent=2)

    print("\n" + "=" * 60)
    print("Database construction completed!")
    print(f"   Output file: {output_file}")
    print("=" * 60)
    return database

if __name__ == "__main__":
    build_database()