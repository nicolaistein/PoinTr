import os
import sys
import open3d as o3d

def process_pcd_file(file_path):
    # Step 1: Load the pcd file
    pcd = o3d.io.read_point_cloud(file_path)

    # Step 2: Add normals for all points
    pcd.estimate_normals()

    # Step 3: Save the pointcloud to the same .pcd file
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pcd"):
                file_path = os.path.join(root, file)
                process_pcd_file(file_path)
                print(f"Processed: {file_path}")

        for dir in dirs:
            process_folder(os.path.join(root, dir))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python precompute.py /path/to/your/folder")
        sys.exit(1)

    folder_path = sys.argv[1]

    if os.path.exists(folder_path):
        process_folder(folder_path)
        print("Processing completed.")
    else:
        print(f"Error: Folder not found - {folder_path}")
