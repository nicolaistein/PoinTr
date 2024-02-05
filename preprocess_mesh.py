import argparse
import os

import open3d as o3d
from tqdm import tqdm


def preprocess_mesh(input_dir, output_dir, num_points):
    file_count = 0
    for root, dirs, files in os.walk(input_dir):
        if files:
            file_count += len(files)

    with tqdm(total=file_count, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".off"):
                    mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))

                    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

                    relative_path = os.path.relpath(root, input_dir)
                    output_file_dir = os.path.join(output_dir, relative_path)
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)
                    o3d.io.write_point_cloud(
                        os.path.join(output_file_dir, file.replace(".off", ".ply")), pcd
                    )

                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ModelNet meshes")
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument(
        "--output_dir", type=str, help="output directory", required=True
    )
    parser.add_argument(
        "--num_points", type=int, default=10000, help="number of points to sample"
    )
    args = parser.parse_args()

    preprocess_mesh(args.input_dir, args.output_dir, args.num_points)
