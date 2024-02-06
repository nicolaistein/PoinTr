import argparse
import os

import open3d as o3d
from tqdm import tqdm


def sample_points_from_mesh(mesh, num_points):
    pcd = mesh.sample_points_uniformly(
        # pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points
    )

    # find the center of the point cloud
    center = pcd.get_center()

    # translate the point cloud so that its center is at the origin
    pcd.translate(-center)
    return pcd


def preprocess_mesh(input_dir, output_dir, num_points):
    file_count = 0
    for root, dirs, files in os.walk(input_dir):
        if files:
            file_count += len(files)

    error_file_paths = []

    with tqdm(total=file_count, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".off"):
                    # get output file paths
                    relative_path = os.path.relpath(root, input_dir)
                    output_file_name = file.replace(".off", ".pcd")
                    output_file_dir_complete = os.path.join(output_dir, "complete")
                    output_file_dir_complete = os.path.join(
                        output_file_dir_complete, relative_path
                    )
                    output_file_path_complete = os.path.join(
                        output_file_dir_complete, output_file_name
                    )
                    output_file_dir_partial = os.path.join(output_dir, "partial")
                    output_file_dir_partial = os.path.join(
                        output_file_dir_partial, relative_path
                    )
                    output_file_path_partial = os.path.join(
                        output_file_dir_partial, output_file_name
                    )

                    # check if the output file already exists
                    if not os.path.exists(
                        output_file_path_complete
                    ) or not os.path.exists(output_file_path_partial):
                        try:
                            # read the mesh
                            mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))

                            num_points_partial = num_points // 10

                            # sample points from the mesh
                            pcd_complete = sample_points_from_mesh(mesh, num_points)
                            pcd_partial = sample_points_from_mesh(
                                mesh, num_points_partial
                            )

                            # perform a cut along the xz plane
                            max_coord_crop = 1000000
                            pcd_partial = pcd_partial.crop(
                                o3d.geometry.AxisAlignedBoundingBox(
                                    min_bound=(-max_coord_crop, 0, -max_coord_crop),
                                    max_bound=(
                                        max_coord_crop,
                                        max_coord_crop,
                                        max_coord_crop,
                                    ),
                                )
                            )

                            # save the point cloud
                            if not os.path.exists(output_file_dir_complete):
                                os.makedirs(output_file_dir_complete)
                            if not os.path.exists(output_file_dir_partial):
                                os.makedirs(output_file_dir_partial)
                            o3d.io.write_point_cloud(
                                os.path.join(
                                    output_file_dir_complete, output_file_name
                                ),
                                pcd_complete,
                            )
                            o3d.io.write_point_cloud(
                                os.path.join(output_file_dir_partial, output_file_name),
                                pcd_partial,
                            )
                        except Exception as e:
                            print(f"Error processing {os.path.join(root, file)}: {e}")
                            error_file_paths.append(os.path.join(root, file))

                    pbar.update(1)

    if error_file_paths:
        print("Error processing the following files:")
        for file_path in error_file_paths:
            print(file_path)


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
