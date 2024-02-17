import argparse
from calendar import c
import json
import os
import numpy as np

import open3d as o3d
from tqdm import tqdm


def sample_points_from_mesh(
    mesh, num_points, translate=None, view_id=None, scale_factor=1.0
):
    pcd = mesh.sample_points_uniformly(
        # pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points
    )

    if translate is not None:
        pcd = pcd.translate(translate)

    pcd.scale(scale=scale_factor, center=(0, 0, 0))

    if view_id is not None:
        # perform a cut along the a plane
        max_coord_crop = 1000000

        crop_coords = [
            -max_coord_crop,
            -max_coord_crop,
            -max_coord_crop,
            max_coord_crop,
            max_coord_crop,
            max_coord_crop,
        ]
        crop_coords[view_id] = 0

        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(crop_coords[0], crop_coords[1], crop_coords[2]),
                max_bound=(crop_coords[3], crop_coords[4], crop_coords[5]),
            )
        )

    return pcd


def preprocess_mesh_dir(
    input_dir, output_dir, num_points, class_id, pbar, overwrite_files
):
    view_id_count = 6

    file_names = []
    error_file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".off"):
                # get output file paths
                input_file_name = file.replace(".off", "")
                output_file_name_complete = input_file_name + ".pcd"

                relative_path = os.path.relpath(root, input_dir)
                output_file_dir_complete = os.path.join(output_dir, relative_path)
                output_file_dir_complete = os.path.join(
                    output_file_dir_complete, "complete"
                )
                output_file_dir_complete = os.path.join(
                    output_file_dir_complete, class_id
                )
                output_file_path_complete = os.path.join(
                    output_file_dir_complete, output_file_name_complete
                )

                output_file_dir_partial = os.path.join(output_dir, relative_path)
                output_file_dir_partial = os.path.join(
                    output_file_dir_partial, "partial"
                )
                output_file_dir_partial = os.path.join(
                    output_file_dir_partial, class_id
                )
                output_file_dir_partial = os.path.join(
                    output_file_dir_partial, input_file_name
                )
                output_file_path_partial_list = [
                    os.path.join(output_file_dir_partial, "0%s.pcd" % view_id)
                    for view_id in range(view_id_count)
                ]

                # check if any of the output files do not exist already
                if overwrite_files or (
                    not os.path.exists(output_file_path_complete)
                    or not all(
                        [os.path.exists(p) for p in output_file_path_partial_list]
                    )
                ):
                    try:
                        # read the mesh
                        mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))

                        # Note: normal computation is not working
                        # # compute the normals of the triangles
                        # mesh.compute_vertex_normals()
                        # mesh.compute_triangle_normals()
                        # triangle_normals = np.asarray(mesh.triangle_normals)

                        # # generate a random view direction
                        # view_dir = np.random.randn(3)
                        # view_dir /= np.linalg.norm(view_dir)

                        # # compute the dot product between the view direction and the triangle normals
                        # dot_products = np.dot(triangle_normals, view_dir)

                        # # remove all triangles not facing the view direction
                        # mesh.triangles = o3d.utility.Vector3iVector(
                        #     np.asarray(mesh.triangles)[dot_products > 0, :]
                        # )
                        # # mesh.remove_triangles_by_mask(dot_products <= 0)

                        # sample points from the mesh
                        pcd_complete = sample_points_from_mesh(mesh, num_points)

                        # find the center of the point cloud
                        center = pcd_complete.get_center()

                        # translate the point cloud so that its center is at the origin
                        pcd_complete = pcd_complete.translate(-center)

                        # nomalize the point cloud to fit in a unit sphere
                        scale_factor = 1 / np.max(pcd_complete.get_max_bound())
                        pcd_complete.scale(scale=scale_factor, center=(0, 0, 0))

                        num_points_partial = num_points // 5
                        pcd_partial_list = []
                        for view_id in range(view_id_count):
                            pcd_partial = sample_points_from_mesh(
                                mesh,
                                num_points_partial,
                                translate=-center,
                                view_id=view_id,
                                scale_factor=scale_factor,
                            )
                            pcd_partial_list.append(pcd_partial)

                        # save the point cloud
                        if not os.path.exists(output_file_dir_complete):
                            os.makedirs(output_file_dir_complete)
                        if not os.path.exists(output_file_dir_partial):
                            os.makedirs(output_file_dir_partial)
                        o3d.io.write_point_cloud(
                            output_file_path_complete, pcd_complete, write_ascii=True
                        )
                        for i, (output_file_path_partial, pcd_partial) in enumerate(
                            zip(output_file_path_partial_list, pcd_partial_list)
                        ):
                            o3d.io.write_point_cloud(
                                output_file_path_partial, pcd_partial, write_ascii=True
                            )

                    except Exception as e:
                        print(f"Error processing {os.path.join(root, file)}: {e}")
                        error_file_paths.append(os.path.join(root, file))
                        pbar.update(1)
                        continue

                file_names.append(input_file_name)
                pbar.update(1)

    return file_names, error_file_paths


def preprocess_mesh(
    input_dir, output_dir, num_points, class_name, class_id, overwrite_files
):
    np.random.seed(0)

    error_file_paths = []

    input_dir_class = os.path.join(input_dir, class_name)
    input_dir_class_train = os.path.join(input_dir_class, "train")
    input_dir_class_val = os.path.join(input_dir_class, "val")
    input_dir_class_test = os.path.join(input_dir_class, "test")

    output_dir_class = output_dir
    output_dir_class_train = os.path.join(output_dir_class, "train")
    output_dir_class_val = os.path.join(output_dir_class, "val")
    output_dir_class_test = os.path.join(output_dir_class, "test")

    # count files in input_dir_class_test and input_dir_class_train directories
    file_count = 0
    for root, dirs, files in os.walk(input_dir_class_test):
        file_count += len(files)
    for root, dirs, files in os.walk(input_dir_class_val):
        file_count += len(files)
    for root, dirs, files in os.walk(input_dir_class_train):
        file_count += len(files)

    with tqdm(total=file_count, desc="Processing files") as pbar:
        file_names_train, error_file_paths_train = preprocess_mesh_dir(
            input_dir_class_train,
            output_dir_class_train,
            num_points,
            class_id,
            pbar,
            overwrite_files,
        )
        file_names_val, error_file_paths_val = preprocess_mesh_dir(
            input_dir_class_val,
            output_dir_class_val,
            num_points,
            class_id,
            pbar,
            overwrite_files,
        )
        file_names_test, error_file_paths_test = preprocess_mesh_dir(
            input_dir_class_test,
            output_dir_class_test,
            num_points,
            class_id,
            pbar,
            overwrite_files,
        )

    json_data = [
        {
            "taxonomy_id": (class_id + ""),
            "taxonomy_name": class_name,
            "train": file_names_train,
            "val": file_names_val,
            "test": file_names_test,
        }
    ]

    with open(os.path.join(output_dir, "ModelNet.json"), "w") as f:
        json.dump(json_data, f)
    print("ModelNet.json file created")

    error_file_paths = (
        error_file_paths_train + error_file_paths_val + error_file_paths_test
    )

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
        "--num_points", type=int, default=16384, help="number of points to sample"
    )
    parser.add_argument(
        "--class_name", type=str, help="dataset class name", required=True
    )
    parser.add_argument("--class_id", type=str, help="dataset class id", required=True)
    parser.add_argument(
        "-O", "--overwrite_files", action="store_true", help="overwrite existing files"
    )
    args = parser.parse_args()

    preprocess_mesh(
        args.input_dir,
        args.output_dir,
        args.num_points,
        args.class_name,
        args.class_id,
        args.overwrite_files,
    )
