import argparse
import json
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm


def np_array_to_uin8_image(x):
    img = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def sample_projected_point_cloud_from_mesh(
    mesh_tensor, partial_points_multiplier, translate=None, scale_factor=1.0
):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10).translate((0, 0, 0))
    # sphere_tensor = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    # scene.add_triangles(sphere_tensor)

    # create random eye vector
    eye = np.random.randn(3)
    eye /= np.linalg.norm(eye)
    eye *= 5

    pixel_multiplier = np.sqrt(partial_points_multiplier)
    width_px = round(190 * pixel_multiplier)
    height_px = round(190 * pixel_multiplier)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=35,
        center=[0, 0, 0],
        eye=eye,
        up=[0, 1, 0],
        width_px=width_px,
        height_px=height_px,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    hit = ans["t_hit"].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points).to_legacy()

    if translate is not None:
        pcd = pcd.translate(translate)

    pcd.scale(scale=scale_factor, center=(0, 0, 0))

    return pcd


def sample_uniform_point_cloud_from_mesh(mesh, num_points):
    pcd = mesh.sample_points_uniformly(
        # pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points
    )

    return pcd


total_complete_point_clouds = 0
total_complete_point_cloud_points = 0
total_partial_point_clouds = 0
total_partial_point_cloud_points = 0


def preprocess_mesh_dir(
    input_dir,
    output_dir,
    num_points,
    partial_points_multiplier,
    class_id,
    pbar,
    overwrite_files,
):
    view_id_count = 8

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
                        input_file_path = os.path.join(root, file)

                        off_file_header = "OFF"
                        # check off file header
                        file_lines = []
                        with open(input_file_path, "r") as f:
                            file_lines = f.readlines()
                        if file_lines[0] != f"{off_file_header}\n":
                            if file_lines[0].startswith(off_file_header):
                                # fix header
                                file_lines[0] = file_lines[0][3:]
                                file_lines.insert(0, "OFF\n")
                                with open(input_file_path, "w") as fout:
                                    contents = "".join(file_lines)
                                    fout.write(contents)
                                print("Fixed OFF header in", input_file_path)
                            else:
                                raise Exception(
                                    f"First line of {input_file_path} is not 'OFF'"
                                )

                        # read the mesh
                        mesh = o3d.io.read_triangle_mesh(input_file_path)

                        # center mesh
                        mesh = mesh.translate(-mesh.get_center())
                        # normalize mesh to fit in a unit sphere
                        mesh = mesh.scale(
                            0.5 / np.max(np.abs(mesh.get_max_bound())), center=(0, 0, 0)
                        )

                        mesh = mesh.rotate(
                            mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)),
                            center=(0, 0, 0),
                        )

                        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                        # sample points from the mesh
                        pcd_complete = sample_uniform_point_cloud_from_mesh(
                            mesh, num_points
                        )

                        # find the center of the point cloud
                        center = pcd_complete.get_center()

                        # translate the point cloud so that its center is at the origin
                        pcd_complete = pcd_complete.translate(-center)

                        # nomalize the point cloud to fit in a unit sphere
                        scale_factor = 1 / np.max(np.abs(pcd_complete.get_max_bound()))
                        pcd_complete.scale(scale=scale_factor, center=(0, 0, 0))

                        # num_points_partial = num_points // 5
                        pcd_partial_list = []
                        for view_id in range(view_id_count):
                            pcd_partial = sample_projected_point_cloud_from_mesh(
                                mesh_tensor,
                                partial_points_multiplier,
                                translate=-center,
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

                        global total_complete_point_clouds
                        global total_complete_point_cloud_points
                        global total_partial_point_clouds
                        global total_partial_point_cloud_points
                        total_complete_point_clouds += 1
                        total_complete_point_cloud_points += len(pcd_complete.points)
                        for pcd in pcd_partial_list:
                            total_partial_point_clouds += 1
                            total_partial_point_cloud_points += len(pcd.points)

                    except Exception as e:
                        print(f"Error processing {os.path.join(root, file)}: {e}")
                        error_file_paths.append(os.path.join(root, file))
                        pbar.update(1)
                        continue

                file_names.append(input_file_name)
                pbar.update(1)

    return file_names, error_file_paths


def preprocess_mesh(
    input_dir,
    output_dir,
    num_points,
    partial_points_multiplier,
    class_ids,
    overwrite_files,
    no_val,
):
    np.random.seed(0)

    error_file_paths = []
    json_data = []

    for i, class_id in enumerate(class_ids):
        print(f"Processing class '{class_id}' [{i+1}/{len(class_ids)}]:")

        input_dir_class = os.path.join(input_dir, class_id)
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
        if not no_val:
            for root, dirs, files in os.walk(input_dir_class_val):
                file_count += len(files)
        for root, dirs, files in os.walk(input_dir_class_train):
            file_count += len(files)

        with tqdm(total=file_count, desc="Processing files") as pbar:
            file_names_train, error_file_paths_train = preprocess_mesh_dir(
                input_dir_class_train,
                output_dir_class_train,
                num_points,
                partial_points_multiplier,
                class_id,
                pbar,
                overwrite_files,
            )
            file_names_val, error_file_paths_val = [], []
            if not no_val:
                file_names_val, error_file_paths_val = preprocess_mesh_dir(
                    input_dir_class_val,
                    output_dir_class_val,
                    num_points,
                    partial_points_multiplier,
                    class_id,
                    pbar,
                    overwrite_files,
                )
            file_names_test, error_file_paths_test = preprocess_mesh_dir(
                input_dir_class_test,
                output_dir_class_test,
                num_points,
                partial_points_multiplier,
                class_id,
                pbar,
                overwrite_files,
            )

        class_json_data = {
            "taxonomy_id": class_id,
            "taxonomy_name": class_id,
            "train": file_names_train,
            "test": file_names_test,
        }
        if not no_val:
            class_json_data["val"] = file_names_val

        json_data.append(class_json_data)

    json_file_name = "ModelNet.json"

    with open(os.path.join(output_dir, json_file_name), "w") as f:
        json.dump(json_data, f)
    print(f"{json_file_name} file created")

    global total_complete_point_clouds
    global total_complete_point_cloud_points
    global total_partial_point_clouds
    global total_partial_point_cloud_points
    if total_complete_point_clouds > 0:
        average_complete_point_cloud_points = (
            total_complete_point_cloud_points / total_complete_point_clouds
        )
        print(
            f"Average number of points in complete point clouds: {average_complete_point_cloud_points}"
        )
    if total_partial_point_clouds > 0:
        average_partial_point_cloud_points = (
            total_partial_point_cloud_points / total_partial_point_clouds
        )
        print(
            f"Average number of points in partial point clouds: {average_partial_point_cloud_points}"
        )

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
        "--partial_points_multiplier",
        type=float,
        default=1.0,
        help="multiplier for the number of points to sample for partial point clouds",
    )
    parser.add_argument(
        "--class_ids", type=str, help="comma-separated list of class IDs", required=True
    )
    parser.add_argument(
        "-O", "--overwrite_files", action="store_true", help="overwrite existing files"
    )
    parser.add_argument(
        "--no_val", action="store_true", help="do not export validation set"
    )
    args = parser.parse_args()

    class_ids = args.class_ids.split(",")
    print(f"Class IDs: {class_ids}")

    preprocess_mesh(
        args.input_dir,
        args.output_dir,
        args.num_points,
        args.partial_points_multiplier,
        class_ids,
        args.overwrite_files,
        args.no_val,
    )
