#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
import torch.nn.functional as F

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic_feature: torch.tensor 
    semantic_feature_path: str 
    semantic_feature_name: str 
    semantic_dim: int = 0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantic_feature_dim: int 

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,
                      semantic_feature_folder, bulk_on_device=True, langsplat_gt=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) 

        
        if langsplat_gt:
            semantic_feature = None
            semantic_feature_path = None
            semantic_feature_name = None
            semantic_dim = None
        else:
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path)
            if not bulk_on_device and'langseg' in semantic_feature_folder:
                semantic_dim = semantic_feature.shape[0]
                # semantic_feature = F.interpolate(
                #             semantic_feature.unsqueeze(0), size=(semantic_feature.shape[1] // 16, semantic_feature.shape[2] // 16),
                #             mode='nearest').squeeze(0)
                semantic_feature = semantic_feature_path
            elif not bulk_on_device:
                if isinstance(semantic_feature, dict):
                    for key in semantic_feature.keys():
                        if isinstance(semantic_feature[key], torch.Tensor):
                            semantic_feature[key] = semantic_feature[key].to('cpu')
                            semantic_dim = semantic_feature[key][0]
                else:
                    semantic_feature = semantic_feature.to('cpu')
                    semantic_dim = semantic_feature[0]
            else:
                semantic_dim = semantic_feature[0] if not isinstance(semantic_feature, dict) \
                                else semantic_feature[list(semantic_feature.keys())[0]][0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name,
                              semantic_dim=semantic_dim) 
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, add_sky=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if add_sky:
        positions, colors = add_sky_points(positions, colors)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def drop_non_existent_images(cam_extrinsics, images_folder, drop_by_feature=False, feature_folder=None):
    keys_to_drop = []
    for key, extr in cam_extrinsics.items():
        if not (Path(images_folder) / os.path.basename(extr.name)).exists():
            keys_to_drop.append(key)
        elif drop_by_feature and not (feature_folder /  Path(Path(os.path.basename(extr.name)).stem + '_fmap_CxHxW.pt')).exists():
            keys_to_drop.append(key)
    for key in keys_to_drop:
        del cam_extrinsics[key]
    return cam_extrinsics
from pathlib import Path

def generate_upper_half_sphere(center, radius, num_points=1000):
    """
    Generate points evenly distributed on the upper half-sphere.

    Parameters:
        center (np.ndarray): The center of the sphere (shape (3,)).
        radius (float): The radius of the sphere.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: An array of shape (num_points, 3) containing the points.
    """
    # Generate polar angles (theta) from 0 to π/2 (upper hemisphere)
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    cos_theta = np.random.uniform(0, 1, num_points)  # Uniform cosine of polar angle for upper half
    theta = np.arccos(cos_theta)  # Polar angle
    
    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    # Offset points by the center
    points = np.stack((x, y, z), axis=-1) + center
    
    return points

def add_sky_points(xyz, rgb, radius_factor=100, num_points=10000):
    minimums = np.min(xyz, axis=0)
    maximums = np.max(xyz, axis=0)
    radius = radius_factor * np.max(maximums - minimums)
    center = np.mean(xyz, axis=0)
    center[-1] = minimums[-1]
    sky_xyz = generate_upper_half_sphere(center, radius, num_points)
    sky_rgb = np.ones_like(sky_xyz) * np.array([[135, 206, 235]]) / 255
    xyz = np.concatenate((xyz, sky_xyz), axis=0)
    rgb = np.concatenate((rgb, sky_rgb), axis=0)
    return xyz, rgb

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8, is_train=True,
                        add_sky=False, bulk_on_device=True, langsplat_gt=False):
    try:
        if (Path(path) / 'sparse' / 'images.bin').exists():
            cameras_extrinsic_file = str(Path(path) / 'sparse' / 'images.bin')
            cameras_intrinsic_file = (Path(path) / 'sparse' / 'cameras.bin')
        else:
            cameras_extrinsic_file = str(Path(path) / 'sparse' / '0' / 'images.bin')
            cameras_intrinsic_file = (Path(path) / 'sparse' / '0' / 'cameras.bin')

        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        if (Path(path) / 'sparse' / 'images.txt').exists():
            cameras_extrinsic_file = str(Path(path) / 'sparse' / 'images.txt')
            cameras_intrinsic_file = (Path(path) / 'sparse' / 'cameras.txt')
        else:
            cameras_extrinsic_file = str(Path(path) / 'sparse' / '0' / 'images.txt')
            cameras_intrinsic_file = (Path(path) / 'sparse' / '0' / 'cameras.txt')
            
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images

    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 
    else:
        semantic_feature_dir = foundation_model
    
    drop_by_feature = (not langsplat_gt) and (not is_train) 
    cam_extrinsics = drop_non_existent_images(cam_extrinsics, os.path.join(path, reading_dir), 
                                              drop_by_feature=drop_by_feature, feature_folder = Path(path) / semantic_feature_dir)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir),
        semantic_feature_folder=os.path.join(path, semantic_feature_dir),
        bulk_on_device=bulk_on_device, langsplat_gt=langsplat_gt)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    ###cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('.')[0])) ### if img name is number
    # cam_infos =cam_infos[:30] ###: for scannet only
    # print(cam_infos)
    if langsplat_gt:
        semantic_feature_dim = 512
    elif isinstance(cam_infos[0].semantic_feature, dict):
        keys = list(cam_infos[0].semantic_feature.keys())
        semantic_feature_dim = cam_infos[0].semantic_feature[keys[0]].shape[0]
    elif isinstance(cam_infos[0].semantic_feature, str):
        semantic_feature_dim = cam_infos[0].semantic_dim
    else:
        semantic_feature_dim = cam_infos[0].semantic_feature.shape[0]

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2] # avoid 1st to be test view
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2] 
        # for i, item in enumerate(test_cam_infos): ### check test set
        #     print('test image:', item[7])
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    
    if (Path(path) / 'sparse' / 'images.bin').exists():
        ply_path = os.path.join(path, "sparse/points3D.ply")
        bin_path = os.path.join(path, "sparse/points3D.bin")
        txt_path = os.path.join(path, "sparse/points3D.txt")
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        if add_sky:
            xyz, rgb = add_sky_points(xyz, rgb)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path, add_sky=add_sky)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_folder, extension=".png"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name)) 
            
    return cam_infos

def readNerfSyntheticInfo(path, foundation_model, white_background, eval, extension=".png"): 
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

def nerstudio_c2w_to_colmap_w2c(c2w_init):
    w2c = c2w_init.copy()
    w2c[2, :] *= -1
    w2c = w2c[np.array([1, 0, 2, 3]), :]
    w2c[0:3, 1:3] *= -1
    w2c = np.linalg.inv(w2c)
    return w2c

def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length



def readKeyframesCameras_lerf(keyframefile, white_background, extension=".jpg"):
    cameras_extrinsic_file = "/storage/shai/3d/data/rgb_data/gal_figurine/images.txt"
    cameras_intrinsic_file = "/storage/shai/3d/data/rgb_data/gal_figurine/cameras.txt"
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        focal_length_x = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
        image_fake = Image.fromarray(np.array(np.ones((height, width, 3)) * 255.0, dtype=np.byte), "RGB")
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_fake,
                                    image_path='', image_name='', width=width,
                                    height=height,
                                    semantic_feature=None,
                            semantic_feature_path=None,
                            semantic_feature_name=None))
    return cam_infos

def readKeyframesCameras_lerf_mycolmap(keyframefile, white_background, extension=".jpg"):
    cam_infos = []

    with open(keyframefile) as json_file:
        contents = json.load(json_file)

        frames = contents["keyframes"]
        img_width = contents['render_width']
        img_height = contents['render_height']

        for idx, frame in enumerate(frames):
            w2c = np.asarray(frame["transformed_mycolmap_w2c_matrix"])
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_fake = Image.fromarray(np.array(np.ones((img_height, img_width, 3)) * 255.0, dtype=np.byte), "RGB")

            fov = frame['fov']
            focal_length = three_js_perspective_camera_focal_length(fov, img_height)
            # # focal_length_x = three_js_perspective_camera_focal_length(fov, img_width)
            FovY = focal2fov(focal_length, img_height)
            FovX = focal2fov(focal_length, img_width)
            # print(f"FovX {FovX}, FovY {FovY}, focal_length {focal_length}, image_height {img_height}, image_width {img_width}")


            # focal_length_x = 1158.0337370751618
            # focal_length_y = 1158.0337370751618
            # FovY = focal2fov(focal_length_y,1080)
            # FovX = focal2fov(focal_length_x,1920)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_fake,
                                    image_path='', image_name='', width=img_width,
                                    height=img_height,
                                    semantic_feature=None,
                            semantic_feature_path=None,
                            semantic_feature_name=None))

    return cam_infos

def readNerfLerfInfo(path, white_background, eval, extension=".jpg"):
    print("Reading Test Transforms")
    train_cam_infos = readKeyframesCameras_lerf_mycolmap(path, white_background, extension="")

    return train_cam_infos


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Lerf": readNerfLerfInfo
}