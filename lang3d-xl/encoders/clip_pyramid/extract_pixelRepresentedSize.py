# import pycolmap
import numpy as np
import random
import struct
from collections import namedtuple
from PIL import Image as PilImage
from PIL import ImageDraw
from pathlib import Path


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_points3D_binary_dict(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    point_dict = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            assert id not in point_dict
            point_dict[id] = xyz
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            
    return point_dict, xyzs

Camera = namedtuple(
    "Camera", ["id", "model", "width", "height", "params", "fx", "fy", "cx", "cy"])
CameraModel = namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            
            if model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]  # Same focal length for SIMPLE_PINHOLE and SIMPLE_RADIAL
                cx, cy = params[1], params[2]  # Principal points
            elif model_name in ["PINHOLE", "OPENCV"]:
                fx, fy = params[0], params[1]  # Different focal lengths
                cx, cy = params[2], params[3]  # Principal points
            else:
                raise ValueError(f"Camera model {model_name} not supported in this example.")

            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params),
                                        fx=fx, fy=fy, cx=cx, cy=cy)
        assert len(cameras) == num_cameras

    return cameras

BaseImage = namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def load_colmap_data(cameras_path, images_path, points3D_path, point3d_as_dict=False):
    cameras = read_intrinsics_binary(cameras_path)
    images = read_extrinsics_binary(images_path)
    
    if point3d_as_dict:
        points_dict, xyz = read_points3D_binary_dict(points3D_path)
        return cameras, images, xyz, points_dict
    
    xyz, rgb, _ = read_points3D_binary(points3D_path)

    return cameras, images, xyz, None

def calculate_point_cloud_center_and_std(points, do_median=False):
    if do_median:
        center = np.median(points, axis=0)
    else:
        center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    std_dev = np.std(distances)
    return center, std_dev

def calculate_pixel_size(camera_position, center, std_dev, factor_param, width, height, fx, fy):
    # Calculate the distance from the camera to the center of the point cloud
    distance_to_center = np.linalg.norm(camera_position - center)

    # Adjust the distance using the standard deviation and factor_param
    adjusted_distance = max(distance_to_center - factor_param * std_dev, 0.01 * std_dev)

    # Calculate FOV in radians
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))

    # Calculate the size of a pixel in meters using the adjusted distance
    ifov_x = fov_x / width  # radians per pixel
    ifov_y = fov_y / height  # radians per pixel
    size_of_pixel_x = 2 * adjusted_distance * np.tan(ifov_x / 2)
    size_of_pixel_y = 2 * adjusted_distance * np.tan(ifov_y / 2)

    return {
        'distance_to_center': distance_to_center,
        'adjusted_distance': adjusted_distance,
        'fov_x': fov_x * (180 / np.pi),  # Convert to degrees
        'fov_y': fov_y * (180 / np.pi),  # Convert to degrees
        'size_of_pixel_x': size_of_pixel_x,
        'size_of_pixel_y': size_of_pixel_y
    }

def get_camera_position_from_image(image):
    R_matrix = image.qvec2rotmat()
    return -np.dot(R_matrix.T, image.tvec)


def calculate_distances_and_fov(images, cameras, points, factor_param):
    center, std_dev = calculate_point_cloud_center_and_std(points)
    results = []

    for image_id, image in images.items():
        camera_position = get_camera_position_from_image(image)

        # Access camera intrinsics
        camera = cameras[image.camera_id]
        fx = camera.fx
        fy = camera.fy
        width = camera.width
        height = camera.height

        # Perform all calculations for pixel size, FOV, and adjusted distance
        pixel_data = calculate_pixel_size(camera_position, center, std_dev, factor_param, width, height, fx, fy)

        results.append({
            'image_id': image_id,
            'distance_to_center': pixel_data['distance_to_center'],
            'adjusted_distance': pixel_data['adjusted_distance'],
            'fov_x': pixel_data['fov_x'],
            'fov_y': pixel_data['fov_y'],
            'size_of_pixel_x': pixel_data['size_of_pixel_x'],
            'size_of_pixel_y': pixel_data['size_of_pixel_y'],
            'width': width,
            'height': height
        })
    
    return results

def calculate_scale_in_pixels(colmap_path, phisical_scales, factor_param=0.5,
                              only_points_in_image=True):
    if (Path(colmap_path) / 'cameras.bin').exists():
        cameras_path = str(Path(colmap_path) / 'cameras.bin')
        images_path = str(Path(colmap_path) / 'images.bin')
        points3D_path = str(Path(colmap_path) / 'points3D.bin')
    else:
        cameras_path = str(Path(colmap_path) / '0' / 'cameras.bin')
        images_path = str(Path(colmap_path) / '0' / 'images.bin')
        points3D_path = str(Path(colmap_path) / '0' / 'points3D.bin')
    
    cameras, images, points3D, point_dict = load_colmap_data(
        cameras_path, images_path, points3D_path, point3d_as_dict=only_points_in_image)
    center, std_dev = calculate_point_cloud_center_and_std(points3D)
    results = {}

    for image_id, image in images.items():
        camera_position = get_camera_position_from_image(image)
        if only_points_in_image:
            images_points = np.array([point_dict[pid] for pid in image.point3D_ids
                                      if pid in point_dict and pid != -1])
            center, _ = calculate_point_cloud_center_and_std(
                images_points, do_median=True)
            std_dev = 0.0

        # Access camera intrinsics
        camera = cameras[image.camera_id]
        fx = camera.fx
        fy = camera.fy
        width = camera.width
        height = camera.height

        # Perform all calculations for pixel size, FOV, and adjusted distance
        pixel_data = calculate_pixel_size(camera_position, center, std_dev, factor_param,
                                          width, height, fx, fy)
        pixel_size = np.mean([pixel_data['size_of_pixel_x'], pixel_data['size_of_pixel_y']])
        results[image.name] = [calculate_square_dimensions_in_pixels(phisical_scale, pixel_size)
                               for phisical_scale in phisical_scales]
    
    return results

def select_images_in_percentage_range(results, percentage_range):
    # Extract pixel sizes in meters (average of x and y size)
    pixel_sizes = np.array([(res['size_of_pixel_x'] + res['size_of_pixel_y']) / 2 for res in results])

    # Calculate the pixel size percentiles
    lower_percentile = np.percentile(pixel_sizes, percentage_range[0])
    upper_percentile = np.percentile(pixel_sizes, percentage_range[1])

    # Filter the images in the given range
    filtered_results = [res for res in results if lower_percentile <= (res['size_of_pixel_x'] + res['size_of_pixel_y']) / 2 <= upper_percentile]

    # Select a random image from the filtered list
    return random.choice(filtered_results) if filtered_results else None

def calculate_square_dimensions_in_pixels(square_size_meters, pixel_size_meters):
    # Calculate square dimensions in pixels for the given square size in meters
    square_size_in_pixels = square_size_meters / pixel_size_meters
    return square_size_in_pixels

def draw_squares_on_image(image_path, output_path, square_sizes, color=(255, 0, 0), thickness=3):
    """
    Draws multiple squares on an image, all centered.

    :param image_path: Path to the input image.
    :param output_path: Path where the output image with squares will be saved.
    :param square_sizes: List of square sizes (length of the sides).
    :param color: Color of the square (default is red).
    :param thickness: Thickness of the square's borders.
    """
    # Open the image
    img = PilImage.open(image_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # Draw each square from the list
    for square_size in square_sizes:
        # Calculate the top-left corner to center the square
        top_left = ((img_width - square_size) // 2, (img_height - square_size) // 2)
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        
        # Draw the square with specified thickness
        for i in range(thickness):
            draw.rectangle([top_left, bottom_right], outline=color)
            top_left = (top_left[0] - 1, top_left[1] - 1)
            bottom_right = (bottom_right[0] + 1, bottom_right[1] + 1)

    # Save the modified image
    img.save(output_path)
    # img.show()  # Display the image (optional)

def draw_squares_on_images(results, square_sizes_meters, percentage_ranges):
    selected_images = []

    for percentage_range in percentage_ranges:
        selected_image = select_images_in_percentage_range(results, percentage_range)
        if selected_image:
            pixel_size_x = selected_image['size_of_pixel_x']
            pixel_size_y = selected_image['size_of_pixel_y']

            # Calculate square dimensions for both 2x2 meters and 15x15 meters squares
            square_2x2_x = calculate_square_dimensions_in_pixels(square_sizes_meters[0], pixel_size_x)
            square_2x2_y = calculate_square_dimensions_in_pixels(square_sizes_meters[0], pixel_size_y)

            square_15x15_x = calculate_square_dimensions_in_pixels(square_sizes_meters[1], pixel_size_x)
            square_15x15_y = calculate_square_dimensions_in_pixels(square_sizes_meters[1], pixel_size_y)

            # Center the squares on the image
            image_center_x = selected_image['width'] / 2
            image_center_y = selected_image['height'] / 2

            selected_images.append({
                'image_id': selected_image['image_id'],
                'square_2x2_x': square_2x2_x,
                'square_2x2_y': square_2x2_y,
                'square_15x15_x': square_15x15_x,
                'square_15x15_y': square_15x15_y,
                'image_center_x': image_center_x,
                'image_center_y': image_center_y
            })

    return selected_images

if __name__ == '__main__':
    # Example usage
    cameras_path = '/root/feature-3dgs/data/milano/sparse/0/cameras.bin'
    images_path = '/root/feature-3dgs/data/milano/sparse/0/images.bin'
    points3D_path = '/root/feature-3dgs/data/milano/sparse/0/points3D.bin'
    jpgs_path = '/root/feature-3dgs/data/milano/images'
    factor_param = 0.5  # You can set this to whatever value you prefer

    cameras, images, points3D = load_colmap_data(cameras_path, images_path, points3D_path)
    results = calculate_distances_and_fov(images, cameras, points3D, factor_param)

    # Define percentage ranges (5%-10%, 25%-30%, 45%-55%, 70%-75%, 90%-95%)
    percentage_ranges = [(5, 10), (25, 30), (45, 55), (70, 75), (90, 95)]

    # Square sizes to display: 2x2 meters and 15x15 meters
    square_sizes_meters = [2e-1, 15e-1]

    # Get selected images and square dimensions for each percentage range
    selected_images_with_squares = draw_squares_on_images(results, square_sizes_meters, percentage_ranges)

    # Print the results
    for image in selected_images_with_squares:
        draw_squares_on_image(str(Path(jpgs_path) / images[image['image_id']].name),
                            images[image['image_id']].name,
                            [np.mean([image['square_2x2_x'], image['square_2x2_y']]),
                            np.mean([image['square_15x15_x'], image['square_15x15_y']])])
        print(f"Image ID: {image['image_id']}")
        print(f"Center: ({image['image_center_x']}, {image['image_center_y']})")
        print(f"2x2 meter square size: ({image['square_2x2_x']:.2f} px, {image['square_2x2_y']:.2f} px)")
        print(f"15x15 meter square size: ({image['square_15x15_x']:.2f} px, {image['square_15x15_y']:.2f} px)\n")
