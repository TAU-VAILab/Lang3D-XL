import numpy as np


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def filter_views_by_distance(views, keep_percent=95):
    """
    Filters out views that are too far from the main cluster of cameras.
    
    Args:
        views (list): List of view objects, each with attributes R (3x3) and T (3,)
        keep_percent (float): Percentage of closest views to keep (default: 95)
        
    Returns:
        filtered_views (list): Views after removing outliers
    """
    # Extract all positions
    positions = np.array([view.T for view in views])  # shape (N, 3)
    
    # Compute robust center (median is less sensitive to outliers)
    center = np.median(positions, axis=0)
    
    # Compute distances from center
    distances = np.linalg.norm(positions - center, axis=1)
    
    # Determine cutoff based on percentile
    threshold = np.percentile(distances, keep_percent)
    
    # Mask: keep only positions within threshold
    mask = distances <= threshold
    
    # Filter views
    filtered_views = [view for i, view in enumerate(views) if mask[i]]
    
    return filtered_views

def filter_poses_by_distance(poses, keep_percent=95):
    positions = poses[:, :3, 3]

    # Compute robust center (median)
    center = np.median(positions, axis=0)

    # Remove extreme outliers
    distances = np.linalg.norm(positions - center, axis=1)
    mask = distances < np.percentile(distances, keep_percent)  # keep 95% closest
    
    return poses[mask]

def render_path_spiral(views, focal=30, zrate=0.5, rots=2, N=120,
                       percent_mean=20, percent_rad=40, offset=(0, 0, 0), rad_alpha=(1,1,1)):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    # poses = np.stack([np.concatenate([view.R.T, view.T[:, None]], 1) for view in views], 0)
    
    poses_for_mean = filter_poses_by_distance(poses, percent_mean)
    c2w = poses_avg(poses_for_mean)
    up = normalize(poses_for_mean[:, :3, 1].sum(0))

    # Get radii for spiral path
    poses = filter_poses_by_distance(poses, percent_rad)
    rads = np.percentile(np.abs(poses[:, :3, 3] - c2w[:3, 3]), 90, 0)
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    offset = np.array(offset + (0,))
    rad_alpha = np.array(rad_alpha + (1,))

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            rad_alpha *(offset + np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        # render_pose[:3] =  np.array([[ 9.9996626e-01, -7.5253481e-03, -3.2866236e-03, -5.6992844e-02],
        #             [-7.7875191e-03, -9.9601853e-01, -8.8805482e-02, -2.9015102e+00],
        #             [-2.6052459e-03,  8.8828087e-02, -9.9604356e-01, -2.3510060e+00]])
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses


def spherify_poses(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_pose = np.eye(4)
        render_pose[:3] = p
        #render_pose[:3, 1:3] *= -1
        new_poses.append(render_pose)

    new_poses = np.stack(new_poses, 0)
    print(new_poses.shape)
    return new_poses
