import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from PIL import Image


def multiprocess_graphs(args):
    make_save_graph(*args)

def make_save_graph(pred_dir, pred_dir2, path, images_dir, gt_dir):
    prec_recall = np.load(str(path), allow_pickle=True).item()
    prec = prec_recall['precision']
    recall = prec_recall['recall']
    plt.figure()
    plt.plot(recall, prec)
            
    path2 = pred_dir2 / path.relative_to(pred_dir)
    if path2.exists():
        prec_recall2 = np.load(str(path2), allow_pickle=True).item()
        prec2 = prec_recall2['precision']
        recall2 = prec_recall2['recall']
        plt.plot(recall2, prec2, 'r')
        save_thresholds_results(
            path2, images_dir, prec_recall2, recall2, prec2, gt_dir)
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(str(path).replace('_pr.npy', '_prec_recall.png'))
    plt.close()

    save_thresholds_results(
        path, images_dir, prec_recall, recall, prec, gt_dir)

def save_thresholds_results(
        path, images_dir, prec_recall, recall, prec, gt_dir):
    
    if (Path(images_dir) / path.name.replace('_pr.npy', '.png')).exists():
        thresholds = prec_recall['thresholds']
        thresholds = np.r_[thresholds, 0]
        scores = Image.open(str(path).replace('_pr.npy', '.jpg'))
        image = Image.open(str(images_dir / path.name.replace('_pr.npy', '.png')))
        image.putalpha(255)
        gt = Image.open(
            str(gt_dir / path.name.replace('_pr.npy', '-gt.jpg'))).convert('L')
        gt = gt.resize(scores.size, resample=Image.Resampling.NEAREST)
        gt = 1 - np.array(gt) / 255
        scores = np.array(scores) / 255
        for val in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
            idx = (np.abs(recall - val)).argmin()
            th = thresholds[idx]
            img = blend_image_heatmap(scores > th, image, gt)
            w, h = img.size
            max_size = max(w, h)
            if max_size > 250:
                factor = 250 / max_size
                img = img.resize((round(w * factor), round(h * factor)))
            img.save(
                str(path).replace('_pr.npy',
                                f"_recall_{str(val).replace('.', '')}_{int(prec[idx] * 1000)}.png"))
        
        
def blend_image_heatmap(scores, original_image, gt):

    """
    # Apply colormap
    colormap = plt.get_cmap('jet')  # 'jet' is a common heatmap colormap
    heatmap = colormap(scores)

    # Convert heatmap to PIL image
    heatmap = np.uint8(heatmap * 255)
    heatmap_image = Image.fromarray(heatmap)
    """
    height, width = scores.shape
    tp_color = np.array([0, 255, 0])[np.newaxis, np.newaxis, :]
    fp_color = np.array([255, 0, 0])[np.newaxis, np.newaxis, :]
    tn_color = np.array([0, 0, 255])[np.newaxis, np.newaxis, :]
    # heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    scores = np.float32(scores)
    gt = np.float32(gt)
    heatmap = (((1 - scores) * gt) > 0)[:, :, np.newaxis] * tn_color \
        + ((scores * (1 - gt)) > 0)[:, :, np.newaxis] * fp_color \
        + ((scores * gt) > 0)[:, :, np.newaxis] * tp_color
    
    above_thresh = (heatmap[:, :, 0] > 0) + (heatmap[:, :, 1] > 0) + (heatmap[:, :, 2] > 0)
    heatmap = heatmap * above_thresh[:, :, np.newaxis] + 255 * (1-above_thresh[:, :, np.newaxis])
    heatmap = np.uint8(heatmap)
    heatmap_image = Image.fromarray(heatmap)

    # Original PIL image (replace with your image)
    # original_image = Image.open('path_to_your_image.jpg')
    original_image = original_image.resize((heatmap.shape[1], heatmap.shape[0]))

    
    # Blend images
    blended_image = np.array(Image.blend(original_image.convert("RGBA"), heatmap_image.convert("RGBA"), alpha=0.7))
    # blended_image = np.uint8(blended_image  * above_thresh[:, :, np.newaxis] + original_image  * (1 - above_thresh[:, :, np.newaxis]))
    blended_image = np.uint8(blended_image)

    return Image.fromarray(blended_image).convert("RGB")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--pred_dir", type=str, help="Directory containing images",
                        default=r'/storage/shai/3d/data/rgb_data/notre_dame/chkpnt_attention_together5_l1seg_less_clip_scale_pyramid_per_image/vis_static_saved_3d_feature')
    parser.add_argument("--pred_dir2", type=str, help="Directory containing images",
                        default=r'/storage/shai/3d/data/rgb_data/notre_dame/clipseg/clipseg_ft')
    parser.add_argument("--images_dir", type=str, help="Directory containing images",
                        default=r'/storage/shai/3d/data/rgb_data/notre_dame/chkpnt_attention_together5_l1seg_less_clip_scale_pyramid_per_image/train/ours_70000/gt')
    parser.add_argument("--gt_dir", type=str, help="Directory containing gt-images",
                        default='')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_dir2 = Path(args.pred_dir2)

    paths_to_process = []
    for category_dir in [path for path in pred_dir.iterdir() if path.is_dir()]:
        for path in [path for path in category_dir.iterdir() if path.name.endswith('_pr.npy')]:
            # make_save_graph(pred_dir, pred_dir2, path, args.images_dir)
            paths_to_process.append((pred_dir, pred_dir2, path,
                                     Path(args.images_dir), Path(args.gt_dir) / category_dir.name))
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(multiprocess_graphs, paths_to_process), total=len(paths_to_process)))
