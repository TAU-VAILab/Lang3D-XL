import os.path
import os
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from collections import namedtuple, Counter
from pathlib import Path
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter

import sys
utils_dir = Path(__file__).resolve().parents[2] / 'utils'
sys.path.append(str(utils_dir))
from data_utils import find_file
from metric_utils import average_precision_score, precision_recall_curve


Metrics = namedtuple("Metrics", "ap ba jaccard dice label")


class MetricCalculator:
    
    GT_THRESHOLD = 0.5
    PRED_THRESHOLD = 0.5

    def __init__(self, path_pred, path_gt, top_k, num_epochs):
        self.path_pred = path_pred
        self.path_gt = path_gt
        self.top_k = str(top_k)
        self.num_epochs = str(num_epochs)
    def process_single_image(self, pred_fn, gt_fn, label=None, verbose=False):

        pred_img = Image.open(os.path.join(self.path_pred, pred_fn)).convert('L')
        pred_array = np.asarray(pred_img) / 255

        gt_img = Image.open(os.path.join(self.path_gt, label, gt_fn)).convert('L').resize((pred_array.shape[1], pred_array.shape[0]))
        gt_array = 1 - np.asarray(gt_img) / 255
        # 1 - x : assumes background is x=1

        gt_uniq = np.unique(gt_array)
        gt_nuniq = len(gt_uniq)
        # if gt_nuniq != 2:
        #     print(f'Warning: Ground truth mask contains {gt_nuniq} unique values:', *gt_uniq)
        #     if verbose:
        #         print('Using threshold', self.GT_THRESHOLD)

        gt_mask = gt_array > self.GT_THRESHOLD

        gt_mask_flat = gt_mask.ravel()
        pred_array_flat = pred_array.ravel()

        pred_array_flat_thresh = pred_array_flat > self.PRED_THRESHOLD

        # calculate metrics
        AP = average_precision_score(gt_mask_flat, pred_array_flat)
        precision, recall, _ = precision_recall_curve(
            gt_mask_flat, pred_array_flat, pos_label=1, sample_weight=None)
        np.save(os.path.join(self.path_pred, pred_fn.split('.')[0] + '_pr.npy'),
                {'precision': precision, 'recall': recall})

        tpr = (gt_mask_flat * pred_array_flat).sum() / max(gt_mask_flat.sum(), 1)
        tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / max((1 - gt_mask_flat).sum(), 1)

        balanced_acc = (tpr + tnr) / 2

        jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

        dice = f1_score(gt_mask_flat, pred_array_flat_thresh)
        
        metrics = Metrics(AP, balanced_acc, jscore, dice, label)

        # if verbose:
        with open(os.path.join(self.path_pred, pred_fn.split('.')[0] + '_metric.txt'),'w')as file:
            file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
            file.write(f'\tAP (average precision):\t {metrics.ap}\n')
            file.write(f'\tBalanced accuracy:\ {metrics.ba}\n')
            file.write(f'\tJaccard score (IoU):\t {metrics.jaccard}, Threshold: {self.PRED_THRESHOLD}\n')
            file.write(f'\tDice score (F1):\t {metrics.dice}, Threshold: {self.PRED_THRESHOLD}')
            file.close()

        return metrics
    
    def process_all_images(self, pred_fns, gt_fns, labels,
                           foreground_masks_dir='', dilation=0, blur=0, load_numpy=False):
        assert len(pred_fns) == len(gt_fns) == len(labels), 'Mismatched number of filenames and/or labels'
        
        all_metrics = []
        images_AP = {}
        """
        for pred_fn, gt_fn, label in zip(pred_fns, gt_fns, labels):
            metric = self.process_single_image(pred_fn, gt_fn, label)
            all_metrics.append(metric)
            images_AP[str((Path(self.path_pred) / pred_fn).relative_to(Path(self.path_pred).parents[2]))] = metric.ap
        """
        multi_args = []
        for pred_fn, gt_fn, label in zip(pred_fns, gt_fns, labels):
            foreground_path = find_file(Path(foreground_masks_dir) / Path(pred_fn).stem) \
                if foreground_masks_dir else None
            multi_args.append((pred_fn, gt_fn, label,
                               self.path_pred, self.path_gt,
                               self.GT_THRESHOLD, self.PRED_THRESHOLD,
                               foreground_path, dilation, blur, load_numpy))
            
        with ProcessPoolExecutor() as executor:
            for rel_path, metric in tqdm(executor.map(
                    process_single_image_wrapper, multi_args), total=len(multi_args)):
                all_metrics.append(metric)
                images_AP[rel_path] = metric.ap
        unique_labels = np.unique(labels)
        counts = Counter(labels)
        
        # print("Unique labels:", *unique_labels)
        # print("Counts:")
        # for label in unique_labels:
        #     print(f"\t{label}: {counts[label]}")
        # print()
        #

        with open(self.path_pred + '/all_category_metric.txt','w')as f:
            f.write("all_category_metric :\n")
            f.write('\tAP (average precision):\t' + str(np.mean([m.ap for m in all_metrics if not np.isnan(m.ap)])) + '\n')
            f.write('\tBalanced accuracy:\t' + str(np.mean([m.ba for m in all_metrics if not np.isnan(m.ba)])) + '\n')
            f.write('\tJaccard score (IoU):\t' + str(
                np.mean([m.jaccard for m in all_metrics if not np.isnan(m.jaccard)])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\tDice score (F1):\t' + str(
                np.mean([m.dice for m in all_metrics if not np.isnan(m.dice)])) + f' (Threshold: {self.PRED_THRESHOLD})\n')
            f.write('\n')

        # print("Micro-averaged metrics (per-sample average):")
        # print('\tAP (average precision):\t', np.mean([m.ap for m in all_metrics]))
        # print('\tBalanced accuracy:\t', np.mean([m.ba for m in all_metrics]))
        # print('\tJaccard score (IoU):\t', np.mean([m.jaccard for m in all_metrics]), f'(Threshold: {self.PRED_THRESHOLD})')
        # print('\tDice score (F1):\t', np.mean([m.dice for m in all_metrics]), f'(Threshold: {self.PRED_THRESHOLD})')
        # print()
        #
        # print("Macro-averaged metrics (per-class average):")
        def macro_average(metric_name):
            values = [
                np.mean([getattr(m, metric_name) for m in all_metrics if m.label == label])
                for label in unique_labels
            ]
            return np.mean(values)
        #
        #
        # print('\tAP (average precision):\t', macro_average('ap'))
        # print('\tBalanced accuracy:\t', macro_average('ba'))
        # print('\tJaccard score (IoU):\t', macro_average('jaccard'), f'(Threshold: {self.PRED_THRESHOLD})')
        # print('\tDice score (F1):\t', macro_average('dice'), f'(Threshold: {self.PRED_THRESHOLD})')
        # print()

        return {k: macro_average(k) for k in ['ap', 'ba', 'jaccard', 'dice']}, images_AP

def process_single_image_wrapper(args):
    pred_fn, gt_fn, label, path_pred, path_gt, \
        GT_THRESHOLD, PRED_THRESHOLD, mask_path, dilation, blur, load_numpy = args

    if load_numpy:
        with open(f'{str(Path(path_pred) / Path(pred_fn).stem)}.npy', 'rb') as f:
            pred_array = np.load(f)
    else:
        pred_img = Image.open(os.path.join(path_pred, pred_fn)).convert('L')
        pred_array = np.asarray(pred_img) / 255
    
    if blur:
        pred_array = gaussian_filter(pred_array, sigma=blur, truncate=8)

    gt_img_path = os.path.join(path_gt, label, gt_fn)
    if not Path(gt_img_path).exists():
        gt_img_path = os.path.join(path_gt, label.split[','][0], gt_fn)
    gt_img = Image.open(gt_img_path).convert('L').resize((pred_array.shape[1], pred_array.shape[0]))
    gt_array = 1 - np.asarray(gt_img) / 255
    # 1 - x : assumes background is x=1

    if dilation:
        kernel = np.ones((dilation, dilation))
        dilated_gt_array = cv2.dilate(gt_array, kernel, iterations=1)
    else:
        dilated_gt_array = None

    if mask_path is not None:
        mask_image = Image.open(str(mask_path))
        mask_image = mask_image.resize(pred_img.size, resample=Image.NEAREST)
        mask = np.round(np.array(mask_image) / 255)
        pred_array = pred_array * mask
        gt_array = gt_array * mask
        if dilated_gt_array is not None:
            dilated_gt_array = dilated_gt_array * mask
            # print()
            # Image.fromarray(((dilated_gt_array + gt_array) * 255 / 2).astype(np.uint8)).save(f'atest_{Path(path_gt).stem}.png')

    gt_uniq = np.unique(gt_array)
    gt_nuniq = len(gt_uniq)
    # if gt_nuniq != 2:
    #     print(f'Warning: Ground truth mask contains {gt_nuniq} unique values:', *gt_uniq)
    #     if verbose:
    #         print('Using threshold', self.GT_THRESHOLD)

    gt_mask = gt_array > GT_THRESHOLD

    gt_mask_flat = gt_mask.ravel()
    pred_array_flat = pred_array.ravel()

    pred_array_flat_thresh = pred_array_flat > PRED_THRESHOLD

    if dilated_gt_array is not None:
        dilated_gt_array = dilated_gt_array > GT_THRESHOLD
        dilated_gt_array = dilated_gt_array.ravel()
    

    # calculate metrics
    AP = average_precision_score(gt_mask_flat, pred_array_flat, dilated_gt_array)
    precision, recall, thresholds = precision_recall_curve(
        gt_mask_flat, pred_array_flat, dilated_gt_array, pos_label=1, sample_weight=None)
    np.save(os.path.join(path_pred, pred_fn.split('.')[0] + '_pr.npy'),
            {'precision': precision, 'recall': recall, 'thresholds': thresholds})

    tpr = (gt_mask_flat * pred_array_flat).sum() / max(gt_mask_flat.sum(), 1)
    tnr = ((1 - gt_mask_flat) * (1 - pred_array_flat)).sum() / max((1 - gt_mask_flat).sum(), 1)

    balanced_acc = (tpr + tnr) / 2

    jscore = jaccard_score(gt_mask_flat, pred_array_flat_thresh)

    dice = f1_score(gt_mask_flat, pred_array_flat_thresh)
    
    metrics = Metrics(AP, balanced_acc, jscore, dice, label)

    # if verbose:
    with open(os.path.join(path_pred, pred_fn.split('.')[0] + '_metric.txt'),'w')as file:
        file.write(f'Metrics for single image (GT: {gt_fn}; preds: {pred_fn})\n')
        file.write(f'\tAP (average precision):\t {metrics.ap}\n')
        file.write(f'\tBalanced accuracy:\ {metrics.ba}\n')
        file.write(f'\tJaccard score (IoU):\t {metrics.jaccard}, Threshold: {PRED_THRESHOLD}\n')
        file.write(f'\tDice score (F1):\t {metrics.dice}, Threshold: {PRED_THRESHOLD}')
        file.close()

    return str((Path(path_pred) / pred_fn).relative_to(Path(path_pred).parents[2])), metrics


def main_metrics(path_pred, path_gt, category, top_k, num_epochs,
                 scene, cat, ts_list, foreground_masks_dir, dilation=0, blur=0,
                 load_numpy=False, full_name_format=False):
    calculator = MetricCalculator(path_pred, path_gt, top_k, num_epochs)

    if full_name_format:
        pred_fns = [name + '.jpg' for name in ts_list]
        gt_fns = [name + '-gt.jpg' for name in ts_list]
    else:
        pred_fns = [str(i).zfill(4) + '.jpg' for i in ts_list]
        # gt_fns = [str(i).zfill(4) + '_mask.jpg' for i in ts_list]
        gt_fns = [str(i).zfill(4) + '-gt.jpg' for i in ts_list]

    labels = [cat] * len(ts_list)

    if labels == []:
        raise ValueError('no category')


    return calculator.process_all_images(
        pred_fns, gt_fns, labels, foreground_masks_dir=foreground_masks_dir,
        dilation=dilation, blur=blur, load_numpy=load_numpy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--gt_dir", type=str, help="Directory containing features")
    parser.add_argument("--pred_dir", type=str, help="Directory containing images")
    parser.add_argument("--foreground_masks_dir", default='', type=str)
    parser.add_argument("--dilation", default=0, type=int)
    parser.add_argument("--blur", default=0, type=int)
    parser.add_argument("--load_numpy", default=False, type=bool)
    parser.add_argument("--classes_file", default='', type=str, help="classes file")
    parser.add_argument("--full_name_format", default=False, type=bool)

    args = parser.parse_args()
    args.foreground_masks_dir = ''
    
    path_gt = args.gt_dir  # '/root/feature-3dgs/data/HolyScenes/cathedral/st_paul'
    path_pred = args.pred_dir  # '/root/feature-3dgs/data/st_paul/clipseg_ft'  # '/root/feature-3dgs/data/st_paul/vis_3d_embeddings_mean_normalized'

    if args.classes_file:
        with open(args.classes_file) as file:
            categories = json.load(file)
        for category in categories:
            gt_categories = [path.name for path in Path(path_gt).iterdir() if path.is_dir()]
            if category not in gt_categories and category.split(',')[0] not in gt_categories:
                raise ValueError(f'{category} not in {path_pred}')
    else:
        categories = [path.name for path in Path(path_gt).iterdir() if path.is_dir()]

    categories = [cat for cat in categories if (Path(path_pred) / cat).exists()]
    print('Categories:', categories)

    for category in categories:
        pred_categories = [path.name for path in Path(path_pred).iterdir() if path.is_dir()]
        if category not in pred_categories:
            raise ValueError(f'{category} not in {path_pred}')
    classes_AP = {}
    for category in categories:
        ts_list = []
        if os.path.exists(os.path.join(path_gt, category)):
            list_dir = os.listdir(os.path.join(path_gt, category))
            if args.full_name_format:
                ts_list = [f.replace('-gt.jpg', '').replace('-img.jpg', '') for f in list_dir if f.endswith('jpg')]
            else:
                ts_list = [int(f[:4]) for f in list_dir if f.endswith('jpg')]
            
        elif os.path.exists(os.path.join(path_gt, category.split(',')[0])):
            list_dir = os.listdir(os.path.join(path_gt, category))
            if args.full_name_format:
                ts_list = [f.replace('-gt.jpg', '').replace('-img.jpg', '') for f in list_dir if f.endswith('jpg')]
            else:
                ts_list = [int(f[:4]) for f in list_dir if f.endswith('jpg')]

        if ts_list == []:
            print("There are no ground truth images in 'path_gt' ")

        _, images_AP = main_metrics(
            path_pred=os.path.join(path_pred, category),
            path_gt=path_gt,
            category=category,
            top_k=len(ts_list),
            num_epochs=1,
            scene=Path(path_gt).stem,
            cat=category,
            ts_list=ts_list,
            foreground_masks_dir=args.foreground_masks_dir,
            dilation=args.dilation,
            blur=args.blur,
            load_numpy=args.load_numpy,
            full_name_format=args.full_name_format)
        classes_AP[category] = images_AP
    
    with open(str(Path(path_pred) / 'data.json'), 'w') as file:
        json.dump(classes_AP, file, indent=4)
