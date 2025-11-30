import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import rcParams
from utils import CLIPVisualizer, blend_image_heatmap, draw_mid_rect

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1)
    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).cpu().numpy()
    if f_samples.shape[0] < 3:
        f_samples = f_samples.repeat(3, axis=0)
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

image_name = '0017'
image_path = f'/storage/shai/3d/data/rgb_data/st_paul/images/{image_name}.jpg'
feat_path = f'/storage/shai/3d/data/rgb_data/st_paul/clip_scale_fixed/{image_name}_fmap_CxHxW.pt'
feat3d_path = f'/storage/shai/3d/data/rgb_data/st_paul/swag_hash_attention_avg1winpyramid_drop70_adjscale_clip_scale_pyramid/train/ours_70000/saved_feature/{image_name}_fmap_CxHxW.pt'
clip_dir = './feature-3dgs/encoders/clip_pyramid/0_CLIPModel'
vis_classes = ['tower', 'sky', 'dome']

clip_vis = CLIPVisualizer(clip_dir, vis_classes)

image = Image.open(image_path)
image.putalpha(255)
w, h = image.size

feat_dict = torch.load(feat_path)
feat3d = torch.load(feat3d_path)

keys = feat_dict.keys()
scale_keys = sorted([key for key in keys if isinstance(key, float)])
display_keys = [scale_keys[i] for i in (0,
                                        round(len(scale_keys) / 4),
                                        round(len(scale_keys) / 2), 
                                        round(len(scale_keys) * 3 / 4),
                                        len(scale_keys) - 1)][:3]
scale_colors = ['magenta', 'cyan', 'yellow', 'orange', 'red'][:3]

# rect_image = image.copy()
# for scale, color in zip(display_keys, scale_colors):
#     rect_dim = int(float(scale) * w)
#     draw_mid_rect(rect_image, (rect_dim, rect_dim), color=color, width=8, loc=(h * 0.36, w * 0.39))
# rect_image.save('/storage/shai/3d/code/HaLo-GS/results/orig.png')

results = {}
avg_feat = None
for key in scale_keys:
    feat = feat_dict[key]
    if avg_feat is None:
        avg_feat = feat
    elif feat.shape[1] > avg_feat.shape[1]:
        avg_feat = feat + F.interpolate(
            avg_feat.unsqueeze(0), size=(feat.shape[1], feat.shape[2]),
            mode='bilinear', align_corners=False).squeeze(0) 
    else:
        avg_feat = avg_feat + F.interpolate(
            feat.unsqueeze(0), size=(avg_feat.shape[1], avg_feat.shape[2]),
            mode='bilinear', align_corners=False).squeeze(0)
avg_feat = avg_feat / len(scale_keys) 

interp_mode = 'bilinear'  # 'nearest'  # 'bilinear'

avg_feat = F.interpolate(
    avg_feat.unsqueeze(0), size=(h, w), mode=interp_mode).squeeze(0)

pca = feature_visualize_saving(avg_feat) * 255
results['avarage'] = {'pca': pca.cpu().numpy().astype(np.uint8)}
for label in clip_vis.quary_texts:
    scores, scores_raw = clip_vis.compute_scores(avg_feat, label, should_combine=False)
    scores = torch.clamp(scores, 0.0, 1.0).squeeze(0) # * 255
    scores = blend_image_heatmap(scores.cpu(), image)
    scores_raw = torch.clamp(scores_raw, 0.0, 1.0).squeeze(0) * 255
    results['avarage'][f'label_{label}'] = {
        'score': scores, #.cpu().numpy().astype(np.uint8),
        'raw_score': scores_raw.cpu().numpy().astype(np.uint8)}

feat3d = feat3d.to(avg_feat.dtype).to(avg_feat.device)

feat3d = F.interpolate(
    feat3d.unsqueeze(0), size=(h, w), mode=interp_mode).squeeze(0)

pca = feature_visualize_saving(feat3d) * 255
results['3d'] = {'pca': pca.cpu().numpy().astype(np.uint8)}
for label in clip_vis.quary_texts:
    scores, scores_raw = clip_vis.compute_scores(feat3d, label, should_combine=False)
    scores = torch.clamp(scores, 0.0, 1.0).squeeze(0) # * 255
    scores = blend_image_heatmap(scores.cpu(), image)
    scores_raw = torch.clamp(scores_raw, 0.0, 1.0).squeeze(0) * 255
    results['3d'][f'label_{label}'] = {
        'score': scores, #.cpu().numpy().astype(np.uint8),
        'raw_score': scores_raw.cpu().numpy().astype(np.uint8)}

for key in display_keys:
    feat = feat_dict[key]
    feat = F.interpolate(
        feat.unsqueeze(0), size=(h, w), mode=interp_mode).squeeze(0)
    pca = feature_visualize_saving(feat) * 255
    results[key] = {'pca': pca.cpu().numpy().astype(np.uint8)}
    for label in clip_vis.quary_texts:
        scores, scores_raw = clip_vis.compute_scores(feat, label, should_combine=False)
        scores = torch.clamp(scores, 0.0, 1.0).squeeze(0) #* 255
        scores = blend_image_heatmap(scores.cpu(), image)
        scores_raw = torch.clamp(scores_raw, 0.0, 1.0).squeeze(0) * 255
        results[key][f'label_{label}'] = {
            'score': scores, #.cpu().numpy().astype(np.uint8),
            'raw_score': scores_raw.cpu().numpy().astype(np.uint8)}

plt.autoscale()
fig, axs = plt.subplots(1 + len(clip_vis.quary_texts), 3 + len(display_keys))
# fig.set_tight_layout(True)
fig.tight_layout()

font_dict = {'fontsize': rcParams['axes.labelsize'],
 'fontweight': rcParams['axes.titleweight'],
 'color': 'k',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}

rect_image = image.copy()
for scale, color in zip(display_keys, scale_colors):
    rect_dim = int(float(scale) * w)
    draw_mid_rect(rect_image, (rect_dim, rect_dim), color=color, width=8)

axs[0, 0].imshow(rect_image)
axs[0, 0].set_title('original image', fontdict=font_dict)
axs[0, 0].axis('off')
rect_image.save('/storage/shai/3d/code/HaLo-GS/results/orig.png')
for j, label in enumerate(clip_vis.quary_texts):
    axs[j + 1, 0].axis('off')

for i, (scale, res) in enumerate(results.items()):
    axs[0, i + 1].imshow(res['pca'])
    Image.fromarray(res['pca']).save(f'/storage/shai/3d/code/HaLo-GS/results/pca_{str(scale).replace(".", "")[:3]}.png')
    if isinstance(scale, str):
        axs[0, i + 1].set_title(f'PCA {scale}', fontdict=font_dict)
    else:
        axs[0, i + 1].set_title(f'PCA scale {scale: .2}', fontdict=font_dict)
    axs[0, i + 1].axis('off')
    for j, label in enumerate(clip_vis.quary_texts):
        axs[j + 1, i + 1].imshow(res[f'label_{label}']['score'])
        axs[j + 1, i + 1].set_title(f'score {label}', fontdict=font_dict)
        axs[j + 1, i + 1].axis('off')
        res[f'label_{label}']['score'].save(f'/storage/shai/3d/code/HaLo-GS/results/score_{label}_{str(scale).replace(".", "")[:3]}.png')
fig.savefig('visualize_missalignment_nearest.png', bbox_inches="tight", dpi=800)