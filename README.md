# Lang3D-XL
We built upon (Feature3DGS)[https://github.com/ShijieZhou-UCLA/feature-3dgs], and thank them for their work. Therefore, any license issues from their repo, applied here.

## Installation
For installation, we recommend creating a conda environment:
```bash
conda create --name lang3d-xl python=3.8
conda activate lang3d-xl
```
Next, install pytorch and gsplat v0.1.10 (for pytorch, install according to the formal docs for your cuda version. if you have issues, specify your torch arch like we did. you can check your cuda arch like this: ```nvidia-smi --query-gpu=compute_cap --format=csv,noheader```): 
```bash
TORCH_CUDA_ARCH_LIST="8.6" pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

TORCH_CUDA_ARCH_LIST="8.6" pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

TORCH_CUDA_ARCH_LIST="8.6" pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.10
```
Also install simple-knn:
```bash
cd submodules/simple-knn
TORCH_CUDA_ARCH_LIST="8.6" pip install .
```

```bash
pip install tqdm plyfile scikit-learn==1.0.2 scikit-image scipy opencv-python tensorboard matplotlib

pip install transformers

pip install git+https://github.com/openai/CLIP.git@3702849800aa56e2223035bccd1c6ef91c704ca8
```

## Pre-Training
Our method assumes you have **images** dir, and **COLMAP** data. So your scene folder should look like that:
```bash
- scene_folder
    - images/
    - sparse/
    - classes.json
```

`classes.json` is optional, but highly recommended. This file should hold a list of strings which are the prompts of interests. It is used during training for the tensorboard visualizations, and can be used also for the querying, for easy specification of the prompts.

### Creating CLIP-Pyramids
First, if you want to use the optimized clip from HaLo-Nerf, download the model folder, from [their github](https://github.com/TAU-VAILab/HaLo-NeRF/tree/main).
```bash
python lang3d-xl/encoders/clip_pyramid/create_scale_pyramid.py $data $output --adjustable_scales "$adjustable_scales" --clip_model_dir $clip_model_dir --colmap_dir $colmap_dir
```
for example: ```python lang3d-xl/encoders/clip_pyramid/create_scale_pyramid.py /path/to/images_dir /path/to/features_saving_dir --adjustable_scales "4e-1,20e-1" --clip_model_dir /path/to/clip_model_dir --colmap_dir /path/to/Sparse```

**Make sure you put the clip saving dir, inside the scene folder, next to the images and COLMAP folders.**

* In case the regular LERF scales (0.05-0.5 of image size) are preferred, input ```--adjustable_scales ""```. in that case, no need to provide colmap_dir, because it is only used to compute the adjustable scales.
* For the regular clip model, input ```--clip_model_dir ""```

The scales we used for the HolyScenes dataset are: Notredame:```4e-1,20e-1```, St.Paul:```5e-1,30e-1```, Milano: ```2e-1,9e-1```, Blue Mosque: ```3e-1,20e-1```, Badshahi Mosque: ```5e-2,8e-1```, Hurba: ```2e-1,30e-1```.

### DINO
We used dino feature maps as a regularization factor. You could use your own DINO features or other features for regularization, or use our own. To create the DINO feature maps that we used, use the following:
First create a new conda environment:
```bash
conda create --name lang3d-xl-pre python=3.10
conda activate lang3d-xl-pre
TORCH_CUDA_ARCH_LIST="8.6" pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers xformers
```
```bash
python lang3d-xl/encoders/dino/dino_encoding.py $images_dir $save_dir
```
We gave the option to choose a different spatial size (```--size```) or a different DINO model (```--dino_model```). However, if the channel dimension size changes due to a different model, you will need to specify the dino_size in the training and rendering scripts later on.

### Segmentation masks
We also use segmentation masks as regularization. You could use your own segmentation masks, use LangSplat from the [original code](https://github.com/minghanqin/LangSplat), or use the following to create LangSplat masks like we did (if you use your own, make sure they are in the same format - we only used the small size masks, but build everything):
First, create the LangSplat environment:
```bash
cd submodules/LangSplat
conda env create --file environment.yml
conda activate
```
Then, run the segmentation:
```bash
python submodules/LangSplat/segment_dataset.py --images $images_dir --save_path $save_path --resolution 1
```

### Buildings segmentation
For the HolyScenes we used building segmentation masks, to eliminate the transient objects during training. We used LangSam to produce the masks. To produce the masks, first build the environment:
```bash
conda create --name langsam python=3.10
conda activate langsam
TORCH_CUDA_ARCH_LIST="8.6" pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

```

Next, run the script to produce masks by text prompt (for example 'building'):
```bash
python segment_by_text.py $images_dir $save_dir $text_prompt
```

## Training
After preparing the pre-training data, move on to the 3D optimization process:
```bash
conda activate lang3d-xl
python ./train.py -s $scene_folder -m $output_dir -f $features_dir_name -r $resolution --feat_enc_hash --wild --dino_dir "$dino" --segmentation_dir "$segdir" --foreground_masks_dir $scene_folder/building_segmentation --iterations $iter --do_attention $train_flags --save_iterations $iter
```

<details>
<summary><b>Training Arguments</b></summary>

### Model Parameters
- `-s, --source_path`: Path to the scene folder containing images and COLMAP data
- `-m, --model_path`: Directory where the trained model and outputs will be saved
- `-f, --foundation_model`: Name of the features directory (CLIP_pyramids)
- `-r, --resolution`: Resolution for training images (-1 for original resolution)

### Wild/Feature Parameters
- `--wild`: Enable wild/in-the-wild training mode (flag)
- `--feat_enc_hash`: Use hash encoding for features (flag - should be specified for our method)
- `--do_attention`: Enable attention mechanism (flag - should be specified for our method)
- `--dino_dir`: Path to DINO feature maps directory (if empty, no DINO regularization)
- `--dino_size`: Size of DINO features (default: 384)
- `--dino_factor`: Weight for DINO regularization loss (default: 0.8)
- `--segmentation_dir`: Path to segmentation masks directory (if empty, no SAM regularization)
- `--foreground_masks_dir`: Path to foreground/building masks directory (if empty, background not eliminated)
- `--max_feat_scale`: Maximum feature scale to be used from CLIP pyramid (default -1.0: use all)
- `--min_feat_scale`: Minimum feature scale to be used from CLIP pyramid (default -1.0: use all)
- `--mlp_feat_decay`: L2 regularization weight for MLP features (default: 1e-5)
- `--attention_dropout`: Dropout rate for attention mechanism (default: 0.0)
- `--seg_loss_weight`: Weight for segmentation loss (default: 0.0005)
- `--seg_loss_mode`: Mode for segmentation loss ('l2' or 'l1')
- `--bulk_on_device`: if 0, not gonna load all images to gpu, but will load every iteration one image (default: 1)
- `--no_vis_enc_hash`: Disable in-the-wild hash encoding method for RGB and opacity (flag)

### Optimization Parameters
- `--iterations`: Total number of training iterations (default: 30,000)
- `--position_lr_init`: Initial learning rate for Gaussian positions (default: 0.000016)
- `--position_lr_final`: Final learning rate for Gaussian positions (default: 0.0000016)
- `--position_lr_max_steps`: Number of steps for position learning rate scheduling (default: 30,000)
- `--semantic_feature_lr`: Learning rate for semantic features (default: 0.001)
- `--wild_lr`: Learning rate for wild/latent codes (default: 0.0005)
- `--wild_feat_lr`: Learning rate for wild features (default: 0.0005)
- `--wild_attention_lr`: Learning rate for attention weights (default: 0.0005)
- `--wild_latent_lr`: Learning rate for in-the-wild latent vectors (default: 0.001)
- `--lambda_dssim`: Weight for DSSIM loss (default: 0.5)

### Advanced Options
- `--warmup`: Enable warmup training without semantic features (flag)
- `--warmup_iterations`: Number of warmup iterations (default: 0)
- `--freeze_after_warmup`: Freeze model after warmup and only train features (flag)
- `--clip_dir`: Downloaded CLIP folder for the tensorboard visualizations. If empty, loads the model from `openai/clip-vit-base-patch32`.

</details>


## Rendering
```bash
conda activate lang3d-xl
python ./render.py -s $scene_folder -m $output -f $features --iteration -1 --skip_test --feat_enc_hash --wild --dino_dir "$dino" --do_attention
```

<details>
<summary><b>Rendering Arguments</b></summary>

### Model Parameters
- `-s, --source_path`: Path to the scene folder containing original images and COLMAP data
- `-m, --model_path`: Path to the trained model directory
- `-f, --foundation_model`: Name of the features directory (CLIP_pyramids)
- `-r, --resolution`: Resolution for rendering (-1 for original resolution)

### Rendering Control
- `--iteration`: Which checkpoint iteration to render (default: -1 for automatically detect the largest iteration saved)
- `--skip_train`: Skip rendering training views (flag)
- `--skip_test`: Skip rendering test views (flag)

### Feature Parameters
- `--wild`: Enable wild/in-the-wild rendering mode (flag)
- `--feat_enc`: Enable feature encoding (flag)
- `--feat_enc_hash`: Use hash encoding for features (flag)
- `--do_attention`: Enable attention mechanism (flag)
- `--dino_dir`: Path to DINO feature maps directory
- `--dino_size`: Size of DINO features (default: 384)
- `--vis_enc_hash`: Use hash encoding for visibility (enabled by default)
- `--no_vis_enc_hash`: Disable hash encoding for visibility (flag)
- `--bulk_on_device`: if 0, not gonna load all images to gpu, but will load every iteration one image (default: 1)

### Output Options
- `--save_features`: Save rendered features (default: True)

</details>


## Querying
```bash
conda activate lang3d-xl
python ./encoders/clip_pyramid/visualize_embeddings.py --feature_dir $output/train/ours_$iter/saved_feature --images_dir $scene_folder/images --save_dir $output/vis_saved_3d_feature --classes_file $classes_file --gt_dir $gt_dir --foreground_masks_dir $scene_folder/building_segmentation --blur $blur --neg_classes $neg_classes
```

<details>
<summary><b>Querying Arguments</b></summary>

### Input/Output Paths
- `--feature_dir`: Directory containing the rendered 3D feature maps from the trained model
- `--images_dir`: Directory containing the original images
- `--save_dir`: Directory where the visualization outputs will be saved
- `--classes_file`: Path to JSON file containing the list of query classes/text prompts to visualize
- `--gt_dir`: Directory containing ground truth data per class (for searching the classes names, if no classes file is given)

### CLIP Model Configuration
- `--clip_model_dir`: Path to the CLIP model directory downloaded from [HaLo-Nerf](https://github.com/TAU-VAILab/HaLo-NeRF/tree/main) (if empty, loading the clip model `openai/clip-vit-base-patch32`)

### Query Processing
- `--neg_classes`: Comma-separated list of negative classes for CLIP relevancy normalization (default: "object,things,stuff,texture")

### Visualization Options
- `--blur`: Gaussian blur sigma for smoothing relevancy maps (default: 2.5)
- `--res`: Resolution downscaling factor for processing (default: 2)
- `--normalize_by_group`: Enable group normalization across all query results' visualization colors (flag)

### Masking
- `--foreground_masks_dir`: Directory containing foreground masks to apply to visualizations. If empty, will not be applied (default: '')

### Performance Options
We use multi processing to make things faster. If you'd like to run on one process only, specify: `--max_workers 1 --chunk_size -1`.
- `--max_workers`: Maximum number of parallel workers for processing (default: 3)
- `--chunk_size`: Number of images to process per chunk (default: 100)

</details>

