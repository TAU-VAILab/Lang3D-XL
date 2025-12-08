set -x &&

cp -r /storage/shai/envs/lang3d-xl miniconda3/envs/. &&

#cd /storage/shai/3d/code/HaLo-GS/feature-3dgs &&
cd /storage/shai/3d/code/new_code/Lang3D-XL &&

source activate /root/miniconda3/envs/lang3d-xl &&

export LD_LIBRARY_PATH=/root/miniconda3/envs/lang3d-xl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "/opt/conda" | grep -v "/usr/local/nvidia" | paste -sd ':' -)
unset PYTORCH_BUILD_NUMBER PYTORCH_BUILD_VERSION PYTORCH_HOME PYTORCH_VERSION


python lang3d-xl/encoders/clip_pyramid/create_scale_pyramid.py /storage/shai/3d/data/rgb_data/milano/images /storage/shai/3d/data/rgb_data/milano/test_clip --adjustable_scales "2e-1,9e-1" --clip_model_dir /storage/shai/3d/code/HaLo-GS/feature-3dgs/encoders/clip_pyramid/0_CLIPModel --colmap_dir /storage/shai/3d/data/rgb_data/milano/sparse