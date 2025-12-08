conda create -y --name lang3d-xl python=3.8 &&
conda activate lang3d-xl &&
python -m pip install --upgrade pip &&
TORCH_CUDA_ARCH_LIST="8.6" pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --no-input &&
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit &&

TORCH_CUDA_ARCH_LIST="8.6" pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-input &&

TORCH_CUDA_ARCH_LIST="8.6" pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.10 --no-input &&

cd submodules/simple-knn &&
TORCH_CUDA_ARCH_LIST="8.6" pip install . --no-input &&

pip install tqdm plyfile scikit-learn==1.0.2 scikit-image scipy opencv-python tensorboard matplotlib --no-input&&

pip install transformers --no-input &&

pip install git+https://github.com/openai/CLIP.git@3702849800aa56e2223035bccd1c6ef91c704ca8 --no-input