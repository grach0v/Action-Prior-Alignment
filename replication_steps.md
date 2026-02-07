# A2 replication steps (local)

This log records the exact steps taken so you can recreate the setup and run the benchmarks.

## Environment setup (uv + Python 3.10)

```bash
uv python install 3.10
uv venv .venv --python 3.10
```

## Dependencies

I created a UV-specific requirements file because several pinned packages in the original `requirements.txt` require Python >=3.9. The adjusted file is `requirements.uv.txt`.

Compatibility pins for Python 3.10:

- `ipython==8.12.3`
- `matplotlib==3.7.5`
- `pandas==1.5.3`
- `scikit-image==0.21.0`
- `networkx==3.1`
- `numpy==1.23.5` (needed for `transforms3d`/`np.float` and Open3D/Scipy)
- `scipy==1.10.1`
- `tifffile==2023.7.10`
- `transforms3d==0.3.1` (required by `graspnetAPI`)

Install base deps (PyTorch will be installed from nightly cu128 next):

```bash
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
  uv pip install -r requirements.uv.txt \
  --python .venv/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

Editable install (skip deps to avoid the `transforms3d` conflict in `setup.py`):

```bash
uv pip install -e . --python .venv/bin/python --no-deps

Install extra runtime deps used in evaluation:

```bash
uv pip install timm --python .venv/bin/python
```
```

## GDrive downloads (gdown 5.2.0)

Older `gdown` failed to parse the current Drive response. I upgraded it:

```bash
uv pip install gdown==5.2.0 --python .venv/bin/python
```

### Testing cases

```bash
mkdir -p testing_cases
./.venv/bin/gdown --folder "https://drive.google.com/drive/folders/1OuTua-69NEeV7RYIi9nzR1jmdZEugB68" --remaining-ok
```

After download, the folder layout is:

```
testing_cases/
  grasp_testing_cases/
  place_testing_cases/
  pp_testing_cases/
```

### Pretrained checkpoint

```bash
mkdir -p logs
./.venv/bin/gdown --folder "https://drive.google.com/drive/folders/1uoDGIgkcSi8okcr8qjKOaF57TyRaHRd_" --remaining-ok
mv logs/a2_pretrained a2_pretrained
```

The test scripts expect `a2_pretrained/checkpoints/sl_checkpoint_199.pth`.

### Assets

```bash
./.venv/bin/gdown --folder "https://drive.google.com/drive/folders/1WxKDFXJktoqiP0jmkDZrMCcNNBx5u-YM" --remaining-ok
```

If gdown creates a nested `assets/assets/` directory, merge it back:

```bash
rsync -a assets/assets/simplified_objects/ assets/simplified_objects/
rm -rf assets/assets
```

If you manually downloaded `assets-*.zip` into the repo root, extract and merge like this:

```bash
unzip -o assets-*.zip -d assets
rsync -a assets/assets/ assets/
rm -rf assets/assets
```

MTL texture validation:

```bash
python - <<'PY'
import re
from pathlib import Path

root = Path('assets')
missing = []
for mtl in root.rglob('*.mtl'):
    text = mtl.read_text(errors='ignore')
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith('map_'):
            tex = re.split(r'\s+', line, maxsplit=1)[1].strip().strip('"')
            if not (mtl.parent / tex).exists():
                missing.append((mtl, tex))
print('missing', len(missing))
for mtl, tex in missing:
    print(mtl, tex)
PY
```

One missing texture in `assets/simplified_objects/068/textured.obj.mtl` was fixed by copying `dummy.png`:

```bash
cp assets/simplified_objects/068/dummy.png assets/simplified_objects/068/close.jpg
```

Note: if gdown fails with permissions/quota for any asset file, download the assets archive manually from the Drive link and extract it into `assets/`. You should end up with:

```
assets/simplified_objects/*.urdf
assets/simplified_objects/000/...
assets/unseen_objects/<name>/collision.obj
```

If only `1abTDHy2qwSxzsrqo3DE4kUXV5xo2Zaud` fails, you can download it directly and place it at:

```
assets/simplified_objects/000/nontextured_simplified.ply
```

## Hugging Face data (optional for evaluation)

The training data is large (10.3 GB). The download was started but timed out before completion. It can be resumed with the same command:

```bash
./.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
repo_id = "KechunXu1/A2_Dataset"
for filename in ["a2_pp_data.npy", "a2_place_adaptive_data.npy"]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False,
    )
    print(path)
PY
```

## CUDA toolkit for extensions (RTX 5090)

RTX 5090 (sm_120) requires a modern CUDA + PyTorch stack. I switched to PyTorch 2.10.0+cu128 (nightly) and used the system CUDA 12.9 toolkit for `nvcc`.

Install PyTorch cu130 (RTX 5090 supported):

```bash
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Install CUDA 13.0 toolkit (no driver):

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
sh cuda_13.0.0_580.65.06_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-13.0 --no-opengl-libs
```

Build extensions with CUDA 13.0:

```bash
TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=$HOME/cuda-13.0 \
  ./.venv/bin/python models/graspnet/pointnet2/setup.py install

TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=$HOME/cuda-13.0 \
  ./.venv/bin/python models/graspnet/knn/setup.py install

Patch for PyTorch 2.10+ (remove deprecated THC includes):

```bash
# remove THC include usage
# files edited:
# - models/graspnet/knn/src/cuda/vision.h
# - models/graspnet/knn/src/knn.h
```

Runtime library path (needed for torch + nvidia libs):

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

# Ensure repo root is on PYTHONPATH for script entrypoints
export PYTHONPATH="$PWD"
export PATH="$PWD/.venv/bin:$PATH"

## RTX 5090 cuBLAS issues (workarounds)

On RTX 5090 + torch nightly, repeated `CUBLAS_STATUS_INVALID_VALUE` errors occurred in batched GEMM calls (attention, projection, and graspnet math). I added CPU fallbacks for these hot spots:

- `models/feature_fusion.py`: CPU fallback for batched `K @ Rt` and `H @ hpts`.
- `models/graspnet/utils/loss_utils.py`: CPU fallback for `batch_viewpoint_params_to_matrix` matmul.
- `models/graspnet/pointnet2/pointnet2_utils.py`: CPU fallback for grouped rotation matmul.
- `models/networks.py`: CPU fallback for cross-attention `multi_head_attention_forward`.

With cu130 wheels + CUDA 13.0 toolkit, the GPU path is stable and CPU fallbacks can be removed.

## Running scripts with output redirection

To avoid large terminal output, redirect logs to files and inspect them:

```bash
PYTHONPATH=$PWD PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib \
bash scripts/test/test_grasp.sh > logs/run_test_grasp.out 2>&1
```

Single-case smoke test (note: grasp cases are `.txt` files):

```bash
PYTHONPATH=$PWD PATH=$PWD/.venv/bin:$PATH \
LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/torch/lib:$HOME/cuda-13.0/lib64:$PWD/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib \
python a2/evaluate/test_pick.py --use_rope --load_model \
  --model_path a2_pretrained/checkpoints/sl_checkpoint_199.pth \
  --testing_case_dir testing_cases/grasp_testing_cases/seen \
  --testing_case case00-round.txt > logs/run_smoke_pick_cu130.out 2>&1
```
```
```

## Run benchmarks

Once the extensions build successfully and assets/testing_cases are in place:

```bash
bash scripts/test/test_grasp.sh
bash scripts/test/test_place.sh
bash scripts/test/test_pickplace.sh
```

The scripts expect CUDA GPUs and use `CUDA_VISIBLE_DEVICES` internally.
