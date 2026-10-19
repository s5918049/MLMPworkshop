# Workshop 02 — AI Project Survival

## 1. Project overview

**Project:** Real-ESRGAN  
**Repository:** https://github.com/xinntao/Real-ESRGAN  
**Task:** Run a pretrained image super-resolution model on a low-resolution image and verify the generated output.

Real-ESRGAN is an image restoration and upscaling project. In this experiment, I used the pretrained `RealESRGAN_x4plus` model to upscale a low-resolution image of Gojo.

## 2. Environment

The experiment was run on a university Linux workstation.

- Operating system: Linux (Red Hat environment)
- Python: 3.10.20
- PyTorch: 2.14.1+cu130
- Torchvision: 0.29.1+cu130
- GPU: NVIDIA GeForce RTX 4080 (16 GB VRAM)
- Virtual environment: Python virtual environment managed using `uv`

CUDA was available, and the Real-ESRGAN imports were verified successfully.

## 3. Model and input

- **Model:** `RealESRGAN_x4plus`
- **Checkpoint:** `RealESRGAN_x4plus.pth`
- **Input image:** `Gojo.jpeg`
- **Input dimensions:** 311 × 162 pixels
- **Input format:** JPEG, RGB

The checkpoint was downloaded from the project's GitHub release:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/
```

## 4. Inference command

From the Real-ESRGAN repository directory, I ran:

```bash
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i inputs/Gojo.jpeg \
  --output results
```

The terminal reported:

```text
Testing 0 Gojo
```

The generated image was saved as `results/Gojo_out.jpeg`.

## 5. Results and verification

| Measurement | Result |
|---|---|
| Input dimensions | 311 × 162 |
| Output dimensions | 1244 × 648 |
| Upscaling factor | 4× |
| Inference status | Successful |

I verified the dimensions using Pillow:

```bash
python -c "from PIL import Image; a=Image.open('inputs/Gojo.jpeg'); b=Image.open('results/Gojo_out.jpeg'); print('Input:', a.size); print('Output:', b.size)"
```

Verified output:

```text
Input: (311, 162)
Output: (1244, 648)
```

The output width and height are both four times the input dimensions.

## 6. Problems encountered and fixes

### Problem 1: Missing Torchvision module

**Error:**

```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

**Cause:** The installed BasicSR code imported `rgb_to_grayscale` from an older Torchvision module path that was unavailable in the installed version.

**Fix:** I changed the import in the BasicSR file inside the virtual environment to:

```python
from torchvision.transforms.functional import rgb_to_grayscale
```

The file was:

`.venv/lib/python3.10/site-packages/basicsr/data/degradations.py`

This compatibility change was made inside the virtual environment, not in the Real-ESRGAN repository source.

### Problem 2: Missing Real-ESRGAN version module

**Error:**

```text
ModuleNotFoundError: No module named 'realesrgan.version'
```

**Fix:** I installed the required packaging tools and used the repository setup script to address the package setup issue. I then verified that the PyTorch, Torchvision, and Real-ESRGAN imports worked.

The final import verification reported:

```text
PyTorch: 2.14.1+cu130
Torchvision: 0.29.1+cu130
CUDA available: True
Real-ESRGAN imports: OK
```

## 7. Evidence

The `evidence/` directory contains:

- `Gojo_input.jpeg` — original low-resolution input image.
- `Gojo_output.jpeg` — generated upscaled image.

Both files were copied from the actual experiment into the submission folder.

## 8. Reproduction notes

The following commands describe the setup used for this experiment. They have not been verified end-to-end from a fresh clone.

### Clone and install dependencies

Run these commands in a Linux terminal with Git, Python 3.10, and `uv` available:

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install pip cython
uv pip install -r requirements.txt
```

### Apply the BasicSR compatibility fix

With the virtual environment activated, run:

```bash
sed -i 's/from torchvision\.transforms\.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' .venv/lib/python3.10/site-packages/basicsr/data/degradations.py
```

### Install Real-ESRGAN

```bash
python setup.py develop
```

The setup script generates `realesrgan/version.py`. The packaging tools must be installed before running it.

### Download the pretrained model

```bash
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights/
```

Place the test image at `inputs/Gojo.jpeg`, then run:

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/Gojo.jpeg --output results
```

Verify the dimensions:

```bash
python -c "from PIL import Image; a=Image.open('inputs/Gojo.jpeg'); b=Image.open('results/Gojo_out.jpeg'); print('Input:', a.size); print('Output:', b.size)"
```

Dependency versions can change, and the BasicSR compatibility patch may not be necessary with other package combinations. A fresh reproduction should verify each installation and import step.

Keep the repository, virtual environment, and model checkpoint outside the coursework submission folder. Submit the README and the input/output images in `evidence/`.

## 9. Conclusion

I successfully ran a pretrained Real-ESRGAN model on a low-resolution image and verified that the output was four times the original width and height. The main challenge was resolving dependency compatibility and package setup issues before inference could run.

The experiment demonstrates a successful inference run. Visual quality should be evaluated by comparing the original and upscaled images; larger dimensions alone do not establish that every detail was reconstructed accurately.