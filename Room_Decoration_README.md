# AI-Powered Room Decoration Tool

Welcome to the AI-Powered Room Decoration Tool! This project leverages the power of semantic segmentation and generative AI to let you reimagine and customize your living spaces. Simply provide an image of a room, and this tool allows you to select objects like furniture, walls, or decor and instantly change their color or texture using text prompts — all without touching any other part of the image.

## Motivation and Importance

Decorating a space is a highly visual and iterative process. Traditionally, imagining how a room would look with a different sofa color, wall texture, or flooring requires either professional design tools or expensive consultations. This project aims to:

- **Democratize Interior Design:** Give anyone the ability to visualize decoration changes instantly using natural language.
- **Reduce Decision Uncertainty:** Allow users to experiment with different styles before committing to a real-world purchase or renovation.
- **Explore State-of-the-Art AI Pipelines:** Combine cutting-edge segmentation and generative models into a coherent, end-to-end application.

## Key Features

- **Object Detection & Segmentation:** Automatically identifies and isolates all objects present in a room image using a transformer-based segmentation model.
- **Interactive Object Selection:** The user specifies the object to modify by entering its label (e.g., `"sofa"`, `"wall"`, `"floor"`), and the corresponding binary mask is automatically extracted.
- **Multiple AI-Powered Generation Approaches:** Three distinct pipelines are explored for applying the modification — plain inpainting, ControlNet text-to-image, and ControlNet image-to-image with depth — allowing a thorough comparison of generation quality.
- **Non-Destructive Editing:** Only the masked region is modified; the rest of the image remains completely untouched.

## How It Works

The tool operates in two main stages:

### Stage 1 — Semantic Segmentation with SegFormer

The input room image is processed by a pre-trained **SegFormer** model from NVIDIA, fine-tuned on the **ADE20K** dataset (a large-scale indoor/outdoor scene understanding benchmark). Two model variants are explored:

- `nvidia/segformer-b5-finetuned-ade-512-512` — Larger, higher-accuracy variant tested initially.
- `nvidia/segformer-b1-finetuned-ade-512-512` — Lighter, faster variant used in the final pipeline for efficiency.

The segmentation pipeline detects all objects in the scene and generates a binary mask for each detected label. When the user selects an object, its mask is extracted and passed to the generation stage.

### Stage 2 — Image Generation with Diffusion Models

Three generation strategies are implemented and compared:

#### Approach 1 — Stable Diffusion Inpainting
- **Model:** `runwayml/stable-diffusion-inpainting`
- **Process:** The original image, the binary object mask, and a text prompt (e.g., `"Change the wall to have a red velvet texture"`) are passed directly to the Stable Diffusion inpainting pipeline. The model redraws only the masked area guided by the prompt, while leaving the rest of the image intact.
- **Hardware:** Runs on GPU with `torch.float16` for memory efficiency.

#### Approach 2 — ControlNet Text-to-Image with Segmentation Mask
- **Model:** `lllyasviel/control_v11p_sd15_seg` + `StableDiffusionControlNetPipeline`
- **Process:** The segmentation mask is used as a **structural conditioning signal** for ControlNet. Instead of simple inpainting, ControlNet guides the generation to respect the spatial layout and shapes of the original scene while applying the textual style description. The scheduler is set to `UniPCMultistepScheduler` for faster, high-quality inference.

#### Approach 3 — ControlNet Image-to-Image with Depth Map
- **Model:** `StableDiffusionControlNetImg2ImgPipeline` with depth conditioning
- **Process:** A **depth estimation model** (`pipeline("depth-estimation")`) first generates a depth map from the original image. This 3-channel depth map is then fed to a ControlNet image-to-image pipeline, providing 3D spatial awareness to the generation. This approach preserves the room's structural geometry and perspective more faithfully than flat segmentation masks alone.

#### Approach 4 — ControlNet Inpainting
- **Model:** `StableDiffusionControlNetInpaintPipeline`
- **Process:** Combines the precision of inpainting (masking only the selected object) with ControlNet's structural conditioning (using the original image as the control signal). This hybrid approach delivers the highest fidelity by simultaneously constraining the generation spatially and limiting it to the masked region.

## Methodology Summary

```
Input Room Image
       │
       ▼
SegFormer Segmentation (ADE20K)
       │
       ├── Detected object labels + binary masks
       │
User selects object label (e.g., "sofa")
       │
       ▼
Binary Mask Extraction
       │
       ├──► Approach 1: SD Inpainting (prompt + mask)
       ├──► Approach 2: ControlNet Text-to-Image (segmentation map)
       ├──► Approach 3: ControlNet Img2Img (depth map)
       └──► Approach 4: ControlNet Inpainting (mask + control image)
                │
                ▼
         Modified Room Image
```

## Technology Stack

- **Segmentation Model:** SegFormer (`nvidia/segformer-b1-finetuned-ade-512-512`, `nvidia/segformer-b5-finetuned-ade-512-512`)
- **Generation Models:** Stable Diffusion Inpainting (`runwayml/stable-diffusion-inpainting`), ControlNet (`lllyasviel/control_v11p_sd15_seg`)
- **Depth Estimation:** Hugging Face `pipeline("depth-estimation")`
- **Core Libraries:** Hugging Face `transformers`, `diffusers`, PyTorch, NumPy, Matplotlib, Pillow
- **Scheduler:** `UniPCMultistepScheduler` for ControlNet inference
- **Environment:** Kaggle Notebook (GPU), Python 3

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amgharhind/AI-powered-room-decoration.git
   ```

2. **Install dependencies:**
   ```bash
   pip install transformers diffusers torch torchvision accelerate pillow matplotlib numpy
   ```

3. **Prepare your image:** Place your room image in the working directory and update the image path in the notebook.

4. **Launch Jupyter Notebook** and open `home-dcoration-diffusion-model-vf.ipynb` to run the full pipeline. A GPU is strongly recommended for Stable Diffusion inference.

## Expected Results

- **Precise Object Isolation:** SegFormer reliably identifies common room elements such as sofas, walls, floors, chairs, and windows.
- **Realistic Texture and Color Transfers:** The diffusion models produce photorealistic modifications that blend naturally with the surrounding, unmodified areas of the image.
- **Side-by-Side Comparison:** Each approach outputs a side-by-side visualization of the original and modified images for easy evaluation.
 
<img width="800" height="283" alt="image" src="https://github.com/user-attachments/assets/a38bfbe3-f743-4075-bc08-bc33a5bd26e2" />


## Limitations and Future Work

- **ADE20K Label Coverage:** The segmentation model is limited to the object classes present in ADE20K. Unusual or very specific furniture items may not be recognized correctly.
- **Mask Precision:** Segmentation masks can have rough boundaries, especially for complex shapes. Fine-tuning SegFormer on a custom interior dataset would improve mask quality.
- **Interactive UI:** The current object selection is text-input based inside the notebook. A future version could provide a visual click-to-select interface using Gradio or Streamlit.
- **Real-Time Processing:** Diffusion model inference is computationally expensive. Optimized pipelines (e.g., using SDXL Turbo or LCM schedulers) could enable near real-time previews.
- **3D Consistency:** While the depth map approach improves spatial awareness, true 3D consistency across multiple edits would require a more comprehensive 3D scene representation.

