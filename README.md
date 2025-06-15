# AI-Powered Room Decoration Tool 

Welcome to the AI-Powered Room Decoration Tool! This project leverages the power of semantic segmentation and generative AI to let you reimagine and customize your living spaces. Simply provide an image of a room, and this tool allows you to select objects like furniture, walls, or decor and instantly change their color or texture using text prompts.

## Key Features

-   **Object Detection & Segmentation:** Automatically identifies and isolates various objects within a room image.
-   **AI-Powered Customization:** Uses a diffusion model to modify specific objects based on your text descriptions.
-   **Seamless Integration:** Combines segmentation and generation into a smooth workflow to produce high-quality, realistic results.

## How It Works

The magic happens in a two-step process:

1.  **Segmentation:** The application first uses **Hugging Face's SegFormer**, a powerful transformer-based model, to analyze the input image and create a semantic map of all the objects it contains. When you choose an object to modify, a precise mask is generated for it.
2.  **Generation:** This mask, along with the original image and a text prompt (e.g., "a blue velvet armchair"), is fed into a **Stable Diffusion** in-painting model. The model then intelligently redraws only the masked area, applying the new style while keeping the rest of the image untouched.

## Technology Stack

-   **Models:** SegFormer, Stable Diffusion
-   **Core Libraries:** Hugging Face `transformers` & `diffusers`, PyTorch
-   **Environment:** Jupyter Notebook, Python

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amgharhind/AI-powered-room-decoration.git
    ```

2.  **Launch Jupyter Notebook** and open the `.ipynb` file to run the project.
