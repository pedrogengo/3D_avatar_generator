# Generate a 3D Avatar of Yourself
by [pedro](https://twitter.com/Predogl) and [ritwik](https://twitter.com/ritwik_raha)

Create a 3D Avatar of Yourself from your images using an IP Adapter and a pretrained image-to-3D Model.

## The 3D Avatar Workflow

Our workflow explores this proposal in the following stages:

- First we upload the image and supply a prompt.
- Next we use an IP Adapter to convert the image in a specific style.
- Finally we use the pretrained TripoSR model from StabilityAI to generate a 3D object and a video.

<figure>
    <img src="https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/3D-Avatar-diagram.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Diagram of the entire workflow</figcaption>
</figure>



## How does the image to 3D workflow work?

> "TripoSR is a fast and feed-forward 3D generative model developed in collaboration between Stability AI and Tripo AI."
-[TripoSR by stability.ai](https://huggingface.co/stabilityai/TripoSR)

TripoSR is meant to follow the pipeline of the LRM model (Large Reconstruction Model for Single Image to 3D) very closely.


<figure>
    <img src="https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/Screenshot%202024-03-25%20at%2011.52.06%E2%80%AFPM.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Source: https://arxiv.org/abs/2311.04400</figcaption>
</figure>

LRM takes an image and tries to turn it into a 3D model. Here's how it works:

1. **Image Understanding:** LRM first uses a pre-trained image recognition model to understand the objects and details in the image. This model pays attention to both the structure and texture of the object.

2. **Camera Awareness:** LRM also considers the camera angle and position that took the picture. This helps to account for any distortion caused by the camera.

3. **Space Slicing:** LRM imagines the 3D space around the object divided into thin slices along three axes (like chopping a box into slabs). This is called a triplane representation.

4. **Image to Triplane Mapping:** LRM uses the image information and camera awareness to project details onto these slices. It does this by letting the slices "talk" to the image features, like matching puzzle pieces.

5. **Fine-tuning the Slices:** LRM refines the slices further by considering the relationships between the details within each slice and across different slices.

6. **3D Prediction:** Finally, LRM uses the information in the refined slices to predict the color and density of every point in 3D space, essentially creating a volume that represents the 3D model.

LRM can also use additional side views of the same object during training to improve the accuracy of the 3D reconstruction.

Let us begin by setting the tools we need for this workflow.

## Setup

We first clone the TripoSR repository into our notebooks and set our device type.

Note: Remember to install the requirements file of the TripoSR model.


```python
!git clone https://github.com/ritwikraha/TripoSR.git
```


```python
import sys
sys.path.append('/content/TripoSR/tsr')
```


```python
%cd TripoSR
```


```python
!pip install -r requirements.txt -q
!pip install -U diffusers accelerate -qq
```


```python
import torch
# Adjust device based on CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Imports


```python
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image
import os
import time
from PIL import Image
import numpy as np
from IPython.display import Video
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
import rembg
```

## Load the Pipelines for Image Preprocessing

We use an Adapter to process the image in a specific style.

An IP-Adapter with only 22M parameters can achieve comparable or even better performance to a fine-tuned image prompt model. IP-Adapter can be generalized not only to other custom models fine-tuned from the same base model, but also to controllable generation using existing controllable tools. Moreover, the image prompt can also work well with the text prompt to accomplish multimodal image generation.



```python
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to(device)
```


### Loading the Base Text-to-Image Pipeline:

- We start by loading a pre-trained text-to-image pipeline model called `AutoPipelineForText2Image` from the `stabilityai/stable-diffusion-xl-base-1.0` model repository.


### Configuring the Schedule:

- We initilaise a new `DDIMScheduler` for controlling different phases of the image generation process.


### Loading and Setting Up the Image Processing Adapter:

- We load an IP adapter model, from a repository called `h94/IP-Adapter`.
- We specify a subfolder `sdxl_models` where the adapter weights are stored.
- We provide two weight names: `ip-adapter-plus_sdxl_vit-h.safetensors` and `ip-adapter-plus-face_sdxl_vit-h.safetensors`.
- We set a scaling factor of `[0.7, 0.3]` for balancing the influence of different adapter components on final images.




```python
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
).to(device)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
)
pipeline.set_ip_adapter_scale([0.7, 0.3])
pipeline.enable_model_cpu_offload()
```

## Timer Class for Utility



We define a `Timer` class to track execution times. It stores start times for named timers in a dictionary. The `start` method records the current time for a given timer name. The `end` method retrieves the start time, calculates the elapsed time (adjusted to milliseconds), and prints the name and duration.

We then create a `Timer` object named `timer` for use in our program.



```python
# Define Timer class
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        print(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()
```

## Upload your data and Generate Avatar Image

Here we upload the image and supply the prompt needed to generate an image in a specified style.

### Upload your picture


```python
from google.colab import files

uploaded = files.upload()
```


```python
# @title Enter a prompt { run: "auto", vertical-output: true, form-width: "10000px", display-mode: "form" }
prompt = "a 3D version of this person" # @param {type:"string"}

```

### Upload the style images to guide the generation

Here, we are using some images hosted on HF that have the funko pop style. If you want, you can change it to use any style you want.


```python
face_image = Image.open(list(uploaded.keys())[0])
style_folder = "https://huggingface.co/datasets/pedrogengo/funkopop_images/resolve/main"
style_images = [load_image(f"{style_folder}/funko{i}.jpeg").resize((1024, 1024)) for i in range(1, 5)]
```

### Generate images using the pipeline

Supply the prompt and the style and object image. We specifiy the `num_inference_steps` to `50`, we can also set that as we need.

A more number of steps will be useful in diluting the `strength` parameter.


```python
generator = torch.Generator(device=device).manual_seed(42)

image = pipeline(
    prompt=prompt,
    ip_adapter_image=[style_images, face_image],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50, num_images_per_prompt=1,
    generator=generator,
).images[0]
```


```python
image.resize((512, 512))
```



<figure>
    <img src="https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/avatar.png"
         alt="Albuquerque, New Mexico">
    <figcaption>Image in the promised style</figcaption>
</figure>


```python
image.resize((512, 512)).save("examples/avatar.jpg")
```

## The TripoSR model for 3D Avatar


```python
# Parameters for running the TripoSR
image_paths = "/content/TripoSR/examples/avatar.jpg"
device = "cuda:0"
pretrained_model_name_or_path = "stabilityai/TripoSR"
chunk_size = 8192
no_remove_bg = True
foreground_ratio = 0.85
output_dir = "output/"
model_save_format = "obj"
render = True
```


```python
output_dir = output_dir.strip()
os.makedirs(output_dir, exist_ok=True)
```

### Generate Images for the 3D model

We'll initialize by loading the powerful TSR model from the disk and configuring its rendering chunk size.  Then, to ensure optimal performance, we'll strategically place the model on either the CPU or GPU. (Go for a GPU or go home)

Next, we'll dive into processing your images.  First, we'll create a handy list to keep track of all the processed ones.  Then, we'll leverage rembg to initiate a background removal session, ensuring your avatar takes center stage.  For each image in your collection, we'll meticulously remove the background and resize the foreground to a specific ratio.  

If the image has an alpha channel, which helps manage transparency, we'll normalize the pixel values and handle alpha blending for a seamless compositing process.  Finally, we'll save the processed image with a clear filename like `input.png` within a designated output directory, before adding it to our list for further processing.

Remember these are your views, that will be projected to the 3D triplane.





```python
# Initialize model
timer.start("Initializing model")
model = TSR.from_pretrained(
    pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)
timer.end("Initializing model")

# Process images
timer.start("Processing images")
images = []


rembg_session = rembg.new_session()

image = remove_background(image, rembg_session)
image = resize_foreground(image, foreground_ratio)

if image.mode == "RGBA":
  image = np.array(image).astype(np.float32) / 255.0
  image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
  image = Image.fromarray((image * 255.0).astype(np.uint8))

image_dir = os.path.join(output_dir, str(0))
os.makedirs(image_dir, exist_ok=True)
image.save(os.path.join(image_dir, "input.png"))
images.append(image)
timer.end("Processing images")
```


```python
# Visualise the image
image
```

<figure>
    <img src="https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/avatar-3d.png"
         alt="Albuquerque, New Mexico">
    <figcaption>3D Image from the image</figcaption>
</figure>

### Render Video from Images


In this stage, we'll process each image individually. We'll loop through your entire collection, providing progress updates as we go. For each image, we'll disable gradients within the model for efficiency before running it to generate scene codes. These scene codes act as the blueprint for your 3D avatar. If rendering is enabled, we'll take things a step further. We'll use the scene codes to create 30 unique views of your avatar from different angles. Each view will be saved as a PNG image, and we'll even compile them into a video showcasing your avatar in motion. Finally, we'll extract the 3D mesh, the core structure of your avatar, from the scene codes and save it in your preferred format.




```python
# Process each image
for i, image in enumerate(images):
    print(f"Running image {i + 1}/{len(images)} ...")

    # Run model
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    # Rendering
    if render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30
        )
        timer.end("Rendering")

    # Export mesh
    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes)
    mesh_file = os.path.join(output_dir, str(i), f"mesh.{model_save_format}")
    meshes[0].export(mesh_file)
    timer.end("Exporting mesh")

print("Processing complete.")
```

## Output Video


```python
# Display the video
Video('output/0/render.mp4', embed=True)
```

# Summary


This project explores an approach to creating 3D avatars directly from your images. This workflow utilizes diffusion tools like IP Adapters and a pretrained image-to-3D model like TripoSR to create 3D version of profile photos.

TripoSR follows a fascinating pipeline inspired by the LRM (Large Reconstruction Model). LRM models typically works through [Image Triplane Mapping](https://helpx.adobe.com/substance-3d-painter/painting/fill-projections/tri-planar-projection.html) which is a way to associate image slices with a specific viewpoint in the 3D space.

**The Workflow:**

1. **Image Upload and Prompting:** You begin by providing a high-quality image of yourself or your desired avatar. Additionally, you can include a text prompt to specify desired characteristics.
2. **Optional Style Transfer:** Want your avatar to have a cartoonish or artistic flair? An IP Adapter can be used to adjust the image style before conversion. Get your own or choose from HuggingFace's wide array of Adapters.
3. **TripoSR Model Processing:**  The core of this process lies with TripoSR, a powerful 3D generative model. It analyzes the image, accounting for camera perspective and object details, to create a 3D representation.
4. **3D Model Rendering:** Finally, the generated 3D model can be visualized and even rendered as a video for a complete showcase.


This project is a simple take on how remixing tools like Adapters, LoRAs and pretrained models can lead to insanely creative generations like a 3D model of your profile picture, a talking head.

Who knows, someday a metaverse character?

# References

- [TripoSR by Stability AI](https://huggingface.co/stabilityai/TripoSR)
- [IP Adapter Face ID](https://huggingface.co/h94/IP-Adapter-FaceID)
- [Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400)
- [Representing scenes as Neural Fields](https://arxiv.org/abs/2003.08934)
