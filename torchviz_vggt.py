import torch
from torchviz import make_dot

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = [
    "./examples/room/images/no_overlap_1.png",
    "./examples/room/images/no_overlap_2.jpg",
    "./examples/room/images/no_overlap_3.jpg",
    "./examples/room/images/no_overlap_4.jpg",
    "./examples/room/images/no_overlap_5.jpg",
    "./examples/room/images/no_overlap_6.jpg",
    "./examples/room/images/no_overlap_7.jpg",
    "./examples/room/images/no_overlap_8.jpg",
]

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

# レイヤーごとに1ノードとなるよう、代表的な出力(depth)のみを可視化
if "depth" in predictions:
    output = predictions["depth"][0]
    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.format = "png"
    dot.render("VGGT_layerwise_nodes")
else:
    print("No suitable outputs found for visualization.")
