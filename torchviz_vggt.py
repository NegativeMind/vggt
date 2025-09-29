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
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)


# "pose_enc"：カメラポーズ（CameraHeadの出力）
pose_enc = make_dot(predictions["pose_enc"], params=dict(model.named_parameters()))
pose_enc.format = "png"
pose_enc.render("VGGT_pose_enc")

# "world_points"：3Dワールド座標（PointHead/DPTHeadの出力）
# "world_points_conf"：3D座標の信頼度（PointHead/DPTHeadの出力）

# "depth"：深度マップ（DepthHead/DPTHeadの出力）
# depth = make_dot(predictions["depth"], params=dict(model.named_parameters()))
# depth.format = "png"
# depth.render("VGGT_depth")

# "depth_conf"：深度の信頼度（DepthHead/DPTHeadの出力）
# "track"：トラッキング結果（TrackHeadの出力）
# "vis"：トラッキング可視性（TrackHeadの出力）
# "conf"：トラッキング信頼度（TrackHeadの出力）
