import torch
from torchinfo import summary
from torchview import draw_graph

from vggt.models.vggt import VGGT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# Define input shape
batch_size = 1  # 可視化用に小さく設定
channels = 3
height = 224
width = 224
input_shape = (batch_size, channels, height, width)

print("=== Model Summary ===")
try:
    # torchinfo でモデル構造を表示
    summary(model, input_size=input_shape, device=device)
except Exception as e:
    print(f"torchinfo summary failed: {e}")

print("\n=== Creating Network Diagram ===")
try:
    # torchview でネットワークグラフを作成
    dummy_input = torch.randn(*input_shape).to(device)
    model_graph = draw_graph(
        model, 
        input_data=dummy_input,
        expand_nested=True,
        depth=3,  # 階層の深さを制限
        device=device
    )
    model_graph.visual_graph.render('VGGT_network_structure', format='png', cleanup=True)
    print("Network diagram saved as: VGGT_network_structure.png")
except Exception as e:
    print(f"torchview visualization failed: {e}")

print("\n=== Model Architecture Text Output ===")
try:
    # モデルの構造をテキストで出力
    with open('VGGT_architecture.txt', 'w') as f:
        f.write(str(model))
        f.write(f"\n\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model architecture saved as: VGGT_architecture.txt")
except Exception as e:
    print(f"Text output failed: {e}")

# 各サブモジュールの構造を個別に可視化
print("\n=== Submodule Analysis ===")
try:
    for name, module in model.named_children():
        print(f"- {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for subname, submodule in module.named_children():
                print(f"  - {name}.{subname}: {type(submodule).__name__}")
except Exception as e:
    print(f"Submodule analysis failed: {e}")
