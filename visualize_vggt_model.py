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
    # torchinfo でモデル構造を表示してファイルに保存
    import contextlib
    import io

    # 標準出力をキャプチャしてファイルに保存
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        summary(model, input_size=input_shape, device=device)
    
    summary_output = f.getvalue()
    print(summary_output)  # コンソールにも表示
    
    # ファイルに保存
    with open('VGGT_torchinfo_summary.txt', 'w', encoding='utf-8') as file:
        file.write("=== VGGT Model Summary (torchinfo) ===\n\n")
        file.write(summary_output)
    print("torchinfo summary saved as: VGGT_torchinfo_summary.txt")
    
except Exception as e:
    print(f"torchinfo summary failed: {e}")

print("\n=== Creating Network Diagram ===")
try:
    # torchview でネットワークグラフを作成（エラーハンドリング強化）
    dummy_input = torch.randn(*input_shape).to(device)
    
    # まず浅い深度で試行
    model_graph = draw_graph(
        model, 
        input_data=dummy_input,
        expand_nested=True,  # ネスト展開を有効化
        depth=1,  # 深度を1に制限
        device=device,
        save_graph=True,
        filename='VGGT_diagram'
    )
    print("torchview diagram saved as: VGGT_diagram")

except Exception as e:
    print(f"torchview visualization failed: {e}")
    print("Falling back to text-based architecture analysis...")
    
    # フォールバック: テキストベースの構造分析
    try:
        with open('VGGT_model_structure_fallback.txt', 'w', encoding='utf-8') as f:
            f.write("=== VGGT Model Structure Analysis ===\n\n")
            f.write("Main Components:\n")
            for name, module in model.named_children():
                f.write(f"- {name}: {type(module).__name__}\n")
                if hasattr(module, 'named_children'):
                    for subname, submodule in list(module.named_children())[:5]:  # 最初の5個のみ
                        f.write(f"  - {subname}: {type(submodule).__name__}\n")
            
            f.write(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        
        print("Fallback structure analysis saved as: VGGT_model_structure_fallback.txt")
    
    except Exception as e2:
        print(f"Fallback analysis also failed: {e2}")
