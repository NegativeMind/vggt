#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT ネットワーク構造図作成スクリプト
torchinfo の出力を解析して、見やすいネットワーク構造図を作成する
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import re
import os

# フォントとサイズの設定
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (20, 16)

def parse_torchinfo_summary(file_path):
    """torchinfo の出力を解析してネットワーク構造を抽出"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 主要なコンポーネントを抽出
    components = []
    
    # 正規表現パターンを使って階層構造を解析
    pattern = r'(├─|└─)([A-Za-z0-9_]+):\s*(\d+)-(\d+)\s+([^\[]+)?\s*(\[[^\]]+\])?\s+([0-9,]+|--)'
    
    lines = content.split('\n')
    for line in lines:
        if '├─' in line or '└─' in line:
            # 階層レベルを判定
            level = 0
            if '│    └─' in line:
                level = 2
            elif '│    ' in line:
                level = 2  
            elif '├─' in line or '└─' in line:
                level = 1
            
            # コンポーネント名と出力形状を抽出
            clean_line = line.strip()
            
            # VGGT の主要部分を特定
            if 'VGGT' in clean_line and level == 0:
                components.append({
                    'name': 'VGGT',
                    'type': 'model',
                    'level': 0,
                    'output_shape': '[1, 1, 3, 224, 224]',
                    'params': '65,941,396'
                })
            elif 'Aggregator:' in clean_line:
                components.append({
                    'name': 'Aggregator',
                    'type': 'aggregator',
                    'level': 1,
                    'output_shape': '[1, 1, 261, 2048]',
                    'params': '--'
                })
            elif 'DinoVisionTransformer:' in clean_line:
                components.append({
                    'name': 'DinoVisionTransformer',
                    'type': 'transformer',
                    'level': 2,
                    'output_shape': '--',
                    'params': '1,409,024'
                })
            elif 'PatchEmbed:' in clean_line:
                components.append({
                    'name': 'PatchEmbed',
                    'type': 'embedding',
                    'level': 3,
                    'output_shape': '[1, 256, 1024]',
                    'params': '603,136'
                })
            elif 'CameraHead:' in clean_line:
                components.append({
                    'name': 'CameraHead',
                    'type': 'head',
                    'level': 1,
                    'output_shape': '[1, 1, 9]',
                    'params': '9'
                })
            elif 'DPTHead:' in clean_line and 'depth' not in [c['name'] for c in components]:
                components.append({
                    'name': 'DPTHead (Depth)',
                    'type': 'head',
                    'level': 1,
                    'output_shape': '[1, 1, 224, 224, 1]',
                    'params': '--'
                })
            elif 'DPTHead:' in clean_line:
                components.append({
                    'name': 'DPTHead (Normal)',
                    'type': 'head',
                    'level': 1,
                    'output_shape': '[1, 1, 224, 224, 3]',
                    'params': '--'
                })
    
    return components

def create_network_diagram(components, output_path):
    """ネットワーク構造図を作成"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # 色の設定
    colors = {
        'model': '#FFE6E6',
        'aggregator': '#E6F3FF', 
        'transformer': '#E6FFE6',
        'embedding': '#FFFFE6',
        'head': '#F0E6FF'
    }
    
    # コンポーネントの位置を計算
    positions = {}
    
    # メインモデル (VGGT)
    vggt_box = FancyBboxPatch((1, 6), 8, 1.2, 
                              boxstyle="round,pad=0.1",
                              facecolor=colors['model'],
                              edgecolor='black',
                              linewidth=2)
    ax.add_patch(vggt_box)
    ax.text(5, 6.6, 'VGGT Model', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 6.2, 'Total Parameters: 1,707,637,164', ha='center', va='center', fontsize=10)
    
    # Aggregator
    agg_box = FancyBboxPatch((1, 4.5), 3.5, 1, 
                             boxstyle="round,pad=0.1",
                             facecolor=colors['aggregator'],
                             edgecolor='blue',
                             linewidth=1.5)
    ax.add_patch(agg_box)
    ax.text(2.75, 5.1, 'Aggregator', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.75, 4.7, 'Output: [1, 1, 261, 2048]', ha='center', va='center', fontsize=9)
    
    # DinoVisionTransformer (within Aggregator)
    dino_box = FancyBboxPatch((1.2, 3), 3.1, 0.8, 
                              boxstyle="round,pad=0.05",
                              facecolor=colors['transformer'],
                              edgecolor='green',
                              linewidth=1)
    ax.add_patch(dino_box)
    ax.text(2.75, 3.5, 'DinoVisionTransformer', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.75, 3.2, 'Params: 1,409,024', ha='center', va='center', fontsize=8)
    
    # PatchEmbed (within DinoVisionTransformer)
    patch_box = FancyBboxPatch((1.4, 1.8), 2.7, 0.6, 
                               boxstyle="round,pad=0.05",
                               facecolor=colors['embedding'],
                               edgecolor='orange',
                               linewidth=1)
    ax.add_patch(patch_box)
    ax.text(2.75, 2.1, 'PatchEmbed', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.75, 1.9, 'Output: [1, 256, 1024]', ha='center', va='center', fontsize=8)
    
    # Transformer Blocks (simplified representation)
    blocks_box = FancyBboxPatch((1.4, 0.8), 2.7, 0.6, 
                                boxstyle="round,pad=0.05",
                                facecolor='#F0F8FF',
                                edgecolor='gray',
                                linewidth=1)
    ax.add_patch(blocks_box)
    ax.text(2.75, 1.1, 'Transformer Blocks', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.75, 0.9, '(Multiple layers)', ha='center', va='center', fontsize=8)
    
    # Output Heads
    # Camera Head
    cam_box = FancyBboxPatch((5.5, 4.5), 1.8, 1, 
                             boxstyle="round,pad=0.1",
                             facecolor=colors['head'],
                             edgecolor='purple',
                             linewidth=1.5)
    ax.add_patch(cam_box)
    ax.text(6.4, 5.1, 'CameraHead', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6.4, 4.7, 'Output: [1, 1, 9]', ha='center', va='center', fontsize=9)
    
    # Depth Head
    depth_box = FancyBboxPatch((5.5, 3), 1.8, 1, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['head'],
                               edgecolor='purple',
                               linewidth=1.5)
    ax.add_patch(depth_box)
    ax.text(6.4, 3.6, 'DPTHead', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6.4, 3.4, '(Depth)', ha='center', va='center', fontsize=10)
    ax.text(6.4, 3.1, '[1, 1, 224, 224, 1]', ha='center', va='center', fontsize=8)
    
    # Normal Head  
    normal_box = FancyBboxPatch((5.5, 1.5), 1.8, 1, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['head'],
                                edgecolor='purple',
                                linewidth=1.5)
    ax.add_patch(normal_box)
    ax.text(6.4, 2.1, 'DPTHead', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6.4, 1.9, '(Normal)', ha='center', va='center', fontsize=10)
    ax.text(6.4, 1.6, '[1, 1, 224, 224, 3]', ha='center', va='center', fontsize=8)
    
    # Input
    input_box = FancyBboxPatch((0.2, 6), 0.6, 1.2, 
                               boxstyle="round,pad=0.05",
                               facecolor='#98FB98',
                               edgecolor='black',
                               linewidth=1)
    ax.add_patch(input_box)
    ax.text(0.5, 6.6, 'Input', ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)
    ax.text(0.5, 6.3, '[1,3,224,224]', ha='center', va='center', fontsize=8, rotation=90)
    
    # 接続線を描画
    # Input -> VGGT
    ax.arrow(0.8, 6.6, 0.15, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # VGGT -> Aggregator
    ax.arrow(2.75, 6, 0, -0.4, head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    
    # Aggregator -> DinoVisionTransformer
    ax.arrow(2.75, 4.5, 0, -0.6, head_width=0.08, head_length=0.05, fc='green', ec='green')
    
    # DinoVisionTransformer -> PatchEmbed
    ax.arrow(2.75, 3, 0, -0.55, head_width=0.06, head_length=0.05, fc='orange', ec='orange')
    
    # PatchEmbed -> Blocks
    ax.arrow(2.75, 1.8, 0, -0.35, head_width=0.06, head_length=0.05, fc='gray', ec='gray')
    
    # Aggregator -> Heads
    ax.arrow(4.5, 5, 0.9, 0, head_width=0.08, head_length=0.05, fc='purple', ec='purple')
    ax.arrow(4.5, 5, 0.7, -1.5, head_width=0.08, head_length=0.05, fc='purple', ec='purple')
    ax.arrow(4.5, 5, 0.7, -3, head_width=0.08, head_length=0.05, fc='purple', ec='purple')
    
    # 図の装飾
    ax.set_title('VGGT Network Architecture\n(Generated from torchinfo output)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # 凡例
    legend_elements = [
        patches.Patch(color=colors['model'], label='Main Model'),
        patches.Patch(color=colors['aggregator'], label='Aggregator'),
        patches.Patch(color=colors['transformer'], label='Transformer'),
        patches.Patch(color=colors['embedding'], label='Embedding'),
        patches.Patch(color=colors['head'], label='Output Heads'),
        patches.Patch(color='#98FB98', label='Input')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # 軸を非表示
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 総パラメータ数などの統計情報をテキストで追加
    stats_text = """
Model Statistics:
• Total Parameters: 1,707,637,164
• Trainable Parameters: 1,707,636,140
• Model Size: ~4.76 GB
• Input Size: [1, 3, 224, 224]
• Multiple Output Heads:
  - Camera: 9 parameters
  - Depth: 224×224×1
  - Normal: 224×224×3
"""
    
    ax.text(7.8, 0.5, stats_text, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    return fig

def create_detailed_block_diagram(output_path):
    """詳細なブロック図を作成"""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 色の設定
    colors = {
        'input': '#98FB98',
        'patch': '#FFFFE6', 
        'transformer': '#E6FFE6',
        'aggregator': '#E6F3FF',
        'heads': '#F0E6FF',
        'output': '#FFE6E6'
    }
    
    # Input
    input_box = FancyBboxPatch((0.5, 8), 1.5, 1, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 8.5, 'Input Image\n[1, 3, 224, 224]', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Patch Embedding
    patch_box = FancyBboxPatch((3, 8), 2, 1, 
                               boxstyle="round,pad=0.1",
                               facecolor=colors['patch'],
                               edgecolor='orange', linewidth=2)
    ax.add_patch(patch_box)
    ax.text(4, 8.5, 'Patch Embedding\n[1, 256, 1024]', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Transformer Blocks (複数のブロックを表現)
    transformer_blocks = []
    for i in range(4):
        y_pos = 6.5 - i * 1.2
        block_box = FancyBboxPatch((3, y_pos), 2, 0.8, 
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['transformer'],
                                   edgecolor='green', linewidth=1.5)
        ax.add_patch(block_box)
        if i == 0:
            ax.text(4, y_pos + 0.4, f'Transformer Block {i+1}\n(Multi-head Attention + MLP)', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(4, y_pos + 0.4, f'Transformer Block {i+1}\n...', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    # "..." を表示して省略を示す
    ax.text(4, 2.5, '... (more blocks)', ha='center', va='center', fontsize=10, style='italic')
    
    # Aggregator
    agg_box = FancyBboxPatch((6, 5), 2.5, 2, 
                             boxstyle="round,pad=0.1",
                             facecolor=colors['aggregator'],
                             edgecolor='blue', linewidth=2)
    ax.add_patch(agg_box)
    ax.text(7.25, 6, 'Aggregator\n[1, 1, 261, 2048]', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Output Heads
    heads_y_positions = [8.5, 6.5, 4.5]
    head_names = ['Camera Head\n[1, 1, 9]', 'Depth Head\n[1, 1, 224, 224, 1]', 'Normal Head\n[1, 1, 224, 224, 3]']
    
    for i, (y_pos, name) in enumerate(zip(heads_y_positions, head_names)):
        head_box = FancyBboxPatch((9.5, y_pos - 0.4), 2, 0.8, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['heads'],
                                  edgecolor='purple', linewidth=2)
        ax.add_patch(head_box)
        ax.text(10.5, y_pos, name, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # 接続線を描画
    # Input -> Patch Embedding
    ax.arrow(2, 8.5, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Patch Embedding -> First Transformer Block
    ax.arrow(4, 8, 0, -1.4, head_width=0.1, head_length=0.1, fc='green', ec='green')
    
    # Between Transformer Blocks
    for i in range(3):
        y_start = 6.5 - i * 1.2
        ax.arrow(4, y_start - 0.4, 0, -0.4, head_width=0.08, head_length=0.08, fc='green', ec='green')
    
    # Last Transformer Block -> Aggregator
    ax.arrow(5, 5, 0.9, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Aggregator -> Heads
    for y_pos in heads_y_positions:
        ax.arrow(8.5, 6, 0.9, y_pos - 6, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
    
    # データフロー説明
    flow_text = """
Data Flow:
1. Input Image [1, 3, 224, 224]
2. Patch Embedding → [1, 256, 1024]
3. Multiple Transformer Blocks
4. Feature Aggregation → [1, 1, 261, 2048]
5. Multiple Task-specific Heads:
   • Camera parameters (9 values)
   • Depth map (224×224×1)
   • Surface normals (224×224×3)
"""
    
    ax.text(0.5, 3, flow_text, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.9))
    
    # タイトル
    ax.set_title('VGGT Detailed Architecture Flow\n(From torchinfo Analysis)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # 軸を非表示
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    return fig

def main():
    """メイン関数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    torchinfo_file = os.path.join(current_dir, 'VGGT_torchinfo_summary.txt')
    
    print("torchinfo 出力を解析してネットワーク構造図を作成中...")
    
    # torchinfo ファイルが存在するかチェック
    if not os.path.exists(torchinfo_file):
        print(f"Error: {torchinfo_file} が見つかりません")
        return
    
    # コンポーネント解析
    components = parse_torchinfo_summary(torchinfo_file)
    print(f"検出されたコンポーネント: {len(components)} 個")
    
    # 概要図作成
    overview_path = os.path.join(current_dir, 'VGGT_network_overview.png')
    fig1 = create_network_diagram(components, overview_path)
    print(f"概要図を保存しました: {overview_path}")
    
    # 詳細図作成
    detailed_path = os.path.join(current_dir, 'VGGT_network_detailed.png')
    fig2 = create_detailed_block_diagram(detailed_path)
    print(f"詳細図を保存しました: {detailed_path}")
    
    # 表示
    plt.show()
    
    print("\nネットワーク構造図の作成が完了しました！")
    print("作成されたファイル:")
    print(f"- {overview_path}")
    print(f"- {overview_path.replace('.png', '.pdf')}")
    print(f"- {detailed_path}")
    print(f"- {detailed_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()