import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8
})

fig, ax = plt.subplots(figsize=(3.5, 4.5)) # Slightly taller for the block diagram
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def draw_box(x, y, w, h, text, facecolor='white', edgecolor='black', linestyle='-', hatch=None, text_color='black'):
    box = patches.Rectangle((x - w/2, y - h/2), w, h, edgecolor=edgecolor, facecolor=facecolor, 
                            linestyle=linestyle, hatch=hatch, linewidth=1, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color=text_color, linespacing=1.3, zorder=3)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle='->', lw=1, color='black'), zorder=1)

# --- Top: Inference ---
ax.text(0, 11.5, '(a) Shared Inference Backbone', fontweight='bold', fontsize=9, va='center')

draw_box(5, 10.2, 4, 0.8, "Input Features (10 s windows)")
draw_arrow(5, 9.8, 5, 9.2)

draw_box(5, 8.8, 6, 0.8, "NEST / Fast Conformer Encoder", facecolor='#E8E8E8')
draw_arrow(5, 8.4, 5, 7.8)

# Split for Offline vs Streaming
draw_box(5, 7.4, 6, 0.8, "Transformer Decoder", facecolor='#E8E8E8')

# Arrows down to specific branches
draw_arrow(3, 7.0, 3, 6.2)
draw_arrow(7, 7.0, 7, 6.2)

draw_box(3, 5.8, 3.5, 0.8, "Offline Output\n(Speaker Slots)")
draw_box(7, 5.8, 3.5, 0.8, "Streaming Output\n+ AOSC Module", facecolor='#D0E0E3') # Slight blue tint to distinguish AOSC


# --- Bottom: Adaptation ---
ax.text(0, 4.2, '(b) Frozen-Encoder Adaptation Path', fontweight='bold', fontsize=9, va='center')

# Highlighted Adaptation Flow
draw_box(5, 3.2, 5, 0.8, "Pretrained Checkpoint Initialized")
draw_arrow(5, 2.8, 5, 2.3)

# Frozen Box
draw_box(5, 1.9, 6.5, 0.8, "NEST / Fast Conformer Encoder\n[ FROZEN ]", 
         facecolor='#D3D3D3', edgecolor='#606060', linestyle='--', hatch='////')
draw_arrow(5, 1.5, 5, 1.0)

# Unfrozen Box
draw_box(5, 0.6, 6.5, 0.8, "Transformer & Output Layers\n[ UPDATED ] (1000 steps)", 
         facecolor='#FFFFFF', edgecolor='black', linestyle='-')

plt.tight_layout()
plt.savefig('results/paper_figures/fig_model_adaptation_overview.pdf',
            format='pdf',
            bbox_inches='tight', dpi=300)
plt.show()