import matplotlib.pyplot as plt
import matplotlib.patches as patches

# IEEE standard figure settings
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8
})

fig, ax = plt.subplots(figsize=(3.5, 3.2)) # IEEE single column width
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# --- PART (a) Windowing Protocol ---
ax.text(0, 9.5, '(a) Windowing Strategy', fontweight='bold', fontsize=9, va='center')

# Continuous audio timeline
ax.plot([1, 19], [8.5, 8.5], color='black', linewidth=1.5)
ax.text(10, 8.8, 'Continuous Audio Recording', ha='center', va='bottom')

# Windows (10s width, 5s shift)
# Window 1
ax.add_patch(patches.Rectangle((1, 7.2), 8, 1, edgecolor='black', facecolor='#E0E0E0', hatch='////', linewidth=1))
ax.text(5, 7.7, 'Window 1 (10 s)', ha='center', va='center')

# Window 2
ax.add_patch(patches.Rectangle((5, 5.8), 8, 1, edgecolor='black', facecolor='#C0C0C0', hatch='\\\\\\\\', linewidth=1))
ax.text(9, 6.3, 'Window 2 (10 s)', ha='center', va='center')

# Window 3
ax.add_patch(patches.Rectangle((9, 4.4), 8, 1, edgecolor='black', facecolor='#E0E0E0', hatch='////', linewidth=1))
ax.text(13, 4.9, 'Window 3 (10 s)', ha='center', va='center')

# Shift arrows
ax.annotate('', xy=(5, 8.3), xytext=(1, 8.3), arrowprops=dict(arrowstyle='<->', lw=0.8))
ax.text(3, 8.1, '5 s shift', ha='center', va='top', fontsize=7)


# --- PART (b) Scoring Protocol ---
ax.text(0, 3.2, '(b) Segment-Level Scoring Protocol', fontweight='bold', fontsize=9, va='center')

# Reference (RTTM)
ax.add_patch(patches.Rectangle((3, 1.8), 10, 0.8, edgecolor='black', facecolor='#808080', linewidth=1))
ax.text(8, 2.2, 'Reference Speech (RTTM)', ha='center', va='center', color='white')

# Hypothesis
ax.add_patch(patches.Rectangle((4.5, 0.6), 9, 0.8, edgecolor='black', facecolor='#D0D0D0', linewidth=1))
ax.text(9, 1.0, 'System Hypothesis', ha='center', va='center')

# Collars (0.25s represented visually)
# Start collar
ax.add_patch(patches.Rectangle((2.6, 1.6), 0.8, 1.2, edgecolor='black', facecolor='none', linestyle='--', linewidth=1))
ax.text(3, 3.0, '0.25 s Collar\n(Ignored)', ha='center', va='bottom', fontsize=7, linespacing=1.2)
ax.plot([3, 3], [1.6, 3.0], color='black', linewidth=0.5, linestyle=':')

# End collar
ax.add_patch(patches.Rectangle((12.6, 1.6), 0.8, 1.2, edgecolor='black', facecolor='none', linestyle='--', linewidth=1))
ax.text(13, 3.0, '0.25 s Collar\n(Ignored)', ha='center', va='bottom', fontsize=7, linespacing=1.2)
ax.plot([13, 13], [1.6, 3.0], color='black', linewidth=0.5, linestyle=':')

# Overlap indication
ax.annotate('Overlap Retained', xy=(8, 1.6), xytext=(8, 0.2), 
            arrowprops=dict(arrowstyle='-', lw=0.8, linestyle=':'), ha='center', va='top', fontsize=7)

plt.tight_layout()
plt.savefig('results/paper_figuresfig_protocol_overview.pdf', format='pdf',
            bbox_inches='tight', dpi=300)
plt.show()