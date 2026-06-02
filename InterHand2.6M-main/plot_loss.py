import re
import matplotlib.pyplot as plt
import numpy as np

log_path = 'output/log/train_logs.txt'

# Parse log: extract epoch, itr, and three losses
pattern = r'Epoch (\d+)/\d+ itr (\d+)/\d+.*loss_joint_heatmap: ([\d.]+) loss_rel_root_depth: ([\d.]+) loss_hand_type: ([\d.]+)'

data = {'epoch': [], 'itr': [], 'loss_joint_heatmap': [], 'loss_rel_root_depth': [], 'loss_hand_type': []}

with open(log_path, 'r') as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            epoch = int(m.group(1))
            data['epoch'].append(epoch)
            data['itr'].append(int(m.group(2)))
            data['loss_joint_heatmap'].append(float(m.group(3)))
            data['loss_rel_root_depth'].append(float(m.group(4)))
            data['loss_hand_type'].append(float(m.group(5)))

# Convert to numpy arrays
epochs = np.array(data['epoch'])
iters = np.array(data['itr'])
loss_jh = np.array(data['loss_joint_heatmap'])
loss_rd = np.array(data['loss_rel_root_depth'])
loss_ht = np.array(data['loss_hand_type'])

# Compute global step (cumulative iterations)
total_iters = len(loss_jh)
global_step = np.arange(total_iters)

# Find epoch boundaries for vertical lines
epoch_boundaries = []
for i in range(1, total_iters):
    if epochs[i] != epochs[i-1]:
        epoch_boundaries.append(i)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

colors = ['#2ecc71', '#e74c3c', '#3498db']
labels = ['Joint Heatmap Loss (MSE)', 'Rel Root Depth Loss (L1)', 'Hand Type Loss (BCE)']
losses = [loss_jh, loss_rd, loss_ht]
ylabels = ['Loss', 'Loss', 'Loss']

for i, (ax, loss, color, label) in enumerate(zip(axes, losses, colors, labels)):
    # Plot raw loss with low alpha
    ax.plot(global_step, loss, color=color, alpha=0.15, linewidth=0.5)

    # Plot smoothed curve (moving average per 500 iterations)
    window = 500
    if len(loss) > window:
        smoothed = np.convolve(loss, np.ones(window)/window, mode='valid')
        smooth_step = global_step[window//2 : window//2 + len(smoothed)]
        ax.plot(smooth_step, smoothed, color=color, linewidth=2, label=f'{label} (smoothed)')

    # Epoch boundaries
    for b in epoch_boundaries:
        ax.axvline(x=b, color='gray', alpha=0.2, linewidth=0.5)

    # Learning rate decay markers
    for ep, lr_text in [(15, 'lr=1e-5'), (17, 'lr=1e-6')]:
        if ep <= epochs[-1]:
            idx = np.where(epochs == ep)[0][0]
            ax.axvline(x=idx, color='orange', alpha=0.6, linewidth=1.2, linestyle='--')
            if i == 0:
                ax.text(idx + 200, ax.get_ylim()[1]*0.95, lr_text, fontsize=8,
                        color='orange', fontweight='bold')

    ax.set_ylabel(label, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_iters)

# X-axis labels as epoch numbers
epoch_ticks = []
epoch_labels = []
for ep in range(0, epochs[-1] + 1):
    if ep == 0:
        epoch_ticks.append(0)
    else:
        idxs = np.where(epochs == ep)[0]
        if len(idxs) > 0:
            epoch_ticks.append(idxs[0])
    epoch_labels.append(str(ep))

axes[2].set_xticks(epoch_ticks)
axes[2].set_xticklabels(epoch_labels)
axes[2].set_xlabel('Epoch', fontsize=12)

fig.suptitle('Training Loss Curves — InterHand2.6M (InterNet + ResNet-50)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('output/loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Plot saved to output/loss_curves.png')
print(f'Total iterations: {total_iters}')
print(f'Epochs: {epochs[0]} to {epochs[-1]}')

# Print per-epoch average losses
print('\nPer-epoch average losses:')
for ep in range(epochs[-1] + 1):
    mask = epochs == ep
    print(f'  Epoch {ep:2d}: jh={loss_jh[mask].mean():.4f}  rd={loss_rd[mask].mean():.4f}  ht={loss_ht[mask].mean():.4f}')
