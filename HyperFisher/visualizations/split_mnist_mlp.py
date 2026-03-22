import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

fopng_data = np.array([
    [0.999, 0.505, 0.476, 0.518, 0.517],
    [0.999, 0.968, 0.480, 0.480, 0.683],
    [0.999, 0.953, 0.994, 0.408, 0.844],
    [0.999, 0.937, 0.986, 0.996, 0.796],
    [0.999, 0.859, 0.988, 0.978, 0.974]
])
ewc_data = np.array([
    [0.999, 0.495, 0.476, 0.439, 0.509],
    [0.357, 0.990, 0.476, 0.358, 0.547],
    [0.462, 0.751, 0.999, 0.517, 0.509],
    [0.463, 0.785, 0.892, 0.998, 0.509],
    [0.454, 0.910, 0.956, 0.979, 0.989]
])
adam_data = np.array([
    [1.000, 0.495, 0.479, 0.484, 0.331],
    [0.193, 0.995, 0.390, 0.517, 0.468],
    [0.463, 0.270, 0.996, 0.625, 0.509],
    [0.785, 0.326, 0.636, 0.997, 0.509],
    [0.568, 0.687, 0.534, 0.798, 0.984]
])

tasks_x = np.array([1, 2, 3, 4, 5], dtype=float)

BG       = '#FFFFFF'
DARK_BG  = "#0D1117"
PANEL_BG = '#F6F8FA'
GRID_CLR = '#D0D7DE'
TEXT_CLR = '#1F2328'
SUBTEXT  = '#57606A'
GREEN    = '#1A7F37'

TASK_COLORS = ['#CF222E', '#9A6700', '#1A7F37', '#0550AE', '#8250DF']

METHODS = [
    ('Baseline (Adam)', adam_data,  '#57606A', '^'),
    ('EWC',             ewc_data,   '#9A6700', 's'),
    ('FOPNG (Ours)',    fopng_data, '#0550AE', 'o'),
]

rcParams.update({'font.family': 'DejaVu Sans'})

MIN_Y_GAP = 0.045
X_STEP    = 0.18

def place_labels(final_vals):
    placed = []
    for y_actual, t_idx in final_vals:
        x_off = 0.0
        collision = True
        while collision:
            collision = False
            for px, py, _ in placed:
                if abs(px - x_off) < X_STEP * 0.9 and abs(py - y_actual) < MIN_Y_GAP:
                    x_off -= X_STEP
                    collision = True
                    break
        placed.append((x_off, y_actual, t_idx))
    return placed

fig, axes = plt.subplots(3, 1, figsize=(12, 12), facecolor=BG, sharex=True)
fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.07, hspace=0.12)

for row, (ax, (label, data, m_col, marker)) in enumerate(zip(axes, METHODS)):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(m_col); sp.set_linewidth(1.2)
    ax.tick_params(colors=SUBTEXT, labelsize=10, length=3)
    ax.set_xlim(0.6, 5.4)
    ax.set_ylim(-0.03, 1.09)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylabel('Accuracy', fontsize=10, color=SUBTEXT, labelpad=6)
    ax.grid(True, axis='y', color=GRID_CLR, lw=0.7, zorder=0)

    for t_idx in range(5):
        col = TASK_COLORS[t_idx]

        pre_xs = tasks_x[:t_idx + 1]
        pre_ys = data[:t_idx + 1, t_idx]
        if len(pre_xs) > 1:
            ax.plot(pre_xs, pre_ys, color=col, lw=1.3, alpha=0.3, ls='--', zorder=2)

        post_xs = tasks_x[t_idx:]
        post_ys = data[t_idx:, t_idx]
        ax.plot(post_xs, post_ys, color=col, lw=2.2, alpha=0.95,
                ls=':', marker=marker, ms=6.5, zorder=3, solid_capstyle='round')

    ax.text(0.012, 0.15, label, transform=ax.transAxes,
            fontsize=12, fontweight='heavy', color=m_col, va='top')

    final_vals = sorted([(data[4, t], t) for t in range(5)], reverse=True)
    placed = place_labels(final_vals)

    for x_off, y_actual, t_idx in placed:
        col = TASK_COLORS[t_idx]
        lx  = 5.0 + x_off

        ax.text(lx - 0.04, y_actual, f'{y_actual:.0%}',
                fontsize=8, color=col, va='center', ha='right',
                fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=PANEL_BG,
                          edgecolor='none', alpha=0.8))
        
    # Task legend inside FOPNG panel (last row)
    if row == 2:
        task_handles = [
            mpatches.Patch(facecolor=TASK_COLORS[i], label=f'Task {i+1}')
            for i in range(5)
        ]
        ax.legend(handles=task_handles, title='Task (color)',
                  loc='lower right',
                  fontsize=9, title_fontsize=9,
                  framealpha=0.8, edgecolor=GRID_CLR,
                  labelcolor=TEXT_CLR, facecolor='#D0D7DE',
                  ncol=5)

axes[-1].set_xticks(tasks_x)
axes[-1].set_xticklabels([f'After task {int(t)}' for t in tasks_x],
                          fontsize=10, color=SUBTEXT)

task_handles = [
    mpatches.Patch(facecolor=TASK_COLORS[i], label=f'Task {i+1}')
    for i in range(5)
]


fig.text(0.08, 0.965,
         'Split-MNIST | MLP | Per-task accuracy trajectories — method comparison',
         fontsize=14, fontweight='bold', color=TEXT_CLR, va='top')
fig.text(0.08, 0.945,
         'dashed = before task introduced  ·  dotted = after introduction',
         fontsize=9, color=SUBTEXT, va='top')

plt.savefig('split_mnist_mlp.pdf', bbox_inches='tight', facecolor=BG)
plt.show()