import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

fopng_data = np.array([
    [1.000, 0.458, 0.524, 0.482, 0.509],  # after task 1
    [1.000, 0.976, 0.524, 0.482, 0.509],  # after task 2
    [1.000, 0.973, 0.988, 0.482, 0.510],  # after task 3
    [1.000, 0.970, 0.989, 0.994, 0.513],  # after task 4
    [1.000, 0.973, 0.928, 0.994, 0.974],  # after task 5
])

ewc_data = np.array([
    [1.000, 0.505, 0.388, 0.518, 0.508],  # after task 1
    [0.161, 0.988, 0.430, 0.518, 0.493],  # after task 2
    [0.196, 0.652, 0.998, 0.564, 0.492],  # after task 3
    [0.069, 0.861, 0.476, 0.995, 0.495],  # after task 4
    [0.517, 0.563, 0.476, 0.975, 0.973],  # after task 5
])

adam_data = np.array([
    [0.999, 0.491, 0.476, 0.518, 0.550],  # after task 1
    [1.000, 0.994, 0.476, 0.482, 0.603],  # after task 2
    [1.000, 0.693, 0.998, 0.482, 0.725],  # after task 3
    [0.997, 0.475, 0.901, 0.999, 0.509],  # after task 4
    [0.997, 0.500, 0.965, 0.470, 0.983],  # after task 5
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

plt.savefig('split_mnist_hyper.pdf', bbox_inches='tight', facecolor=BG)
plt.show()