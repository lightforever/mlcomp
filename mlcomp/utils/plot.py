import io
from PIL import Image

import numpy as np

from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def figure_to_binary(figure: Figure, **kwargs):
    buf = io.BytesIO()
    figure.savefig(buf, format='jpg', bbox_inches='tight', **kwargs)
    buf.seek(0)
    content = buf.read()
    buf.close()
    return content


def figure_to_img(figure: Figure, **kwargs):
    buf = io.BytesIO()
    figure.savefig(buf, format='jpg', bbox_inches='tight', **kwargs)
    buf.seek(0)
    im = Image.open(buf)
    im.show()
    buf.close()
    return im


def binary_to_img(b: bytes):
    buf = io.BytesIO()
    buf.write(b)
    buf.seek(0)
    im = Image.open(buf)
    im.show()
    return im


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(
            pc.get_paths(),
            pc.get_facecolors(),
            pc.get_array()
    ):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(
        AUC,
        xlabel,
        ylabel,
        xticklabels,
        yticklabels,
        figure_width=40,
        figure_height=20,
        correct_orientation=False,
        cmap='RdBu'
):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(
        AUC,
        edgecolors='k',
        linestyle='dashed',
        linewidths=0.2,
        cmap=cmap
    )

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick1line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick1line.set_visible(False)

    # Add color bar
    fig.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig.set_size_inches(cm2inch(figure_width, figure_height))
    return fig


def plot_classification_report(classification_report, cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2:len(lines)]:
        if 'avg' in line or line.strip() == '' or 'accuracy' in line:
            continue
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1:len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = [
        '{0} ({1})'.format(class_names[idx],
                           sup) for idx,
        sup in enumerate(support)
    ]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    return heatmap(
        np.array(plotMat),
        xlabel,
        ylabel,
        xticklabels,
        yticklabels,
        figure_width,
        figure_height,
        correct_orientation,
        cmap=cmap
    )
