# For dataframes and plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle

# For visuals
from IPython.display import HTML, display

fig_num = 1
table_num = 1

def truncated_svd(X):
    q, s, p = np.linalg.svd(X)
    nssd = (s / np.linalg.norm(s))**2
    s = np.diag(s)
    return q, s, p.T, nssd


def project_svd(q, s, k):
    return q[:, :k] @ s[:k, :k]


def plot_svd(X_new, features, p):
    """
    Plot transformed data and features on to the first two singular vectors

    Parameters
    ----------
    X_new : array
        Transformed data
    featurs : sequence of str
        Feature names
    p : array
        P matrix
    """
    fig, ax = plt.subplots(1, 2,
                           gridspec_kw=dict(wspace=0.4), dpi=300)
    ax[0].scatter(X_new[:, 0], X_new[:, 1])
    ax[0].set_xlabel('SV1')
    ax[0].set_ylabel('SV2')

    for feature, vec in zip(features, p):
        ax[1].arrow(0, 0, 2*vec[0], 2*vec[1], width=0.01, ec='none', fc='r')
        ax[1].text(2.2*vec[0], 2.2*vec[1], feature,
                   ha='center', color='r', fontsize=5)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('SV1')
    ax[1].set_ylabel('SV2')


def pca(X):
    """Return the transformed PCA matrix, vectors, and variance explained."""
    X_bar = np.mean(X, axis=0)
    X = X - X_bar
    V = (X.T @ X) / len(X)

    eigenvals, w = np.linalg.eig(V)

    X_new = X @ w
    variance_explained = eigenvals / eigenvals.sum()
    sorter = variance_explained.argsort()[::-1]

    variance_explained = variance_explained[sorter]
    w = w[:, sorter]
    X_new = X_new[:, sorter]

    return X_new, w, variance_explained


def plot_theme(sv, p, nssd, features, title='SV', left_paintings=[],
               right_paintings=[], highlights=[]):
    """Plot the top features for an SV with sample images."""
    fig, ax = plt.subplots(figsize=(20, 5), dpi=300)
    order = np.argsort(np.abs(p[:, sv]))[-10:]
    f = [features[o] for o in order]
    ax.barh(f, p[order, sv], color='xkcd:dark cyan')

    for lp, y in zip(left_paintings, [8.20, 4.50, 0.80]):
        imagebox = OffsetImage(image.imread(f'paintings/SV_{sv+1}/negative/{lp}.webp'),
                               zoom=0.25)
        ab = AnnotationBbox(imagebox, (-0.30, y), frameon=False, zorder=0)
        ax.add_artist(ab)

    for rp, y in zip(right_paintings, [8.20, 4.50, 0.80]):
        imagebox = OffsetImage(image.imread(f'paintings/SV_{sv+1}/positive/{rp}.webp'),
                               zoom=0.25)
        ab = AnnotationBbox(imagebox, (0.30, y), frameon=False, zorder=0)
        ax.add_artist(ab)

    for n, v in enumerate(p[order, sv]):
        if v > 0:
            ax.text(v, n-0.1, f[n]+'   ', color='#F8FFDE', fontweight='bold',
                    ha='right')
            ax.text(0.04, n-0.1, '  '+str(v)[0:5], color='#F8FFDE',
                    ha='right')
        if v < 0:
            ax.text(v, n-0.1, '   '+f[n], color='#F8FFDE', fontweight='bold',
                    ha='left')
            ax.text(-0.04, n-0.1, str(v)[0:5]+'  ', color='#F8FFDE',
                    ha='left')

    ax.set_xlim([-0.6, 0.6])
    ax.set_yticks([])
    ax.axvline(0, linestyle='--', color='green')
    ax.set_title(f'THEME {sv+1}: {title} ({nssd[sv]*100:.2f}%)',
                 fontweight='bold')
    plt.axis('off')

    for h in highlights:
        pos = f.index(h)
        ax.patches[pos].set_facecolor('#aa3333')

    plt.show()


def caption(text, subtext, fig=True):
    """Print a caption header and subtitle for a table or figure in HTML."""
    global fig_num
    global table_num
    if fig:
        cap = display(HTML(f'<br><span style="font-size:16px;color=#0a3b05">\
                    <b>Figure   {fig_num}. {text}</b></span><br>{subtext}'))
        fig_num += 1
    else:
        cap = display(HTML(
            f'<center><br><span style="font-size:16px;#0a3b05">\
            <b>Table   {table_num}. {text}</b></span><br>{subtext}</center>'))
        table_num += 1
    return cap


def show_df(df):
    show = display(HTML(f'<center>{df.to_html()}</center>'))
    return show


def plot_annot_svd(X_new, features, p):
    """
    Plot transformed data and features on to the first two singular vectors, with annotation.
    """
    fig, ax = plt.subplots(1, 2,  
                           gridspec_kw=dict(wspace=0.4), dpi=300)
    ax[0].scatter(X_new[:,0], X_new[:,1])
    width = round((max(X_new[:,0])- min(X_new[:,0]))/2,0)
    height = round((max(X_new[:,1])-min(X_new[:,1]))/2,0)
    ax[0].add_patch(Rectangle((-5,-1), width, 3, facecolor='blue', 
                              edgecolor=None, alpha=0.3))
    ax[0].add_patch(Rectangle((-5,-3), width, 2, facecolor='gray', 
                              edgecolor=None, alpha=0.3))
    ax[0].set_xlabel('Barrenness Range')
    ax[0].set_ylabel('Color Contrast Range')
    ax[0].text(-3, 1.5, ' HIGH CONTRAST', size=7, ha='left')
    ax[0].text(-3, 1.1, ' Range of high\n contrast colors', 
               size=5, ha='left')
    ax[0].text(-3, -2.5, ' LOW BARRENNESS', size=7, ha='left')
    ax[0].text(-3, -2.9, ' Preference for\n lush paintings', 
               size=5, ha='left')

    for feature, vec in zip(features, p):
        ax[1].arrow(0, 0, 2*vec[0], 2*vec[1], width=0.01, ec='none', fc='r')
        ax[1].text(2.2*vec[0], 2.2*vec[1], feature, ha='center', color='r', 
                   fontsize=5)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('Barrenness Range')
    ax[1].set_ylabel('Color Contrast Range')