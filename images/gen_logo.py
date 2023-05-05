import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image
import numpy as np

# This sets how many boxes to draw in x, y, z
axes = [5, 5, 5]
data = np.ones(axes, dtype=bool)

# Fix the seed for a reproducible image
np.random.seed(1165)

# JAX colors (now they are not precisely the JAX ones)
colors = ["#ffffff", "#5e97f6", "#3367d6", "#26a69a", "#00695c", "#ea80fc", "#6a1b9a"]

# Probability that a square is white
p_white = 0.3

# The remaining probabilities are equiprobable
p = np.full(len(colors[1:]), (1.0 - p_white) / len(colors[1:]))
p = np.concatenate([[p_white], p])

# Convert to RGBA because `voxels` pretends so
colors = np.array([to_rgba(c.upper()) for c in colors])

# Sample the colors with the chosen probability
indices = np.random.choice(np.arange(len(colors)), size=data.shape, p=p)
colors = colors[indices]


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection="3d")
ax.axis("off")
ax.view_init(elev=30, azim=-45 - 90)

ax.voxels(data, facecolors=colors, edgecolors="lightgrey")

fig.savefig("moldex_logo.svg", bbox_inches="tight", transparent=True)
fig.savefig("moldex_logo.png", bbox_inches="tight", transparent=True, dpi=300)


img = Image.open("moldex_logo.png")
logo_size = img.size
img = img.resize((250, 250), Image.Resampling.LANCZOS)
img.save("moldex_logo_250px.png")
