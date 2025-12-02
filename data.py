import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# True 2D function f(x, y)
# -----------------------------

def f_true_xy(X):
    """
    True synthetic surface f(x, y).

    Parameters
    ----------
    X : array, shape (N, 2)
        Each row is [x, y] with x,y in [0, 1].

    Returns
    -------
    f : array, shape (N,)
        Function values f(x, y).
    """
    x = X[:, 0]
    y = X[:, 1]

    f = np.empty_like(x)
    
    ### Regions ###
    
    # Region A: Smooth, low frequency
    mask_a = (x < 0.5) & (y >= 0.5)
    # Region B: Higher frequency in x coordinate
    mask_b = (x >= 0.5) & (y >= 0.5)
    # Region C: Mixture of patterns 
    mask_c = y < 0.5
    
    # Filling each region of the grid 
    # A: 0.7 sin(2πx) + 0.7 sin(2πy)
    f[mask_a] = (
        0.7 * np.sin(2 * np.pi * x[mask_a])
        + 0.7 * np.sin(2 * np.pi * y[mask_a])
    )

    # B: 0.7 sin(8πx) + 0.7 sin(2πy)
    f[mask_b] = (
        0.7 * np.sin(8 * np.pi * x[mask_b])
        + 0.7 * np.sin(2 * np.pi * y[mask_b])
    )

    # C: 0.7 sin(2πx) + 0.4 sin(10πy)
    f[mask_c] = (
        0.7 * np.sin(2 * np.pi * x[mask_c])
        + 0.4 * np.sin(10 * np.pi * y[mask_c])
    )

    return f

# ----------------------------------------
# Grid utilities for evaluation/plots
# ----------------------------------------

def make_grid_2d(n_per_axis=50):
    """
    Create a regular 2D grid over [0,1] x [0,1].

    Returns
    -------
    xs : array, shape (n_per_axis,)
        1D grid along x and y (same).
    X_mesh, Y_mesh : arrays, shape (n_per_axis, n_per_axis)
        Meshgrid for plotting.
    X_flat : array, shape (n_per_axis**2, 2)
        Flattened coordinates, each row = [x, y].
    """
    xs = np.linspace(0.0, 1.0, n_per_axis)
    X_mesh, Y_mesh = np.meshgrid(xs, xs, indexing="xy")
    X_flat = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
    return xs, X_mesh, Y_mesh, X_flat

def eval_true_on_grid(n_per_axis=50):
    """
    Evaluate f_true_xy on a regular grid.

    Returns
    -------
    xs, X_mesh, Y_mesh : grid coordinates
    Z : array, shape (n_per_axis, n_per_axis)
        Function values on the grid for plotting.
    """
    xs, X_mesh, Y_mesh, X_flat = make_grid_2d(n_per_axis)
    Z_flat = f_true_xy(X_flat)
    Z = Z_flat.reshape(n_per_axis, n_per_axis)
    return xs, X_mesh, Y_mesh, Z


# -----------------------------
# visualization
# Please call this function in the notebook to visualize the code above
# -----------------------------

def plot_true_terrain(n_per_axis=50, cmap="viridis"):
    xs, X_mesh, Y_mesh, Z = eval_true_on_grid(n_per_axis)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        Z,
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="equal",
        cmap=cmap,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("True synthetic terrain f(x, y)")
    fig.colorbar(im, ax=ax, label="f(x, y)")

    # Optional: show region boundaries
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()