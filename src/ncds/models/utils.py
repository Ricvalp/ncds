import matplotlib.pyplot as plt
import numpy as np


def plot_vector_field(
    vector_field_fn,
    training_traj: np.ndarray = None,
    xlim: tuple = (-2.5, 0.5),
    ylim: tuple = (-0.5, 2.5),
    grid_size: int = 20,
    save_path: str = None,
) -> None:
    """
    Plots a 2D vector field using quiver plot.

    Args:
        vector_field_fn: A function that takes in a (N, 2) array of points and returns
                         a (N, 2) array of vectors.
        xlim: Tuple specifying the x-axis limits.
        ylim: Tuple specifying the y-axis limits.
        grid_size: Number of points along each axis.
    """
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    vectors = vector_field_fn(points)
    U = vectors[:, 0].reshape(X.shape)
    V = vectors[:, 1].reshape(Y.shape)

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V)
    if training_traj is not None:
        for traj in training_traj:
            plt.plot(traj[:, 0], traj[:, 1], "r-", alpha=0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Vector Field")
    plt.grid()
    if save_path:
        plt.savefig(save_path)


def plot_solution_trajectories(
    trajectories: np.ndarray,
    title: str = "Solution Trajectories",
    save_path: str = None,
) -> None:
    """
    Plots solution trajectories in 2D.

    Args:
        trajectories: A (N, T, 2) array where N is the number of trajectories,
                      T is the number of time steps, and 2 corresponds to (x, y).
        title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis("equal")
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
