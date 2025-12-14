from pathlib import Path
from typing import List
import numpy as np
import scipy.io

def load_lasa_shape(path: str, normalize: bool = False) -> list:
    """
    Returns a list of trajectories.
    Each trajectory has shape (T, 2).
    """
    data = scipy.io.loadmat(path)
    
    try:
        dt = float(data["dt"][0, 0])
    except Exception:
        print("dt field not found in the .mat file; defaulting to dt=0.01")
        dt = 0.01

    demos: List[np.ndarray] = []
    shape_struct = data.get("demos")
    if shape_struct is None:
        raise ValueError(f"'demos' field not found in {path}")

    # `demos` is a 1 x N cell array; each entry has a `pos` field shaped (2, T).
    for i in range(shape_struct.shape[1]):
        demo = shape_struct[0, i]
        pos = demo["pos"][0, 0]  # unwrap the nested cell to a (2, T) ndarray
        demos.append(pos.T)  # return as (T, 2)
    demos = np.stack(demos)
    
    if normalize:
        std = np.std(demos)
        demos = demos / std
        
    return {'trajectories': demos, 'dt': dt}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Resolve paths relative to repository root so the script can be run from anywhere.
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / "dataset" / "lasahandwritingdataset" / "DataSet"
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    shapes_to_plot = ["CShape", "Sine", "WShape"]

    for shape in shapes_to_plot:
        mat_path = dataset_dir / f"{shape}.mat"
        demos_dict = load_lasa_shape(str(mat_path), normalize=True)
        demos = demos_dict['trajectories']
        dt = demos_dict['dt']
        
        fig, ax = plt.subplots()
        for demo in demos:
            ax.plot(demo[:, 0], demo[:, 1], linewidth=1.5)

        ax.set_title(f"{shape} demos")
        ax.set_aspect("equal", "box")
        ax.invert_yaxis()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        output_path = figures_dir / f"lasa_{shape.lower()}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    
