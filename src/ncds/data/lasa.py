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
