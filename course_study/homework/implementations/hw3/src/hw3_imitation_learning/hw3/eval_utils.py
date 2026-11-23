from ethz_starter.hw3_imitation_learning.hw3.eval_utils import *
from hw3.dataset import target_offset

# The course evaluator and its inference helpers share this feature registry.
ZARR_KEY_TO_OBS.update(
    {
        "state_target_offset": lambda obs: target_offset(
            obs["cubes_xyz"].reshape(3, 3), obs["goal"], obs["ee_pos"]
        ),
        "state_bin_offset": lambda obs: obs["goal_pos"][:3] - obs["ee_pos"],
    }
)
