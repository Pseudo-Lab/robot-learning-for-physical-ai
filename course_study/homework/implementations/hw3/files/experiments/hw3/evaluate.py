"""Evaluate the saved study policy with the official HW3 harness from the starter."""
import argparse
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
course = here.parents[1] / "hw3_imitation_learning"
sys.path[:0] = [str(course / "student_eval"), str(course)]
import eval_harness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", type=int, choices=(1, 2, 3))
    parser.add_argument("output", type=Path, help="new .hwresult path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    model = "multi_cube.py" if args.exercise == 3 else "single_cube.py"
    eval_harness.run_eval(exercise=args.exercise, model_py=str(here / model),
                          checkpoint=str(here / f"ex{args.exercise}.pt"),
                          output_path=str(args.output.resolve()), num_episodes=100,
                          seed=42, verbose=True)
