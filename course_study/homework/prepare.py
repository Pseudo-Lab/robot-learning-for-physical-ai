"""Build an executable study copy from the pinned course checkout and the study edits."""
import argparse
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PIN = "2e089e97d84968182c1ffa2e1bc8177530feb6fe"
ROOT = Path(__file__).resolve().parent
PACKAGES = ("hw1_pytorch_tutorial", "hw2_robot_control_mdps",
            "hw3_imitation_learning", "hw4_reinforcement_learning")
CELL = re.compile(r"^# %% study-cell ([0-9]+)\n", re.MULTILINE)


def edit(source, delta):
    """Apply diff -n instructions. Line numbers always refer to the original file."""
    original = source.splitlines(keepends=True)
    commands = iter(delta.splitlines(keepends=True))
    result = []
    cursor = 0
    for command in commands:
        match = re.fullmatch(r"([ad])(\d+) (\d+)\n", command)
        if match is None:
            raise ValueError(f"Invalid edit: {command!r}")
        operation, line, length = match.groups()
        start = int(line) - (operation == "d")
        count = int(length)
        if not cursor <= start <= len(original) or count <= 0:
            raise ValueError(f"Invalid line range: {command!r}")
        result.extend(original[cursor:start])
        cursor = start
        if operation == "d":
            cursor += count
            if cursor > len(original):
                raise ValueError("Deletion extends past the source")
        else:
            for _ in range(count):
                result.append(next(commands))
    return "".join(result + original[cursor:])


def cell_text(notebook):
    return "".join(f"# %% study-cell {i}\n" + "".join(cell["source"]).rstrip("\n") + "\n"
                   for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code")


def apply_delta(path, delta):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        blocks = CELL.split(edit(cell_text(notebook), delta))
        indexes = [i for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
        if blocks[0] or list(map(int, blocks[1::2])) != indexes:
            raise ValueError(f"Unexpected notebook cell layout: {path.name}")
        for index, block in zip(indexes, blocks[2::2], strict=True):
            notebook["cells"][index].update(source=block.rstrip("\n").splitlines(keepends=True),
                                            outputs=[], execution_count=None)
        text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    else:
        text = edit(text, delta)
    path.write_text(text, encoding="utf-8")


def prepare(starter, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    archive = subprocess.check_output(["git", "-C", str(starter), "archive", PIN, "--", *PACKAGES])
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tree = Path(tmp) / "course"
        tree.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as src:
            src.extractall(tree, filter="data")
        for number in range(1, 5):
            lesson = ROOT / "implementations" / f"hw{number}"
            for delta in sorted((lesson / "changes").rglob("*.rcs")):
                target = tree / delta.relative_to(lesson / "changes").with_suffix("")
                apply_delta(target, delta.read_text(encoding="utf-8"))
            shutil.copytree(lesson / "files", tree, dirs_exist_ok=True)
        (tree / "tests").mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "conftest.py", tree / "tests/conftest.py")
        tree.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("starter", type=Path, help="Git checkout containing the official course commit")
    parser.add_argument("output", type=Path, help="new directory under an existing parent")
    args = parser.parse_args()
    prepare(args.starter.resolve(), args.output.absolute())
