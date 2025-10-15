import os
import sys


class PathDict(dict):
    """Custom dictionary that creates directories on demand when output paths are accessed."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_keys = {"output", "output_data", "output_meshes", "output_opt"}
    
    def __getitem__(self, key):
        path = super().__getitem__(key)
        if key in self._output_keys:
            os.makedirs(path, exist_ok=True)
        return path


def get_name():
    import ipynbname
    try:
        name = ipynbname.name()  # Get the notebook name without .ipynb extension
    except Exception:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Fallback to script name if not in a notebook
    return name


def setup_paths(notebook_name, output_root_dir=os.getcwd(), test_name=None):
    # Find the root directory
    current_path = os.path.abspath(os.getcwd())
    split = current_path.split("C-tubes")
    if len(split) < 2:
        raise ValueError("Please rename the repository 'C-tubes'")
    root = os.path.join(split[0], "C-tubes")

    # Determine subdirectory path
    subdir = notebook_name
    if test_name is not None:
        subdir = os.path.join(notebook_name, test_name)
    name = test_name if test_name is not None else notebook_name

    # Set up all 
    paths = PathDict({
        "name": name,
        "data": os.path.join(root, "data"),
        "output": os.path.join(output_root_dir, "output", subdir),
        "output_data": os.path.join(output_root_dir, "output", subdir, "data"),
        "output_meshes": os.path.join(output_root_dir, "output", subdir, "meshes"),
        "output_opt": os.path.join(output_root_dir, "output", subdir, "opt"),
    })

    return paths