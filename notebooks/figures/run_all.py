"""
run_all.py

Simple runner to convert every notebook in this folder to a python script and execute it.

Usage (from this folder):

    python run_all.py

Options:
    --maxiter N        : replace occurrences of 'maxiter' in notebook scripts (useful for quick tests)
    --continue         : continue running remaining notebooks if one fails (default: True)
    --notebooks PATTERN: glob pattern to select notebooks (default: "*.ipynb")

Outputs:
    - generated scripts: notebooks/figures/scripts/*.py
    - stdout logs:       notebooks/figures/output/_stdout/*.out

"""

from pathlib import Path
import subprocess
import re
import argparse
import time
import sys
import os

NOTEBOOK_DIR = Path(__file__).parent
SCRIPTS_DIR = NOTEBOOK_DIR / "scripts"
OUTPUT_DIR = NOTEBOOK_DIR / "output"
STDOUT_DIR = OUTPUT_DIR / "_stdout"

DEFAULT_REPLACEMENTS = {
    r"get_ipython\(\)\.run_line_magic\(.*?\)": "",  # remove IPython magics
    r"import matplotlib\.pyplot as plt": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    r"plt\.show\(\)": "# plt.show()",
}

# ANSI colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'


def convert_notebook(nb_path: Path) -> Path:
    """Convert a notebook to a .py script using nbconvert. Returns the path to the generated .py file."""
    print(f"Converting {nb_path} -> script")
    subprocess.run(["jupyter", "nbconvert", "--to", "script", str(nb_path)], check=True)
    py_path = nb_path.with_suffix('.py')
    return py_path


def sanitize_script(py_path: Path, out_path: Path, replacements: dict, maxiter: int | None = None):
    content = py_path.read_text(encoding='utf-8')
    # apply replacements
    for pat, rep in replacements.items():
        content = re.sub(pat, rep, content)
    # optional: tweak maxiter to a small value for fast runs
    if maxiter is not None:
        content = re.sub(r"('maxiter'\s*:\s*)\d+", lambda m: f"{m.group(1)}{maxiter}", content)
        content = re.sub(r"(\"maxiter\"\s*:\s*)\d+", lambda m: f"{m.group(1)}{maxiter}", content)
    out_path.write_text(content, encoding='utf-8')


def run_script(script_path: Path, stdout_path: Path, python_exe: str):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {script_path} -> {stdout_path}")
    with stdout_path.open('w', encoding='utf-8') as out_f:
        proc = subprocess.run([python_exe, str(script_path)], stdout=out_f, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Convert & run figure notebooks")
    parser.add_argument('--maxiter', type=int, default=None,
                        help='Optional: replace maxiter values in scripts (useful to shorten runs)')
    parser.add_argument('--continue', dest='cont', action='store_true', default=True,
                        help='Continue if a notebook script fails (default: continue)')
    parser.add_argument('--stop-on-fail', dest='cont', action='store_false', default=True,
                        help='Stop if a notebook script fails')
    parser.add_argument('--notebooks', type=str, default='*.ipynb',
                        help='Glob pattern to select notebooks in this folder')
    args = parser.parse_args()

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_DIR.mkdir(parents=True, exist_ok=True)

    nb_paths = sorted(NOTEBOOK_DIR.glob(args.notebooks))
    if not nb_paths:
        print("No notebooks found.")
        return

    python_exe = sys.executable

    total_time = 0.0
    results = []

    for nb in nb_paths:
        if nb.name.startswith('.') or nb.name == Path(__file__).name:
            continue
        print('\n' + '='*72)
        print(f"Notebook: {nb.name}")
        print('='*72)

        try:
            py_generated = convert_notebook(nb)
            out_script = SCRIPTS_DIR / py_generated.name
            sanitize_script(py_generated, out_script, DEFAULT_REPLACEMENTS, maxiter=args.maxiter)
            # remove intermediate .py in notebook folder to avoid clutter
            try:
                py_generated.unlink()
            except Exception:
                pass

            stdout_file = STDOUT_DIR / (nb.stem + '.out')
            start = time.time()
            rc = run_script(out_script, stdout_file, python_exe)
            elapsed = time.time() - start
            total_time += elapsed
            results.append((nb.name, rc, elapsed, str(stdout_file)))
            if rc == 0:
                print(f"{GREEN}SUCCESS{RESET}")
                print(f"{nb.name} outputs saved in: {stdout_file}")
            else:
                print(f"{RED}FAILED{RESET}")
                print(f"{nb.name} (rc={rc}) log saved in: {stdout_file}")
            print(f"Time: {elapsed:.2f}s, returncode={rc}")
            if rc != 0 and not args.cont:
                print("Stopping on failure as requested.")
                break

        except subprocess.CalledProcessError as e:
            print(f"{RED}ERROR while processing {nb.name}:{RESET}", e)
            results.append((nb.name, -1, 0.0, ''))
            if not args.cont:
                break

    # summary
    print('\n' + '='*72)
    print("Run summary:")
    for name, rc, t, log in results:
        if rc == 0:
            status = f"{GREEN}OK{RESET}"
            print(f"{name:30} {status:12} time={t:.2f}s  outputs={log}")
        else:
            status = f"{RED}FAIL(rc={rc}){RESET}"
            print(f"{name:30} {status:12} time={t:.2f}s  log={log}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    main()
