"""Execute the tutorial with the current Python, using only project-local Jupyter state."""
from pathlib import Path
import base64
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".jupyter_runtime"
LOCAL_PACKAGES = ROOT / ".notebook_runtime"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(LOCAL_PACKAGES), str(ROOT), os.environ.get("PYTHONPATH", "")]
    )
for variable, directory in {
    "JUPYTER_DATA_DIR": RUNTIME / "share",
    "JUPYTER_RUNTIME_DIR": RUNTIME / "run",
    "JUPYTER_CONFIG_DIR": RUNTIME / "config",
    "IPYTHONDIR": RUNTIME / "ipython",
    "MPLCONFIGDIR": RUNTIME / "matplotlib",
}.items():
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
kernel_dir = RUNTIME / "share/kernels/aqua_align"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Python (aqua_align)", "language": "python",
}), encoding="utf-8")

import nbformat
from nbclient import NotebookClient

path = ROOT / "notebooks/01_degradation_aware_fgdpa_tutorial.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name="aqua_align",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
print(f"Executed {len(code_cells)} code cells: {path}")
figure_dir = ROOT / "outputs/notebook_qa"
figure_dir.mkdir(parents=True, exist_ok=True)
figure_index = 0
for cell in code_cells:
    for output in cell.outputs:
        if output.output_type == "stream":
            print(output.text)
        if "image/png" in output.get("data", {}):
            (figure_dir / f"figure_{figure_index}.png").write_bytes(
                base64.b64decode(output.data["image/png"])
            )
            figure_index += 1
print(f"Saved {figure_index} figures for visual verification: {figure_dir}")
