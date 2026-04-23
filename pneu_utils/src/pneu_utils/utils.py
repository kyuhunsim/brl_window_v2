import sys
import os
import yaml
from pathlib import Path

from typing import Optional, Any, Dict

def get_pkg_path(
    pkg: str
):
    workspace_root = Path(__file__).resolve().parents[3]
    pkg_path = workspace_root / pkg
    if not pkg_path.is_dir():
        raise FileNotFoundError(f"Package path not found: {pkg_path}")
    return str(pkg_path)

def is_dir(folder_path: str) -> bool:
    return os.path.isdir(folder_path)

def checker(
    content: str,
    header: Optional[str] = None
):
    if header is None:
        print(f'[ INFO] checker ==> {content}')
    else:
        print(f'[ INFO] {header} ==> {content}')

def delete_lines(
    num: int
):
    for _ in range(num):
        sys.stdout.write("\033[F")  # Move the cursor up one line
        sys.stdout.write("\033[K")  # Clear the line

def color(
    line: str,
    color: str
):
    if color == 'blue':
        return '\033[94m' + f'{line}' + '\033[0m'
    elif color == 'yellow':
        return '\033[33m' + f'{line}' + '\033[0m'
    elif color == 'red':
        return '\033[91m' + f'{line}' + '\033[0m'
    else:
        return line

def save_yaml(
    folder_name: str,
    kwargs: Dict[str, Any],
    file_name: str = 'cfg.yaml'
):
    model_path = f'{get_pkg_path("pneu_rl")}/models/{folder_name}'
    with open(f'{model_path}/{file_name}', 'w') as f:
        yaml.dump(kwargs, f)

def load_yaml(
    folder_name: str
) -> Dict[str, Any]:
    model_path = f'{get_pkg_path("pneu_rl")}/models/{folder_name}'
    with open(f'{model_path}/cfg.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    return data
    
def save_kwargs(
    path: str,
    kwargs: Dict[str, Any]
) -> None:
    with open(path, "w") as f:
        yaml.dump(kwargs, f)

def setup_plot_style(overrides: Optional[Dict[str, Any]] = None) -> None:
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "figure.titlesize": 22,
    }
    if overrides:
        style.update(overrides)

    import matplotlib.pyplot as plt

    plt.rcParams.update(style)
