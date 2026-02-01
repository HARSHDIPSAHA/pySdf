import importlib.util
import sys
from pathlib import Path


def load_module(module_name, rel_path):
    base = Path(__file__).resolve().parent.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    target = base / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {target}")
    spec.loader.exec_module(module)
    return module
