import importlib.util
from pathlib import Path


def load_module(module_name, rel_path):
    base = Path(__file__).resolve().parent.parent
    target = base / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {target}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
