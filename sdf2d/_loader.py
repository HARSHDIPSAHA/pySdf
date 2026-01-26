"""Dynamic module loader for sdf2d package."""
import importlib.util
from pathlib import Path


def load_module(module_name, rel_path):
    """
    Load a module from a relative path outside the package.
    
    Args:
        module_name: Name to give the loaded module
        rel_path: Relative path from project root (e.g., "2d/geometry_2d.py")
    
    Returns:
        Loaded module object
    """
    base = Path(__file__).resolve().parent.parent
    target = base / rel_path
    
    if not target.exists():
        raise ImportError(f"Cannot find module file: {target}")
    
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {target}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module
