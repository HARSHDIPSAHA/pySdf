"""sdf3d.examples — example 3D geometry assemblies.

Implemented assemblies
----------------------
:func:`NATOFragment`
    NATO STANAG-4496 fragment: cylinder with ogive nose cone.

:func:`RocketAssembly`
    Parametric rocket with body, nose cone, and fins.
"""

from .nato_stanag import NATOFragment
from .rocket_assembly import RocketAssembly

__all__ = ["NATOFragment", "RocketAssembly"]
