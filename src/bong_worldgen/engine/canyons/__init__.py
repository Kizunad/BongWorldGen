"""大峡谷的独立地表生成模块。"""

from .generator import apply_canyons
from .topology import generate_canyon_paths

__all__ = ["apply_canyons", "generate_canyon_paths"]
