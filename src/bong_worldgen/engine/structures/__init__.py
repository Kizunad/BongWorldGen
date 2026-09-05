"""可导入的 Minecraft 结构资源。"""

from .catalog import (
    STRUCTURE_NOTES,
    StructureNote,
    structure_anchor_y,
    structure_block_kind,
    structure_note,
)
from .schematic import (
    SchematicBlock,
    SchematicStructure,
    load_litematic,
    load_schematic,
    load_schematic_directory,
    load_structure_directory,
)
from .interiors import schematic_interior_spawn

__all__ = [
    "SchematicBlock",
    "SchematicStructure",
    "STRUCTURE_NOTES",
    "StructureNote",
    "load_litematic",
    "load_schematic",
    "load_schematic_directory",
    "load_structure_directory",
    "structure_anchor_y",
    "structure_block_kind",
    "structure_note",
    "schematic_interior_spawn",
]
