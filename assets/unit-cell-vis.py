from typing import TypeAlias, Callable, Mapping
from functools import wraps

from ase import Atoms
from tce.constants import LatticeStructure, STRUCTURE_TO_ATOMIC_BASIS
import numpy as np
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import StaticSource, Pipeline
from ovito.vis import Viewport, CoordinateTripodOverlay
from ovito.data import DataCollection


OvitoModifier: TypeAlias = Callable[[int, DataCollection], None]


def change_colors_modifier(
    color_map: Mapping[str, tuple[float, float, float]]
) -> OvitoModifier:

    @wraps(change_colors_modifier)
    def wrapper(frame: int, data: DataCollection) -> None:

        types = data.particles_.particle_types_

        for key, color in color_map.items():
            try:
                types.type_by_name_(key).color = color
            except KeyError:
                continue

    return wrapper


def main():

    lattice_parameter = 10.0

    for lattice_structure in LatticeStructure:

        basis = STRUCTURE_TO_ATOMIC_BASIS[lattice_structure]

        atoms = Atoms(
            "X" * len(basis),
            positions=lattice_parameter * basis,
            cell=lattice_parameter * np.ones(3),
            pbc=True,
        )

        data = ase_to_ovito(atoms)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(change_colors_modifier(color_map={
            "X": (0.32, 0.18, 0.50)
        }))
        pipeline.add_to_scene()
        vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(-2, -1, -1))
        vp.overlays.append(
            CoordinateTripodOverlay(
                axis1_color=(0.0, 0.0, 0.0),
                axis2_color=(0.0, 0.0, 0.0),
                axis3_color=(0.0, 0.0, 0.0),
                axis1_label="(100)",
                axis2_label="(010)",
                axis3_label="(001)",
                offset_x=0.05,
                offset_y=0.05
            )
        )
        vp.zoom_all((600, 600))
        vp.render_image(filename=f"lattice-structures/{lattice_structure.name.lower()}.png", size=(600, 600))
        pipeline.remove_from_scene()


if __name__ == "__main__":

    main()
