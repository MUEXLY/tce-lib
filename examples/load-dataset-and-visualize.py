from pathlib import Path
import os
os.environ["OVITO_GUI_MODE"] = "1"
from functools import wraps
from typing import Mapping, Callable, TypeAlias

from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
from ovito.vis import ColorLegendOverlay, Viewport
from ovito.qt_compat import QtCore
from ovito.data import DataCollection
import numpy as np

# from tce.datasets import Dataset, available_datasets
from tce.datasets import Dataset, PresetDataset


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

    # you can print out the available datasets
    for dataset_name in PresetDataset:
        print(dataset_name)

    dataset = Dataset.from_preset(PresetDataset.TUNGSTEN_TANTALUM_GENETIC)
    configurations = dataset.configurations
    
    len_x, len_y = 25, 10
    spacing = 10
    x_min, x_max, x_points = -len_x * spacing, len_x * spacing, len_x
    y_min, y_max, y_points = -len_y * spacing, len_y * spacing, len_y
    z_val = 0.0
    assert len_x * len_y == len(configurations)

    x_vals = np.linspace(x_min, x_max, x_points)
    y_vals = np.linspace(y_min, y_max, y_points)

    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    points = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, z_val)))
    axis = np.array([1, -1, 0])
    axis = axis / np.linalg.norm(axis) * 30 * np.pi / 180
    pipeline = None
    for point, configuration in zip(points, configurations):
        data = ase_to_ovito(configuration)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(change_colors_modifier(color_map={
            "Ta": (0.0, 0.7, 0.7),
            "W": (0.85, 0.65, 0.13)
        }))
        pipeline.add_to_scene(translation=point, rotation=axis)
    if not pipeline:
        raise ValueError

    vp = Viewport()
    legend = ColorLegendOverlay(
        property="particles/Particle Type",
        alignment=QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter,
        orientation=QtCore.Qt.Orientation.Horizontal,
        title=" ",
        pipeline=pipeline
    )
    vp.overlays.append(legend)
    size = (2000, 1000)
    vp.zoom_all(size)
    vp.render_image(filename="visualized.png", size=size)
    


if __name__ == "__main__":

    main()
