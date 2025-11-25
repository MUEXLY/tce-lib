r"""
This module is a convenience wrapper for loading predefined toy datasets that come with an installation of `tce-lib`.
These are useful for largely for educational reasons, i.e., seeing how `tce-lib` expects training data to look.
"""

from dataclasses import dataclass, fields
from pathlib import Path
import json
from importlib.resources import files, as_file
import logging
import warnings

from ase import Atoms, io
from aenum import Enum

from tce.constants import LatticeStructure
from tce.training import get_type_map


LOGGER = logging.getLogger(__name__)


TCE_MODULE_TRAVERSABLE = files("tce")
"""@private"""


class PresetDataset(Enum):

    CANTOR_ALLOY_SURROGATE = "cantor_alloy_surrogate"
    IRON_CHROME_ALUMINUM_SURROGATE = "iron_chrome_aluminum_surrogate"
    NOBLE_HEA_SURROGATE = "noble_hea_surrogate"
    TUNGSTEN_TANTALUM_GENETIC = "tungsten_tantalum_genetic"


@dataclass
class Dataset:

    r"""
    dataset class that can load a pre-defined dataset. to see available datasets:

    ```py
    from tce.datasets import Dataset, available_datasets

    for dataset_name in available_datasets:
        print(dataset)
    ```

    to load a given dataset:

    ```py
    from pathlib import Path

    from tce.datasets import Dataset, available_datasets

    for dataset_name in available_datasets:
        dataset = Dataset.load(Path(dataset_name))
    ```
    
    see [here](https://muexly.github.io/tce-lib/tce.html#loading-and-visualizing-datasets) for a more concrete example
    showing what you can do with the `tce.datasets.Dataset` object!
    """

    lattice_parameter: float
    lattice_structure: LatticeStructure
    description: str
    contact_info: str
    configurations: list[Atoms]

    def __repr__(self):
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "configurations":
                parts.append(f"{f.name}=[...]")
            else:
                parts.append(f"{f.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def from_dir(cls, directory: Path) -> "Dataset":

        warnings.warn(
            f"{cls.__name__}.from_dir is deprecated. Use {cls.__name__}.from_preset instead.",
            DeprecationWarning
        )

        with as_file(TCE_MODULE_TRAVERSABLE) as module_dir:
            dataset_dir = module_dir / "datasets"

            with (dataset_dir / directory / "metadata.json").open("r") as file:
                metadata = json.load(file)

            metadata["lattice_structure"] = getattr(LatticeStructure, metadata["lattice_structure"].upper())
            
            configurations = []
            for path in (dataset_dir / directory).glob("*.xyz"):
                configuration = io.read(path, format="extxyz")
                if isinstance(configuration, list):
                    raise ValueError(f"path {path} contained multiple frames")
                configurations.append(configuration)

        instance = cls(**metadata, configurations=configurations)
        LOGGER.debug(
            f"loaded dataset from {directory} with {len(instance.configurations):.0f} configurations with types "
            f"{', '.join(get_type_map(configurations))}."
        )
        return instance

    @classmethod
    def from_preset(cls, preset: PresetDataset) -> "Dataset":

        directory = Path(preset.value)
        with as_file(TCE_MODULE_TRAVERSABLE) as module_dir:
            dataset_dir = module_dir / "datasets"

            with (dataset_dir / directory / "metadata.json").open("r") as file:
                metadata = json.load(file)

            metadata["lattice_structure"] = getattr(LatticeStructure, metadata["lattice_structure"].upper())

            configurations = []
            for path in (dataset_dir / directory).glob("*.xyz"):
                configuration = io.read(path, format="extxyz")
                if isinstance(configuration, list):
                    raise ValueError(f"path {path} contained multiple frames")
                configurations.append(configuration)

        instance = cls(**metadata, configurations=configurations)
        LOGGER.debug(
            f"loaded dataset from {directory} with {len(instance.configurations):.0f} configurations with types "
            f"{', '.join(get_type_map(configurations))}."
        )
        return instance


def get_available_datasets() -> list[PresetDataset]:

    return list(PresetDataset)


def available_datasets() -> list[str]:

    warnings.warn(
        "available_datasets is deprecated. Please use get_available_datasets instead.",
        DeprecationWarning
    )

    with as_file(TCE_MODULE_TRAVERSABLE) as module_dir:
        dataset_dir = module_dir / "datasets"

        return list(x.name for x in dataset_dir.iterdir())