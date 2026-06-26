from ase import build
import numpy as np

from tce.calculator import TCECalculator
from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS


def main():

    # define the lattice and species
    structure = LatticeStructure.BCC
    lattice_parameter = 2.9
    size = (5, 5, 5)
    species = ["Fe", "Cr"]

    rng = np.random.default_rng(seed=0)
    cutoffs = lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[structure][:2]

    calculator = TCECalculator(
        neighbor_cutoffs=cutoffs,
        many_body_features=[(0, 0, 1)],
        species=species
    )

    ase_supercell = build.bulk(
        species[0],
        crystalstructure=structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat(size)
    ase_supercell.symbols = rng.choice(species, p=[0.93, 0.07], size=len(ase_supercell))

    feature_vector = calculator.get_feature_vector(ase_supercell)
    print(feature_vector)


if __name__ == "__main__":

    main()
