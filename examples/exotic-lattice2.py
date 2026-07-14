from ase import build
import numpy as np

from tce.calculator import TCECalculator


def main():

    fluorite_unit_cell = build.bulk("UO2", crystalstructure="fluorite", a=1.0, cubic=True)
    supercell = fluorite_unit_cell.repeat((3, 3, 3))
    distances = np.unique(supercell.get_all_distances(mic=True).flatten())

    unique_tol = []
    for x in distances:
        if not any(np.isclose(x, u, atol=1.0e-3) for u in unique_tol):
            unique_tol.append(x)

    unique_tol = np.sort(unique_tol)[1:]

    lattice_parameter = 5.6
    cutoffs = unique_tol[:3] * lattice_parameter

    atoms = build.bulk(
        "UO2",
        crystalstructure="fluorite",
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))
    cations = ["U", "Th"]
    rng = np.random.default_rng(seed=0)
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "O":
            continue
        atoms[i].symbol = rng.choice(cations)

    species = cations + ["O"]

    calculator = TCECalculator(
        neighbor_cutoffs=cutoffs,
        many_body_features=[
            (0, 0, 1),
            (0, 0, 2)
        ],
        species=species
    )

    feature_vector = calculator.get_feature_vector(atoms)
    print(feature_vector)


if __name__ == "__main__":

    main()
