import numpy as np
from ase import build

from tce.constants import (
    LatticeStructure,
    register_new_lattice_structure,
    ClusterBasis,
    STRUCTURE_TO_CUTOFF_LISTS
)
from tce.topology import topological_feature_vector_factory


def main():

    register_new_lattice_structure(
        name="DIAMOND",
        atomic_basis=np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25]
        ]),
        cutoff_list=np.array([
            0.25 * np.sqrt(3.0), 0.5 * np.sqrt(2.0), 0.25 * np.sqrt(11.0), 1.0
        ])
    )

    species = np.array(["Si", "Ge"])
    lattice_parameter = 5.5

    feature_vector_computer = topological_feature_vector_factory(
        basis=ClusterBasis(
            LatticeStructure.DIAMOND,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=2,
            max_triplet_order=1
        ),
        type_map=species
    )

    rng = np.random.default_rng(seed=0)
    atoms = build.bulk(
        species[0],
        crystalstructure="diamond",
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))
    atoms.symbols = rng.choice(species, p=[0.3, 0.7], size=len(atoms))

    feature_vector = feature_vector_computer(atoms)
    print(feature_vector)

    # check the number of nearest neighbors
    # for cubic diamond, we should see 4th 1st nearest, 12 2nd nearest, 12 3rd nearest, and 6th 4th nearest
    # we should also see that there's no dispersity in the nearest neighbor counts
    distances = atoms.get_all_distances(mic=True)
    cutoffs = lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.DIAMOND]

    tol = 1.0e-3
    for i, cutoff in enumerate(cutoffs):
        num_neighbors = np.logical_and(
            (1.0 - tol) * cutoff < distances, distances < (1.0 + tol) * cutoff
        ).sum(axis=0)
        print(num_neighbors.mean(), num_neighbors.std())


if __name__ == "__main__":

    main()
