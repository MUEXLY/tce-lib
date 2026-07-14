import numpy as np
from ase import build

from tce.calculator import TCECalculator


def main():

    species = ["Si", "Ge"]
    lattice_parameter = 5.5

    rng = np.random.default_rng(seed=0)
    atoms = build.bulk(
        species[0],
        crystalstructure="diamond",
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))
    atoms.symbols = rng.choice(species, p=[0.3, 0.7], size=len(atoms))

    cutoffs = np.array([
        0.25 * np.sqrt(3.0),
        0.5 * np.sqrt(2.0),
        0.25 * np.sqrt(11.0),
        1.0
    ]) * lattice_parameter

    calculator = TCECalculator(
        neighbor_cutoffs=cutoffs,
        many_body_features=[
            (0, 0, 1),
            (1, 1, 1),
            (0, 0, 0, 1, 1, 1)
        ],
        species=species
    )

    feature_vector = calculator.get_feature_vector(atoms)
    print(feature_vector)

    # verify the user-provided cutoffs using cached adjacency tensors
    topological_tensors = calculator.get_topological_tensors(atoms)
    adjacency_tensors = topological_tensors[2]
    neighbor_counts = adjacency_tensors.sum(axis=2).todense()

    for shell_index, counts in enumerate(neighbor_counts):
        print(f"shell {shell_index + 1}: mean={counts.mean():.2f}, std={counts.std():.2f}")


if __name__ == "__main__":

    main()
