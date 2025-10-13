import numpy as np
import matplotlib.pyplot as plt
from ase import build
from scipy.spatial import KDTree

from tce.constants import STRUCTURE_TO_CUTOFF_LISTS, LatticeStructure
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector


def main():
    vertical_lengths = np.arange(3, 9)
    num_atoms = np.zeros_like(vertical_lengths)
    feature_sizes = np.zeros_like(vertical_lengths)
    rng = np.random.default_rng(seed=0)

    lattice_structure = LatticeStructure.BCC
    lattice_parameter = 3.5

    for i, length in enumerate(vertical_lengths):
        # construct the supercell
        supercell = build.bulk(
            "Cu",
            crystalstructure=lattice_structure.name.lower(),
            a=lattice_parameter,
            cubic=True
        ).repeat((3, 3, length))
        num_atoms[i] = len(supercell)
        supercell.symbols = rng.choice(["Cu", "Pd"], p=[0.5, 0.5], size=len(supercell))

        state_matrix = np.zeros((len(supercell), 2))
        for site, t in enumerate(supercell.symbols):
            if t == "Cu":
                j = 0
            elif t == "Pd":
                j = 1
            else:
                raise ValueError
            state_matrix[site, j] = 1.0

        # compute the feature vector for that state matrix
        adjacency_tensors = get_adjacency_tensors(
            tree=KDTree(data=supercell.positions, boxsize=np.diag(supercell.get_cell()[:])),
            cutoffs=lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[lattice_structure][:3]
        )
        three_body_tensors = get_three_body_tensors(
            lattice_structure=lattice_structure,
            adjacency_tensors=adjacency_tensors,
            max_three_body_order=2
        )
        feature_vector = get_feature_vector(
            adjacency_tensors=adjacency_tensors,
            three_body_tensors=three_body_tensors,
            state_matrix=state_matrix,
        )
        feature_sizes[i] = np.linalg.norm(feature_vector)

    plt.scatter(num_atoms, feature_sizes, edgecolor="black", facecolor="turquoise", zorder=7)
    plt.xlabel("Number of atoms")
    plt.ylabel(r"Feature magnitude $\|\mathbf{t}\|$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("size-dependence.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
