import numpy as np
import matplotlib.pyplot as plt
from ase import build

from tce.calculator import TCECalculator


def main():
    vertical_lengths = np.arange(3, 9)
    num_atoms = np.zeros_like(vertical_lengths)
    feature_sizes = np.zeros_like(vertical_lengths)
    rng = np.random.default_rng(seed=0)

    a = 3.5

    calc = TCECalculator(
        neighbor_cutoffs=[
            0.5 * np.sqrt(2.0) * a,
            1.0 * a,
            np.sqrt(1.5) * a
        ],
        many_body_features=[
            (0, 0, 0),
            (0, 0, 1)
        ],
        species=["Cu", "Pd"]
    )

    for i, length in enumerate(vertical_lengths):
        # construct the supercell
        supercell = build.bulk(
            "Cu",
            crystalstructure="fcc",
            a=a,
            cubic=True
        ).repeat((3, 3, length))
        num_atoms[i] = len(supercell)
        supercell.symbols = rng.choice(["Cu", "Pd"], p=[0.5, 0.5], size=len(supercell))

        feature_sizes[i] = np.linalg.norm(calc.get_feature_vector(supercell))

    plt.scatter(num_atoms, feature_sizes, edgecolor="black", facecolor="turquoise", zorder=7)
    plt.xlabel("Number of atoms")
    plt.ylabel(r"Feature magnitude $\|\mathbf{t}\|$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("size-dependence.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
