from ase import build
from ase.calculators.eam import EAM
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tce.calculator import TCECalculator


def main():

    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        "Cu",
        crystalstructure="fcc",
        a=3.6,
        cubic=True
    ).repeat((3, 3, 3))

    num_configurations = 50
    configurations = []
    for _ in range(num_configurations):
        configuration = atoms.copy()
        x_cu = generator.random()
        configuration.symbols = generator.choice(
            a=["Cu", "Ni"],
            p=[x_cu, 1.0 - x_cu],
            size=len(configuration)
        )

        configuration.calc = EAM(potential="Cu_Ni_Fischer_2018.eam.alloy")
        configuration.get_potential_energy()
        configurations.append(configuration)

    alpha_values = np.logspace(-4.0, 3.0, 20)
    prop_considered_clusters = np.zeros_like(alpha_values)

    for i, alpha in enumerate(alpha_values):

        calc = TCECalculator(
            models={
                "energy": Pipeline([
                    ("scale", StandardScaler()),
                    ("reduce", PCA()),
                    ("fit", Lasso(alpha=alpha))
                ])
            },
            neighbor_cutoffs=[
                0.5 * np.sqrt(2.0) * 3.6, 
                1.0 * 3.6, 
                np.sqrt(1.5) * 3.6
            ],
            many_body_features=[
                (0, 0, 0),
                (0, 0, 1)
            ],
            species=np.array(["Cu", "Ni"])
        )
        calc.train(configurations)

        prop_considered_clusters[i] = np.logical_not(
            np.isclose(calc.models["energy"].named_steps["fit"].coef_, 0.0)
        ).mean()

    plt.plot(alpha_values, 100 * prop_considered_clusters, color="orchid")
    plt.xscale("log")
    plt.xlabel("regularization parameter")
    plt.ylabel("proportion of considered clusters (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("regularization.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
