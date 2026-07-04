from pathlib import Path
import pickle

from ase import build
from ase.calculators.eam import EAM
import numpy as np
import requests
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tce.calculator import TCECalculator


# from https://doi.org/10.1016/j.actamat.2019.06.027
EAM_POTENTIAL_URL = "https://www.ctcms.nist.gov/potentials/Download/2019--Fischer-F-Schmitz-G-Eich-S-M--Cu-Ni/3/Cu_Ni_Fischer_2018.eam.alloy"


def main():

    species = np.array(["Cu", "Ni"])
    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        name=species[0],
        crystalstructure="bcc",
        a=3.56,
        cubic=True
    ).repeat((3, 3, 3))

    num_configurations = 50
    configurations = []
    for _ in range(num_configurations):
        configuration = atoms.copy()
        x_cu = generator.random()
        configuration.symbols = generator.choice(
            a=species,
            p=[x_cu, 1.0 - x_cu],
            size=len(configuration)
        )

        potential = "Cu_Ni_Fischer_2018.eam.alloy"
        if not Path(potential).exists():

            response = requests.get(EAM_POTENTIAL_URL)
            with open(potential, "w") as file:
                file.write(response.text)

        configuration.calc = EAM(potential=potential)
        configuration.get_potential_energy()
        configurations.append(configuration)

    calc = TCECalculator(
        neighbor_cutoffs=[3.08, 3.56, 5.03],
        many_body_features=[
            (0, 0, 1),
            (0, 0, 2)
        ],
        species=species
    )

    train, test = train_test_split(configurations, test_size=0.2, random_state=0)

    calc.train(train)

    train_energies_actual = [t.get_potential_energy() for t in train]
    train_energies_predicted = [calc.get_potential_energy(t) for t in train]
    plt.scatter(train_energies_actual, train_energies_predicted, label="train", zorder=7)

    test_energies_actual = [t.get_potential_energy() for t in test]
    test_energies_predicted = [calc.get_potential_energy(t) for t in test]
    plt.scatter(test_energies_actual, test_energies_predicted, label="test", zorder=8)

    plt.xlabel("actual energy (eV)")
    plt.ylabel("predicted energy (eV)")
    plt.legend()
    plt.grid()
    gca = plt.gca()
    plt.plot(gca.get_xlim(), gca.get_xlim(), ls="--", color="black")
    plt.savefig("out.png")

    with open("copper_nickel_tce.pkl", "wb") as file:
        pickle.dump(calc, file)


if __name__ == "__main__":

    main()
