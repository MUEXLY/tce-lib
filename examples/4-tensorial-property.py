from ase import build
from ase.calculators.eam import EAM
import numpy as np

from tce.training import LimitingRidge
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
        configuration.get_stress()
        configurations.append(configuration)

    calc = TCECalculator(
        models={"energy": LimitingRidge(), "stress": LimitingRidge()},
        neighbor_cutoffs=[
            0.5 * np.sqrt(2.0) * 3.6, 
            1.0 * 3.6, 
            np.sqrt(1.5) * 3.6
        ],
        many_body_features=[
            (0, 0, 0),
            (0, 0, 1)
        ],
        species=np.array(["Cu", "Ni"]),
        intensive={"energy": False, "stress": True}
    ).train(configurations)

    # predict a larger stress

    larger_system = build.bulk(
        "Cu",
        crystalstructure="fcc",
        a=3.6,
        cubic=True
    ).repeat((10, 10, 10))
    larger_system.symbols = generator.choice(["Cu", "Ni"], p=[0.7, 0.3], size=len(larger_system))
    larger_system.calc = calc
    print(larger_system.get_stress())
    print(larger_system.get_potential_energy())

    # now we can predict enthalpy too
    stress = larger_system.get_stress()
    energy = larger_system.get_potential_energy()
    volume = larger_system.get_volume()

    # trace of stress tensor
    pressure = -np.mean(stress[:3])
    enthalpy = energy + pressure * volume
    print(enthalpy)


if __name__ == "__main__":

    main()
