import logging
import sys
from typing import Callable
from functools import wraps
from collections import Counter

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ase import build, Atoms

from tce.calculator import TCECalculator
from tce.monte_carlo import monte_carlo, MCStep


def one_particle_swap_factory(
    generator: np.random.Generator,
    atomic_numbers: NDArray[np.int64]
) -> MCStep:

    @wraps(one_particle_swap_factory)
    def wrapper(atoms: Atoms) -> Atoms:

        new_atoms = atoms.copy()
        i, = generator.integers(len(atoms), size=1)
        
        new_atoms.numbers[i] = generator.choice(atomic_numbers)
        return new_atoms

    return wrapper


def energy_modifier_factory(
    chemical_potentials: dict[int, float]
) -> Callable[[Atoms, Atoms], float]:

    @wraps(energy_modifier_factory)
    def wrapper(
        initial: Atoms,
        final: Atoms
    ) -> float:

        types_initial = Counter(initial.numbers)
        types_final = Counter(final.numbers)

        legendre_transform = -sum(
            mu * (types_final[t] - types_initial[t]) for t, mu in chemical_potentials.items()
        )

        return legendre_transform

    return wrapper


def main():

    rng = np.random.default_rng(seed=0)

    calculator = TCECalculator.load("copper_nickel_tce.pkl")

    chemical_potentials_cu = np.linspace(0.5, 1.2, 25)
    atomic_fractions_cu = np.zeros_like(chemical_potentials_cu)

    pure_ni = build.bulk(
        "Ni",
        a=3.6,
        crystalstructure="fcc",
        cubic=True
    ).repeat((10, 10, 10))
    pure_ni.symbols = rng.choice(["Cu", "Ni"], size=len(pure_ni))

    for i, chemical_potential_cu in enumerate(chemical_potentials_cu):

        trajectory = monte_carlo(
            initial_configuration=pure_ni,
            tce_calculator=calculator,
            num_steps=10_000,
            beta=19.341,
            save_every=1_000,
            energy_modifier=energy_modifier_factory(
                chemical_potentials={29: chemical_potential_cu, 28: 0.0}
            ),
            mc_step=one_particle_swap_factory(
                generator=rng, 
                atomic_numbers=np.array([28, 29])
            ),
            callback=lambda x, y: None
        )
        final_types = np.array(trajectory[-1].get_chemical_symbols())
        atomic_fractions_cu[i] = (final_types == "Cu").mean()

    plt.plot(chemical_potentials_cu, 100 * atomic_fractions_cu, color="orangered")
    plt.xlabel(r"$\mu_\text{Cu} - \mu_\text{Ni}$ (eV)")
    plt.ylabel(r"Cu concentration (at. %)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("cu-ni-sgcmc.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
