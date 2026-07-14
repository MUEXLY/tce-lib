from pathlib import Path
import logging
import sys

import numpy as np
from ase import io, build

from tce.calculator import TCECalculator
from tce.monte_carlo import monte_carlo


def main():

    rng = np.random.default_rng(seed=0)

    calculator = TCECalculator.load("copper_nickel_tce.pkl")

    atoms = build.bulk(
        "Cu",
        a=3.6,
        crystalstructure="fcc",
        cubic=True
    ).repeat((15, 15, 15))
    atoms.symbols = rng.choice(["Cu", "Ni"], size=len(atoms))

    trajectory = monte_carlo(
        initial_configuration=atoms,
        tce_calculator=calculator,
        num_steps=10_000,
        beta=19.341,
        save_every=100
    )

    for i, frame in enumerate(trajectory):
        path = Path(f"copper-nickel/frame_{i:.0f}.xyz")
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write(path, frame)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
