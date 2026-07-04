from pathlib import Path
import logging
import sys
import pickle

import numpy as np
from ase import io, build

#from tce.training import ClusterExpansion
#from tce.monte_carlo import monte_carlo


def main():

    rng = np.random.default_rng(seed=0)

    #cluster_expansion = ClusterExpansion.load(Path("CuNi.pkl"))

    atoms = build.bulk(
        "Cu",
        a=3.56,
        crystalstructure="bcc",
        cubic=True
    ).repeat((10, 10, 10))
    atoms.symbols = rng.choice(["Cu", "Ni"], size=len(atoms))

    with open("copper_nickel_tce.pkl", "rb") as file:
        calc = pickle.load(file)
    
    atoms.calc = calc
    print(f"Initial energy: {atoms.get_potential_energy()} eV")
    raise ValueError

    trajectory = monte_carlo(
        initial_configuration=atoms,
        cluster_expansion=cluster_expansion,
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
