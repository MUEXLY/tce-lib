import os
import pickle

import numpy as np
import requests
from ase import build

from tce.monte_carlo import monte_carlo


def discord_webhook_callback(
    step: int,
    num_steps: int,
    env_var: str = "DISCORD_WEBHOOK_URL",
    message: str = "CuNi mc run finished"
):

    if not os.getenv(env_var) or step + 1 < num_steps:
        return

    response = requests.post(url=os.getenv(env_var), json={"content": message})
    response.raise_for_status()


def main():

    rng = np.random.default_rng(seed=0)

    with open("copper_nickel_tce.pkl", "rb") as file:
        calc = pickle.load(file)

    atoms = build.bulk(
        "Cu",
        a=3.6,
        crystalstructure="fcc",
        cubic=True
    ).repeat((10, 10, 10))
    atoms.symbols = rng.choice(["Cu", "Ni"], size=len(atoms))

    trajectory = monte_carlo(
        initial_configuration=atoms,
        tce_calculator=calc,
        num_steps=10_000,
        beta=19.341,
        save_every=100,
        callback=discord_webhook_callback,
    )

if __name__ == "__main__":

    main()
