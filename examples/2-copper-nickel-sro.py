from pathlib import Path

import ovito
from cowley_sro_parameters import nearest_neighbor_topology, sro_modifier
import matplotlib.pyplot as plt
from ase import io, Atoms


def energy_per_atom(atoms: Atoms) -> float:

    return atoms.get_potential_energy() / len(atoms)


def main():

    pipeline = ovito.io.import_file("copper-nickel/frame_*.xyz")
    pipeline.modifiers.append(nearest_neighbor_topology(num_neighbors=12))
    pipeline.modifiers.append(sro_modifier())

    sro_parameters = [data.attributes["sro_12"] for data in pipeline.frames]

    # sort by number in filename, lexographical sorting done by Path.glob does not work
    paths = list(Path("copper-nickel").glob("frame_*.xyz"))
    paths.sort(key=lambda path: int(path.stem.split("_")[1]))
    energies = [energy_per_atom(io.read(p)) for p in paths]

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(sro_parameters, color="mediumvioletred")
    axs[0].grid()
    axs[0].set_ylabel("Cu-Ni Cowley \n short range order parameter")

    axs[1].plot(energies, color="dodgerblue")
    axs[1].grid()
    axs[1].set_xlabel("Monte Carlo frame")
    axs[1].set_ylabel("potential energy (eV / atom)")
    fig.tight_layout()
    fig.savefig("cu-ni-sro.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
