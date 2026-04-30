# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty, Transform
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import create_chem_graph_from_composition
from mattergen.diffusion.data.batched_data import BatchedData

ConditionLoader = Iterable[tuple[BatchedData, dict[str, torch.Tensor]] | None]


def _atoms_to_chemgraph(atoms) -> ChemGraph:
    """Convert ASE Atoms object to ChemGraph."""
    atoms: ase.Atoms
    frac_coords = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    atomic_numbers = atoms.get_atomic_numbers()
    num_atoms = len(atoms)

    return ChemGraph(
        pos=torch.from_numpy(frac_coords).float() % 1.0,
        cell=torch.from_numpy(np.array(cell)).float().unsqueeze(0),
        atomic_numbers=torch.from_numpy(atomic_numbers).long(),
        num_atoms=torch.tensor(num_atoms, dtype=torch.long),
        num_nodes=torch.tensor(num_atoms, dtype=torch.long),
    )


def get_proposal_data_loader(
    rss_proposal_path: str,
    batch_size: int,
    max_sampling_num: int | None = None,
    shuffle: bool = False,
) -> ConditionLoader:
    """
    Create a data loader from xyz files in the rss_proposal_path directory.

    Args:
        rss_proposal_path: Path to directory containing xyz files
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data

    Returns:
        ConditionLoader that yields batches of ChemGraph objects
    """
    proposal_path = Path(rss_proposal_path)

    # Find all xyz files in the directory
    xyz_files = list(proposal_path.glob("*.xyz"))

    if not xyz_files:
        raise ValueError(f"No xyz files found in {rss_proposal_path}")

    # Read all xyz files and convert to ChemGraph objects
    dataset_ = []
    for xyz_file in xyz_files:
        try:
            # Read atoms from xyz file
            atoms_list = ase.io.read(str(xyz_file), index=":")
            # Convert to ChemGraph
            for atoms in atoms_list:
                chemgraph = _atoms_to_chemgraph(atoms)
                dataset_.append(chemgraph)
        except Exception as e:
            print(f"Warning: Failed to read {xyz_file}: {e}")
            continue

    np.random.seed(42)
    if max_sampling_num is not None:
        random_idx = np.random.permutation(len(dataset_))[:max_sampling_num]
        dataset_ = [dataset_[i] for i in random_idx]

    if not dataset_:
        raise ValueError(f"No valid xyz files could be loaded from {rss_proposal_path}")

    # Create dataset
    dataset = ChemGraphlistDataset(dataset_)
    print(f"Dataset created with {len(dataset_)} chemgraphs from {rss_proposal_path}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


def _collate_fn(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
) -> tuple[BatchedData, None]:
    return collate_fn(batch), None


def get_number_of_atoms_condition_loader(
    num_atoms_distribution: str,
    num_samples: int,
    batch_size: int,
    shuffle: bool = True,
    transforms: list[Transform] | None = None,
    properties: TargetProperty | None = None,
) -> ConditionLoader:
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    assert num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS, (
        f"Invalid num_atoms_distribution: {num_atoms_distribution}"
    )
    dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
        num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution],
        num_samples=num_samples,
        transforms=transforms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


def get_composition_data_loader(
    target_compositions_dict: list[dict[str, float]],
    num_structures_to_generate_per_composition: int,
    batch_size: int,
) -> ConditionLoader:
    """
    Given a list of target compositions, generate a dataset of chemgraphs
    where each chemgraph contains atoms corresponding to the target composition
    without positions or cell information.
    Returns a torch dataloader equipped with the correct collate function containing such dataset.
    """

    dataset_ = []
    for compostion in target_compositions_dict:
        chemgraphs = [
            create_chem_graph_from_composition(compostion)
        ] * num_structures_to_generate_per_composition
        dataset_.extend(chemgraphs)

    dataset = ChemGraphlistDataset(dataset_)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=False,
    )


class ChemGraphlistDataset(Dataset):
    def __init__(self, data: list[ChemGraph]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ChemGraph:
        return self.data[index]
