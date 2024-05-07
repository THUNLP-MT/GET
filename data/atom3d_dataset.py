#!/usr/bin/python
# -*- coding:utf-8 -*-
from tqdm import tqdm

import torch
import scipy
import numpy as np
import pandas as pd
from atom3d.datasets import LMDBDataset

from utils import neighbors as nb

from .dataset import BlockGeoAffDataset, df_to_blocks, VOCAB, Block, blocks_to_data, Atom, blocks_interface
from .tokenizer.tokenize_3d import tokenize_3d
from .converter.atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks


class LEPDataset(BlockGeoAffDataset):
    # binary classification

    def __init__(self, data_file, database=None, dist_th=6, n_cpu=4, fragment=None):
        self.fragment = fragment
        suffix = fragment if fragment else ''
        print(f'fragmentation {self.fragment}')
        super().__init__(data_file, database, dist_th, n_cpu, suffix)

    def _load_data_file(self):
        return LMDBDataset(self.data_file)
    
    def _post_process(self, items, processed_data):
        indexes = [{'id': item['id'], 'label': item['label'] } for item in processed_data if item is not None]
        data = [item for item in processed_data if item is not None]
        return indexes, data
    
    def _preprocess(self, item):
        result = {}
        if item['label'] == 'A':
            activate = 1
        elif item['label'] == 'I':
            activate = 0
        else:
            raise ValueError(f'Activation label {item["label"]} not recognized.')
        result['id'] = item['id']
        result['label'] = activate
        for i, name in enumerate(['atoms_inactive', 'atoms_active']):
            blocks1 = df_to_blocks(item[name], key_atom_name='name')
            blocks2 = []
            assert blocks1[-1].symbol == VOCAB.UNK  # all atoms in the molecule are grouped in a single "residue" in the original data
            for atom in blocks1[-1].units:
                blocks2.append(Block(atom.element.lower(), [atom]))
            blocks1 = blocks1[:-1]
            blocks1, _ = blocks_interface(blocks1, blocks2, self.dist_th)  # blocks2 (small molecule) need to be included as a whole
            if len(blocks1) == 0:
                return None
            
            if self.fragment is not None:
                # fragmentation
                try:
                    blocks2 = atom_blocks_to_frag_blocks(
                        blocks2,
                        item['SMILES'].split('.')[0]) # detached ions will not be included in the structure
                except AssertionError as e:
                    print(e)
                    return None

            result[i] = blocks_to_data(blocks1, blocks2)
        result['len'] = len(result[0]['B']) + len(result[1]['B'])
        return result
    
    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[0][key], dtype=_type))
                val.append(torch.tensor(item[1][key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        res['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        lengths = []
        for item in batch:
            lengths.append(len(item[0]['B']))
            lengths.append(len(item[1]['B']))
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['X'] = res['X'].unsqueeze(-2)  # number of channel is 1
        return res
    

class LBADataset(BlockGeoAffDataset):

    def __init__(self, data_file, database=None, dist_th=6, n_cpu=4, fragment=False):
        self.fragment = fragment
        suffix = fragment if fragment else ''
        print(f'fragmentation {self.fragment}')
        super().__init__(data_file, database, dist_th, n_cpu, suffix)

    def _load_data_file(self):
        return LMDBDataset(self.data_file)

    def _post_process(self, items, processed_data):
        indexes = [{'id': item['id'], 'label': item['label'] } for item in processed_data]
        # # additionally save the index file
        # index_file = splitext(splitext(self.proc_file)[0])[0] + '.jsonl'
        # with open(index_file, 'w') as fout:
        #     for index in indexes:
        #         fout.write(json.dumps(index) + '\n')
        return indexes, processed_data  # no need to save the LMDB reader
    
    def _preprocess(self, item):

        # receptor
        blocks1 = df_to_blocks(item['atoms_pocket'], key_atom_name='name')
        
        # ligand (each block is an atom)
        blocks2 = []
        for row in item['atoms_ligand'].itertuples():
            atom = Atom(
                atom_name=getattr(row, 'name'),  # e.g. C1, C2, ..., these position code will be a unified encoding such as <sm> (small molecule) in our framework
                coordinate=[getattr(row, axis) for axis in ['x', 'y', 'z']],
                element=getattr(row, 'element'),
                pos_code=VOCAB.atom_pos_sm
            )
            blocks2.append(Block(
                symbol=atom.element.lower(),
                units=[atom]
            ))

        if self.fragment is not None:
            # fragmentation
            bonds = []
            for row in item['bonds'].itertuples():
                bond_type = int(getattr(row, 'type'))
                if bond_type == 1.5:
                    bond_type = 4  # aromatic
                bonds.append((getattr(row, 'atom1'), getattr(row, 'atom2'), bond_type))
            blocks2 = atom_blocks_to_frag_blocks(blocks2, bonds=bonds)
        
        result = blocks_to_data(blocks1, blocks2)
        # result = BlockGeoAffDataset._residues_to_data(rec_blocks, lig_blocks)
        result['label'] = item['scores']['neglog_aff']
        result['id'] = item['id']

        return result


if __name__ == '__main__':
    import sys
    dataset = LEPDataset(sys.argv[1])
    print(len(dataset))
    length = [len(item[0]['B']) + len(item[1]['B']) for item in dataset]
    print(f'interface length: min {min(length)}, max {max(length)}, mean {sum(length) / len(length)}')
    atom_length = [len(item[0]['A']) + len(item[1]['A']) for item in dataset]
    print(f'atom number: min {min(atom_length)}, max {max(atom_length)}, mean {sum(atom_length) / len(atom_length)}')