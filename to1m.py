import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from grid_chip import (fill_cell, get_seq_counts, to_indices,
                       mask_chip, chip_to_csv, get_sorted_seqs,
                       write_seqs_to_file)
from grid_chip import FILLER_PROBES, MASK_FILE, CHIP_SHAPE, SUPPORTED_CHIP_DESIGNS

SUPPORTED_GRID_SIZES = [21, 10, 8]
CELL_SHAPE = {
    21: (120, 109),
    10: (3, 3),  # debug
    8:  (328, 192), # 1: (1824, 534)
}
COLS_LOC = {
    21: [(i, i + CELL_SHAPE[21][1] - 1) for i in (0, 212, 424)],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
    8:  [(i, i + CELL_SHAPE[8][1] - 1) for i in (0, 342)],
}
ROWS_LOC = {
    21: [(i, i + CELL_SHAPE[21][0] - 1) for i in (0, 244, 488, 732, 976, 1220, 1464)] + [(1708, CHIP_SHAPE['1m'][0] - 1)],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
    8:  [(i, i + CELL_SHAPE[8][0] - 1) for i in (0, 499, 998, 1496)],
}

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare grid chips')
    parser.add_argument('-f', '--features_file', type=str, required=True,
                        help='Path to tdt features file downloaded from agilent website')
    parser.add_argument('-b', '--blocks_file', type=str, required=True,
                        help='Path to csv file with seq to block number mapping (seq_name,block_number)')
    parser.add_argument('-g', '--grid_size', type=int, choices=SUPPORTED_GRID_SIZES, default=21,
                        help='Grid size in cells, default: 24')
    parser.add_argument('-c', '--chip_design', type=str, choices=SUPPORTED_CHIP_DESIGNS, default='1m',
                        help='Chip design in words, default: 1m')
    parser.add_argument('-o', '--out_file', type=str,
                        help='Path to output file, default: <features_file>_grid_<grid_size>.tdt')
    args = parser.parse_args()
    return args


def map_block_to_seqs(blocks_file):
    """Read a CSV (seq_name,block_number) and return a dict of block_number → [seq_names]."""
    mapp = {
        'CEBPa': 0,
        'Foxl2': 1,
        'GAGA': 2,
        'GATA4': 3,
        'GE': 4,
        'GR': 5,
        'HHEX': 6,
        'HSF1': 7,
        'Human-GAPDH': 8,
        'Human-PGK1': 9,
        'IRF1': 10,
        'KLF3': 11,
        'Mouse-GAPDH': 8,
        'Mouse-PGK1': 9,
        'NANOG': 12,
        'PAX': 13,
        'PU1': 14,
        'Rat-GAPDH': 8,
        'Rat-PGK1': 9,
        'SF1': 15,
        'STAT1': 16,
        'WT1': 17,
        'YY1': 18,
    }
    block_to_seqs = {}
    with open(blocks_file) as f:
        for line in f:
            seq, block = line.strip().split(',')
            # seq = line.strip()
            # block = mapp.get(seq.split('_')[0], -1)
            block_to_seqs.setdefault(int(block), []).append(seq)
    return block_to_seqs


def make_chip(cells, grid_size, chip_design):
    """Assemble a chip from a list of pre-filled cell arrays.

    Unlike grid_chip.make_chip (which tiles a single cell), this version accepts
    one cell per (row, col) position in raster order.
    """
    rows_loc, cols_loc = ROWS_LOC[grid_size], COLS_LOC[grid_size]
    chip = np.empty(CHIP_SHAPE[chip_design])
    # cartesian multiplication of rows and cols
    for i, row in enumerate(rows_loc):
        for j, col in enumerate(cols_loc):
            cell = cells[i * len(cols_loc) + j]
            rows = row[1] - row[0] + 1
            cols = col[1] - col[0] + 1
            part_cell = cell
            # if first row, cut chip from the top
            if rows < cell.shape[0]:
                if i == 0:
                    part_cell = part_cell[-rows:, :]
                elif i == len(rows_loc) - 1:
                    part_cell = part_cell[:rows, :]
                else:
                    raise ValueError("Problem in cell size, it's not your fault")
            if cols < cell.shape[1]:
                if j == 0:
                    part_cell = part_cell[:, -cols:]
                elif j == len(cols_loc) - 1:
                    part_cell = part_cell[:, :cols]
                else:
                    raise ValueError("Problem in cell size, it's not your fault")
            chip[row[0]:row[1]+1, col[0]:col[1]+1] = part_cell
    return chip


def get_masked_count(i, grid_size):
    """Return the number of masked positions in cell i (row and col share the same index)."""
    rows, cols = ROWS_LOC[grid_size][i], COLS_LOC[grid_size][i]
    mask = pd.read_csv(MASK_FILE[grid_size], header=0, index_col=0)
    return mask.iloc[rows[0]:rows[1]+1, cols[0]:cols[1]+1].sum().sum()


def fix_by_counts(chip, counts, rows_loc, cols_loc):
    """Adjust probe counts within a single cell region so they match `counts`.

    Over-represented probes are replaced with under-represented ones.
    The replacement targets are consumed in order, cycling through all deficient
    probes rather than always replacing with the same one.
	"""
    part_chip = chip[rows_loc[0]:rows_loc[1]+1, cols_loc[0]:cols_loc[1]+1]
    unique_vals, val_counts = np.unique(part_chip, return_counts=True)
    cur_counts = pd.Series(val_counts, index=unique_vals)

    # Ensure every probe in the target appears in cur_counts (with 0 if absent)
    not_in_cur = counts.index.difference(cur_counts.index)
    cur_counts = pd.concat([cur_counts, pd.Series(0, index=not_in_cur)], verify_integrity=True)
    cur_counts = cur_counts[~cur_counts.index.isna()]
    diffs = (cur_counts - counts).astype(int)

    # Build an ordered list of replacement values: one entry per missing copy
    fill_values = [
        val
        for val, diff in diffs.items()
        if diff < 0
        for _ in range(-diff)
    ]
    fill_idx = 0

    for probe_val, diff in diffs.items():
        if diff <= 0:
            continue
        # Find positions in the *full* chip (not just part_chip) that hold this probe
        excess_rows, excess_cols = np.where(chip == probe_val)
        replaced = 0
        for row, col in zip(excess_rows, excess_cols):
            if replaced >= diff:
                break
            chip[row, col] = fill_values[fill_idx]
            fill_idx += 1
            replaced += 1


def fix_global_counts(chip, target_counts, protected_mask=None):
    """Fix global probe counts across the entire chip.

    Parameters
    ----------
    chip : np.ndarray
        Chip array containing probe indices.

    target_counts : pd.Series
        Target global counts indexed by probe index.

    protected_mask : np.ndarray[bool], optional
        Boolean mask of positions that should not be modified.
        Same shape as chip. True = protected.
    """
    flat_chip = chip.ravel()

    if protected_mask is None:
        protected_mask = np.zeros(chip.shape, dtype=bool)

    flat_mask = protected_mask.ravel()

    unique_vals, val_counts = np.unique(flat_chip, return_counts=True)
    cur_counts = pd.Series(val_counts, index=unique_vals)

    # Ensure all targets exist
    missing = target_counts.index.difference(cur_counts.index)
    if len(missing):
        cur_counts = pd.concat([
            cur_counts,
            pd.Series(0, index=missing)
        ])

    cur_counts = cur_counts[target_counts.index].fillna(0).astype(int)
    diffs = (cur_counts - target_counts).astype(int)

    # Values missing globally
    deficit_values = []
    for val, diff in diffs.items():
        if diff < 0:
            deficit_values.extend([val] * (-diff))

    if not deficit_values:
        return chip

    deficit_idx = 0

    # Replace excess values
    for val, diff in diffs.items():
        if diff <= 0:
            continue

        positions = np.where((flat_chip == val) & (~flat_mask))[0]

        replaced = 0
        for pos in positions:
            if replaced >= diff or deficit_idx >= len(deficit_values):
                break

            flat_chip[pos] = deficit_values[deficit_idx]
            deficit_idx += 1
            replaced += 1

    return chip


def prepare_grid_chip(features_file, blocks_file, grid_size, chip_design, out_file):
    # - get cell size, from a dict by grid size
    # - understand how many replicates are there in each cell
    # - fill the cell with the replicates
    # - order the features file
    print('[*] Mapping block to sequences')
    counts, translator = get_seq_counts(features_file)
    block_to_seqs = map_block_to_seqs(blocks_file)
    remove_probes = to_indices(FILLER_PROBES, translator)
    print('[*] Processing blocks')
    blocks = []
    rows_num, cols_num = len(ROWS_LOC[grid_size]), len(COLS_LOC[grid_size])
    for block in tqdm(range(rows_num * cols_num)):
        # get relevant counts
        block_seqs = block_to_seqs.get(block, [])
        # per_cell_counts is a df with seqs as index and counts as values
        # per_cell_counts = pd.Series({
        #     translator[translator.Name == seq].index[0]: translator[translator.Name == seq]['count'].iloc[0] for seq in block_seqs
        # })
        filtered_translator = translator[translator.CustomerId.isin(block_seqs) & ~translator.CustomerId.isin(FILLER_PROBES)]
        per_cell_counts = pd.Series(filtered_translator['count'].values, index=filtered_translator.index)

        try:
            rows, cols = ROWS_LOC[grid_size][block // cols_num], COLS_LOC[grid_size][block % cols_num]
        except Exception:
            print('ERROR:', grid_size, block, cols_num)
            raise
        shape = rows[1] - rows[0] + 1, cols[1] - cols[0] + 1
        cell = fill_cell(per_cell_counts, shape)
        blocks.append(cell)
    chip = make_chip(blocks, grid_size, chip_design)
    print('[*] Fixing block counts after removing ctrl probes')
    chip = mask_chip(chip, chip_design)
    for block in tqdm(range(rows_num * cols_num)):
        rows, cols = ROWS_LOC[grid_size][block // cols_num], COLS_LOC[grid_size][block % cols_num]
        filtered_translator = translator[translator.CustomerId.isin(block_to_seqs.get(block, []))]
        per_cell_counts = pd.Series(filtered_translator['count'].values, index=filtered_translator.index)
        fix_by_counts(chip, per_cell_counts, rows, cols)
    print('[*] Fixing global counts')
    global_counts = pd.Series(
        translator['count'].values,
        index=translator.index
    )
    chip = fix_global_counts(chip, global_counts)
    print(chip)
    print('[*] Writing csv')
    chip_to_csv(chip, translator, out_file[:-3] + 'csv')
    print('[*] Writing tdt')
    seqs = get_sorted_seqs(chip, chip_design)
    # write_seqs_to_file(out_file, seqs, translator)


if __name__ == '__main__':
    args = parse_args()
    prepare_grid_chip(args.features_file, args.blocks_file, args.grid_size, args.chip_design,
                      args.out_file or f'{os.path.splitext(args.features_file)[0]}_grid_{args.grid_size}.tdt')
