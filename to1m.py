import os
import time
import argparse
from collections import defaultdict

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
    block_to_seqs = {}
    with open(blocks_file) as f:
        for line in f:
            seq, block = line.strip().split(',')
            block_to_seqs.setdefault(int(block), []).append(seq)
    return block_to_seqs


def make_chip(cells, grid_size, chip_design):
    """Assemble a chip from a list of pre-filled cell arrays.

    Unlike grid_chip.make_chip (which tiles a single cell), this version accepts
    one cell per (row, col) position in raster order.
    """
    rows_loc, cols_loc = ROWS_LOC[grid_size], COLS_LOC[grid_size]
    chip = np.full(CHIP_SHAPE[chip_design], -1)
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


def fix_counts(arr, target_counts, mask=None, randomize=True, prioritize_cells=False, cell_slices=None):
    """
    Adjust values in `arr` so global counts match `target_counts`.
    Over-represented values are replaced with under-represented ones.
    Parameters
    ----------
    arr : np.ndarray
        Array to modify in place.
    target_counts : pd.Series
        Desired counts indexed by value.
    mask : np.ndarray[bool], optional
        Boolean mask of protected positions.
        True = do not modify.
    randomize : bool, default=True
        Whether to randomize replacement positions.
    prioritize_cells : bool
        If True, remove excess probes first from cells with the
        highest local abundance of that probe.
    cell_slices : list
        Required if prioritize_cells=True.
        Format: [((r0, r1), (c0, c1)),...]

    Returns
    -------
    np.ndarray
        Modified array (same object as input).
    """
    if prioritize_cells and cell_slices is None:
        raise ValueError("cell_slices required when prioritize_cells=True")
    flat_arr = arr.ravel()
    if mask is None:
        flat_mask = np.zeros(flat_arr.shape, dtype=bool)
    else:
        if mask.shape != arr.shape:
            raise ValueError("mask must have same shape as arr")
        flat_mask = mask.ravel().copy()
    # Always protect NaN cells
    flat_mask |= pd.isna(flat_arr)
    # Count only non-NaN values
    valid_vals = flat_arr[~flat_mask]

    unique_vals, val_counts = np.unique(valid_vals, return_counts=True)
    cur_counts = pd.Series(val_counts, index=unique_vals)
    # Align to target index
    cur_counts = cur_counts.reindex(target_counts.index, fill_value=0)
    # Difference: positive = excess, negative = deficit
    diffs = (cur_counts - target_counts).astype(int)

    # Build replacement pool
    fill_values = [val for val, diff in diffs.items() if diff < 0 for _ in range(-diff)]
    if not fill_values:
        return arr
    fill_idx = 0
    # Replace excess values
    if prioritize_cells:
        mask_2d = flat_mask.reshape(arr.shape)
        for val, diff in diffs.items():
            if diff <= 0:
                continue
            positions = []
            cell_infos = []
            for (r0, r1), (c0, c1) in cell_slices:
                sub_arr = arr[r0:r1 + 1, c0:c1 + 1]
                sub_mask = mask_2d[r0:r1 + 1, c0:c1 + 1]
                local_pos = np.where((sub_arr == val) & (~sub_mask))
                count = len(local_pos[0])
                if count == 0:
                    continue
                flat_positions = np.ravel_multi_index((local_pos[0] + r0, local_pos[1] + c0), arr.shape)
                if randomize:
                    np.random.shuffle(flat_positions)
                cell_infos.append((count, flat_positions))
            cell_infos.sort(key=lambda x: x[0], reverse=True)
            max_len = cell_infos[0][0]
            positions = [
                sublist[i - (max_len - len(sublist))]
                for i in range(max_len)
                for count, sublist in cell_infos
                if i >= max_len - count
            ]
            for pos in positions[:diff]:
                arr.flat[pos] = fill_values[fill_idx]
                fill_idx += 1
                if fill_idx == len(fill_values):
                    return arr
    else:
        positions_by_value = defaultdict(list)
        for idx, val in enumerate(flat_arr):
            if not flat_mask[idx]:
                positions_by_value[val].append(idx)
        for val, diff in diffs.items():
            if diff <= 0:
                continue
            positions = np.asarray(positions_by_value[val])
            if randomize:
                np.random.shuffle(positions)
            for pos in positions[:diff]:
                arr.flat[pos] = fill_values[fill_idx]
                fill_idx += 1
                if fill_idx == len(fill_values):
                    return arr
    return arr

def get_non_cell_positions(grid_size, chip_design):
    non_cell_positions = np.ones(CHIP_SHAPE[chip_design], dtype=bool)
    for row in ROWS_LOC[grid_size]:
        for col in COLS_LOC[grid_size]:
            non_cell_positions[row[0]:row[1] + 1, col[0]:col[1] + 1] = False
    mask_chip(non_cell_positions, chip_design, False)
    return non_cell_positions


def fill_non_cells(chip, grid_size, chip_design, filler_probes, translator):
    # fill all non-cells by FILLER_PROBES which are in translator.CustomerId
    non_cell_positions = get_non_cell_positions(grid_size, chip_design)
    # fill with FILLER_PROBES by their counts in translator, normalized to the number of non-cell positions
    filler_counts = translator.loc[to_indices(filler_probes, translator), 'count']
    filler_counts = filler_counts * non_cell_positions.sum() / filler_counts.sum()
    filler_values = np.repeat(filler_counts.index.values, (filler_counts.values + .5).astype(int))
    np.random.shuffle(filler_values)
    flat_chip = chip.ravel()
    flat_non_cell_positions = non_cell_positions.ravel()
    flat_chip[flat_non_cell_positions] = filler_values[:flat_non_cell_positions.sum()]

def get_block_counts(block_seqs, translator, block_size):
    filtered_translator = translator[
        translator.CustomerId.isin(block_seqs) & ~translator.CustomerId.isin(FILLER_PROBES)]
    per_cell_counts = pd.Series(filtered_translator['count'].values, index=filtered_translator.index)
    # normalize per_cell_counts to shape
    per_cell_counts = per_cell_counts * block_size / per_cell_counts.sum()
    return (per_cell_counts + .5).astype(int)


def find_missing_probes(chip, grid_size, block_to_seqs, translator):
    missing_probes = {}
    rows_num, cols_num = len(ROWS_LOC[grid_size]), len(COLS_LOC[grid_size])
    for block in tqdm(range(rows_num * cols_num)):
        rows, cols = ROWS_LOC[grid_size][block // cols_num], COLS_LOC[grid_size][block % cols_num]
        non_masked_positions = (~np.isnan(chip[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1])).sum()
        block_counts = get_block_counts(block_to_seqs.get(block, []), translator, non_masked_positions)
        real_counts = np.unique(chip[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1], return_counts=True)
        # print probes with block_counts - real_counts >= 3 or real_counts == 0
        for probe, expected_count in block_counts.items():
            real_count = real_counts[1][real_counts[0] == probe][0] if probe in real_counts[0] else 0
            if expected_count - real_count >= 3 or real_count == 0:
                missing_probes.setdefault(block, []).append((probe, expected_count, real_count))
    return missing_probes


def prepare_grid_chip(features_file, blocks_file, grid_size, chip_design, out_file):
    # - get cell size, from a dict by grid size
    # - understand how many replicates are there in each cell
    # - fill the cell with the replicates
    # - order the features file
    print('[*] Mapping block to sequences')
    counts, translator = get_seq_counts(features_file)
    block_to_seqs = map_block_to_seqs(blocks_file)
    print('[*] Processing blocks')
    blocks = []
    rows_num, cols_num = len(ROWS_LOC[grid_size]), len(COLS_LOC[grid_size])
    for block in tqdm(range(rows_num * cols_num)):
        # get relevant counts
        rows, cols = ROWS_LOC[grid_size][block // cols_num], COLS_LOC[grid_size][block % cols_num]
        shape = rows[1] - rows[0] + 1, cols[1] - cols[0] + 1
        block_counts = get_block_counts(block_to_seqs.get(block, []), translator, shape[0] * shape[1])
        cell = fill_cell(block_counts, shape)
        blocks.append(cell)
    chip = make_chip(blocks, grid_size, chip_design)
    print('[*] Fixing block counts after removing ctrl probes')
    chip = mask_chip(chip, chip_design)
    for block in tqdm(range(rows_num * cols_num)):
        (r0, r1), (c0, c1) = ROWS_LOC[grid_size][block // cols_num], COLS_LOC[grid_size][block % cols_num]
        sub_chip = chip[r0:r1+1, c0:c1+1]
        non_masked_positions = (~np.isnan(sub_chip)).sum()
        block_counts = get_block_counts(block_to_seqs.get(block, []), translator, non_masked_positions)
        fix_counts(sub_chip, block_counts)
    print('[*] Fill by fillers')
    fill_non_cells(chip, grid_size, chip_design, FILLER_PROBES, translator)
    print('[*] Fixing global counts')
    global_counts = pd.Series(translator['count'].values, index=translator.index)
    # fix count of experiment probes, if there are mistakes
    chip = fix_counts(chip, global_counts[~global_counts.index.isin(to_indices(FILLER_PROBES, translator))],
                        mask=get_non_cell_positions(grid_size, chip_design))
    # second` pass without protected mask to fix any remaining issues
    chip = fix_counts(chip, global_counts, prioritize_cells=True,
                        cell_slices=[(rows, cols) for rows in ROWS_LOC[grid_size] for cols in COLS_LOC[grid_size]])
    print(chip)
    print('[*] Writing csv')
    chip_to_csv(chip, translator, out_file[:-3] + 'csv')
    print('[*] Writing tdt')
    seqs = get_sorted_seqs(chip, chip_design)
    write_seqs_to_file(out_file, seqs, translator)
    print('[*] Searching for missing probes')
    # for each cell, find counts that are 3+ less than expected, or 0
    missing_probes = find_missing_probes(chip, grid_size, block_to_seqs, translator)
    for block, probes in missing_probes.items():
        for (probe,  expected_count, real_count) in probes:
            print(f'Block {block}, probe {translator.loc[probe, "CustomerId"]}: expected {expected_count}, got {real_count}')


if __name__ == '__main__':
    args = parse_args()
    prepare_grid_chip(args.features_file, args.blocks_file, args.grid_size, args.chip_design,
                      args.out_file or f'{os.path.splitext(args.features_file)[0]}_grid_{args.grid_size}.tdt')
