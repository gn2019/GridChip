import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

SEED = 42
HEADER = ['ProbeId', 'CustomerId', 'Sequence', 'ProbeGroupName', 'ProbeGroupId']
FILLER_PROBES = ['A_12_P113985']
SUPPORTED_GRID_SIZES = [24, 10]
GRID_SHAPE = {
    24: (1824, 534),
    10: (10, 10),  # debug
}
CELL_SHAPE = {
    24: (200, 131),
    10: (3, 3),  # debug
}
COLS_LOC = {
    24: [(13, 143), (202, 332), (394, 524),],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
}
ROWS_LOC = {
    24: [
        (0, 199),  # actually (0, 139)
        (223, 422), (506, 705), (789, 988), (1072, 1271), (1355, 1554),
        (1624, 1823),  # actually (0, 139)
    ],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
}
MASK_FILE = {
    24: 'mask_24.csv',
    10: 'mask_10.csv',  # debug
}
ORDER_FILE = {
    24: 'order_24.txt',
    10: 'mask_10.csv',  # debug
}
SAFETY_DISTANCE = {
    24: 15,
    10: 1,  # debug
}


def get_cell_shape(grid_size):
    return CELL_SHAPE[grid_size]


def get_cell_size(grid_size):
    cell_shape = get_cell_shape(grid_size)
    return cell_shape[0] * cell_shape[1]


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare grid chips')
    parser.add_argument('-f', '--features_file', type=str, required=True,
                        help='Path to tdt features file downloaded from agilent website')
    parser.add_argument('-g', '--grid_size', type=int, choices=SUPPORTED_GRID_SIZES, default=24,
                        help='Grid size in cells, default: 24')
    parser.add_argument('-o', '--out_file', type=str,
                        help='Path to output file, default: <features_file>_grid_<grid_size>.tdt')
    args = parser.parse_args()
    return args


def get_seq_counts(features_file):
    csv = pd.read_csv(features_file, sep='\t')
    # agg by all columns and count
    counts = csv.groupby(csv.columns.tolist()).size()
    # make columns from index
    counts = counts.reset_index()
    # rename count column
    counts = counts.rename(columns={0: 'count'})
    return counts['count'], counts


def get_per_cell_counts(counts, cell_size, remove_probes=None):
    if counts.shape[0] > cell_size:
        raise ValueError(f'Number of sequences ({counts.shape[0]}) is larger than cell size ({cell_size})')

    scaling_factor = (cell_size - counts.shape[0]) / counts.sum()
    proportional_counts = (counts * scaling_factor).sort_values().astype(int)
    proportional_counts += 1
    if remove_probes is not None:
        proportional_counts[remove_probes] = 0
    bonus = cell_size - proportional_counts.sum()
    # increase last bonus cells by 1
    if bonus:
        if remove_probes is None:
            proportional_counts[-bonus:] += 1
        else:  # fillers don't get bonus
            get_bonus = list(proportional_counts.index)
            for probe in remove_probes:
                get_bonus.remove(remove_probes)
            proportional_counts[get_bonus[-bonus:]] += 1
    return proportional_counts


def fill_cell(per_cell_counts, cell_shape):
    # make np.array of cell_shape
    grid = np.zeros(cell_shape)
    # get list of per_cell_counts indexes, eah one by its count
    seqs = np.array([])
    for key, value in per_cell_counts.to_dict().items():
        if value > 0:
            seqs = np.append(seqs, np.full((value - 1,), key))
    # shuffle the list with seed
    np.random.seed(SEED)
    np.random.shuffle(seqs)
    unique_seqs = np.unique(per_cell_counts[per_cell_counts > 0].index)
    np.random.shuffle(unique_seqs)
    # all unique seqs should be first in the list
    seqs = np.append(unique_seqs, seqs)
    # fill the grid with the sequences
    grid.flat = seqs
    return grid


def fill_columns(chip, grid_size):
    cols_loc = COLS_LOC[grid_size].copy()
    space_loc = [(i[1] + 1, j[0] - 1) for i, j in zip(cols_loc[:-1], cols_loc[1:])]
    if cols_loc[0][0] != 0:
        space_loc.insert(0, (0, cols_loc[0][0] - 1))
    if cols_loc[-1][1] != chip.shape[1] - 1:
        space_loc.append((cols_loc[-1][1] + 1, chip.shape[1] - 1))

    # split middle spaces to 2
    space_loc_2 = [space_loc[0]]
    for i, j in space_loc[1:-1]:
        mid = (j + i) // 2
        space_loc_2.extend([(i, mid), (mid+1, j)])
    space_loc_2.append(space_loc[-1])

    # to pairs
    space_loc_pairs = [(space_loc_2[i], space_loc_2[i+1]) for i in range(0, len(space_loc_2), 2)]

    # fill the spaces
    for (c1, c2), ((s11, s12), (s21, s22)) in zip(COLS_LOC[grid_size], space_loc_pairs):
        if s11 is not None:
            chip[:, s11:s12+1] = chip[:, c2-(s12-s11):c2+1]
        if s22 is not None:
            chip[:, s21:s22+1] = chip[:, c1:c1+(s22-s21)+1]


def fill_rows(chip, grid_size):
    rows_loc = ROWS_LOC[grid_size].copy()
    space_loc = [(i[1] + 1, j[0] - 1) for i, j in zip(rows_loc[:-1], rows_loc[1:])]
    if rows_loc[0][0] != 0:
        space_loc.insert(0, (0, rows_loc[0][0] - 1))
    if rows_loc[-1][1] != chip.shape[0] - 1:
        space_loc.append((rows_loc[-1][1] + 1, chip.shape[0] - 1))

    # split middle spaces to 2
    space_loc_2 = [space_loc[0]]
    for i, j in space_loc[1:-1]:
        mid = (j + i) // 2
        space_loc_2.extend([(i, mid), (mid+1, j)])
    space_loc_2.append(space_loc[-1])

    # to pairs
    space_loc_pairs = [(space_loc_2[i], space_loc_2[i+1]) for i in range(0, len(space_loc_2), 2)]

    # fill the spaces
    for (r1, r2), ((s11, s12), (s21, s22)) in zip(ROWS_LOC[grid_size], space_loc_pairs):
        if s11 is not None:
            chip[s11:s12+1, :] = chip[r2-(s12-s11):r2+1, :]
        if s22 is not None:
            chip[s21:s22+1, :] = chip[r1:r1+(s22-s21)+1, :]


def make_chip(cell, grid_size):
    rows_loc, cols_loc = ROWS_LOC[grid_size], COLS_LOC[grid_size]
    chip = np.zeros(GRID_SHAPE[grid_size])
    # cartesian multiplication of rows and cols
    for i, row in enumerate(rows_loc):
        for j, col in enumerate(cols_loc):
            rows = row[1] - row[0] + 1
            cols = col[1] - col[0] + 1
            part_cell = cell
            # if first row, cut chip from the top
            if rows < cell.shape[0]:
                if i == 0: part_cell = part_cell[-rows:,:]
                elif i == len(rows_loc) - 1: part_cell = part_cell[:rows,:]
                else: raise ValueError("Problem in cell size, it's not your fault")
            if cols < cell.shape[1]:
                if j == 0: part_cell = part_cell[:,-cols:]
                elif j == len(cols_loc) - 1: part_cell = part_cell[:,:cols]
                else: raise ValueError("Problem in cell size, it's not your fault")
            chip[row[0]:row[1]+1, col[0]:col[1]+1] = part_cell
    return chip


def mask_chip(chip, grid_size):
    mask = pd.read_csv(MASK_FILE[grid_size], header=0, index_col=0)
    masked = np.where(mask, np.nan, chip)
    return masked


def get_sorted_seqs(chip : np.ndarray, grid_size):
    # read chip from up to down, from left to right, rows first
    seqs = chip.flatten(order='C')
    # remove nan
    seqs = seqs[~np.isnan(seqs)]
    order = pd.read_csv(ORDER_FILE[grid_size], header=None).values.flatten()
    seqs = seqs[order]

    return seqs.astype(int)


def write_seqs_to_file(out_file, seqs, translator):
    content = ('\t'.join(HEADER) + '\n' +
               '\n'.join(tqdm('\t'.join(translator.iloc[seq].values[:-1]) for seq in seqs)) +
               '\n')
    with open(out_file, 'w') as f:
        f.write(content)


def get_unused_cells(grid_size, expand=0):
    dist = SAFETY_DISTANCE[grid_size] - expand
    if dist < 0:
        raise ValueError('Safety distance is too small')

    rows, cols = GRID_SHAPE[grid_size]

    space_loc_rows = [(i[1] + 1, j[0] - 1) for i, j in zip(ROWS_LOC[grid_size][:-1], ROWS_LOC[grid_size][1:])]
    space_loc_rows_unused = [(i+dist, j-dist) for i, j in space_loc_rows if j - i > dist*2]
    unused_rows = np.concatenate([np.arange(i, j+1) for i, j in space_loc_rows_unused])
    space_loc_cols = [(i[1] + 1, j[0] - 1) for i, j in zip(COLS_LOC[grid_size][:-1], COLS_LOC[grid_size][1:])]
    space_loc_cols_unused = [(i+dist, j-dist) for i, j in space_loc_cols if j - i > dist*2]
    unused_cols = np.concatenate([np.arange(i, j+1) for i, j in space_loc_cols_unused])
    # all cell indices for unused rows only, not regarding unused_cols
    row_indices = np.repeat(unused_rows[:, np.newaxis], cols)
    col_indices = np.tile(np.arange(cols), len(unused_rows))
    # all cell indices for unused cols only, without unused rows
    used_rows = np.setdiff1d(np.arange(rows), unused_rows)
    row_indices_2 = np.repeat(used_rows[:, np.newaxis], len(unused_cols))
    col_indices_2 = np.tile(unused_cols, len(used_rows))

    rows, cols = np.concatenate([row_indices, row_indices_2]), np.concatenate([col_indices, col_indices_2])
    return rows, cols

    # space_loc = [(i[1] + 1, j[0] - 1) for i, j in zip(COLS_LOC[grid_size][:-1], COLS_LOC[grid_size][1:])]
    # space_loc_unused = [(i + dist, j - dist) for i, j in space_loc if j - i > dist * 2]
    # unused_cols = np.concatenate([np.arange(i, j + 1) for i, j in space_loc_unused])
    # return unused_rows


def fix_cells(chip, per_cell_counts, grid_size):
    nans = np.isnan(chip)
    unique_per_cell_seqs = per_cell_counts[per_cell_counts > 0].index
    rows_loc, cols_loc = ROWS_LOC[grid_size], COLS_LOC[grid_size]
    for i, row in enumerate(rows_loc):
        for j, col in enumerate(cols_loc):
            cell = chip[row[0]:row[1] + 1, col[0]:col[1] + 1]
            # count values
            unique_values, value_counts = np.unique(cell[~np.isnan(cell)], return_counts=True)
            # values that are not in the cell
            orig_missing_values = np.setdiff1d(unique_per_cell_seqs, unique_values)
            # values that are in the cell but not in the counts
            extra_values = np.setdiff1d(unique_values, unique_per_cell_seqs)
            missing_values = orig_missing_values
            while missing_values.shape[0]:
                print(f'Cell ({i},{j}): missing: {missing_values.shape[0]}, extra: {extra_values.shape[0]}')
                # union missing_values with orig_missing_values, drop dups
                missing_values = np.unique(np.concatenate([missing_values, orig_missing_values]))
                # if there are missing values, fill them is the end of the cell
                # get last missing_values.shape[0] non NaN cells in the cell
                indices_to_fill = np.flatnonzero(~np.isnan(cell))[-missing_values.shape[0]:]
                cell.flat[indices_to_fill] = missing_values
                unique_values, value_counts = np.unique(cell[~np.isnan(cell)], return_counts=True)
                missing_values = np.setdiff1d(unique_per_cell_seqs, unique_values)

    # assert all nans are still nans
    assert np.alltrue(np.isnan(chip) == nans)


def fix_by_counts(chip, counts, grid_size, removed_probes=None):
    # count chip
    to_remove = len(chip[~np.isnan(chip)]) - counts.sum()
    if to_remove > 0:
        # remove from end of chip
        indices_to_remove = np.flatnonzero(~np.isnan(chip))[-to_remove:]
        chip.flat[indices_to_remove] = np.nan
    # count values on chip
    unique_values, value_counts = np.unique(chip[~np.isnan(chip)], return_counts=True)
    cur_counts = pd.Series(value_counts, index=unique_values)
    cur_counts = cur_counts.append(pd.Series(0, index=removed_probes), verify_integrity=True)
    diffs = (cur_counts - counts).astype(int)
    fill = np.concatenate([np.full(-diff, val) for val, diff in diffs.iteritems() if diff < 0])
    # get from unused_cells diffs[i] cells of seq i
    for i, diff in tqdm(diffs.iteritems(), total=diffs.shape[0]):
        if diff <= 0: continue
        ind = []
        expand = 0
        while len(ind) < diff:
            unused_cells = get_unused_cells(grid_size, expand)
            ind = np.where(chip[unused_cells] == i)[0]
            expand += 1
        chip[unused_cells[0][ind][:diff], unused_cells[1][ind][:diff]] = fill[:diff]
        fill = np.delete(fill, np.arange(diff))

    unique_values, value_counts = np.unique(chip, return_counts=True)
    cur_counts = pd.Series(value_counts, index=unique_values)
    diffs = cur_counts - counts
    assert diffs.sum() == 0


def chip_to_csv(chip, translator, outfile):
    chip_df = pd.DataFrame(chip)
    chip_df = chip_df.replace(np.nan, 'CTRL')
    for i, j in tqdm(translator.iterrows(), total=translator.shape[0]):
        chip_df = chip_df.replace(i, j[1])
    chip_df.to_csv(outfile, header=False, index=False)


def to_indices(probes, translator):
    return translator[translator.CustomerId.isin(probes)].index


def prepare_grid_chip(features_file, grid_size, out_file):
    # - get cell size, from a dict by grid size
    # - understand how many replicates are there in each cell
    # - fill the cell with the replicates
    # - fill column spaces with their matching columns
    # - fill row spaces with their matching rows
    # - order the features file
    counts, translator = get_seq_counts(features_file)
    remove_probes = to_indices(FILLER_PROBES, translator)
    per_cell_counts = get_per_cell_counts(counts, get_cell_size(grid_size), remove_probes=remove_probes)
    print(per_cell_counts)
    # print(per_cell_counts)
    cell = fill_cell(per_cell_counts, get_cell_shape(grid_size))
    chip = make_chip(cell, grid_size)
    # print(chip, '\n\n')
    fill_columns(chip, grid_size)
    # print(chip, '\n\n')
    fill_rows(chip, grid_size)
    # print(chip, '\n\n')
    chip = mask_chip(chip, grid_size)
    print(chip)
    fix_cells(chip, per_cell_counts, grid_size)
    fix_by_counts(chip, counts, grid_size, removed_probes=remove_probes)
    chip_to_csv(chip, translator, out_file[:-3] + 'csv')
    seqs = get_sorted_seqs(chip, grid_size)
    write_seqs_to_file(out_file, seqs, translator)


if __name__ == '__main__':
    args = parse_args()
    prepare_grid_chip(args.features_file, args.grid_size,
                      args.out_file or f'{os.path.splitext(args.features_file)[0]}_grid_{args.grid_size}.tdt')
