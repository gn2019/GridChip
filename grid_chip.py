import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

SEED = 42
HEADER = ['ProbeId', 'CustomerId', 'Sequence', 'ProbeGroupName', 'ProbeGroupId']
FILLER_PROBES = ['A_12_P113985']
SUPPORTED_GRID_SIZES = [21, 10]
SUPPORTED_CHIP_DESIGNS = ['1m', '10dbg']
CHIP_SHAPE = {
    '1m': (1824, 534),
    '10dbg': (10, 10),  # debug
}
CELL_SHAPE = {
    21: (200, 131),
    10: (3, 3),  # debug
}
COLS_LOC = {
    21: [(13, 143), (202, 332), (394, 524),],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
}
ROWS_LOC = {
    21: [
        (0, 199),  # actually (0, 139)
        (223, 422), (506, 705), (789, 988), (1072, 1271), (1355, 1554),
        (1624, 1823),  # actually (1638, 1823)
    ],
    10: [(0, 1), (3, 5), (8, 9)],  # debug
}
MASK_FILE = {
    '1m': 'mask_1m.csv',
    '10dbg': 'mask_10.csv',  # debug
}
ORDER_FILE = {
    '1m': 'order_1m.txt',
    '10dbg': 'order_10.txt',  # debug
}
SAFETY_DISTANCE = {
    21: 15,
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
    parser.add_argument('-g', '--grid_size', type=int, choices=SUPPORTED_GRID_SIZES, default=21,
                        help='Grid size in cells, default: 21')
    parser.add_argument('-c', '--chip_design', type=str, choices=SUPPORTED_CHIP_DESIGNS, default='1m',
                        help='Chip design in words, default: 1m')
    parser.add_argument('-o', '--out_file', type=str,
                        help='Path to output file, default: <features_file>_grid_<grid_size>.tdt')
    args = parser.parse_args()
    return args


def get_seq_counts(features_file):
    csv = pd.read_csv(features_file, sep='\t', dtype=str)
    # agg by all columns and count
    counts = csv.groupby(csv.columns.tolist()).size()
    # counts = csv.groupby('CustomerId').size()
    # make columns from index
    counts = counts.reset_index()
    # rename count column
    counts = counts.rename(columns={0: 'count'})
    return counts['count'], counts


def get_per_cell_counts(counts, cell_size, remove_probes=None):
    if counts.shape[0] > cell_size:
        raise ValueError(f'Number of sequences ({counts.shape[0]}) is larger than cell size ({cell_size})')

    scaling_factor = (cell_size - counts.shape[0]) / counts.sum()
    proportional_counts = (counts * scaling_factor).sort_values().astype(int) + 1
    if remove_probes is not None:
        proportional_counts.loc[remove_probes] = 0
    bonus = cell_size - proportional_counts.sum()
    # increase last bonus cells by 1
    if bonus > 0:
        if remove_probes is None:
            proportional_counts.iloc[-bonus:] += 1
        else:
            mask = ~proportional_counts.index.isin(remove_probes)
            proportional_counts.loc[proportional_counts[mask].index[-bonus:]] += 1
    return proportional_counts


def fill_cell(per_cell_counts, cell_shape):
    # get list of per_cell_counts indexes, each one by its count
    positive = per_cell_counts[per_cell_counts > 0]
    repeated = np.repeat(positive.index.to_numpy(), positive.to_numpy() - 1)
    # shuffle the list with seed
    np.random.seed(SEED)
    np.random.shuffle(repeated)
    unique_seqs = np.unique(positive.index)
    np.random.shuffle(unique_seqs)
    # all unique seqs should be first in the list
    seqs = np.concatenate([unique_seqs, repeated])
    # fill the grid with the sequences
    grid = np.empty(cell_shape, dtype=seqs.dtype)
    grid.flat = seqs
    return grid


def _fill_axis(chip, locs, axis):
    size = chip.shape[1 if axis == 1 else 0]

    spaces = [(a[1] + 1, b[0] - 1) for a, b in zip(locs[:-1], locs[1:])]

    if locs[0][0] > 0:
        spaces.insert(0, (0, locs[0][0] - 1))

    if locs[-1][1] < size - 1:
        spaces.append((locs[-1][1] + 1, size - 1))

    split_spaces = [spaces[0]]

    for start, end in spaces[1:-1]:
        mid = (start + end) // 2
        split_spaces.extend([(start, mid), (mid + 1, end)])

    split_spaces.append(spaces[-1])

    for (p1, p2), ((s11, s12), (s21, s22)) in zip(
        locs,
        zip(split_spaces[::2], split_spaces[1::2])
    ):
        if axis == 1:
            if s11 <= s12:
                chip[:, s11:s12+1] = chip[:, p2-(s12-s11):p2+1]

            if s21 <= s22:
                chip[:, s21:s22+1] = chip[:, p1:p1+(s22-s21)+1]

        else:
            if s11 <= s12:
                chip[s11:s12+1, :] = chip[p2-(s12-s11):p2+1, :]

            if s21 <= s22:
                chip[s21:s22+1, :] = chip[p1:p1+(s22-s21)+1, :]


def fill_columns(chip, grid_size):
    _fill_axis(chip, COLS_LOC[grid_size], axis=1)


def fill_rows(chip, grid_size):
    _fill_axis(chip, ROWS_LOC[grid_size], axis=0)


def make_chip(cell, grid_size, chip_design):
    rows_loc, cols_loc = ROWS_LOC[grid_size], COLS_LOC[grid_size]
    cell_rows, cell_cols = cell.shape
    chip = np.empty(CHIP_SHAPE[chip_design], dtype=cell.dtype)

    for i, (r0, r1) in enumerate(rows_loc):
        rows = r1 - r0 + 1
        if rows == cell_rows:
            row_slice = slice(None)
        elif i == 0:
            row_slice = slice(cell_rows - rows, None)
        elif i == len(rows_loc) - 1:
            row_slice = slice(0, rows)
        else:
            raise ValueError("Invalid row layout")

        for j, (c0, c1) in enumerate(cols_loc):
            cols = c1 - c0 + 1
            if cols == cell_cols:
                col_slice = slice(None)
            elif j == 0:
                col_slice = slice(cell_cols - cols, None)
            elif j == len(cols_loc) - 1:
                col_slice = slice(0, cols)
            else:
                raise ValueError("Invalid column layout")
            chip[r0:r1+1, c0:c1+1] = cell[row_slice, col_slice]

    return chip


def mask_chip(chip, chip_design):
    mask = pd.read_csv(MASK_FILE[chip_design], header=0, index_col=0)
    masked = np.where(mask, np.nan, chip)
    return masked


def get_sorted_seqs(chip : np.ndarray, chip_design):
    # read chip from up to down, from left to right, rows first
    seqs = chip.flatten(order='C')
    # remove nan
    seqs = seqs[~np.isnan(seqs)]
    order = pd.read_csv(ORDER_FILE[chip_design], header=None).values.flatten()
    seqs = seqs[order]

    return seqs.astype(int)


def write_seqs_to_file(out_file, seqs, translator):
    content = ('\t'.join(HEADER) + '\n' +
               '\n'.join(tqdm(('\t'.join(translator.iloc[seq].values[:-1]) for seq in seqs), total=len(seqs))) +
               '\n')
    with open(out_file, 'w') as f:
        f.write(content)


def get_unused_cells(grid_size, chip_design, expand=0):
    dist = SAFETY_DISTANCE[grid_size] - expand
    if dist < 0:
        raise ValueError('Safety distance is too small')

    rows, cols = CHIP_SHAPE[chip_design]

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
    required = per_cell_counts[per_cell_counts > 0].index.to_numpy()
    rows_loc = ROWS_LOC[grid_size]
    cols_loc = COLS_LOC[grid_size]
    for i, (r0, r1) in enumerate(rows_loc):
        for j, (c0, c1) in enumerate(cols_loc):
            cell = chip[r0:r1+1, c0:c1+1]
            flat = cell.ravel()
            mask = ~np.isnan(flat)
            values = flat[mask]
            if values.size == 0:
                continue
            unique_vals, counts = np.unique(values, return_counts=True)
            missing = np.setdiff1d(required, unique_vals, assume_unique=False)
            extra = np.setdiff1d(unique_vals, required, assume_unique=False)
            if missing.size == 0 and extra.size == 0:
                continue
            # fill from end (same logic, but single pass)
            if missing.size > 0:
                fill_idx = np.where(mask)[0][-missing.size:]
                flat[fill_idx] = missing
    assert np.all(np.isnan(chip) == nans)


def fix_by_counts(chip, counts, grid_size, chip_design, removed_probes=None):
    # count chip
    to_remove = len(chip[~np.isnan(chip)]) - counts.sum()
    if to_remove > 0:
        # remove from end of chip
        indices_to_remove = np.flatnonzero(~np.isnan(chip))[-to_remove:]
        chip.flat[indices_to_remove] = np.nan
    # count values on chip
    unique_values, value_counts = np.unique(chip[~np.isnan(chip)], return_counts=True)
    cur_counts = pd.Series(value_counts, index=unique_values)
    # for each probe in counts that is not in curr_counts, add it with value 0
    not_in_cur_counts = counts.index.difference(cur_counts.index)
    cur_counts = pd.concat([cur_counts, pd.Series(0, index=not_in_cur_counts)], verify_integrity=True)
    diffs = (cur_counts - counts).astype(int)
    if diffs.size:
        fill = np.concatenate([np.full(-diff, val) for val, diff in diffs.items() if diff < 0])
    else:
        fill = np.array([])
    # get from unused_cells diffs[i] cells of seq i
    for i, diff in tqdm(diffs.items(), total=diffs.shape[0]):
        if diff <= 0: continue
        ind = []
        expand = 0
        while len(ind) < diff:
            unused_cells = get_unused_cells(grid_size, chip_design, expand)
            ind = np.where(chip[unused_cells] == i)[0]
            expand += 1
        chip[unused_cells[0][ind][:diff], unused_cells[1][ind][:diff]] = fill[:diff]
        fill = np.delete(fill, np.arange(diff))

    unique_values, value_counts = np.unique(chip, return_counts=True)
    cur_counts = pd.Series(value_counts, index=unique_values)
    diffs = cur_counts - counts
    assert diffs.sum() == 0


def chip_to_csv(chip, translator, outfile, chunk_size=1000):
    # Build replacement dictionary once
    replace_dict = translator["CustomerId"].to_dict()
    # Replace NaN separately
    chip_df = pd.DataFrame(chip).fillna("CTRL")
    arr = chip_df.to_numpy(dtype=object)
    # Vectorized replacement
    mask = arr != "CTRL"
    arr[mask] = np.vectorize(lambda x: replace_dict.get(x, x), otypes=[object])(arr[mask])
    pd.DataFrame(arr).to_csv(outfile, header=False, index=False)


def to_indices(probes, translator):
    return translator[translator.CustomerId.isin(probes)].index


def prepare_grid_chip(features_file, grid_size, chip_design, out_file):
    # - get cell size, from a dict by grid size
    # - understand how many replicates are there in each cell
    # - fill the cell with the replicates
    # - fill column spaces with their matching columns
    # - fill row spaces with their matching rows
    # - order the features file
    print('[*] Computing per cell counts')
    counts, translator = get_seq_counts(features_file)
    remove_probes = to_indices(FILLER_PROBES, translator)
    per_cell_counts = get_per_cell_counts(counts, get_cell_size(grid_size), remove_probes=remove_probes)
    print(per_cell_counts)
    # print(per_cell_counts)
    print('[*] Generating cells')
    cell = fill_cell(per_cell_counts, get_cell_shape(grid_size))
    chip = make_chip(cell, grid_size, chip_design)
    # print(chip, '\n\n')
    print('[*] Filling the gaps')
    fill_columns(chip, grid_size)
    # print(chip, '\n\n')
    fill_rows(chip, grid_size)
    # print(chip, '\n\n')
    print('[*] Fixing counts after removing ctrl probes')
    chip = mask_chip(chip, chip_design)
    fix_cells(chip, per_cell_counts, grid_size)
    fix_by_counts(chip, counts, grid_size, chip_design, removed_probes=remove_probes)
    print(chip)
    print('[*] Writing csv')
    chip_to_csv(chip, translator, out_file[:-3] + 'csv')
    print('[*] Writing tdt')
    seqs = get_sorted_seqs(chip, chip_design)
    write_seqs_to_file(out_file, seqs, translator)
    print('[*] Done')


if __name__ == '__main__':
    args = parse_args()
    prepare_grid_chip(args.features_file, args.grid_size, args.chip_design,
                      args.out_file or f'{os.path.splitext(args.features_file)[0]}_grid_{args.grid_size}.tdt')
