import os
import numpy as np

# Define input folder name
REACTION = "spnf"

# Define bond atom pairs
BOND_ATOMS = [
    (16, 62),
    (15, 56),
    (13, 41)
]

# Thresholds
BREAK_THRESHOLD = 3.2
FORMATION_THRESHOLD = 1.7

# Directories
INPUT_DIR = f'./{REACTION}'
OUTPUT_DIRS = {
    'r2r': './r2r',
    'r2p1': './r2p1',
    'r2p2': './r2p2',
    'p2p': './p2p'
}

# Ensure output directories exist
for path in OUTPUT_DIRS.values():
    os.makedirs(path, exist_ok=True)

def parse_xyz_trajectory(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    atom_count = int(lines[0])
    frame_size = atom_count + 2
    frames = [lines[i:i+frame_size] for i in range(0, len(lines), frame_size)]
    return frames, atom_count

def calculate_bond_lengths(frame_lines, atom_count):
    coordinates = [np.array(line.split()[1:], dtype=float) for line in frame_lines[1:1+atom_count]]
    bond_lengths = [
        np.linalg.norm(coordinates[a1] - coordinates[a2])
        for a1, a2 in BOND_ATOMS
    ]
    return bond_lengths

def split_trajectory(frames, atom_count):
    traj1, traj2 = [], []
    bond_data_1, bond_data_2 = [], []
    division_found = False

    for frame in frames:
        division_flag = frame[1].split()[6]
        bonds = calculate_bond_lengths(frame, atom_count)

        if not division_found and division_flag == '1':
            division_found = True

        if not division_found:
            traj1.append(frame)
            bond_data_1.append(bonds)
        else:
            traj2.append(frame)
            bond_data_2.append(bonds)

    return traj1, traj2, bond_data_1, bond_data_2

def write_trajectory(output_dir, index, traj_part1, traj_part2, reverse_first_part=False):
    filepath = os.path.join(output_dir, f'traj{index}.xyz')
    with open(filepath, 'w') as file:
        if reverse_first_part:
            for frame in reversed(traj_part1):
                file.writelines(frame)
            for frame in traj_part2:
                file.writelines(frame)
        else:
            for frame in reversed(traj_part2):
                file.writelines(frame)
            for frame in traj_part1:
                file.writelines(frame)

def classify_and_save_trajectory(index, traj1, traj2, bond_data_1, bond_data_2):
    b11, b12, b13 = bond_data_1[-1]
    b21, b22, b23 = bond_data_2[-1]

    if b11 > BREAK_THRESHOLD and b21 > BREAK_THRESHOLD:
        write_trajectory(OUTPUT_DIRS['r2r'], index, traj1, traj2, reverse_first_part=True)
    elif b11 > BREAK_THRESHOLD and b22 < FORMATION_THRESHOLD:
        write_trajectory(OUTPUT_DIRS['r2p1'], index, traj1, traj2, reverse_first_part=True)
    elif b21 > BREAK_THRESHOLD and b12 < FORMATION_THRESHOLD:
        write_trajectory(OUTPUT_DIRS['r2p1'], index, traj1, traj2, reverse_first_part=False)
    elif b11 > BREAK_THRESHOLD and b23 < FORMATION_THRESHOLD:
        write_trajectory(OUTPUT_DIRS['r2p2'], index, traj1, traj2, reverse_first_part=True)
    elif b21 > BREAK_THRESHOLD and b13 < FORMATION_THRESHOLD:
        write_trajectory(OUTPUT_DIRS['r2p2'], index, traj1, traj2, reverse_first_part=False)
    else:
        write_trajectory(OUTPUT_DIRS['p2p'], index, traj1, traj2, reverse_first_part=True)

def main():
    trajectory_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xyz')]

    for index, traj_file in enumerate(trajectory_files, start=1):
        frames, atom_count = parse_xyz_trajectory(os.path.join(INPUT_DIR, traj_file))
        traj1, traj2, bond_data_1, bond_data_2 = split_trajectory(frames, atom_count)
        classify_and_save_trajectory(index, traj1, traj2, bond_data_1, bond_data_2)

if __name__ == "__main__":
    main()
