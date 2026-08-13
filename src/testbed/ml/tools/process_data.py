import os
import sys
import csv

RANGES = 300

def main():
    if len(sys.argv) < 2:
        print('Usage: python combine_csv.py <folderPath>')
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print('Provided path is not a directory.')
        sys.exit(1)

    # Get all CSV files in the folder (including subdirectories), excluding combined.csv
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith('.csv') and f != 'combined.csv':
                files.append(os.path.join(root, f))
    if not files:
        print('No CSV files found in the folder.')
        sys.exit(1)
    files.sort()

    combined_rows = []
    header = None

    for file_path in files:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            if not reader:
                continue
            if header is None:
                header = reader[0]
                combined_rows.append(header)
            data_rows = reader[1:]
            for i in range(0, len(data_rows), RANGES):
                chunk = data_rows[i:i+RANGES]
                chunk = sorted(chunk, key=lambda x: (x[0]))  # Sort by the first column
                if len(chunk) == RANGES:
                    combined_rows.extend(chunk)
                else:
                    print(f'File "{file_path}" chunk starting at row {i + 1} does not have {RANGES} rows, skipping.')

    output_path = os.path.abspath(os.path.join(folder_path, '..', 'combined.csv'))
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(combined_rows)
    print(f'Combined CSV written to {output_path}')

if __name__ == '__main__':
    main()