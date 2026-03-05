import os
import sys
import csv

RANGES = 100

def main():
    if len(sys.argv) < 2:
        print('Usage: python combine_csv.py <folderPath>')
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print('Provided path is not a directory.')
        sys.exit(1)

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        print('No CSV files found in the folder.')
        sys.exit(1)

    combined_rows = []
    header = None

    for file in files:
        if file == 'combined.csv':
            print(f'Skipping file "{file}" as it is the output file.')
            continue
        file_path = os.path.join(folder_path, file)
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
                    print(f'File "{file}" chunk starting at row {i + 1} does not have {RANGES} rows, skipping.')

    output_path = os.path.abspath(os.path.join(folder_path, '..', 'combined.csv'))
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(combined_rows)
    print(f'Combined CSV written to {output_path}')

if __name__ == '__main__':
    main()