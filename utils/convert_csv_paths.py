import csv

def convert_paths(input_file, output_file):
    """
    Convert paths in a space-delimited CSV file from one absolute format to another.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file with updated paths.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile, delimiter=' ')

        for row in reader:
            if len(row) == 2:  # Ensure the row has the expected structure
                old_path, number = row
                # Replace the base path
                new_path = old_path.replace("/nfs/turbo/jjparkcv-turbo-large/miyen", "/home/madhavan/jepa/data")
                writer.writerow([new_path, number])
            else:
                print(f"Skipping malformed row: {row}")

# Example usage:
input_csv = "/home/madhavan/jepa/data/droid_videos_2/video_labels_old.csv"  # Replace with your input CSV file name
output_csv = "/home/madhavan/jepa/data/droid_videos_2/video_labels.csv"  # Replace with your desired output CSV file name
convert_paths(input_csv, output_csv)
