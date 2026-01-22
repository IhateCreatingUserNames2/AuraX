import os

# Specify the directory containing your .py files
source_directory = "E:\\ProjetosPython\\CeafV3"

# Specify the output file name
output_file = "AuraCEAFV3.txt"

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Loop through the directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            # Check if the file is a Python file
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Get the relative folder path from the source directory
                relative_folder = os.path.relpath(root, source_directory)
                if relative_folder == '.':
                    folder_info = "Root Directory"
                else:
                    folder_info = f"Folder: {relative_folder}"

                # Open each Python file and read its contents
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(f"# {'-' * 20} {file} ({folder_info}) {'-' * 20}\n\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # Add spacing between files

print(f"All Python files have been combined into {output_file}")