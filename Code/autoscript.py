# coding: utf-8

from required import *

input_dest, input_req ,out_dest = config_fetch()

# Input directory
dir_ = input_dest
output_dir = out_dest + r'\Output'

file_path = Path(dir)

ls_files = glob.glob(str(file_path) + r'/*.xlsx')
latest_file = [os.path.basename(filename) for filename in ls_files]

old_files = pd.read_csv(input_req + r'\Data\Requirements\OldFileNames.csv')
old_file_ls = list(old_files.OldFileNames.unique())

if all (filesname in old_file_ls for filesname in latest_file):
    print("No new file found")
else:
    print("New file found")
    latest_file = [files for files in latest_file if files not in old_file_ls]

    print("\nStarting Pridiction Module")
    subprocess.call(["python", input_req + r'\Code\predict.py'])

print("\nPrediction completed and output excel is saved at output location")
new_files = pd.DataFrame(list(map(os.path.basename, ls_files)), columns = ["OldFileNames"])
dff_names = pd.concat([old_files, new_files]).drop_duplicates('OldFileNames').reset_index(drop=True)
dff_names.to_csv(str(input_req) + r'\Data\Requirements\OldFileNames.csv', index = False)