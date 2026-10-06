import os
import shutil
import pandas as pd
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor

def download_pdb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    if not pdb_id or len(pdb_id) != 4:
        return
    path_pdb = "../pdb-files"
    target_pdb = os.path.join(path_pdb, f"{pdb_id}.pdb")
    target_ent = os.path.join(path_pdb, f"pdb{pdb_id}.ent")
    
    if os.path.isfile(target_pdb):
        return f"{pdb_id} already exists"
        
    try:
        pdbl = PDBList()
        # Retrieve PDB file (this creates target_ent)
        pdbl.retrieve_pdb_file(pdb_id, pdir=path_pdb, file_format="pdb")
        if os.path.isfile(target_ent):
            shutil.copy(target_ent, target_pdb)
            return f"Successfully downloaded and copied {pdb_id}"
        else:
            return f"Downloaded {pdb_id} but .ent file not found"
    except Exception as e:
        return f"Failed to download {pdb_id}: {str(e)}"

def main():
    path_dataset = "../Database Redox Pot Fe2S2 proteins.xlsx"
    df = pd.read_excel(path_dataset)
    pdb_codes = df["PDB Code(s)"].dropna().unique()
    pdb_codes = [code.strip().lower() for code in pdb_codes if len(str(code).strip()) == 4]
    
    print(f"Total unique PDB IDs to check: {len(pdb_codes)}")
    
    os.makedirs("../pdb-files", exist_ok=True)
    
    # Use ThreadPoolExecutor for parallel downloading
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_pdb, pdb_codes))
        
    success_count = sum(1 for res in results if res and "Successfully" in res)
    failed_count = sum(1 for res in results if res and "Failed" in res)
    already_count = sum(1 for res in results if res and "already exists" in res)
    
    print(f"Summary: {success_count} downloaded, {failed_count} failed, {already_count} already existed.")

if __name__ == "__main__":
    main()
