#!/usr/bin/env python
import subprocess
import argparse
import os

def parse_batch_file(batch_file):
    """Parse the batch SMILES file to extract drug names/IDs and SMILES strings."""
    results = []
    with open(batch_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                drug_id = parts[0]  # First part is the drug ID/name
                smiles = parts[1]   # Second part is the SMILES string
                results.append((drug_id, smiles))
    return results

def run_docking(drug_id, smiles, pdb_option, pdb_value, chain):
    """Run dockpipe.py with the given arguments."""
    cmd = [
        "python", "dockpipe3.py",
        f"--{pdb_option}", pdb_value,
        "--smiles", smiles,
        "--prefix", drug_id,
        "--chain", chain
    ]
    
    print(f"Running docking for {drug_id}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Completed docking for {drug_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error running docking for {drug_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch molecular docking using dockpipe.py")
    
    # Only one of these arguments should be provided
    pdb_group = parser.add_mutually_exclusive_group(required=True)
    pdb_group.add_argument("--pdb", help="PDB identifier")
    pdb_group.add_argument("--local_pdb", help="Path to local PDB file")
    
    parser.add_argument("--batch_file", required=True, help="File containing drug IDs and SMILES strings")
    parser.add_argument("--chain", required=True, help="Chain identifier")
    
    args = parser.parse_args()
    
    # Determine which PDB option to use
    if args.pdb:
        pdb_option = "pdb"
        pdb_value = args.pdb
    else:
        pdb_option = "local_pdb"
        pdb_value = args.local_pdb
    
    # Parse the batch file
    entries = parse_batch_file(args.batch_file)
    
    if not entries:
        print("No valid entries found in batch file.")
        return
    
    print(f"Found {len(entries)} compounds to dock.")
    
    # Run docking for each entry
    for i, (drug_id, smiles) in enumerate(entries, 1):
        print(f"\nProcessing compound {i}/{len(entries)}")
        run_docking(drug_id, smiles, pdb_option, pdb_value, args.chain)

if __name__ == "__main__":
    main()


