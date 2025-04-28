#!/usr/bin/env python
"""
Molecular Docking Script with Binding Pocket Prediction and Grid Box Optimization

This script downloads a protein PDB file, predicts its binding pockets using PRANK,
prepares the ligand and computes its radius-of-gyration to set a base grid box size (2.9×Rg),
generates seven candidate grid boxes (base–10, base–5, base, base+5, base+10, base+15, base+20),
performs docking with AutoDock Vina using all available CPU cores, and finally retains
only the best docking output while removing all intermediate files.
"""

import argparse
import os
import sys
import platform
import subprocess
from pathlib import Path
import pandas as pd
import multiprocessing
import glob
import MDAnalysis as mda
import prolif as plf


from prody import *
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D

reduce2_path = Path("/data/opus/.pixi/envs/pymol/lib/python3.10/site-packages/mmtbx/command_line/reduce2.py") #replace with your environment path like conda
geostd_path = Path("/data/opus/dock/geostd") #replace with your environment path like conda
# Helper functions
def run_command(command):
    """Executes a shell command and prints stdout/stderr."""
    print(f"Executing: {command}")
    process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    if process.stdout:
        print("STDOUT:\n", process.stdout)
    if process.stderr:
        print("STDERR:\n", process.stderr)
    print("-" * 20)

def download_pdb(pdb_id):
    """Download PDB file from RCSB PDB database."""
    pdb_file = f"{pdb_id}.pdb"
    if os.path.exists(pdb_file):
        print(f"PDB file {pdb_file} already exists.")
        return pdb_file
    try:
        import requests
        print(f"Downloading {pdb_id}.pdb from RCSB...")
        response = requests.get(f"http://files.rcsb.org/view/{pdb_id}.pdb")
        response.raise_for_status()
        with open(pdb_file, 'w') as f:
            f.write(response.text)
        print("Download complete.")
    except (ImportError, requests.exceptions.RequestException) as e:
        print(f"Error with requests: {e}")
        print("Attempting download with curl...")
        cmd_curl = f"curl \"http://files.rcsb.org/view/{pdb_id}.pdb\" -o \"{pdb_file}\""
        run_command(cmd_curl)
    return pdb_file

def predict_binding_pockets(pdb_file, prefix=''):
    """Predict binding pockets using PRANK."""
    output_dir = f"{prefix}prank_output" if prefix else "prank_output"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(pdb_file)
    output_prefix = os.path.join(output_dir, os.path.splitext(base_name)[0])
    cmd = f"prank predict -f {pdb_file} -o {output_prefix}"
    run_command(cmd)
    prank_csv = os.path.join(output_dir, os.path.splitext(base_name)[0],
                             f"{base_name}_predictions.csv")
    if not os.path.exists(prank_csv):
        print(f"PRANK output CSV file not found: {prank_csv}")
        alt_prank_csv = os.path.join(output_dir, os.path.splitext(base_name)[0],
                                     f"{os.path.splitext(base_name)[0]}_predictions.csv")
        if os.path.exists(alt_prank_csv):
            prank_csv = alt_prank_csv
            print(f"Found alternative PRANK output at: {prank_csv}")
        else:
            sys.exit(1)
    print(f"Using PRANK predictions from: {prank_csv}")
    return prank_csv

def extract_pocket_coordinates(prank_csv):
    """Extract top pocket coordinates from PRANK output CSV."""
    try:
        df = pd.read_csv(prank_csv)
        df.columns = df.columns.str.strip()
        top_pocket = df.iloc[0]
        center_x = float(top_pocket['center_x'])
        center_y = float(top_pocket['center_y'])
        center_z = float(top_pocket['center_z'])
        print(f"Selected binding pocket center: ({center_x}, {center_y}, {center_z})")
        return center_x, center_y, center_z
    except Exception as e:
        print(f"Error parsing PRANK output: {e}")
        pdb_id = os.path.basename(prank_csv).split('_')[0]
        pdb_file = f"{pdb_id}.pdb"
        atoms_from_pdb = parsePDB(pdb_file)
        protein_atoms = atoms_from_pdb.select('protein')
        center_x, center_y, center_z = calcCenter(protein_atoms)
        print(f"Using protein centroid as fallback: ({center_x}, {center_y}, {center_z})")
        return center_x, center_y, center_z

def calculate_optimal_box_size(sdf_file):
    """
    Calculate the ligand's radius of gyration and determine a base grid box size as 2.9 × Rg.
    """
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    if not suppl or len(suppl) == 0 or suppl[0] is None:
        raise ValueError(f"Could not read molecule from {sdf_file}")
    mol = suppl[0]
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
    rg = Descriptors3D.RadiusOfGyration(mol)
    optimal_box_size = 2.9 * rg
    print(f"Calculated ligand radius of gyration: {rg:.2f}")
    print(f"Optimal base box size (2.9 × Rg): {optimal_box_size:.2f}")
    return optimal_box_size


def interaction_analysis(protein_pdb, ligand_pdbqt_output, prefix=''):
    """
    Analyze protein-ligand interactions using prolif.
    
    Parameters:
    - protein_pdb: The preprocessed protein PDB file
    - ligand_pdbqt_output: The docking output PDBQT file from Vina
    - prefix: Prefix to add to all output files
    """
    # Load protein with MDAnalysis
    u = mda.Universe(protein_pdb)
    protein_mol = plf.Molecule.from_mda(u)
    
    # Convert PDBQT output to SDF for analysis
    ligand_sdf = f"{prefix}ligand_best_pose.sdf"
    cmd = f"mk_export.py {ligand_pdbqt_output} -s {ligand_sdf}"
    process = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    
    # Load ligand from SDF
    ligand_mol = plf.sdf_supplier(ligand_sdf)[0]
    
    # List available fingerprint types
    plf.Fingerprint.list_available()
    
    # Use default interactions
    fp = plf.Fingerprint()
    
    # Run on your poses
    fp.run_from_iterable([ligand_mol], protein_mol)
    # Create an empty list to store all interaction details
    interaction_details = []

    # Get the frame index (0 in this case)
    frame = 0

    # Iterate through all residue pairs in the fingerprint data
    for residue_pair in fp.ifp[frame].keys():
        ligand_res, protein_res = residue_pair
        
        # Get all interactions for this residue pair
        interactions = fp.ifp[frame][residue_pair]
        
        # Iterate through each interaction type (HBDonor, Cationic, etc.)
        for interaction_type, details_tuple in interactions.items():
            # Each interaction might have multiple instances (stored as a tuple)
            for details in details_tuple:
                # Create a basic record for this interaction
                record = {
                    'frame': frame,
                    'ligand': ligand_res,
                    'protein': protein_res,
                    'interaction': interaction_type
                }
                
                # Add the distance and any other metrics (excluding indices)
                for key, value in details.items():
                    if key not in ['indices', 'parent_indices']:
                        record[key] = value
                
                # Add this record to our collection
                interaction_details.append(record)

    # Convert to DataFrame
    result_df = pd.DataFrame(interaction_details)

    # Save to CSV
    result_df.to_csv(f'{prefix}protein_ligand_interactions.csv', index=False)

    # Display the first few rows
    print(result_df.head())

    
    # Save the plot to an HTML file
    html_output = fp.plot_lignetwork(ligand_mol, kind="frame", frame=0, width='1800px', height='1000px')
    with open(f"{prefix}ligand_network.html", "w") as f:
        f.write(html_output.data)  # .data contains the HTML content
    
    # Generate 3D visualization
    view = fp.plot_3d(ligand_mol, protein_mol, frame=0, display_all=False, size=(1800, 1000))
    
    # Save the 3D visualization to an HTML file
    view.write_html(f"{prefix}protein_ligand_3d.html")
    print("Visualization files saved!")
    
    return ligand_sdf

def parse_vina_score(output_file):
    """Parse the Vina docking output file to extract the best (lowest) energy score."""
    best_score = None
    try:
        with open(output_file) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.strip().split()
                    score = float(parts[3])
                    if best_score is None or score < best_score:
                        best_score = score
    except Exception as e:
        print(f"Error parsing Vina score from {output_file}: {e}")
    return best_score

def perform_docking(pdb_id, ligand_smiles, ph=7.4, exhaustiveness=128, prefix='', chain ='A'):
    """
    Main function to perform the complete docking workflow with grid box optimization.
    It prepares the ligand, calculates the optimal box size from its radius-of-gyration,
    generates candidate grid boxes, docks with Vina using all available CPUs, 
    retains only the best docking output, and removes all intermediate files.
    
    Parameters:
    - pdb_id: PDB ID for the protein
    - ligand_smiles: SMILES string for the ligand
    - ph: pH for ligand preparation
    - exhaustiveness: Exhaustiveness parameter for Vina docking
    - prefix: Prefix to add to all output files
    """
    scrub = "scrub.py"
    mk_prepare_ligand = "mk_prepare_ligand.py"
    mk_prepare_receptor = "mk_prepare_receptor.py"
    
    full_py_version = platform.python_version()
    major_and_minor = ".".join(full_py_version.split(".")[:2])
    
    # 1. Ligand Preparation
    print("\n# 1. Ligand Preparation")
    ligand_name = f"{prefix}ligand" if prefix else "ligand"
    ligand_pdbqt = f"{ligand_name}.pdbqt"
    args = "--skip_tautomer"
    ligand_sdf = f"{ligand_name}_scrubbed.sdf"
    cmd_scrub = f"{scrub} \"{ligand_smiles}\" -o {ligand_sdf} --ph {ph} {args}"
    run_command(cmd_scrub)
    cmd_prep_lig = f"{mk_prepare_ligand} -i {ligand_sdf} -o {ligand_pdbqt}"
    run_command(cmd_prep_lig)
    
    # Calculate optimal box size from ligand SDF via radius of gyration
    optimal_box_size = calculate_optimal_box_size(ligand_sdf)
    # Create candidate grid box sizes
    # candidate_sizes = [max(optimal_box_size-5, 5), optimal_box_size, optimal_box_size+3, optimal_box_size + 5, optimal_box_size + 10, optimal_box_size + 15, optimal_box_size + 20]
    candidate_sizes = [ optimal_box_size, optimal_box_size+3]
    print(f"Candidate grid box sizes: {', '.join(f'{s:.2f}' for s in candidate_sizes)}")
    
    # 2. Protein Structure Download
    print("\n# 2. Protein Structure Download")
    pdb_file = download_pdb(pdb_id)
    
    # 3. Predict Binding Pockets
    print("\n# 3. Binding Pocket Prediction")
    prank_csv = predict_binding_pockets(pdb_file, prefix)
    
    # 4. Extract Pocket Coordinates
    print("\n# 4. Extract Pocket Information")
    center_x, center_y, center_z = extract_pocket_coordinates(prank_csv)
    
    # 5. Receptor Preparation
    print("\n# 5. Receptor Preparation")
    atoms_from_pdb = parsePDB(pdb_file)
    receptor_selection = f"protein and chain {chain} not water and not hetero"
    receptor_atoms = atoms_from_pdb.select(receptor_selection)
    prody_receptor_pdb = f"{prefix}{pdb_id}_receptor_atoms.pdb"
    writePDB(prody_receptor_pdb, receptor_atoms)
    
    reduce_input_pdb = f"{prefix}{pdb_id}_receptor.pdb"
    try:
        shell_command = f"cat <(grep 'CRYST1' '{pdb_file}') '{prody_receptor_pdb}' > '{reduce_input_pdb}'"
        subprocess.run(shell_command, shell=True, check=True, executable='/bin/bash', capture_output=True, text=True)
    except Exception as e:
        print(f"Error using process substitution: {e}")
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        cryst_line = next((line for line in lines if line.startswith("CRYST1")), "")
        with open(prody_receptor_pdb, 'r') as f:
            receptor_lines = f.readlines()
        with open(reduce_input_pdb, 'w') as f:
            if cryst_line:
                f.write(cryst_line)
            f.writelines(receptor_lines)
    
    reduce_opts = "approach=add add_flip_movers=True overwrite=True"
    env = os.environ.copy()
    env['MMTBX_CCP4_MONOMER_LIB'] = str(geostd_path)
    opts_list = reduce_opts.split()
    cmd_reduce = [sys.executable, str(reduce2_path), str(reduce_input_pdb)] + opts_list
    
    try:
        # Execute the command, checking for errors and capturing output
        completed_process = subprocess.run(
            cmd_reduce,
            env=env,
            check=True,           # Raise an exception if the command fails
            capture_output=True,  # Capture stdout and stderr
            text=True             # Decode stdout/stderr as text
        )
        # If the command succeeds, you can optionally print its output
        print("Command executed successfully:")
        if completed_process.stdout:
            print("STDOUT:\n", completed_process.stdout)
        if completed_process.stderr:
            print("STDERR:\n", completed_process.stderr)

    except subprocess.CalledProcessError as e:
        # If the command fails (non-zero exit status), catch the exception
        print(f"Command '{' '.join(e.cmd)}' failed with return code {e.returncode}")
        # Print the captured standard output (if any)
        if e.stdout:
            print("Captured STDOUT:\n", e.stdout)
        # Print the captured standard error (this usually contains the error message)
        if e.stderr:
            print("Captured STDERR (Error Details):\n", e.stderr)


    prepare_in_pdb = f"{prefix}{pdb_id}_receptorFH.pdb"
    
    best_score = None
    best_output = None
    intermediate_files = []
    
    cpus = multiprocessing.cpu_count()
    print(f"Using {cpus} CPU cores for Vina docking.")
    
    # Loop over each candidate grid box size and perform docking
    for i, candidate_size in enumerate(candidate_sizes):
        print(f"\n# 6. Docking with Candidate {i+1} (Box size = {candidate_size:.2f})")
        candidate_prefix = f"{prefix}{pdb_id}_receptorFH_candidate_{i+1}"
        candidate_receptor = f"{candidate_prefix}.pdbqt"
        candidate_config = f"{candidate_prefix}.box.txt"
        candidate_box_pdb = f"{candidate_prefix}.box.pdb"
        candidate_output = f"{prefix}{pdb_id}_{ligand_name}_vina_out_candidate_{i+1}.pdbqt"
        
        # Add files to the intermediate files list
        intermediate_files.extend([
            candidate_receptor, 
            candidate_config, 
            candidate_box_pdb, 
            candidate_output
        ])
        
        cmd_receptor = [
            mk_prepare_receptor,
            "-i", str(prepare_in_pdb),
            "-o", candidate_prefix,
            "--allow_bad_res",
            "-p",
            "-v",
            "--box_center", str(center_x), str(center_y), str(center_z),
            "--box_size", str(candidate_size), str(candidate_size), str(candidate_size)
        ]
        subprocess.run(cmd_receptor, check=True, capture_output=True, text=True)
        
        cmd_vina = [
            "vina",
            "--receptor", candidate_receptor,
            "--ligand", ligand_pdbqt,
            "--config", candidate_config,
            "--cpu", str(cpus),
            "--exhaustiveness", str(exhaustiveness),
            "--out", candidate_output
        ]
        subprocess.run(cmd_vina, check=True, capture_output=True, text=True)
        score = parse_vina_score(candidate_output)
        print(f"Candidate {i+1} docking score: {score}")
        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_output = candidate_output

    if best_output is not None:
        final_output = f"{pdb_id}_{ligand_name}_vina_best.pdbqt"
        run_command(f"cp {best_output} {final_output}")
        print(f"\nBest docking model saved as {final_output} with score {best_score}")
        
        # Analyze protein-ligand interactions
        print("\n# 7. Protein-Ligand Interaction Analysis")
        prepare_in_pdb = f"{prefix}{pdb_id}_receptorFH.pdb"  # This is the prepared protein with hydrogens
        interaction_analysis(prepare_in_pdb, final_output, prefix)
        
        # Remove all intermediate files
        removed_count = 0
        for file in intermediate_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
        
        print(f"Removed {removed_count} intermediate files.")
        return final_output

def main():
    parser = argparse.ArgumentParser(description='Molecular docking with binding pocket prediction and grid box optimization')
    parser.add_argument('--pdb', required=True, help='PDB ID for the protein')
    parser.add_argument('--smiles', required=True, help='SMILES string for the ligand')
    parser.add_argument('--ph', type=float, default=7.4, help='pH for ligand preparation')
    parser.add_argument('--exhaustiveness', type=int, default=128, help='Exhaustiveness for Vina docking')
    parser.add_argument('--prefix', default='', help='Prefix to add to all output files')
    parser.add_argument('--chain', default='A', help='Chain of protein to use for docking (default: %(default)s)')
    args = parser.parse_args()
    perform_docking(args.pdb, args.smiles, args.ph, args.exhaustiveness, args.prefix, args.chain)

if __name__ == "__main__":
    main()


