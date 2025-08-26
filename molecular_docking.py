#!/usr/bin/env python

"""
Molecular Docking Script with Binding Pocket Prediction and Grid Box Optimization

This script uses a class-based approach to perform molecular docking. It downloads a protein PDB file,
predicts binding pockets using PRANK, prepares the ligand, computes its radius-of-gyration to set a
base grid box size (2.9×Rg), performs docking with AutoDock Vina using all available CPU cores, and
retains only the best docking output while removing intermediate files.

Adjustments:
- box_buffer=5
- rg_scale_factors=[0.8, 1.0, 1.2, 1.5]
- protein_scale_factors=[0.75, 1.0]
- exhaustiveness=128
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
import numpy as np
import Bio.PDB
from prody import *
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
import shutil

class MolecularDocker:
    """Class to handle molecular docking with binding pocket prediction and grid box optimization."""
    
    def __init__(self, pdb_id=None, ligand_smiles=None, ph=7.4, exhaustiveness=128, prefix='', chain='A', local_pdb=None):
        """Initialize the docking parameters and paths."""
        self.pdb_id = pdb_id
        self.ligand_smiles = ligand_smiles
        self.ph = ph
        self.exhaustiveness = exhaustiveness
        self.prefix = prefix
        self.chain = chain
        self.original_dir = os.getcwd()
        # Convert local_pdb to absolute path if provided
        self.local_pdb = os.path.abspath(local_pdb) if local_pdb else None
        self.reduce2_path = Path("/data/opus/.pixi/envs/easydocker/lib/python3.10/site-packages/mmtbx/command_line/reduce2.py")
        self.geostd_path = Path("/data/opus/dock/geostd")
        self.rg_scale_factors = [0.8, 1.0, 1.2, 1.5]
        self.protein_scale_factors = [0.75, 1.0]
        self.box_buffer = 5
        self.intermediate_files = []
        
        # Setup output directory
        if self.prefix:
            os.makedirs(self.prefix, exist_ok=True)
            os.chdir(self.prefix)
            print(f"All output files will be saved to directory: {self.prefix}")

    def run_command(self, command):
        """Execute a shell command and print stdout/stderr."""
        print(f"Executing: {command}")
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if process.stdout:
            print("STDOUT:\n", process.stdout)
        if process.stderr:
            print("STDERR:\n", process.stderr)
        print("-" * 20)

    def download_pdb(self):
        """Download PDB file from RCSB PDB database."""
        pdb_file = f"{self.pdb_id}.pdb"
        if os.path.exists(pdb_file):
            print(f"PDB file {pdb_file} already exists.")
            return pdb_file
        try:
            import requests
            print(f"Downloading {self.pdb_id}.pdb from RCSB...")
            response = requests.get(f"http://files.rcsb.org/view/{self.pdb_id}.pdb")
            response.raise_for_status()
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            print("Download complete.")
        except (ImportError, requests.exceptions.RequestException) as e:
            print(f"Error with requests: {e}")
            print("Attempting download with curl...")
            cmd_curl = f"curl \"http://files.rcsb.org/view/{self.pdb_id}.pdb\" -o \"{pdb_file}\""
            self.run_command(cmd_curl)
        return pdb_file

    def predict_binding_pockets(self, pdb_file):
        """Predict binding pockets using PRANK."""
        output_dir = "prank_output"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(pdb_file)
        output_prefix = os.path.join(output_dir, os.path.splitext(base_name)[0])
        cmd = f"prank predict -f {pdb_file} -o {output_prefix}"
        self.run_command(cmd)
        prank_csv = os.path.join(output_dir, os.path.splitext(base_name)[0], f"{base_name}_predictions.csv")
        if not os.path.exists(prank_csv):
            alt_prank_csv = os.path.join(output_dir, os.path.splitext(base_name)[0], 
                                       f"{os.path.splitext(base_name)[0]}_predictions.csv")
            if os.path.exists(alt_prank_csv):
                prank_csv = alt_prank_csv
                print(f"Found alternative PRANK output at: {prank_csv}")
            else:
                sys.exit(1)
        print(f"Using PRANK predictions from: {prank_csv}")
        return prank_csv

    def extract_pocket_coordinates(self, prank_csv):
        """Extract top pocket coordinates from PRANK output CSV."""
        try:
            df = pd.read_csv(prank_csv)
            df.columns = df.columns.str.strip()
            top_pocket = df.iloc[0]
            center = (float(top_pocket['center_x']), float(top_pocket['center_y']), float(top_pocket['center_z']))
            print(f"Selected binding pocket center: {center}")
            return center
        except Exception as e:
            print(f"Error parsing PRANK output: {e}")
            pdb_file = f"{self.pdb_id}.pdb"
            atoms_from_pdb = parsePDB(pdb_file)
            protein_atoms = atoms_from_pdb.select('protein')
            center = calcCenter(protein_atoms)
            print(f"Using protein centroid as fallback: {center}")
            return center

    def calculate_optimal_box_size(self, sdf_file):
        """Calculate ligand's radius of gyration and determine base grid box size (2.9 × Rg)."""
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

    def calculate_docking_box(self, pdb_file):
        """Calculate center and size of bounding box for all atoms in PDB structure."""
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{pdb_file}'")
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        coords = [atom.coord for model in structure for chain in model for residue in chain for atom in residue]
        if not coords:
            raise ValueError(f"No atomic coordinates found in PDB file: {pdb_file}")
        coords = np.array(coords)
        min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
        center = (max_coords + min_coords) / 2.0
        size = max_coords - min_coords + self.box_buffer
        return center, size

    def interaction_analysis(self, protein_pdb, ligand_pdbqt_output):
        """Analyze protein-ligand interactions using prolif."""
        u = mda.Universe(protein_pdb)
        protein_mol = plf.Molecule.from_mda(u)
        ligand_sdf = "ligand_best_pose.sdf"
        cmd = f"mk_export.py {ligand_pdbqt_output} -s {ligand_sdf}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        ligand_mol = plf.sdf_supplier(ligand_sdf)[0]
        fp = plf.Fingerprint()
        fp.run_from_iterable([ligand_mol], protein_mol)
        interaction_details = []
        for residue_pair in fp.ifp[0].keys():
            ligand_res, protein_res = residue_pair
            for interaction_type, details_tuple in fp.ifp[0][residue_pair].items():
                for details in details_tuple:
                    record = {'frame': 0, 'ligand': ligand_res, 'protein': protein_res, 'interaction': interaction_type}
                    record.update({k: v for k, v in details.items() if k not in ['indices', 'parent_indices']})
                    interaction_details.append(record)
        result_df = pd.DataFrame(interaction_details)
        result_df.to_csv('protein_ligand_interactions.csv', index=False)
        print(result_df.head())
        html_output = fp.plot_lignetwork(ligand_mol, kind="frame", frame=0, width='1800px', height='1000px')
        with open("ligand_network.html", "w") as f:
            f.write(html_output.data)
        view = fp.plot_3d(ligand_mol, protein_mol, frame=0, display_all=False, size=(1800, 1000))
        view.write_html("protein_ligand_3d.html")
        print("Visualization files saved!")
        return ligand_sdf

    def parse_vina_score(self, output_file):
        """Parse Vina docking output to extract best energy score."""
        best_score = None
        try:
            with open(output_file) as f:
                for line in f:
                    if line.startswith("REMARK VINA RESULT:"):
                        score = float(line.strip().split()[3])
                        if best_score is None or score < best_score:
                            best_score = score
        except Exception as e:
            print(f"Error parsing Vina score from {output_file}: {e}")
        return best_score

    def prepare_ligand(self):
        """Prepare ligand from SMILES string."""
        print("\n# 1. Ligand Preparation")
        ligand_name = "ligand"
        ligand_pdbqt = f"{ligand_name}.pdbqt"
        ligand_sdf = f"{ligand_name}_scrubbed.sdf"
        cmd_scrub = f"scrub.py \"{self.ligand_smiles}\" -o {ligand_sdf} --ph {self.ph} --skip_tautomer"
        cmd_prep_lig = f"mk_prepare_ligand.py -i {ligand_sdf} -o {ligand_pdbqt}"
        self.run_command(cmd_scrub)
        self.run_command(cmd_prep_lig)
        return ligand_pdbqt, ligand_sdf

    def prepare_receptor(self, pdb_file):
        """Prepare receptor from PDB file."""
        print("\n# 7. Receptor Preparation")
        atoms_from_pdb = parsePDB(pdb_file)
        receptor_selection = f"protein and chain {self.chain} not water and not hetero"
        receptor_atoms = atoms_from_pdb.select(receptor_selection)
        prody_receptor_pdb = f"{self.pdb_id}_receptor_atoms.pdb"
        writePDB(prody_receptor_pdb, receptor_atoms)
        reduce_input_pdb = f"{self.pdb_id}_receptor.pdb"
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
        env['MMTBX_CCP4_MONOMER_LIB'] = str(self.geostd_path)
        cmd_reduce = [sys.executable, str(self.reduce2_path), str(reduce_input_pdb)] + reduce_opts.split()
        subprocess.run(cmd_reduce, env=env, check=True, capture_output=True, text=True)
        return f"{self.pdb_id}_receptorFH.pdb"

    def perform_docking(self):
        """Perform the complete docking workflow."""
        try:
            # Step 1: Ligand Preparation
            ligand_pdbqt, ligand_sdf = self.prepare_ligand()

            # Step 2: Protein Structure Acquisition
            print("\n# 2. Protein Structure Acquisition")
            if self.local_pdb:
                local_pdb_basename = os.path.basename(self.local_pdb)
                pdb_file = local_pdb_basename
                # Copy the local PDB file to the current directory (baiyp) if it doesn't exist
                if not os.path.exists(pdb_file):
                    if not os.path.exists(self.local_pdb):
                        raise FileNotFoundError(f"Local PDB file not found: {self.local_pdb}")
                    shutil.copy2(self.local_pdb, pdb_file)
                print(f"Using local PDB file: {pdb_file}")
                if self.pdb_id is None:
                    self.pdb_id = os.path.splitext(local_pdb_basename)[0]
            else:
                pdb_file = self.download_pdb()

            # Step 3: Calculate Optimal Box Size
            print("\n# 3. Calculate optimal box size from ligand radius of gyration")
            optimal_box_size = self.calculate_optimal_box_size(ligand_sdf)
            rg_scaled_sizes = [optimal_box_size * factor for factor in self.rg_scale_factors]
            print(f"RG-scaled box sizes: {', '.join(f'{s:.2f}' for s in rg_scaled_sizes)}")

            # Step 4 & 5: Binding Pocket Prediction and Coordinate Extraction
            print("\n# 4. Binding Pocket Prediction using PRANK")
            prank_csv = self.predict_binding_pockets(pdb_file)
            print("\n# 5. Extract Pocket Information from PRANK")
            prank_center = self.extract_pocket_coordinates(prank_csv)

            # Step 6: Calculate Protein Center and Box Size
            print("\n# 6. Calculate protein center and box size")
            protein_center, protein_box_size = self.calculate_docking_box(pdb_file)
            print(f"Protein-based box center: ({protein_center[0]:.2f}, {protein_center[1]:.2f}, {protein_center[2]:.2f})")
            print(f"Protein-based box size: ({protein_box_size[0]:.2f}, {protein_box_size[1]:.2f}, {protein_box_size[2]:.2f})")

            # Step 7: Receptor Preparation
            prepare_in_pdb = self.prepare_receptor(pdb_file)

            # Step 8 & 9: Docking with PRANK and Protein Centers
            best_score, best_output = None, None
            cpus = multiprocessing.cpu_count()
            print(f"Using {cpus} CPU cores for Vina docking.")
            candidate_counter = 1

            # Docking with PRANK center and RG-scaled sizes
            for i, size in enumerate(rg_scaled_sizes):
                print(f"\n# 8. Docking with Candidate {candidate_counter} (Center: PRANK, Size: RG-scaled-{self.rg_scale_factors[i]})")
                candidate_prefix = f"{self.pdb_id}_receptorFH_candidate_{candidate_counter}"
                candidate_files = self._dock_candidate(prepare_in_pdb, ligand_pdbqt, candidate_prefix, 
                                                    prank_center, [size] * 3, cpus)
                self.intermediate_files.extend(candidate_files)
                score = self.parse_vina_score(candidate_files[-1])
                print(f"Candidate {candidate_counter} docking score: {score}")
                if score is not None and (best_score is None or score < best_score):
                    best_score, best_output = score, candidate_files[-1]
                candidate_counter += 1

            # Docking with protein center and protein-scaled sizes
            for i, factor in enumerate(self.protein_scale_factors):
                print(f"\n# 9. Docking with Candidate {candidate_counter} (Center: Protein, Size: Protein-scaled-{factor})")
                candidate_prefix = f"{self.pdb_id}_receptorFH_candidate_{candidate_counter}"
                scaled_size = [dim * factor for dim in protein_box_size]
                candidate_files = self._dock_candidate(prepare_in_pdb, ligand_pdbqt, candidate_prefix, 
                                                    protein_center, scaled_size, cpus)
                self.intermediate_files.extend(candidate_files)
                score = self.parse_vina_score(candidate_files[-1])
                print(f"Candidate {candidate_counter} docking score: {score}")
                if score is not None and (best_score is None or score < best_score):
                    best_score, best_output = score, candidate_files[-1]
                candidate_counter += 1

            # Step 10: Process Best Result
            if best_output:
                final_output = f"{self.pdb_id}_ligand_vina_best.pdbqt"
                self.run_command(f"cp {best_output} {final_output}")
                print(f"\nBest docking model saved as {final_output} with score {best_score}")
                
                # Interaction Analysis
                print("\n# 10. Protein-Ligand Interaction Analysis")
                self.interaction_analysis(prepare_in_pdb, final_output)
                
                # Cleanup
                removed_count = sum(1 for file in self.intermediate_files if os.path.exists(file) and not os.remove(file))
                print(f"Removed {removed_count} intermediate files.")
                
                return os.path.join(self.prefix, final_output) if self.prefix else final_output
            else:
                print("Error: No successful docking results found.")
                return None
        finally:
            if self.prefix:
                os.chdir(self.original_dir)

    def _dock_candidate(self, receptor_pdb, ligand_pdbqt, prefix, center, size, cpus):
        """Helper method to dock a single candidate."""
        receptor = f"{prefix}.pdbqt"
        config = f"{prefix}.box.txt"
        box_pdb = f"{prefix}.box.pdb"
        output = f"{self.pdb_id}_ligand_vina_out_candidate_{prefix.split('_')[-1]}.pdbqt"
        
        cmd_receptor = [
            "mk_prepare_receptor.py",
            "-i", str(receptor_pdb),
            "-o", prefix,
            "--allow_bad_res",
            "-p",
            "-v",
            "--box_center", str(center[0]), str(center[1]), str(center[2]),
            "--box_size", str(size[0]), str(size[1]), str(size[2])
        ]
        subprocess.run(cmd_receptor, check=True, capture_output=True, text=True)
        
        cmd_vina = [
            "vina",
            "--receptor", receptor,
            "--ligand", ligand_pdbqt,
            "--config", config,
            "--cpu", str(cpus),
            "--exhaustiveness", str(self.exhaustiveness),
            "--seed", "42",
            "--out", output
        ]
        subprocess.run(cmd_vina, check=True, capture_output=True, text=True)
        
        return [receptor, config, box_pdb, output]

def main():
    parser = argparse.ArgumentParser(description='Molecular docking with binding pocket prediction and grid box optimization')
    pdb_group = parser.add_mutually_exclusive_group(required=True)
    pdb_group.add_argument('--pdb', help='PDB ID for the protein')
    pdb_group.add_argument('--local_pdb', help='Path to local protein PDB file')
    parser.add_argument('--smiles', required=True, help='SMILES string for the ligand')
    parser.add_argument('--ph', type=float, default=7.4, help='pH for ligand preparation')
    parser.add_argument('--exhaustiveness', type=int, default=128, help='Exhaustiveness for Vina docking')
    parser.add_argument('--prefix', default='', help='Name for output folder where all files will be saved')
    parser.add_argument('--chain', default='A', help='Chain of protein to use for docking (default: %(default)s)')
    
    args = parser.parse_args()
    
    docker = MolecularDocker(
        pdb_id=args.pdb,
        ligand_smiles=args.smiles,
        ph=args.ph,
        exhaustiveness=args.exhaustiveness,
        prefix=args.prefix,
        chain=args.chain,
        local_pdb=args.local_pdb
    )
    docker.perform_docking()

if __name__ == "__main__":
    main()



