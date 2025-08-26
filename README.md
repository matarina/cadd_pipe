# Molecular Docking and Protein Structure Optimization

This repository contains three Python scripts for molecular docking and protein structure optimization:

- `molecular_docking.py`: For docking a single ligand to a protein.
- `batch_docking.py`: For docking multiple ligands in batch mode.
- `graft_protein_with_optimization.py`: For grafting a missing protein segment from an AlphaFold model into an experimental structure and optimizing the result.

These scripts leverage computational chemistry tools (e.g., AutoDock Vina, PRANK, Biopython, PyRosetta) to perform docking, predict binding pockets, optimize grid boxes, analyze interactions, and refine protein structures.

## Overview

### `molecular_docking.py`
Performs molecular docking for a single ligand (specified via a SMILES string) against a protein (provided as a PDB ID or local PDB file). It predicts binding pockets, optimizes the docking grid box, performs docking with AutoDock Vina, and analyzes interactions using ProLIF.

### `batch_docking.py`
Automates docking for multiple ligands by reading a batch file containing drug IDs and SMILES strings. It calls `molecular_docking.py` for each ligand, producing separate output directories for each docking run.

### `graft_protein_with_optimization.py`
Grafts a missing segment (e.g., a loop) from an AlphaFold model into an experimental protein structure with missing residues, using Biopython. Optionally optimizes the grafted structure using PyRosetta's FastRelax protocol to improve structural stability.

## Features

- **Protein Acquisition**: Downloads a protein structure from the RCSB PDB database or uses a local PDB file.
- **Binding Pocket Prediction**: Uses PRANK to predict potential binding pockets on the protein.
- **Ligand Preparation**: Processes SMILES strings to generate docking-ready ligand files (PDBQT format).
- **Grid Box Optimization**: Calculates an optimal grid box size based on the ligand's radius of gyration (2.9×Rg) and tests multiple box sizes and centers.
- **Docking**: Performs docking with AutoDock Vina using all available CPU cores for efficiency.
- **Interaction Analysis**: Analyzes protein-ligand interactions using ProLIF and generates visualizations (2D ligand network and 3D interaction plots).
- **Cleanup**: Retains only the best docking result and removes intermediate files.
- **Batch Processing**: Automates docking for multiple ligands via `batch_docking.py`.
- **Structure Grafting and Optimization**: Grafts missing protein segments from AlphaFold models and optimizes the structure using PyRosetta via `graft_protein_with_optimization.py`.

## Installation Pre-requisites

### For Conda Users
Install with `environment.yaml`:
```bash
conda env create -f environment.yaml
```

### For Pixi Users
Install with `pixi.toml`:
```bash
pixi shell -e easydocker
```

### Path Configuration
Specify the `geostd` paths in the `molecular_docking.py` file at line 51:
```python
self.geostd_path = Path("/data/geostd")
```
- Manually clone the `geostd` repository from [https://github.com/phenix-project/geostd.git](https://github.com/phenix-project/geostd.git) and set the path accordingly.

## Usage

### Single Docking with `molecular_docking.py`

#### Command-Line Syntax
```bash
python molecular_docking.py [OPTIONS]
```

#### Optional adjustable Parameters
Parameters not set via command line can be modified in the script (It is not recommended to make adjustments unless you know what you are doing!) :
- **Box Buffer**: 5 Å added to the protein-based box size.
- **Radius of Gyration (Rg) Scale Factors**: [0.8, 1.0, 1.2, 1.5] for ligand-based box sizes.
- **Protein Scale Factors**: [0.75, 1.0] for protein-based box sizes.

#### Options
- `--pdb <PDB_ID>`: PDB ID to download from RCSB (e.g., `11GS`). Mutually exclusive with `--local_pdb`.
- `--local_pdb <PATH>`: Path to a local PDB file (e.g., `11GS.pdb`). Mutually exclusive with `--pdb`. (For some complex proteins that fail to process automatically with `--pdb`, manual preprocessing and saving as a local PDB file may be required.)
- `--smiles <SMILES>`: SMILES string of the ligand (required).
- `--ph <FLOAT>`: pH for ligand preparation (default: 7.4).
- `--exhaustiveness <INT>`: Exhaustiveness for Vina docking (default: 128).
- `--prefix <DIR>`: Output directory name.
- `--chain <CHAR>`: Protein chain to use (default: `A`).

#### Example Commands
1. **Docking with a PDB ID**:
   ```bash
   python molecular_docking.py --pdb 1EOU --smiles "CC(=O)Nc1ccc(cc1)S(=O)(=O)N" --prefix output_1eou --exhaustiveness 64
   ```
   - Downloads `1EOU.pdb` from RCSB.
   - Outputs to `output_1eou`.
   - Uses exhaustiveness of 64.

2. **Docking with a Local PDB File**:
   ```bash
   python molecular_docking.py --local_pdb 11GS.pdb --smiles "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2C(=O)NC4=C(C=CC(=C4)C(F)(F)F)C(F)(F)F)CC[C@@H]5[C@@]3(C=CC(=O)N5)C" --prefix test --chain A
   ```
   - Uses `11GS.pdb` from the current directory.
   - Outputs to the `test` directory.
   - Docks to chain A.


**Files in `test`**:
```
11GS_ligand_vina_best.pdbqt
protein_ligand_interactions.csv
ligand_network.html
protein_ligand_3d.html
```

### Batch Docking with `batch_docking.py`

#### Purpose
Automates docking for multiple ligands by reading a batch file with drug IDs and SMILES strings, calling `molecular_docking.py` for each ligand.

#### Command-Line Syntax
```bash
python batch_docking.py [OPTIONS]
```

#### Options
- `--pdb <PDB_ID>`: PDB ID to download from RCSB (e.g., `11GS`). Mutually exclusive with `--local_pdb`.
- `--local_pdb <PATH>`: Path to a local PDB file (e.g., `11GS.pdb`). Mutually exclusive with `--pdb`.
- `--batch_file <PATH>`: Path to a text file with drug IDs and SMILES strings (required). Example file: `batch_smiles.txt`.
- `--chain <CHAR>`: Protein chain to use (required).

#### Batch File Format
A text file where each line contains:
- A drug ID/name (no spaces).
- A space or tab separator.
- A valid SMILES string.

**Example** (`batch_smiles.txt`):
```
Drug1 C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2C(=O)NC4=C(C=CC(=C4)C(F)(F)F)C(F)(F)F)CC[C@@H]5[C@@]3(C=CC(=O)N5)C
Drug2 CC(=O)Nc1ccc(cc1)S(=O)(=O)N
Drug3 c1cc(c(c(c1)Cl)NC(=O)C)Cl
```

#### Example Commands
1. **Batch Docking with a Local PDB**:
   ```bash
   python batch_docking.py --local_pdb 11GS.pdb --batch_file batch_smiles.txt --chain A
   ```


**Directory Structure**:
```
Drug1/
  11GS_ligand_vina_best.pdbqt
  protein_ligand_interactions.csv
  ligand_network.html
  protein_ligand_3d.html
Drug2/
  11GS_ligand_vina_best.pdbqt
  protein_ligand_interactions.csv
  ligand_network.html
  protein_ligand_3d.html
Drug3/
  11GS_ligand_vina_best.pdbqt
  protein_ligand_interactions.csv
  ligand_network.html
  protein_ligand_3d.html
```

### Protein Structure Grafting and Optimization with `graft_protein_with_optimization.py`

#### Purpose
Grafts a missing segment (e.g., a loop) from an AlphaFold model into an experimental protein structure with missing residues, using Biopython. Optionally optimizes the grafted structure using PyRosetta's FastRelax protocol to improve structural stability and energy.

#### Command-Line Syntax
```bash
python graft_protein_with_optimization.py <exp_pdb> <af_pdb> [chain_id] [missing_start] [missing_end] [output_pdb]
```

#### Arguments
- `<exp_pdb>`: Path to the experimental PDB file with missing residues (required).
- `<af_pdb>`: Path to the AlphaFold PDB file with the complete structure (required).
- `[chain_id]`: Chain ID where grafting occurs (default: `A`).
- `[missing_start]`: Starting residue index of the missing segment (default: 48).
- `[missing_end]`: Ending residue index of the missing segment (default: 62).
- `[output_pdb]`: Output PDB file name (default: `grafted_model.pdb`).

#### Additional Notes
- The script always performs grafting and saves the grafted structure to `<output_pdb>`.
- Optimization is enabled by default (`optimize=True`) using PyRosetta's `ref2015_cart` score function with 5 cycles.
- The optimized structure is saved to a file with `_optimized` appended to the base name (e.g., `grafted_model_optimized.pdb`) unless overridden by the `opt_output_pdb` parameter in the script.

#### Example Commands
1. **Grafting and Optimization with a Missing Loop**:
   ```bash
   python graft_protein_with_optimization.py experimental.pdb alphafold.pdb B 77 110 grafted_model.pdb
   ```


**Files**:
```
grafted_model.pdb
grafted_model_optimized.pdb
```

## Workflow Pipelines

### `molecular_docking.py`
1. **Initialization**: Sets up parameters (PDB ID or local PDB, SMILES, chain, etc.) and creates an output directory if specified.
2. **Ligand Preparation**:
   - Converts SMILES to SDF using `scrub.py`.
   - Generates PDBQT file using `mk_prepare_ligand.py`.
3. **Protein Acquisition**:
   - Downloads PDB from RCSB (if `--pdb`) or copies local PDB (if `--local_pdb`).
4. **Binding Pocket Prediction**:
   - Runs PRANK to predict binding pockets.
   - Extracts top pocket coordinates.
5. **Grid Box Optimization**:
   - Calculates ligand radius of gyration (Rg) to set base box size (2.9×Rg).
   - Tests multiple box sizes (Rg-scaled and protein-scaled) and centers (PRANK and protein centroid).
6. **Receptor Preparation**:
   - Selects specified chain, removes water/heteroatoms.
   - Adds hydrogens using `reduce2.py`.
7. **Docking**:
   - Performs AutoDock Vina docking for each box configuration using all CPU cores.
   - Selects the best docking pose based on the lowest energy score.
8. **Interaction Analysis**:
   - Analyzes protein-ligand interactions using ProLIF.
   - Generates CSV (interaction details) and HTML files (2D/3D visualizations).
9. **Cleanup**:
   - Retains best docking result (`<PDB_ID>_ligand_vina_best.pdbqt`).
   - Removes intermediate files.

### `batch_docking.py`
1. **Initialization**: Parses command-line arguments (PDB ID or local PDB, batch file, chain).
2. **Batch File Parsing**:
   - Reads drug IDs and SMILES strings from the batch file.
3. **Iterative Docking**:
   - For each ligand, constructs a command to run `molecular_docking.py` with:
     - Specified protein (PDB ID or local file).
     - SMILES string.
     - Chain ID.
     - Output directory named after the drug ID.
   - Executes each docking run sequentially.
4. **Error Handling**:
   - Captures errors for individual docking runs and continues with remaining ligands.
5. **Output**:
   - Creates a separate directory per drug ID, each containing the outputs of `molecular_docking.py`.

### `graft_protein_with_optimization.py`
1. **Initialization**: Parses command-line arguments (experimental PDB, AlphaFold PDB, chain, residue range, output file).
2. **Structure Parsing**:
   - Loads experimental and AlphaFold PDB files using Biopython's `PDBParser`.
3. **Chain Selection**:
   - Identifies the specified chain in both structures.
4. **Residue Verification**:
   - Checks which residues in the specified range are missing in the experimental structure.
   - Confirms their presence in the AlphaFold structure.
5. **Grafting**:
   - Removes any existing residues in the missing range (if any).
   - Copies missing residues from the AlphaFold model to the experimental structure.
   - Sorts residues to maintain correct order.
6. **Saving Grafted Structure**:
   - Saves the grafted structure to the specified output file (e.g., `grafted_model.pdb`).
7. **Optimization** (if enabled):
   - Loads the grafted structure into PyRosetta.
   - Applies FastRelax protocol with `ref2015_cart` score function for 5 cycles.
   - Provides grafted and optimized PDB files, with console output showing initial and final PyRosetta scores.
