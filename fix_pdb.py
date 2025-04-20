# fix_pdb.py
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input PDB file")
parser.add_argument("-o", "--output", required=True, help="Output fixed PDB file")
parser.add_argument("--ph", type=float, default=7.0, help="pH for adding hydrogens")
args = parser.parse_args()

print(f"Reading {args.input}")
# Initialize PDBFixer. Use PDB file, not PDBx/mmCIF, if possible.
fixer = PDBFixer(filename=args.input)

print("Finding missing residues...")
fixer.findMissingResidues() # Finds chain breaks

print("Finding nonstandard residues...")
fixer.findNonstandardResidues() # Identifies non-standard AA, ligands etc.

print("Replacing nonstandard residues (if applicable)...")
# This replaces common variants like HSE/HSD with HIS based on guessed protonation
fixer.replaceNonstandardResidues()

print("Removing heterogens (excluding water for now)...")
# Set keepWater=False if you intend to add solvent later anyway
# Set keepWater=True if your input PDB has water you want to keep initially
fixer.removeHeterogens(keepWater=False)

print("Finding missing atoms...")
fixer.findMissingAtoms() # Finds missing atoms within standard residues

print("Adding missing atoms...")
fixer.addMissingAtoms() # Adds them based on residue templates

print(f"Adding missing hydrogens at pH {args.ph}...")
# This is crucial for correct protonation states (including terminals)
fixer.addMissingHydrogens(args.ph)

print(f"Writing fixed PDB to {args.output}")
# Use keepIds=True to preserve original atom/residue numbering where possible
with open(args.output, 'w') as outfile:
    PDBFile.writeFile(fixer.topology, fixer.positions, outfile, keepIds=True)

print("PDB fixing complete.")

