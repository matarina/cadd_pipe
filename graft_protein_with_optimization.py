#!/usr/bin/env python
import sys
import os
from Bio.PDB import PDBParser, PDBIO
from copy import deepcopy

# Import PyRosetta
import pyrosetta
pyrosetta.init(silent=True)  # Initialize PyRosetta

def optimize_with_pyrosetta(input_pdb, output_pdb, scorefxn_name='ref2015_cart', cycles=5):
    """
    Optimizes protein structure using PyRosetta's FastRelax protocol.
    
    Parameters:
    -----------
    input_pdb : str
        Path to the input PDB file to optimize
    output_pdb : str
        Path to save the optimized structure
    scorefxn_name : str
        Name of the score function to use (default: 'ref2015_cart')
    cycles : int
        Number of relax cycles to perform (default: 5)
    
    Returns:
    --------
    bool
        True if optimization was successful, False otherwise
    """
    try:
        print(f"Starting structure optimization with PyRosetta...")
        print(f"  Input PDB: {input_pdb}")
        print(f"  Output PDB: {output_pdb}")
        print(f"  Score function: {scorefxn_name}")
        print(f"  Cycles: {cycles}")
        print("=" * 50)
        
        # Load the PDB file into a Rosetta pose
        pose = pyrosetta.pose_from_pdb(input_pdb)
        print(f"Loaded structure with {pose.total_residue()} residues")
        
        # Create score function
        scorefxn = pyrosetta.create_score_function(scorefxn_name)
        print(f"Created score function: {scorefxn_name}")
        
        # Calculate initial score
        initial_score = scorefxn(pose)
        print(f"Initial score: {initial_score:.2f}")
        
        # Setup FastRelax mover
        relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn, cycles)
        relax.cartesian(True)  # Use cartesian space minimization
        relax.minimize_bond_angles(True)
        relax.minimize_bond_lengths(True)
        print(f"FastRelax mover configured with {cycles} cycles")
        
        # Apply relaxation
        print("Applying FastRelax protocol...")
        relax.apply(pose)
        
        # Calculate final score
        final_score = scorefxn(pose)
        print(f"Final score: {final_score:.2f}")
        print(f"Score improvement: {initial_score - final_score:.2f}")
        
        # Save the optimized structure
        pose.dump_pdb(output_pdb)
        print(f"Optimized structure saved to {output_pdb}")
        
        return True
        
    except Exception as e:
        print(f"Error during PyRosetta optimization: {e}")
        return False

def graft_missing_segment(exp_pdb_path, af_pdb_path, chain_id='B', 
                          missing_start=77, missing_end=110, output_pdb="grafted_model.pdb",
                          optimize=True, opt_output_pdb=None, 
                          scorefxn_name='ref2015_cart', cycles=5):
    """
    Grafts a segment from an AlphaFold model into an experimental structure with missing residues.
    Optionally optimizes the resulting structure using PyRosetta.
    
    Parameters:
    -----------
    exp_pdb_path : str
        Path to the experimental PDB file with missing residues
    af_pdb_path : str
        Path to the AlphaFold PDB file with complete structure
    chain_id : str
        Chain ID where grafting will occur (default: 'B')
    missing_start : int
        Starting residue index of the missing segment
    missing_end : int
        Ending residue index of the missing segment
    output_pdb : str
        Name of the output PDB file
    optimize : bool
        Whether to perform structure optimization after grafting
    opt_output_pdb : str or None
        Name of the optimized output PDB file. If None, will use output_pdb with "_optimized" suffix
    scorefxn_name : str
        Name of the score function to use for PyRosetta optimization
    cycles : int
        Number of relax cycles for PyRosetta optimization
    """
    print(f"Starting Biopython grafting process...")
    print(f"  Experimental PDB: {exp_pdb_path}")
    print(f"  AlphaFold PDB: {af_pdb_path}")
    print(f"  Chain ID: {chain_id}")
    print(f"  Missing region: {missing_start}-{missing_end}")
    print(f"  Output file: {output_pdb}")
    print("=" * 50)
    
    # Parse structures
    parser = PDBParser(QUIET=True)
    print(f"Parsing experimental structure: {exp_pdb_path}")
    exp_structure = parser.get_structure("exp", exp_pdb_path)
    
    print(f"Parsing AlphaFold structure: {af_pdb_path}")
    af_structure = parser.get_structure("af", af_pdb_path)
    
    # Get chains
    exp_chain = None
    af_chain = None
    
    for chain in exp_structure[0]:
        if chain.id == chain_id:
            exp_chain = chain
            break
    
    for chain in af_structure[0]:
        if chain.id == chain_id:
            af_chain = chain
            break
    
    if exp_chain is None:
        raise ValueError(f"Chain {chain_id} not found in experimental structure")
    
    if af_chain is None:
        raise ValueError(f"Chain {chain_id} not found in AlphaFold structure")
    
    print(f"Found chain {chain_id} in both structures")
    
    # Create a new structure for the output
    print("Creating new structure for grafting...")
    new_structure = deepcopy(exp_structure)
    new_chain = None
    
    for chain in new_structure[0]:
        if chain.id == chain_id:
            new_chain = chain
            break
    
    # Check existing residues in experimental structure
    exp_res_ids = [res.id[1] for res in exp_chain]
    missing_res_ids = list(range(missing_start, missing_end + 1))
    
    print(f"Checking for residues {missing_start}-{missing_end} in experimental structure...")
    actually_missing = [res_id for res_id in missing_res_ids if res_id not in exp_res_ids]
    
    if len(actually_missing) != len(missing_res_ids):
        print(f"Warning: Some residues in range {missing_start}-{missing_end} already exist in experimental structure")
        print(f"Actually missing: {actually_missing}")
    
    if not actually_missing:
        print("No residues are actually missing. Nothing to graft.")
        return False
    
    # Check if the missing residues exist in the AlphaFold structure
    af_res_ids = [res.id[1] for res in af_chain]
    missing_in_af = [res_id for res_id in actually_missing if res_id not in af_res_ids]
    
    if missing_in_af:
        print(f"Warning: Some missing residues are not found in AlphaFold model: {missing_in_af}")
        print("Will only graft residues that exist in AlphaFold model")
    
    # Remove any existing residues in the missing range (shouldn't be any, but just in case)
    to_delete = []
    for residue in new_chain:
        if missing_start <= residue.id[1] <= missing_end:
            to_delete.append(residue.id)
    
    if to_delete:
        print(f"Removing {len(to_delete)} residues from experimental structure that fall within missing range")
        for res_id in to_delete:
            new_chain.detach_child(res_id)
    
    # Add residues from AlphaFold model
    print(f"Grafting residues from AlphaFold model...")
    added_count = 0
    
    # Get all the residues we want to add from AlphaFold
    residues_to_add = []
    for residue in af_chain:
        if missing_start <= residue.id[1] <= missing_end:
            residues_to_add.append(deepcopy(residue))
    
    # Sort residues by ID to ensure correct order
    residues_to_add.sort(key=lambda r: r.id[1])
    
    # Add residues to the chain
    for residue in residues_to_add:
        try:
            new_chain.add(residue)
            added_count += 1
        except Exception as e:
            print(f"Warning: Could not add residue {residue.id[1]}: {e}")
    
    print(f"Successfully added {added_count} residues from AlphaFold model")
    
    # Sort all residues by ID to ensure correct order
    new_chain.child_list.sort(key=lambda r: r.id[1])
    
    # Save the new structure
    print(f"Saving grafted structure to {output_pdb}")
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)

    print(f"Grafting complete! AlphaFold segment ({missing_start}-{missing_end}) successfully added to experimental structure")
    print(f"Output saved to {output_pdb}")
    
    # Optimize the structure if requested
    if optimize:
        # Determine output path for optimized structure
        if opt_output_pdb is None:
            # Generate a default name by inserting "_optimized" before the file extension
            base, ext = os.path.splitext(output_pdb)
            opt_output_pdb = f"{base}_optimized{ext}"
        
        # Run the PyRosetta optimization
        if optimize_with_pyrosetta(output_pdb, opt_output_pdb, 
                                  scorefxn_name=scorefxn_name, 
                                  cycles=cycles):
            print(f"Structure optimization complete!")
            print(f"Optimized structure saved to {opt_output_pdb}")
        else:
            print(f"Warning: Structure optimization failed.")
    
    return True

if __name__ == "__main__":
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        exp_pdb_path = sys.argv[1]
        af_pdb_path = sys.argv[2]
        chain_id = sys.argv[3] if len(sys.argv) > 3 else 'A'
        missing_start = int(sys.argv[4]) if len(sys.argv) > 4 else 77
        missing_end = int(sys.argv[5]) if len(sys.argv) > 5 else 110
        output_pdb = sys.argv[6] if len(sys.argv) > 6 else "grafted_model.pdb"
    else:
        # Default values
        exp_pdb_path = "exp.pdb"
        af_pdb_path = "alpha.pdb"
        chain_id = 'A'
        missing_start = 48
        missing_end = 62
        output_pdb = "grafted_model.pdb"
    
    try:
        graft_missing_segment(
            exp_pdb_path=exp_pdb_path,
            af_pdb_path=af_pdb_path,
            chain_id=chain_id,
            missing_start=missing_start,
            missing_end=missing_end,
            output_pdb=output_pdb,
            optimize=True,
            opt_output_pdb=None,
            scorefxn_name='ref2015_cart',
            cycles=5
        )
    except Exception as e:
        print(f"ERROR: Grafting failed: {e}")
        sys.exit(1)

