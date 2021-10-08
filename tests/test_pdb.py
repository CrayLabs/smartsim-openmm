import parmed as pmd
import MDAnalysis as mda 

import MDAnalysis.coordinates as MDCoords

from MDAnalysis.lib.util import NamedStream

import io, os


filename = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'MD_exps', 'fs-pep', 'pdb', '100-fs-peptide-400K.pdb')
traj_file = ''  # put a valid dcd file here


pdb_from_file = pmd.read_PDB(filename)
mda_traj_from_file = mda.Universe(topology=filename, coordinates=traj_file, format='PDB')
mda_traj_from_file.trajectory[1]
PDB_file = mda.Writer('test.pdb')
PDB_file.write(mda_traj_from_file.atoms)
PDB_file.close()


# Avoid files
with open('test.pdb') as output_file:
    output_file.seek(0)
    pdb_file_lines = output_file.readlines()

with open(filename, 'r') as pdb_file:
    pdb_lines = pdb_file.readlines()

pdb_from_lines = pmd.read_PDB(pdb_lines)
pdb_stream = io.StringIO("\n".join(pdb_lines))
pseudo_pdb = NamedStream(pdb_stream, 'pseudo.pdb')
mda_traj_from_lines = mda.Universe(topology=pseudo_pdb, coordinates=traj_file, format='PDB')
mda_traj_from_lines.trajectory[1]

output_stream = io.StringIO()
pseudo_output = NamedStream(output_stream, 'pseudo_output.pdb')

PDB = MDCoords.PDB.PDBWriter(pseudo_output, multiframe=True)
PDB.write(mda_traj_from_lines.atoms)
PDB.close()

pseudo_output.seek(0)
pdb_pseudo_lines = pseudo_output.readlines()

with open('new.pdb', 'w') as file:
    for line in pdb_pseudo_lines:
        file.write(line)

print(len(pdb_file_lines), len(pdb_pseudo_lines))
# Check
assert([line for line in pdb_file_lines] == [line for line in  pdb_pseudo_lines])


input("Press ENTER to end test and read OpenMM errors")