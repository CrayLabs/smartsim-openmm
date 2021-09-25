import parmed as pmd
import MDAnalysis as mda 

import MDAnalysis.coordinates as MDCoords

from MDAnalysis.lib.util import NamedStream

import io

class PseudoStringIO(io.StringIO):
    def write(self, s):
        print(s)
        super().write(s)

filename = '/lus/scratch/arigazzi/smartsim-dev/smartsim-openmm/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb'

pdb_from_file = pmd.read_PDB(filename)

pdb_file = open(filename, 'r')
pdb_lines = pdb_file.readlines()

pdb_from_lines = pmd.read_PDB(pdb_lines)

traj_file = '/lus/scratch/arigazzi/smartsim-dev/smartsim-openmm/SmartSim-DDMD/omm_out/omm_runs_00_1632508428/output.dcd'
pdb_stream = io.StringIO("\n".join(pdb_lines))

pseudo_pdb = NamedStream(pdb_stream, 'pseudo.pdb')
mda_traj_from_lines = mda.Universe(topology=pseudo_pdb, coordinates=traj_file, format='PDB')
print(mda_traj_from_lines)
mda_traj_from_lines.trajectory[1] 

output_stream = io.StringIO()
pseudo_output = NamedStream(output_stream, 'pseudo_output.pdb')

pseudo_output.flush()
print(str(pseudo_output))

PDB = MDCoords.PDB.PDBWriter(pseudo_output)
PDB.write(mda_traj_from_lines.atoms) 

pseudo_output.seek(0)
print(pseudo_output.readlines())

del mda_traj_from_lines
del PDB

input("Press ENTER to end my misery")