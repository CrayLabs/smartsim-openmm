from smartsim_utils import put_text_file, put_strings_as_file, get_text_file, save_text_file

from smartsim.database import SlurmOrchestrator
from smartsim import Experiment
from smartsim.launcher import slurm

from smartredis import Client

import os

exp = Experiment(name="test-file", launcher="slurm")
orchestrator = SlurmOrchestrator(db_nodes=1, time="00:10:00", interface="ipogif0", batch=True)
exp.generate(orchestrator)
exp.start(orchestrator)

db_address = orchestrator.get_address()[0]

client = Client(address=db_address, cluster=False)

original_file = '/lus/scratch/arigazzi/smartsim-dev/smartsim-openmm/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb'
dest_file = os.path.join(exp.exp_path, 'files', "_"+os.path.basename(original_file))

put_text_file(original_file, client, overwrite=False)

input("A look at DB? You find it at: " + db_address)

strings = get_text_file(original_file, client)

put_strings_as_file(dest_file, strings, client)

save_text_file(dest_file, client, exist_ok=True)

input("Last look at DB?")