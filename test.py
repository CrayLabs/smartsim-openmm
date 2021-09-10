from typing_extensions import runtime
from smartsim.database import SlurmOrchestrator
from smartsim.settings import SrunSettings
from smartsim import Experiment
from smartsim.launcher import slurm
import os


exp = Experiment(name="test-dataset", launcher="slurm")
orchestrator = SlurmOrchestrator(db_nodes=1, time="02:00:00")
exp.generate(orchestrator)
exp.start(orchestrator)

base_path = os.path.abspath('.')

alloc = slurm.get_allocation()

run_settings = SrunSettings(exe=f"python",
                            exe_args=f"{base_path}/dataset_client.py", alloc=alloc)

model = exp.create_model("test_model", run_settings=run_settings)

exp.generate(model, overwrite=True)
exp.start(model)


input("Press Enter to terminate and kill the orchestrator (if it is still running)...")

slurm.release_allocation(alloc)
exp.stop(orchestrator)