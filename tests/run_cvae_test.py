from smartsim.database import SlurmOrchestrator
from smartsim.settings import SrunSettings
from smartsim import Experiment
from smartsim.launcher import slurm
import os


exp = Experiment(name="test-cvae", launcher="slurm")
orchestrator = SlurmOrchestrator(db_nodes=1, time="00:10:00", interface="ipogif0")
exp.generate(orchestrator)
exp.start(orchestrator)

base_path = os.path.abspath('.')

alloc = slurm.get_allocation(time="00:08:00", options={"constraint": "P100"})

pythonpath = os.getenv("PYTHONPATH")

if pythonpath is None:
    pythonpath = os.path.join(base_path, "CVAE_exps", "cvae")
else:
    pythonpath = pythonpath+":"+os.path.join(base_path, "CVAE_exps", "cvae")

run_settings = SrunSettings(exe=f"python",
                            exe_args=f"{base_path}/test_cvae.py", alloc=alloc, env_vars={"PYTHONPATH": pythonpath})

model = exp.create_model("cvae", run_settings=run_settings)

exp.generate(model, overwrite=True)
exp.start(model)


input("Press Enter to terminate and kill the orchestrator (if it is still running)...")

slurm.release_allocation(alloc)
exp.stop(orchestrator)