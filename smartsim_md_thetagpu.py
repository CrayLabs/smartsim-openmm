import os, time
import numpy as np
from smartsim import Experiment

from smartredis import Client, Dataset
from smartsim_utils import put_text_file

from thetagpu.thetagpu_utils import generate_rankfiles, assign_hosts

TINY = False
base_path = os.path.abspath(os.curdir)
gpus_per_node = 8 #  1 on Cray XC-50

INTERFACE="enp226s0"

# Set to 1 if binary files should be always written to disk
# Set to 0 if binary files should be cached into DB -- this is an experimental feature
# and it is not guaranteed to work
BINARY_FILES = "0"

launcher='cobalt'

base_dim = 3

if TINY:
    LEN_initial = 3
    LEN_iter = 3
    md_counts = gpus_per_node*1
    ml_counts = gpus_per_node*1
    db_node_count = 1
else:
    LEN_initial = 10
    LEN_iter = 5  # this is mainly to test asynchronous behavior
    md_counts = gpus_per_node*2
    ml_counts = gpus_per_node*2
    db_node_count = 3

md_node_count = int(np.ceil(md_counts//gpus_per_node))
ml_node_count = int(np.ceil(ml_counts//gpus_per_node))

# Theta GPU setup
db_hosts, md_hosts, ml_hosts, outlier_hosts = assign_hosts(db_node_count, md_node_count, ml_node_count)
rankfile_dir = generate_rankfiles(md_hosts, ml_hosts, gpus_per_node, base_path)

print("-"*49)
print(" "*21 + "WELCOME")
print("-"*49 + "\n")

class TrainingPipeline:
    def __init__(self):
        self.exp = Experiment(name="SmartSim-DDMD", launcher=launcher)
        self.exp.generate(overwrite=True)
        self.cluster_db = db_node_count>1

    def start_orchestrator(self):
        self.orchestrator = self.exp.create_database(db_nodes=db_node_count,
                                                     interface=INTERFACE,
                                                     batch=False,
                                                     run_command='mpirun',
                                                     hosts=[host+'.mcp' for host in db_hosts])
        self.exp.generate(self.orchestrator)
        self.exp.start(self.orchestrator)
        self.client = Client(address=self.orchestrator.get_address()[0], cluster=self.cluster_db)

        return


    def generate_MD_stage(self, num_MD=1):
        """
        Function to generate MD stage.
        """

        old_python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}:{base_path}/MD_exps:{base_path}/MD_exps/MD_utils_fspep:" + old_python_path
        os.environ["PYTHONPATH"]=python_path

        md_run_settings = self.exp.create_run_settings(exe="python",
                                                       exe_args=f"{base_path}/MD_exps/fs-pep/run_openmm.py",
                                                       run_args=None,
                                                       run_command="mpirun",
                                                       env_vars={"PYTHONPATH": python_path,
                                                                 "SS_CLUSTER": str(int(self.cluster_db)),
                                                                 "SS_BINARY_FILES": BINARY_FILES})
        md_run_settings.set_tasks(1)

        md_ensemble = self.exp.create_ensemble("openmm", run_settings=md_run_settings, replicas=num_MD)

        md_ensemble.enable_key_prefixing()
        self.exp.generate(md_ensemble, overwrite=True)

        for i, md in enumerate(md_ensemble):
            md.run_settings.set_hostlist([md_hosts[i//gpus_per_node]])
            md.run_settings.update_env({"CUDA_VISIBLE_DEVICES": str(i%gpus_per_node)})
            # ThetaGPU specific
            md.run_settings.run_args["rankfile"] = os.path.join(rankfile_dir, f"md_ranks_{i}")


        self.client.set_script_from_file("cvae_script",
                                        f"{base_path}/MD_to_CVAE/MD_to_CVAE_scripts.py",
                                        device="CPU")

        return md_ensemble


    # This function has been relocated to the Outlier Search stage,
    # here we just keep the initial_MD phase
    def init_MD_exe_args(self):

        iter_id = 0
        put_text_file(f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', self.client)
        for (i, omm) in enumerate(self.md_stage.entities):
            input_dataset_key = omm.name + "_input"
            input_dataset = Dataset(input_dataset_key)

            exe_args = []
            exe_args.extend(["--output_path",
                            os.path.join(self.exp.exp_path,"omm_out",f"omm_runs_{i:02d}_{iter_id:06d}_{int(time.time())}"),
                            "-g", "0",
                            '--pdb_file', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                            '--length', str(LEN_initial)])

            for exe_arg in exe_args:
                input_dataset.add_meta_string("args", exe_arg)

            self.client.put_dataset(input_dataset)


    def generate_ML_stage(self, num_ML=1):
        """
        Function to generate the learning stage
        """


        old_python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + old_python_path


        os.environ["PYTHONPATH"]=python_path
        ml_rs_exe_args = [f'{base_path}/CVAE_exps/train_cvae.py', '--dim', "SmartSim"]
        ml_rs_env_vars = {"PYTHONPATH": python_path,
                          "SS_CLUSTER": str(int(self.cluster_db)),
                          "SS_BINARY_FILES": BINARY_FILES}
        ml_run_settings = self.exp.create_run_settings(run_command="mpirun",
                                                       exe="python",
                                                       exe_args=ml_rs_exe_args,
                                                       env_vars=ml_rs_env_vars)
        ml_run_settings.set_tasks(1)

        ml_ensemble = self.exp.create_ensemble("cvae", run_settings=ml_run_settings, replicas=num_ML)
        for i in range(num_ML):
            self.client.put_tensor(f"cvae_{i}_dim", np.asarray([base_dim+i]).astype(int))
        for entity in self.md_stage:
            ml_ensemble.register_incoming_entity(entity)

        self.exp.generate(ml_ensemble, overwrite=True)
        for i, ml in enumerate(ml_ensemble):
            ml.run_settings.set_hostlist(ml_hosts[i//gpus_per_node])
            ml.run_settings.update_env({"CUDA_VISIBLE_DEVICES": str(i%gpus_per_node)})
            ml.run_settings.run_args["rankfile"] = os.path.join(rankfile_dir, f"ml_ranks_{i}")



        return ml_ensemble


    def generate_interfacing_stage(self):

        python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + python_path

        rs_exe_args = ['outlier_locator.py',
                       '--md', os.path.join(self.exp.exp_path, 'omm_out'),
                       '--pdb', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                       '--ref', f'{base_path}/MD_exps/fs-pep/pdb/fs-peptide.pdb',
                       '--len_initial', str(LEN_initial),
                       '--len_iter', str(LEN_iter),
                       '--exp_path', self.exp.exp_path]

        rs_env_vars = {"PYTHONPATH": python_path,
                       "SS_CLUSTER": str(int(self.cluster_db)),
                       "PYTHONUNBUFFERED": "1",
                       "SS_BINARY_FILES": BINARY_FILES}

        interfacing_run_settings = self.exp.create_run_settings(exe = 'python',
                                                                exe_args=rs_exe_args,
                                                                run_command="mpirun",
                                                                env_vars=rs_env_vars)
        interfacing_run_settings.set_tasks(1)
        interfacing_run_settings.set_hostlist(outlier_hosts)

        interfacing_run_settings.update_env({"PYTHONUNBUFFERED": "1"})


        interfacing_model = self.exp.create_model('SmartSim-Outlier_search', run_settings=interfacing_run_settings)
        interfacing_model.attach_generator_files(to_copy = [os.path.join(base_path, "Outlier_search", "outlier_locator.py"),
                                                            os.path.join(base_path, "Outlier_search", "utils.py"),
                                                            os.path.join(base_path,'smartsim_utils.py')])

        self.exp.generate(interfacing_model, overwrite=True)

        [interfacing_model.register_incoming_entity(entity) for entity in self.ml_stage]
        [interfacing_model.register_incoming_entity(entity) for entity in self.md_stage]

        return interfacing_model


    def run_pipeline(self):

        self.start_orchestrator()
        # --------------------------
        # MD stage
        self.md_stage = self.generate_MD_stage(num_MD=md_counts)
        self.init_MD_exe_args()
        print("STARTING MD")
        self.exp.start(self.md_stage, block=False)

        # --------------------------
        # Learning stage
        self.ml_stage = self.generate_ML_stage(num_ML=ml_counts)
        while not any([self.client.dataset_exists(md.name) for md in self.md_stage]):
            time.sleep(5)
        print("STARTING ML")
        self.exp.start(self.ml_stage, block=False)

        # --------------------------
        # Outlier identification stage
        self.interfacing_stage = self.generate_interfacing_stage()
        while not any([self.client.dataset_exists(ml.name) for ml in self.ml_stage]):
            time.sleep(5)
        print("STARTING OUTLIER SEARCH", flush=True)
        self.exp.start(self.interfacing_stage, block=False)

        while True:
            # Here possibly plot info about simulation
            print("Simulation is running", flush=True)
            time.sleep(120)


    def __del__(self):
        try:
            self.exp.stop(self.interfacing_stage, self.ml_stage, self.md_stage)
        except:
            print("Some stage could not be stopped, please stop processes manually")
        try:
            self.exp.stop(self.orchestrator)
        except:
            print("Orchestrator could not be stopped, please stop it manually")

if __name__ == '__main__':

    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

