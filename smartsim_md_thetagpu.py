import os, time 
import numpy as np
from smartsim import Experiment
from smartsim.settings import MpirunSettings, CobaltBatchSettings
from smartsim.database import CobaltOrchestrator

from smartredis import Client, Dataset
from smartsim_utils import put_text_file

# Assumptions:
# - # of MD steps: 2

TINY = True
BATCH = True

HOME = os.environ.get('HOME')
conda_path = os.environ.get('CONDA_PREFIX')
base_path = os.path.abspath(os.curdir)
conda_sh = '/lus/scratch/arigazzi/anaconda3/etc/profile.d/conda.sh'
INTERFACE="enp226s0"

launcher='cobalt'

base_dim = 3

db_node_count = 3

if TINY:
    LEN_initial = 3
    LEN_iter = 3
    md_counts = 2
    ml_counts = 2
else:
    LEN_initial = 10
    LEN_iter = 5  # this is mainly to test asynchronous behavior
    md_counts = 6
    ml_counts = 6


nodefile_name = os.getenv('COBALT_NODEFILE')
with open(nodefile_name, 'r') as nodefile:
    hosts = [node.strip("\n") for node in nodefile.readlines()]

base_host = 0
db_hosts = hosts[base_host:db_node_count]
base_host += db_node_count
md_hosts = hosts[base_host:base_host+md_counts]
base_host += md_counts
ml_hosts = hosts[base_host:base_host+ml_counts]
base_host += ml_counts
outlier_hosts = hosts[-1]


print("-"*49)
print(" "*21 + "WELCOME")
print("-"*49 + "\n")

class TrainingPipeline:
    def __init__(self):
        self.exp = Experiment(name="SmartSim-DDMD", launcher=launcher)
        self.exp.generate(overwrite=True)
        self.cluster_db = db_node_count>1
        
    def start_orchestrator(self, attach=False):
        checkpoint = os.path.join(self.exp.exp_path, "database", "smartsim_db.dat")
        if attach and os.path.exists(checkpoint):
            print("Found orchestrator checkpoint, reconnecting")
            self.orchestrator = self.exp.reconnect_orchestrator(checkpoint)
        else:
            self.orchestrator = CobaltOrchestrator(db_nodes=db_node_count, time="02:30:00", 
                                                    interface=INTERFACE, batch=BATCH, run_command='mpirun',
                                                    hosts=[host+'.mcp' for host in db_hosts])
            self.exp.generate(self.orchestrator)
            self.exp.start(self.orchestrator)
        self.client = Client(address=self.orchestrator.get_address()[0], cluster=self.cluster_db)

        return


    def generate_MD_stage(self, num_MD=1): 
        """
        Function to generate MD stage. 
        """
        
        if BATCH:
            md_batch_settings = CobaltBatchSettings(time="02:00:00")
            md_batch_settings.set_nodes(md_counts)
            md_batch_settings.set_tasks(md_counts)
            md_batch_settings.set_queue("full-node")
            md_batch_settings.set_hostlist(md_hosts)
            md_batch_settings.add_preamble(f'. {conda_sh}')
            md_batch_settings.add_preamble(f'conda activate {conda_path}')
            md_batch_settings.add_preamble('module load cudatoolkit')

        old_python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}:{base_path}/MD_exps:{base_path}/MD_exps/MD_utils_fspep:" + old_python_path
        os.environ["PYTHONPATH"]=python_path

        md_run_settings = MpirunSettings(exe="python",
                                        exe_args=f"{base_path}/MD_exps/fs-pep/run_openmm.py",
                                        run_args=None,
                                        env_vars={"PYTHONPATH": python_path,
                                        "SS_CLUSTER": str(int(self.cluster_db))})
        md_run_settings.set_tasks(1)

        if BATCH:
            md_ensemble = self.exp.create_ensemble("SmartSim-fs-pep", batch_settings=md_batch_settings)
            for i in range(num_MD): 
                # Add the MD task to the simulating stage
                md_model = self.exp.create_model(f"openmm_{i}", run_settings=md_run_settings)
                
                md_model.enable_key_prefixing()
                md_ensemble.add_model(md_model)
            
            self.exp.generate(md_ensemble, overwrite=True)
        else:
            md_ensemble = self.exp.create_ensemble("openmm", run_settings=md_run_settings, replicas=num_MD)
            
            md_ensemble.enable_key_prefixing()
            self.exp.generate(md_ensemble, overwrite=True)
        
        for i, md in enumerate(md_ensemble):
            md.run_settings.set_hostlist([md_hosts[i]])


        self.client.set_script_from_file("cvae_script",
                                        f"{base_path}/MD_to_CVAE/MD_to_CVAE_scripts.py",
                                        device="GPU")

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
                            "-g", str(i%gpus_per_node),
                            '--pdb_file', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                            '--length', str(LEN_initial)])

            for exe_arg in exe_args:
                input_dataset.add_meta_string("args", exe_arg)
            
            self.client.put_dataset(input_dataset)


    def generate_ML_stage(self, num_ML=1): 
        """
        Function to generate the learning stage
        """

        if BATCH:
            ml_batch_settings = CobaltBatchSettings(time="02:00:00")
            ml_batch_settings.set_nodes(num_ML)
            ml_batch_settings.set_tasks(num_ML)
            ml_batch_settings.set_queue('full-node')
            ml_batch_settings.set_hostlist(ml_hosts)
            
            ml_batch_settings.add_preamble([f'. {conda_sh}', 'module load cudatoolkit', f'conda activate {conda_path}' ])

        old_python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + old_python_path
        if BATCH:
            ml_ensemble = self.exp.create_ensemble("SmartSim-ML", batch_settings=ml_batch_settings)
            # learn task
            for i in range(num_ML): 
                dim = i + base_dim
                ml_run_settings = MpirunSettings('python', [f'{base_path}/CVAE_exps/train_cvae.py', 
                                                            '--dim', str(dim)],
                                                            env_vars={"PYTHONPATH": python_path, 
                                                            "SS_CLUSTER": str(int(self.cluster_db))})
                ml_run_settings.set_tasks(1)
                ml_model = self.exp.create_model(name=f"cvae_{i}", run_settings=ml_run_settings)

                ml_model.enable_key_prefixing()
                for entity in self.md_stage:
                    ml_model.register_incoming_entity(entity)
                
                ml_ensemble.add_model(ml_model)
        else:
            os.environ["PYTHONPATH"]=python_path
            ml_run_settings = MpirunSettings('python', [f'{base_path}/CVAE_exps/train_cvae.py', 
                                            '--dim', "SmartSim"],
                                            env_vars={"PYTHONPATH": python_path,
                                            "SS_CLUSTER": str(int(self.cluster_db))}) 
            ml_run_settings.set_tasks(1)

            ml_ensemble = self.exp.create_ensemble("cvae", run_settings=ml_run_settings, replicas=num_ML)
            for i in range(num_ML):
                self.client.put_tensor(f"cvae_{i}_dim", np.asarray([base_dim+i]).astype(int))
            for entity in self.md_stage:
                ml_ensemble.register_incoming_entity(entity)


        self.exp.generate(ml_ensemble, overwrite=True)
        for i, ml in enumerate(ml_ensemble):
            ml.run_settings.set_hostlist(ml_hosts[i])

        
        return ml_ensemble 


    def generate_interfacing_stage(self): 
        
        python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + python_path
        
        interfacing_run_settings = MpirunSettings('python', 
                                                    exe_args=['outlier_locator.py',
                                                    '--md', os.path.join(self.exp.exp_path, 'omm_out'), 
                                                    '--pdb', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', 
                                                    '--ref', f'{base_path}/MD_exps/fs-pep/pdb/fs-peptide.pdb',
                                                    '--len_initial', str(LEN_initial),
                                                    '--len_iter', str(LEN_iter),
                                                    '--exp_path', self.exp.exp_path],
                                                    env_vars={"PYTHONPATH": python_path,
                                                              "SS_CLUSTER": str(int(self.cluster_db)),
                                                              "PYTHONUNBUFFERED": "1"})
        interfacing_run_settings.set_tasks(1)
        interfacing_run_settings.set_hostlist(hosts[-1])

        interfacing_run_settings.update_env({"PYTHONUNBUFFERED": "1"})

        if BATCH:
            interfacing_batch_settings = CobaltBatchSettings(time="02:00:00")
            interfacing_batch_settings.set_queue("full-node")
            interfacing_batch_settings.set_nodes(1)
            interfacing_batch_settings.set_tasks(1)
            interfacing_batch_settings.set_hostlist(hosts[-1])
            interfacing_batch_settings.add_preamble([f'. {conda_sh}',
                                                    'module load cudatoolkit',
                                                    f'conda activate {conda_path}'])
            # Scanning for outliers and prepare the next stage of MDs 
            interfacing_ensemble = self.exp.create_ensemble('SmartSim-Outlier_search', batch_settings=interfacing_batch_settings)

        interfacing_model = self.exp.create_model('SmartSim-Outlier_search', run_settings=interfacing_run_settings)
        interfacing_model.attach_generator_files(to_copy = [os.path.join(base_path, "Outlier_search", "outlier_locator.py"),
                                                            os.path.join(base_path, "Outlier_search", "utils.py"),
                                                            os.path.join(base_path,'smartsim_utils.py')])

        if BATCH:
            interfacing_ensemble.add_model(interfacing_model)
            self.exp.generate(interfacing_ensemble, overwrite=True)
        else:
            self.exp.generate(interfacing_model, overwrite=True)

        [interfacing_model.register_incoming_entity(entity) for entity in self.ml_stage]
        [interfacing_model.register_incoming_entity(entity) for entity in self.md_stage]
        if BATCH:
            return interfacing_ensemble
        else:
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
        while not any([self.client.key_exists(md.name) for md in self.md_stage]):
            time.sleep(5)
        print("STARTING ML")
        self.exp.start(self.ml_stage, block=False)

        # --------------------------
        # Outlier identification stage
        self.interfacing_stage = self.generate_interfacing_stage() 
        while not any([self.client.key_exists(ml.name) for ml in self.ml_stage]):
            time.sleep(5)
        print("STARTING OUTLIER SEARCH")
        self.exp.start(self.interfacing_stage, block=False)

        while True:
            #self.update_MD_exe_args()
            # Here possibly plot info about simulation
            print("Simulation is running")
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
    