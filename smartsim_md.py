import os, time 
from smartsim import Experiment
from smartsim.settings import SrunSettings, SbatchSettings
from smartsim.database import SlurmOrchestrator

from smartredis import Client, Dataset

# Assumptions:
# - # of MD steps: 2
# - Each MD step runtime: 15 minutes
# - Summit's scheduling policy [1]
#
# Resource request:
# - 4 <= nodes with 2h walltime.
#
# Workflow [2]
#
# [1] https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/scheduling-policy
# [2] https://docs.google.com/document/d/1XFgg4rlh7Y2nckH0fkiZTxfauadZn_zSn3sh51kNyKE/
#


gpus_per_node = 1  # 6 on Summit, 1 on Horizon
TINY = True

HOME = os.environ.get('HOME')
conda_path = os.environ.get('CONDA_PREFIX')
base_path = os.path.abspath('.')
conda_sh = '/lus/scratch/arigazzi/anaconda3/etc/profile.d/conda.sh'
INTERFACE="ib0"

if TINY:
    LEN_initial = 3
    LEN_iter = 3
    md_counts = gpus_per_node*2
    ml_counts = 2
    RETRAIN_FREQ = 2
    MAX_STAGE = 4
else:
    LEN_initial = 10
    LEN_iter = 10
    md_counts = 6
    ml_counts = 6
    RETRAIN_FREQ = 5
    MAX_STAGE = 10

node_counts = md_counts // gpus_per_node

print("-"*49)
print(" "*21 + "WELCOME")
print("-"*49 + "\n")

class TrainingPipeline:
    def __init__(self):
        self.exp = Experiment(name="SmartSim-DDMD", launcher="slurm")
        self.exp.generate(overwrite=True)
        self.cluster_db = True
        self.used_outliers = []
        
    def start_orchestrator(self, attach=False):
        checkpoint = os.path.join(self.exp.exp_path, "database", "smartsim_db.dat")
        if attach and os.path.exists(checkpoint):
            print("Found orchestrator checkpoint, reconnecting")
            self.orchestrator = self.exp.reconnect_orchestrator(checkpoint)
        else:
            self.orchestrator = SlurmOrchestrator(db_nodes=3 if self.cluster_db else 1, time="02:30:00", interface=INTERFACE)
            self.exp.generate(self.orchestrator)
            self.exp.start(self.orchestrator)
        self.client = Client(address=self.orchestrator.get_address()[0], cluster=self.cluster_db)

        used_files = Dataset('used_files')
        used_files.add_meta_string('pdbs', '100-fs-peptide-400K.pdb')
        used_files.add_meta_string('checkpoints', '_.chk')  # Fake, just to initialize field
        self.client.put_dataset(used_files)
        return


    def generate_MD_stage(self, num_MD=1): 
        """
        Function to generate MD stage. 
        """
        
        md_batch_args = {"nodes": node_counts, "ntasks-per-node": 1, "constraint": "V100", "exclusive": None}
        md_batch_settings = SbatchSettings(time="02:00:00", batch_args=md_batch_args)
        md_batch_settings.set_partition("spider")
        md_batch_settings.add_preamble(f'. {conda_sh}')
        md_batch_settings.add_preamble(f'conda activate {conda_path}')
        md_batch_settings.add_preamble('module load cudatoolkit')
        python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/MD_exps:{base_path}/MD_exps/MD_utils_fspep:" + python_path
        md_ensemble = self.exp.create_ensemble("SmartSim-fs-pep", batch_settings=md_batch_settings)
        self.exp.generate(md_ensemble)
        for i in range(num_MD):
            md_run_settings = SrunSettings(exe=f"python",
                                            exe_args=f"{base_path}/MD_exps/fs-pep/run_openmm.py",
                                            run_args={"exclusive": None}, env_vars={"PYTHONPATH": python_path, "SS_CLUSTER": str(int(self.cluster_db))})
            md_run_settings.set_nodes(1)
            md_run_settings.set_tasks(1)
            md_run_settings.set_tasks_per_node(1)
            os.makedirs(os.path.join(self.exp.exp_path,"omm_out"), exist_ok=True)
         
            # Add the MD task to the simulating stage
            md_model = self.exp.create_model(f"openmm_{i}", run_settings=md_run_settings)
            
            md_model.enable_key_prefixing()
            md_ensemble.add_model(md_model)
        
        self.exp.generate(md_ensemble, overwrite=True)
        return md_ensemble

    def update_MD_exe_args(self):
        
        initial_MD = True

        if self.client.key_exists('outliers'):
            outliers = self.client.get_dataset('outliers')
            try:
                outlier_list = outliers.get_meta_strings('points')
                # Filter out used files -- we need this because we are async
                [outlier_list.remove(outlier) for outlier in outlier_list if outlier in self.used_outliers]
                num_outliers = len(outlier_list)
            except:
                outlier_list = []
                num_outliers = 0
        else:
            num_outliers = 0
        
        initial_MD = num_outliers == 0

        # MD tasks
        time_stamp = int(time.time())

        outlier_idx = 0
        for (i, omm) in enumerate(self.md_stage.entities):
            if not initial_MD:
                outlier = outlier_list[outlier_idx]

            input_dataset_key = omm.name + "_input"
            if self.client.key_exists(input_dataset_key):
                continue
            
            input_dataset = Dataset(input_dataset_key)

            exe_args = []
            exe_args.extend(["--output_path",
                            os.path.join(self.exp.exp_path,"omm_out",f"omm_runs_{i:02d}_{time_stamp+i}"),
                            "-g", str(i%gpus_per_node)])

            # pick initial point of simulation 
            if initial_MD or outlier_idx >= len(outlier_list): 
                exe_args.extend(['--pdb_file', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb'])

            elif outlier.endswith('pdb'): 
                exe_args.extend(['--pdb_file', outlier])

                self.used_outliers.append(outlier)
                used_files = self.client.get_dataset('used_files')
                used_files.add_meta_string('pdbs', os.path.basename(outlier))
                super(type(self.client), self.client).put_dataset(used_files)

                outlier_idx += 1

            elif outlier.endswith('chk'): 
                exe_args.extend(['--pdb_file',
                                f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                                '-c', outlier] )

                self.used_outliers.append(outlier)
                used_files = self.client.get_dataset('used_files')
                used_files.add_meta_string('checkpoints', os.path.basename(outlier))
                super(type(self.client), self.client).put_dataset(used_files)

                outlier_idx += 1

            # how long to run the simulation 
            if initial_MD: 
                exe_args.extend(['--length', str(LEN_initial)])
            else: 
                exe_args.extend(['--length', str(LEN_iter)])

            for exe_arg in exe_args:
                input_dataset.add_meta_string("args", exe_arg)
            
            self.client.put_dataset(input_dataset)
            print("Updated " + input_dataset_key)
            

    def generate_ML_stage(self, num_ML=1): 
        """
        Function to generate the learning stage
        """

        ml_batch_settings = SbatchSettings(time="02:00:00", batch_args={"nodes": num_ML, "ntasks-per-node": 1, "constraint": "V100"})
        ml_batch_settings.set_partition("spider")
        ml_batch_settings.add_preamble([f'. {conda_sh}', 'module load cudatoolkit', f'conda activate {conda_path}' ])
        python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + python_path
        ml_ensemble = self.exp.create_ensemble("SmartSim-ML", batch_settings=ml_batch_settings)
        # learn task
        for i in range(num_ML): 
            dim = i + 3 
            ml_run_settings = SrunSettings('python', [f'{base_path}/CVAE_exps/train_cvae.py', 
                    '--dim', str(dim)],
                    env_vars={"PYTHONPATH": python_path, "SS_CLUSTER": str(int(self.cluster_db))})
            ml_run_settings.set_tasks_per_node(1)
            ml_run_settings.set_tasks(1)
            ml_run_settings.set_nodes(1)
            ml_model = self.exp.create_model(name=f"cvae_{i}", run_settings=ml_run_settings)

            ml_model.enable_key_prefixing()
            for entity in self.md_stage:
                ml_model.register_incoming_entity(entity)
            
            ml_ensemble.add_model(ml_model)

        self.exp.generate(ml_ensemble, overwrite=True)
        return ml_ensemble 


    def generate_interfacing_stage(self): 
        
        python_path = os.getenv("PYTHONPATH", "")
        python_path = f"{base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:" + python_path
        interfacing_run_settings = SrunSettings('python', 
                                                exe_args=['outlier_locator.py',
                                                '--md', os.path.join(self.exp.exp_path, 'omm_out'), 
                                                '--pdb', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', 
                                                '--ref', f'{base_path}/MD_exps/fs-pep/pdb/fs-peptide.pdb'],
                                                env_vars={"PYTHONPATH": python_path, "SS_CLUSTER": str(int(self.cluster_db))})
        interfacing_run_settings.set_nodes(1)
        interfacing_run_settings.set_tasks_per_node(1)
        interfacing_run_settings.set_tasks(1)
        interfacing_batch_settings = SbatchSettings(time="02:00:00",
                                                    batch_args = {"nodes": 1, "ntasks-per-node": 1, "constraint": "V100"})
        interfacing_batch_settings.add_preamble([f'. {conda_sh}',
                                                 'module load cudatoolkit',
                                                 f'conda activate {conda_path}',
                                                ])
        interfacing_batch_settings.set_partition("spider")
        # Scanning for outliers and prepare the next stage of MDs 
        
        interfacing_model = self.exp.create_model('SmartSim-Outlier_search', run_settings=interfacing_run_settings)
        interfacing_ensemble = self.exp.create_ensemble('SmartSim-Outlier_search', batch_settings=interfacing_batch_settings)
        interfacing_model.attach_generator_files(to_copy = [os.path.join(base_path, "Outlier_search", "outlier_locator.py"),
                                                            os.path.join(base_path, "Outlier_search", "utils.py")])
        interfacing_ensemble.add_model(interfacing_model)

        self.exp.generate(interfacing_ensemble, overwrite=True)

        [interfacing_model.register_incoming_entity(entity) for entity in self.ml_stage]
        [interfacing_model.register_incoming_entity(entity) for entity in self.md_stage]
        return interfacing_ensemble


    def run_pipeline(self):

        self.start_orchestrator()
        # --------------------------
        # MD stage, re-initialized at every iteration
        self.md_stage = self.generate_MD_stage(num_MD=md_counts)
        self.update_MD_exe_args()
        print("STARTING MD")
        self.exp.start(self.md_stage, block=False)

        # --------------------------
        # Learning stage, re-initialized at every retrain iteration
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
            self.update_MD_exe_args()
            time.sleep(15)


    def __del__(self):
        self.exp.stop(self.interfacing_stage, self.ml_stage, self.md_stage)
        self.exp.stop(self.orchestrator)

if __name__ == '__main__':

    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    
