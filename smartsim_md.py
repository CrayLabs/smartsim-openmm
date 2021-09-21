import os, json, time 
from smartsim import Experiment
from smartsim.settings import SrunSettings, SbatchSettings
from smartsim.database import SlurmOrchestrator

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
TINY = False

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
        
    
    def start_orchestrator(self, attach=False):
        checkpoint = os.path.join(self.exp.exp_path, "database", "smartsim_db.dat")
        if attach and os.path.exists(checkpoint):
            print("Found orchestrator checkpoint, reconnecting")
            self.orchestrator = self.exp.reconnect_orchestrator(checkpoint)
        else:
            self.orchestrator = SlurmOrchestrator(db_nodes=3 if self.cluster_db else 1, time="02:00:00", interface=INTERFACE)
            self.exp.generate(self.orchestrator)
            self.exp.start(self.orchestrator)
        return


    def generate_MD_stage(self, num_MD=1): 
        """
        Function to generate MD stage. 
        """
        
        initial_MD = True
        outlier_filepath = f'{self.exp.exp_path}/SmartSim-Outlier_search/SmartSim-Outlier_search/restart_points.json'

        if os.path.exists(outlier_filepath):
            outlier_file = open(outlier_filepath, 'r') 
            outlier_list = json.load(outlier_file) 
            outlier_file.close()
            num_outliers = len(outlier_list)
            print(f"Found {num_outliers} outliers")
            initial_MD = num_outliers > 0
            if num_outliers == 0:
                print("No outlier in file")
        else:
            print("No outlier file found")

        # MD tasks
        time_stamp = int(time.time())

        md_batch_args = {"nodes": node_counts, "ntasks-per-node": 1, "constraint": "V100", "exclusive": None}
        md_batch_settings = SbatchSettings(time="01:00:00", batch_args=md_batch_args)
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
            md_run_settings.add_exe_args(["--output_path", os.path.join(self.exp.exp_path,"omm_out",f"omm_runs_{i:02d}_{time_stamp+i}")])

            # pick initial point of simulation 
            if initial_MD or i >= len(outlier_list): 
                md_run_settings.add_exe_args(['--pdb_file', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb'])
            elif outlier_list[i].endswith('pdb'): 
                md_run_settings.add_exe_args(['--pdb_file', outlier_list[i]])
            elif outlier_list[i].endswith('chk'): 
                md_run_settings.add_exe_args( ['--pdb_file',
                        f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                        '-c', outlier_list[i]] )

            # how long to run the simulation 
            if initial_MD: 
                md_run_settings.add_exe_args(['--length', str(LEN_initial)])
            else: 
                md_run_settings.add_exe_args(['--length', str(LEN_iter)])
                              
            # Add the MD task to the simulating stage
            md_model = self.exp.create_model(f"openmm_{i}", run_settings=md_run_settings)
            if not (initial_MD or i >= len(outlier_list)) and (outlier_list[i].endswith('pdb') or outlier_list[i].endswith('chk')):
                md_model.attach_generator_files(to_copy=[outlier_list[i]])
            
            md_model.enable_key_prefixing()
            md_ensemble.add_model(md_model)
        
        # [md_ensemble.register_incoming_entity(entity) for entity in md_ensemble.entities]
        self.exp.generate(md_ensemble, overwrite=True)
        time.sleep(10)
        return md_ensemble


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
            ml_model = self.exp.create_model(name=f"cvae_{i}", path=f'{base_path}/CVAE_exps', run_settings=ml_run_settings)

            ml_model.enable_key_prefixing()
            for entity in self.md_stage:
                ml_model.register_incoming_entity(entity)
            
            ml_ensemble.add_model(ml_model)

        self.exp.generate(ml_ensemble, overwrite=True)
        time.sleep(5) # apparently lustre sometimes needs a rest
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
        interfacing_batch_settings = SbatchSettings(time="00:10:00",
                                                    batch_args = {"nodes": node_counts, "ntasks-per-node": 1, "constraint": "V100"})
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

        [interfacing_model.register_incoming_entity(entity) for entity in self.ml_stage.entities]
        [interfacing_model.register_incoming_entity(entity) for entity in self.md_stage.entities]
        return interfacing_ensemble


    def run_pipeline(self):

        self.start_orchestrator()

        for CUR_STAGE in range(MAX_STAGE):
            print ('Running stage %d of %d' % (CUR_STAGE, MAX_STAGE))
            
            # --------------------------
            # MD stage, re-initialized at every iteration
            self.md_stage = self.generate_MD_stage(num_MD=md_counts)
            self.exp.start(self.md_stage)

            # --------------------------
            # Learning stage, re-initialized at every retrain iteration
            if CUR_STAGE % RETRAIN_FREQ == 0: 
                self.ml_stage = self.generate_ML_stage(num_ML=ml_counts)
                self.exp.start(self.ml_stage)

            # --------------------------
            # Outlier identification stage
            if CUR_STAGE==0:
                self.interfacing_stage = self.generate_interfacing_stage() 
            self.exp.start(self.interfacing_stage)

        input("Press Enter to terminate and kill the orchestrator (if it is still running)...")
        # self.exp.stop(self.orchestrator)

    def __del__(self):
        self.exp.stop(self.orchestrator)

if __name__ == '__main__':

    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    
