import os, json, time 
from smartsim import Experiment
from smartsim.settings import SrunSettings, SbatchSettings
from smartsim.settings.settings import BatchSettings

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
md_counts = 12
ml_counts = 10
node_counts = md_counts // gpus_per_node


HOME = os.environ.get('HOME')
conda_path = os.environ.get('CONDA_PREFIX')
base_path = os.path.abspath('.') # '/gpfs/alpine/proj-shared/bip179/entk/hyperspace/microscope/experiments/'
conda_sh = '/lus/scratch/arigazzi/anaconda3/etc/profile.d/conda.sh'

CUR_STAGE=0
MAX_STAGE=10
RETRAIN_FREQ = 5

LEN_initial = 10
LEN_iter = 10 

def run_training_pipeline(exp):
    """
    Function to generate the CVAE_MD pipeline
    """

    def generate_MD_stage(num_MD=1): 
        """
        Function to generate MD stage. 
        """
        
        initial_MD = True 
        outlier_filepath = '%s/Outlier_search/restart_points.json' % base_path

        if os.path.exists(outlier_filepath): 
            initial_MD = False 
            outlier_file = open(outlier_filepath, 'r') 
            outlier_list = json.load(outlier_file) 
            outlier_file.close() 

        # MD tasks
        time_stamp = int(time.time())

        md_batch_args = {"N": node_counts, "ntasks-per-node": 1}
        md_batch_settings = SbatchSettings(time="01:00:00", batch_args=md_batch_args)
        md_batch_settings.add_preamble(f'. {conda_sh}')
        md_batch_settings.add_preamble(f'conda activate {conda_path}')
        md_batch_settings.add_preamble('module load cudatoolkit')
        md_batch_settings.add_preamble(f'export PYTHONPATH={base_path}/MD_exps:{base_path}/MD_exps/MD_utils:$PYTHONPATH')
        md_ensemble = exp.create_ensemble("SmartSim-fs-pep", batch_settings=md_batch_settings)
        for i in range(num_MD):
            md_run_settings = SrunSettings(exe=f"python", exe_args=f"{base_path}/MD_exps/fs-pep/run_openmm.py")
            

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
                md_run_settings.add_exe_args(['--length', LEN_initial])
            else: 
                md_run_settings.add_exe_args(['--length', LEN_iter])
                              
            # Add the MD task to the simulating stage
            md_model = exp.create_model(f"omm_runs_{time_stamp+i}", run_settings=md_run_settings, path=f"{base_path}/MD_exps/fs-pep")
            if outlier_list[i].endswith('pdb') or outlier_list[i].endswith('chk'):
                md_model.attach_generator_files(to_copy=[outlier_list[i]])
            
            md_ensemble.add_model(md_model)
        
        exp.generate(md_ensemble)
        return md_ensemble


    def generate_aggregating_stage(): 
        """ 
        Function to concatenate the MD trajectory (h5 contact map) 
        """ 
        aggr_run_settings = SrunSettings('python',
                                         ['%s/MD_to_CVAE/MD_to_CVAE.py' % base_path, 
                                          '--sim_path', '%s/MD_exps/fs-pep' % base_path] )

        aggr_batch_settings = SbatchSettings(time="00:10:00", batch_args = {"N": 1, "n": 1})

        aggr_batch_settings.add_preamble([f'. {conda_sh}', f'conda activate {conda_path}'])

        # Add the aggregation task to the aggreagating stage
        aggregating_model = exp.create_model('SmartSim-MD_to_CVAE', run_settings=aggr_run_settings, path=f'{base_path}/MD_to_CVAE')
        aggregating_ensemble = exp.create_ensemble("SmartSim-MD_to_CVAE", batch_settings=aggr_batch_settings)
        aggregating_ensemble.add_model(aggregating_model)

        return aggregating_ensemble


    def generate_ML_stage(num_ML=1): 
        """
        Function to generate the learning stage
        """

        time_stamp = int(time.time())
        ml_batch_settings = SbatchSettings(time="02:00:00", batch_args={"N": num_ML, "ntasks-per-node": 1})
        ml_batch_settings.add_preamble([f'. {conda_sh}', 'module load cudatoolkit', f'conda activate {conda_path}' ])
        ml_batch_settings.add_preamble(f'export PYTHONPATH={base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:$PYTHONPATH')
        ml_ensemble = exp.create_ensemble("SmartSim-ML", batch_settings=ml_batch_settings)
        # learn task
        for i in range(num_ML): 
            dim = i + 3 
            cvae_dir = 'cvae_runs_%.2d_%d' % (dim, time_stamp+i) 
            ml_run_settings = SrunSettings('python', [f'{base_path}/CVAE_exps/train_cvae.py', 
                    '--h5_file', f'{base_path}/MD_to_CVAE/cvae_input.h5', 
                    '--dim', dim] )
            ml_model = exp.create_model(name=cvae_dir, path=f'{base_path}/CVAE_exps', run_settings=ml_run_settings)

            ml_ensemble.add_model(ml_model)

        exp.generate(ml_ensemble)
        return ml_ensemble 


    def generate_interfacing_stage(): 
        
        interfacing_run_settings = SrunSettings('python', 
                                                exe_args=['outlier_locator.py',
                                                '--md', '../MD_exps/fs-pep', 
                                                '--cvae', '../CVAE_exps', 
                                                '--pdb', '../MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', 
                                                '--ref', '../MD_exps/fs-pep/pdb/fs-peptide.pdb'])
        interfacing_batch_settings = SbatchSettings(time="00:10:00", batch_args = {"N": 1, "n": 1})
        interfacing_batch_settings.add_preamble([f'. {conda_sh}',
                                                 'module load cudatoolkit'
                                                 f'conda activate {conda_path}'
                                                 f'export PYTHONPATH={base_path}/CVAE_exps:{base_path}/CVAE_exps/cvae:$PYTHONPATH'
                                                ])
        # Scaning for outliers and prepare the next stage of MDs 
        
        interfacing_model = exp.create_model('Outlier_search', path=None, run_settings=interfacing_run_settings)
        interfacing_ensemble = exp.create_ensemble('Outlier_search', path='.', BatchSettings=interfacing_batch_settings)

        interfacing_ensemble.add_model(interfacing_model)

        exp.generate(interfacing_ensemble)
        return interfacing_ensemble



    for CUR_STAGE in range(MAX_STAGE):
        print ('finishing stage %d of %d' % (CUR_STAGE, MAX_STAGE))
        
        # --------------------------
        # MD stage
        md_stage = generate_MD_stage(num_MD=md_counts)
        # Add simulating stage to the training pipeline
        exp.run(md_stage)

        if CUR_STAGE % RETRAIN_FREQ == 0: 
            # --------------------------
            # Aggregate stage
            aggregating_stage = generate_aggregating_stage() 
            # Add the aggregating stage to the training pipeline
            exp.run(aggregating_stage)

            # --------------------------
            # Learning stage
            ml_stage = generate_ML_stage(num_ML=ml_counts) 
            # Add the learning stage to the pipeline
            exp.run(ml_stage)

        # --------------------------
        # Outlier identification stage
        interfacing_stage = generate_interfacing_stage() 
        exp.run(interfacing_stage)
    

    return 


if __name__ == '__main__':

    exp = Experiment(name="SmartSim-MD", launcher="slurm", exp_path='.')

    run_training_pipeline(exp)
    
    
