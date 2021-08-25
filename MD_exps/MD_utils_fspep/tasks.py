from celery import Celery
import simtk.unit as u 
from CVAE import run_cvae 
from keras import backend as K
import sys, os, shutil, gc 
import errno 

# sys.path.append('$HOME/Research/molecules/molecules_git/build/lib')
from molecules.sim.openmm_simulation import openmm_simulate_charmm_nvt, openmm_simulate_amber_fs_pep

app = Celery('tasks', broker='pyamqp://guest@localhost//', backend='rpc://', broker_pool_limit = None) 

app.conf.update(
    broker_pool_limit = 0, 
    timezone='US/Eastern',
    enable_utc=True,
)

@app.task
def run_omm_with_celery(run_id, gpu_index, top_file, pdb_file, check_point=None): 
    work_dir = os.getcwd()
    iter_dir = os.path.join(work_dir, "omm_run_%d" % int(run_id))
    
    try:
        os.mkdir(iter_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(iter_dir) 
    shutil.copy2(top_file, iter_dir)
    shutil.copy2(pdb_file, iter_dir)
    openmm_simulate_charmm_nvt(top_file, pdb_file, 
                               check_point = check_point, 
                               GPU_index=gpu_index, 
                               output_traj="output.dcd", 
                               output_log="output.log", 
                               output_cm='output_cm.h5',
                               report_time=50*u.picoseconds, 
                               sim_time=10000*u.nanoseconds) 
    
    
@app.task
def run_omm_with_celery_fs_pep(run_id, gpu_index, pdb_file, check_point=None): 
    work_dir = os.getcwd()
    iter_dir = os.path.join(work_dir, "omm_run_%d" % int(run_id))
    
    try:
        os.mkdir(iter_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(iter_dir) 
    shutil.copy2(pdb_file, iter_dir)
    openmm_simulate_amber_fs_pep(pdb_file, 
                                 check_point = check_point, 
                                 GPU_index=gpu_index, 
                                 output_traj="output.dcd", 
                                 output_log="output.log", 
                                 output_cm='output_cm.h5',
                                 report_time=50*u.picoseconds, 
                                 sim_time=10000*u.nanoseconds) 
    

@app.task(autoretry_for=(OSError,))
def run_cvae_with_celery(job_id, gpu_id, cvae_input, hyper_dim=3): 
    cvae_input = os.path.abspath(cvae_input)
    work_dir = os.getcwd() 
    model_dir = os.path.join(work_dir, "cvae_model_%d_%d" % (hyper_dim, int(job_id)))
    
    try:
        os.mkdir(model_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    os.chdir(model_dir) 
    
    cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim) 
    
    model_weight = os.path.join(model_dir, 'cvae_weight.h5') 
    model_file = os.path.join(model_dir, 'cvae_model.h5')
    
    cvae.model.save_weights(model_weight)
    cvae.save(model_file) 
    K.clear_session()
    del cvae
    sys.exit()
    # return model_weight, model_file
    
