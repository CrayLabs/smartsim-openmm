from tasks import run_omm_with_celery, run_omm_with_celery_fs_pep, run_cvae_with_celery
from celery.bin import worker
import numpy as np
import threading, h5py
import subprocess, errno, os
import warnings
from sklearn.cluster import DBSCAN
import MDAnalysis as mda
from keras import backend as K
from molecules.utils.matrix_op import triu_to_full

from CVAE import CVAE

def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 

def start_rabbit(rabbitmq_log): 
    """
    A function starting the rabbitmq server within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    rabbitmq_log : ``str``
        log file contains the screen output of rabbitmq server 
    
    """
    log = open(rabbitmq_log, 'w')
    subprocess.Popen('rabbitmq-server &'.split(' '), stdout=log, stderr=log) 

def start_worker(celery_worker_log): 
    """
    A function starting the celery works within the python script and sending
    the worker running at the background. 
    
    Parameters: 
    -----------
    celery_worker_log : ``str``
        log file contains the screen output of celery worker 
    
    """
    
    celery_cmdline = "celery worker -A tasks" 
    log = open(celery_worker_log, 'w')
    subprocess.Popen(celery_cmdline.split(" "), stdout=log, stderr=log) 
    # This format of starting the workers used mess up the print function in notebook. 
#     celery_worker = worker.worker(app=celery_app)
#     threaded_celery_worker = threading.Thread(target=celery_worker.run) 
#     threaded_celery_worker.start() 
#     return threaded_celery_worker
    

def start_flower_monitor(address='127.0.0.1', port=5555): 
    """
    A function starting the flower moniter for celery servers and workers. 
    The information is available at http://127.0.0.1:5555 by default
    
    Parameters: 
    -----------
    address : ``string``
        The address to the flower server
    port : ``int``
        The port to open or port the server 
        
    """ 
    
    celery_flower_cmdline = 'celery flower -A tasks --address={0} --port={1}'.format(address, port)
    subprocess.Popen(celery_flower_cmdline.split(" "))
    

def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def job_on_gpu(gpu_id, jobs): 
    """
    Find job on GPU gpu_id
    
    Parameters: 
    -----------
    gpu_id : ``int`` 
    jobs : ``list of celery tasks``
    """
    for job in jobs: 
        if job.gpu_id == gpu_id: 
            return job 
        

class omm_job(object): 
    """
    A OpenMM simulation job. 
    
    Parameters: 
    -----------
    job_id : ``int`` 
        A int number to track the job, according to which the job will create a directory 
        and store the log, trajectory and contact maps h5 files 
    gpu_id : ``int``
        The id of GPU, on which the OpenMM will be running 
    top_file : ``str``
        The location of input topology file for OpenMM 
    pdb_file : ``str``
        The location of input coordinate file for OpenMM 
        
    """
    def __init__(self, job_id=0, gpu_id=0, top_file=None, pdb_file=None, check_point=None): 
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.top_file = top_file
        self.pdb_file = pdb_file 
        self.check_point = None 
        self.type = 'omm'
        self.state = 'RECEIVED'
        self.save_path = 'omm_run_%d' % job_id
        self.job = None 
        
    def start(self): 
        """
        A function to start the job and store the `class :: celery.result.AsyncResult` 
        in the omm_job.job 
        """
        if self.top_file: 
            sim_job = run_omm_with_celery.delay(self.job_id, self.gpu_id, 
                                                self.top_file, self.pdb_file,
                                                self.check_point) 
        else: 
            sim_job = run_omm_with_celery_fs_pep.delay(self.job_id, self.gpu_id, self.pdb_file, 
                                                 self.check_point) 
        
        self.state = 'RUNNING'
        self.job = sim_job
    
    def stop(self): 
        """
        A function to stop the job and return the available gpu_id 
        """
        if self.job: 
            self.state = 'STOPPED'
            self.job.revoke(terminate=True) 
        else: 
            warnings.warn('Attempt to stop a job, which is not running. \n')
        return self.gpu_id 
    

    
class cvae_job(object): 
    """
    A CVAE job. 
    
    Parameters: 
    -----------
    job_id : ``int`` 
        A int number to track the job, according to which the job will create a directory 
        and store the weight files 
    gpu_id : ``int``
        The id of GPU, on which the CVAE will be running 
    input_data_file : ``str`` file location
        The location of h5 file for CVAE input  
    hyper_dim : ``int``
        The number of latent space dimension 
        
    """
    def __init__(self, job_id, gpu_id=0, cvae_input=None, hyper_dim=3): 
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.cvae_input = cvae_input
        self.hyper_dim = hyper_dim 
        self.type = 'cvae'
        self.state = 'RECEIVED'
        self.model_weight = os.path.join("cvae_model_%d_%d" % (hyper_dim, int(job_id)), 'cvae_weight.h5')
        self.job = None 
        
    def start(self): 
        """
        A function to start the job and store the `class :: celery.result.AsyncResult` 
        in the cvae_job.job 
        """
        sim_job = run_cvae_with_celery.delay(self.job_id, self.gpu_id, 
                                             self.cvae_input, hyper_dim=self.hyper_dim)
        self.state = 'RUNNING' 
        self.job = sim_job 
        
    def cave_model(self): 
        pass
#         if self.job.
    
    def stop(self): 
        """
        A function to stop the job and return the available gpu_id 
        """
        if self.job: 
            self.state = 'STOPPED' 
            self.job.revoke(terminate=True) 
        else: 
            warnings.warn('Attempt to stop a job, which is not running. \n')

            
class job_list(list): 
    """
    This create a list that allows to easily tracking the status of Celery jobs
    """
    def __init__(self): 
        pass
    
    def get_running_jobs(self): 
        running_list = []
        for job in self: 
            if job.job:
                if job.state == u'RUNNING':  
                    running_list.append(job)
        return running_list 
    
    def get_job_from_gpu_id(self, gpu_id): 
        for job in self.get_running_jobs(): 
            if job.gpu_id == gpu_id: 
                return job 
            
    def get_omm_jobs(self): 
        omm_list = [job for job in self if job.type == 'omm']
        return omm_list 
    
    def get_cvae_jobs(self): 
        cvae_list = [job for job in self if job.type == 'cvae']
        return cvae_list 
    
    def get_available_gpu(self, gpu_list): 
        avail_gpu = gpu_list[:]
        for job in self.get_running_jobs():
            avail_gpu.remove(job.gpu_id)
        return avail_gpu 
    
    def get_running_omm_jobs(self): 
        running_omm_list = [job for job in self.get_running_jobs() if job.type == 'omm'] 
        return running_omm_list  
    
    def get_finished_cave_jobs(self): 
        finished_cvae_list = [job for job in self.get_cvae_jobs() if job.job.status == u'SUCCESS']
        return finished_cvae_list
    
    
def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
    
def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[key]) < 0: 
            dir_name = os.path.dirname(key) 
            traj_file = os.path.join(dir_name, 'output.dcd')             
            return traj_file, local_frame
        else: 
            local_frame -= int(traj_dict[key])
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb

def make_dir_p(path_name): 
    try:
        os.mkdir(path_name)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def outliers_from_cvae(model_weight, cvae_input, hyper_dim=3, eps=0.35): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(0)  
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cvae.model.load_weights(model_weight)
    cm_predict = cvae.return_embeddings(cvae_input) 
    db = DBSCAN(eps=eps, min_samples=10).fit(cm_predict)
    db_label = db.labels_
    outlier_list = np.where(db_label == -1)
    K.clear_session()
    return outlier_list

def predict_from_cvae(model_weight, cvae_input, hyper_dim=3): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(0)  
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cvae.model.load_weights(model_weight)
    cm_predict = cvae.return_embeddings(cvae_input) 
    return cm_predict
