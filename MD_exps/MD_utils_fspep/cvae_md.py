from __future__ import print_function
print('============================================================== ')
print('Running the MD simulation according to the CVAE classification ')
print('============================================================== ')

from glob import glob
import numpy as np
import sys, os, h5py, time, errno, random
import GPUtil
import subprocess
from sklearn.cluster import DBSCAN
import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD

from utils import start_rabbit, start_worker, start_flower_monitor, read_h5py_file, cm_to_cvae, job_on_gpu
from utils import find_frame, write_pdb_frame, make_dir_p, job_list, outliers_from_cvae
from utils import omm_job, cvae_job 

from CVAE import CVAE

# n_gpus = 16
# number of cvae jobs, starting from hyper_dim 3 
n_cvae = 4
GPU_ids = [gpu.id for gpu in GPUtil.getGPUs()] 
print('Available GPUs', GPU_ids) 

os.environ["RABBITMQ_MNESIA_BASE"] = "~/.rabbit_base"
os.environ["RABBITMQ_LOG_BASE"] = "~/.rabbit_base/"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# top_file = os.path.abspath('../P27-all/C1B48/C1B48.top.gz')
# pdb_file = os.path.abspath('../P27-all/C1B48/C1B48.pdb.gz')
top_file = None
pdb_file = os.path.abspath('./pdb/100-fs-peptide-400K.pdb')
ref_pdb_file = os.path.abspath('./pdb/fs-peptide.pdb')

 
work_dir = os.path.abspath('./')

# create folders for store results
log_dir = os.path.join(work_dir, 'scheduler_logs') 
make_dir_p(log_dir)

# Starting RabbitMQ and Celery schedule framework. 
rabbitmq_log = os.path.join(log_dir, 'rabbit_server_log.txt') 
start_rabbit(rabbitmq_log)
time.sleep(5)

celery_worker_log = os.path.join(log_dir, 'celery_worker_log.txt') 
start_worker(celery_worker_log)
# start_flower_monitor() 
print('Waiting 10 seconds for the server to stablize. \n')
time.sleep(10)


# Starting MD simulation on all GPUs using OpenMM 
jobs = job_list() 
for gpu_id in GPU_ids: 
    job = omm_job(job_id=int(time.time()), gpu_id=gpu_id, top_file=top_file, pdb_file=pdb_file)
    job.start() 
    jobs.append(job) 
    print(' Started OpenMM jobs on GPU', gpu_id)
    time.sleep(2)
    
print('\nWaiting 2 mins for omm to write valid contact map .h5 files ')
time.sleep(120) 


# Read all the contact map .h5 file in local dir
cm_files = sorted(glob('omm*/*_cm.h5'))
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 

print('Waiting for the OpenMM to complete the first 100,000 frames as CVAE training data')
frame_number = lambda lists: sum([cm.shape[1] for cm in lists]) 

frame_marker = 0 
# number of training frames for cvae, change to 1e5 later, HM 
while frame_number(cm_data_lists) < 100000: 
    for cm in cm_data_lists: 
        cm.refresh() 
    if frame_number(cm_data_lists) >= frame_marker: 
        print('Current number of frames from OpenMM:', frame_number(cm_data_lists)) 
        frame_marker = int((10000 + frame_marker) / 10000) * 10000
        print('    Next report at frame', frame_marker) 

print('Ready for CAVE with total number of frames:', frame_number(cm_data_lists)) 

# Compress all .h5 files into one in cvae format 
cvae_input = cm_to_cvae(cm_data_lists) 
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
cvae_data_length = len(cvae_input)

# Write the traj info 
omm_log = os.path.join(log_dir, 'openmm_log.txt') 
log = open(omm_log, 'w') 
for i, n_frame in enumerate(train_data_length): 
    log.writelines("{} {}\n".format(cm_files[i], n_frame))    
log.close()

# Create .h5 input for cvae
cvae_input_dir = os.path.join(work_dir, 'cvae_input') 
make_dir_p(cvae_input_dir)

cvae_input_file = os.path.join(cvae_input_dir, 'cvae_input.h5')
cvae_input_save = h5py.File(cvae_input_file, 'w')
cvae_input_save.create_dataset('contact_maps', data=cvae_input)
cvae_input_save.close() 

# CVAE
hyper_dims = np.arange(n_cvae) + 3
print('\n===========================================')
print('Running CVAE for hyper dimension:', hyper_dims) 

for i in range(n_cvae): 
    cvae_j = cvae_job(time.time(), i, cvae_input_file, hyper_dim=hyper_dims[i]) 
    stop_jobs = jobs.get_job_from_gpu_id(i) 
    stop_jobs.stop()  
    print(' Started CVAE for hyper dimension:', hyper_dims[i])
    time.sleep(2)
    cvae_j.start() 
    jobs.append(cvae_j) 
    time.sleep(2)

    
while [os.path.isfile(cvae_j.model_weight) for cvae_j in jobs.get_cvae_jobs()] != [True] * len(jobs.get_cvae_jobs()): 
    time.sleep(.5)
print('CVAE jobs done. ') 

for cvae_j in jobs.get_cvae_jobs(): 
    cvae_j.state = 'FINISHED'

# get all the weights
model_weights = [cvae_j.model_weight for cvae_j in jobs.get_cvae_jobs()]
outliers_pdb_path = os.path.join(work_dir, 'outlier_pdbs')
make_dir_p(outliers_pdb_path)

outlier_pdb_files = []
restarted_points = []
cvae_input_length = len(cvae_input) 


# monitoring new outlier
iter_record = 0 
while True: 
    print('\n\n===================================')
    print('Starting iteration: ', iter_record)

    print('Counting number of frames in current step.') 
    cm_files_iter = sorted(glob('omm*/*_cm.h5')) 

    for cm_file in cm_files_iter: 
        if cm_file not in cm_files: 
            cm_files.append(cm_file)
            cm_data_lists.append(read_h5py_file(cm_file)) 
    
    for cm in cm_data_lists: 
        cm.refresh() 

    print(' Current number of total frames: ', frame_number(cm_data_lists)) 
    
    # Create openmm log file to track openmm traj information to backtrack 
    train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
    log = open(omm_log, 'w') 
    for i, n_frame in enumerate(train_data_length): 
        log.writelines("{} {}\n".format(cm_files[i], n_frame))    
    log.close()
    
    # Prep data fro cvae prediction
    cvae_input = cm_to_cvae(cm_data_lists)
    iter_cvae_data_length = len(cvae_input) 

    if iter_record % 100 == 0: 
        cvae_input_file = os.path.join(cvae_input_dir, 'cvae_input_{}.h5'.format(iter_record))
        cvae_input_save = h5py.File(cvae_input_file, 'w')
        cvae_input_save.create_dataset('contact_maps', data=cvae_input)
        cvae_input_save.close() 

    outlier_list = []
    for model_weight in model_weights: 
        print('Model latent dimension: ', int(model_weight[11]))
        for eps in np.arange(0.20, 2.0, 0.05): 
            outliers = np.squeeze(outliers_from_cvae(model_weight, cvae_input, hyper_dim=int(model_weight[11]), eps=eps))
            n_outlier = len(outliers)
            print('dimension = {0}, eps = {1:.2f}, number of outlier found: {2}'.format(
                model_weight[11], eps, n_outlier))
            if n_outlier <= 50: 
                outlier_list.append(outliers)
                break
    
    outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 
    
    print('\nPreparing to write new pdb files') 
    # write the pdb according the outlier indices
    traj_info = open('./scheduler_logs/openmm_log.txt', 'r').read().split()
    traj_dict = dict(zip(traj_info[::2], np.array(traj_info[1::2]).astype(int)))
    
    # Write the new outliers 
    n_outlier_iter = 0
    new_outlier_list = []
    for outlier in outlier_list_uni: 
        traj_file, num_frame = find_frame(traj_dict, outlier) 
        if num_frame == 0: 
            print('Detected initial point as outlier, skipping...') 
            continue
        outlier_pdb_file = os.path.join(outliers_pdb_path, '{}_{:06d}.pdb'.format(traj_file[:18], num_frame))
        new_outlier_list.append(outlier_pdb_file) 
        if outlier_pdb_file not in outlier_pdb_files: 
            print('Found a new outlier# {} at frame {} of {}'.format(outlier, num_frame, traj_file))
            outlier_pdb = write_pdb_frame(traj_file, pdb_file, num_frame, outlier_pdb_file) 
            print('     Written as {}'.format(outlier_pdb_file))
            outlier_pdb_files.append(outlier_pdb_file) 
            n_outlier_iter += 1

    for outlier_pdb_file in outlier_pdb_files: 
        if outlier_pdb_file not in new_outlier_list: 
            print('Old outlier {} is now connected to a cluster and removing it from the outlier list '.format(outlier_pdb_file[-29:]))
            outlier_pdb_files.remove(outlier_pdb_file) 

    # Sort the outliers according to their RMSD to the native structure 
    # Calculate the RMSD
    if ref_pdb_file: 
        outlier_traj = mda.Universe(outlier_pdb_files[0], outlier_pdb_files) 
        ref_traj = mda.Universe(ref_pdb_file) 
        R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
        R.run()    
        # Make a dict contains outliers and their RMSD
        outlier_pdb_RMSD = dict(zip(outlier_pdb_files, R.rmsd[:,2]))

    # Stop a simulation if len(traj) > 10k and no outlier in past 5k frames
    for job in jobs.get_running_omm_jobs(): 
        job_h5 = os.path.join(job.save_path, 'output_cm.h5') 
        assert (job_h5 in cm_files)
        job_n_frames = cm_data_lists[cm_files.index(job_h5)].shape[1] 
        print('The running job under {} has completed {} frames. '.format(job.save_path, job_n_frames))
        job_outlier_frames = [int(outlier[-10:-4]) for outlier in outlier_pdb_files if job.save_path in outlier] 
        if job_outlier_frames: 
            latest_outlier_pdb = max(job_outlier_frames) 
        else: 
            latest_outlier_pdb = 1e20
        if job_n_frames >= 2e4 and job_n_frames - latest_outlier_pdb >= 5e3: 
            print('Stopping running job under ', job.save_path) 
            job.stop()
            time.sleep(2) 

    if iter_cvae_data_length > float(cvae_data_length) * 1.6: 
        print('\nThe OpenMM simulation generated 1.6 times of original training data, retraining the cvae models. \n') 
        cvae_data_length = iter_cvae_data_length 

    # Start a new openmm simulation if there's GPU available 
    if jobs.get_available_gpu(GPU_ids): 
        print('\nRestarting OpenMM simulation on GPU', jobs.get_available_gpu(GPU_ids))
        if ref_pdb_file:
            print('Starting the simulation according the RMSD seq') 
            for gpu_id in jobs.get_available_gpu(GPU_ids): 
                omm_pdb_files_sorted = [outlier for outlier in sorted(outlier_pdb_RMSD, key=outlier_pdb_RMSD.get) \
                        if outlier not in restarted_points] 
                if len(restarted_points) % 4 == 3: 
                    omm_pdb_file = omm_pdb_files_sorted[-1] 
                    print('Starting simulation on highest RMSD outlier conformer: ', omm_pdb_file[-29:]) 
                elif len(restarted_points) % 4 == 2: 
                    omm_pdb_file = random.choice(omm_pdb_files_sorted) # [random.randrange(len(omm_pdb_files_sorted))]
                    print('Starting simulation on random outlier conformer: ', omm_pdb_file[-29:]) 
                else: 
                    omm_pdb_file = omm_pdb_files_sorted[0]
                    print('Starting simulation on the lowest RMSD outlier conformer: ', omm_pdb_file[-29:]) 
                job = omm_job(job_id=int(time.time()), gpu_id=gpu_id, top_file=top_file, pdb_file=omm_pdb_file)
                restarted_points.append(omm_pdb_file) 
                job.start()
                print('Restarted OMM simulation on {}'.format(gpu_id))
                jobs.append(job) 
                time.sleep(2)

            # Total 4; 1, 2: lowest RMSD; 3: highest RMSD; 4 random, tracked by restarted_points
        else: 
            for gpu_id in jobs.get_available_gpu(GPU_ids): 
                omm_pdb_files = [outlier for outlier in outlier_pdb_files if outlier not in restarted_points]
                omm_pdb_file = random.choice(omm_pdb_files)
                print('Starting simulation on random outlier conformer: ', omm_pdb_file[-29:]) 
                job = omm_job(job_id=int(time.time()), gpu_id=gpu_id, top_file=top_file, pdb_file=omm_pdb_file)
                restarted_points.append(omm_pdb_file) 
                job.start()
                print('Restarted OMM simulation on {}'.format(gpu_id))
                jobs.append(job) 
                time.sleep(2)
    else: 
        print('No GPU available, continuing...') 

    if n_outlier_iter: 
        print('Successfully found {} new outliers.'.format(n_outlier_iter)) 
    else: 
        print('Nothing found, next iter.') 

    iter_record += 1 
    print('Waiting for 5 min for next iter. \n\n')
    time.sleep(300)
 
print('Finishing and cleaning up the jobs. ')
subprocess.Popen('bash prerun_clean.sh &> closing.log'.split(" "))

