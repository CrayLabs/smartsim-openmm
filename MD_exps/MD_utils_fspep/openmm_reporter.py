import time
import simtk.unit as u 
from smartredis import Client, Dataset
from smartredis.util import Dtypes

import numpy as np 
import h5py 
import os

from MDAnalysis.analysis import distances


class ContactMapReporter(object):
    def __init__(self, file, reportInterval):
        self._file = h5py.File(file, 'w', libver='latest')
        self._file.swmr_mode = True
        self._out = self._file.create_dataset('contact_maps', shape=(2,0), maxshape=(None, None))
        self._reportInterval = reportInterval

    def __del__(self):
        print(f"Destroying reporter, final size of contact map: {self._out.shape}")
        self._file.close()


    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        ca_indices = []
        for atom in simulation.topology.atoms():
            if atom.name == 'CA':
                ca_indices.append(atom.index)
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions_ca = positions[ca_indices].astype(np.float32)
        distance_matrix = distances.self_distance_array(positions_ca)
        contact_map = (distance_matrix < 8.0) * 1.0 
        new_shape = (len(contact_map), self._out.shape[1] + 1) 
        self._out.resize(new_shape)
        self._out[:, new_shape[1]-1] = contact_map
        self._file.flush()

class SmartSimContactMapReporter(object):
    def __init__(self, worker_id, reportInterval):
        self._reportInterval = reportInterval
        self._client = Client(address=None, cluster=bool(int(os.getenv("SS_CLUSTER", False))))
        self._worker_id = worker_id
        dataset_name = "openmm_" + str(self._worker_id)
        self.dataset_name = dataset_name
        if self._client.key_exists(dataset_name):
           self._dataset = self._client.get_dataset(dataset_name)
           self._append = True
        else:
           self._dataset = Dataset(dataset_name)
           self._append = False
        self._out = None
        self._timestamp = str(time.time())
        if not self._client.model_exists("cvae_script"):
            self._client.set_script_from_file("cvae_script",
                "/lus/scratch/arigazzi/smartsim-dev/smartsim-openmm/MD_to_CVAE/MD_to_CVAE_scripts.py",
                device="CPU")

    def __del__(self):
        out = np.transpose(self._out).copy().astype(np.float32)
        traj_length = int(out.shape[1])
        if not self._append:
            self._dataset.add_tensor(self._timestamp, self._out)
            self._client.put_tensor(f"batch_{self._worker_id}", out)
            self._dataset.add_meta_scalar("cm_lengths", traj_length)
        else:
            dtype = Dtypes.tensor_from_numpy(out)
            self._dataset.add_tensor(self._timestamp, out, dtype)

            batch = self._client.get_tensor(f"batch_{self._worker_id}")
            batch = np.hstack((batch, out)).copy().astype(np.float32)
            self._client.delete_tensor(f"batch_{self._worker_id}")
            self._client.put_tensor(f"batch_{self._worker_id}", batch)
            self._dataset.add_meta_scalar("cm_lengths", np.asarray(traj_length), Dtypes.tensor_from_numpy(np.asarray(traj_length)))

        print(f"Destroying reporter, final size of contact map: {out.shape}")
    
        self._dataset.add_meta_string("timestamps", self._timestamp)
        print(self._dataset.get_meta_strings("timestamps"))
        print(self._dataset.get_meta_scalars("cm_lengths"))
        if not self._append:
            self._client.put_dataset(self._dataset)
        else:
            super(type(self._client), self._client).put_dataset(self._dataset)
        self._client.run_script("cvae_script",
                                "cm_to_cvae",
                                f"batch_{self._worker_id}",
                                f"preproc_{self._worker_id}")
        stored = self._client.get_dataset(self.dataset_name)
        print(stored.get_meta_scalars("cm_lengths"))

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        ca_indices = []
        for atom in simulation.topology.atoms():
            if atom.name == 'CA':
                ca_indices.append(atom.index)
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions_ca = positions[ca_indices].astype(np.float32)
        distance_matrix = distances.self_distance_array(positions_ca)
        contact_map = (distance_matrix < 8.0) * 1.0 
        if self._out is None:
            self._out = np.empty(shape=(1, len(contact_map)))
            self._out[0,:] = np.transpose(contact_map)
        else:
            self._out = np.vstack((self._out, np.transpose(contact_map)))
