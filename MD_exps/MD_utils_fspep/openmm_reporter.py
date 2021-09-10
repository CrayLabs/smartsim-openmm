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
        # time = int(np.round(state.getTime().value_in_unit(u.picosecond)))
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
        self._client = Client(cluster=True)
        dataset_name = "openmm_" + str(worker_id)
        try:
            self._dataset = self._client.get_dataset(dataset_name)
            self._append = True
        except:
            self._dataset = Dataset(dataset_name)
            self._append = False
        self._out = np.empty(shape=(2,0))
        self._timestamp = str(time.time())
        print(os.environ["SSDB"])

    def __del__(self):
        if not self._append:
            self._dataset.add_tensor(self._timestamp, self._out)
        else:
            dtype = Dtypes.tensor_from_numpy(self._out)
            self._dataset.add_tensor(self._timestamp, self._out, dtype)
    
        self._dataset.add_meta_string("timestamps", self._timestamp)
        if not self._append:
            self._client.put_dataset(self._dataset)
        else:
            super(type(self._client), self._client).put_dataset(self._dataset)
        print(f"Destroying reporter, final size of contact map: {self._out.shape}")

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
        # TODO: move aggregation to here?