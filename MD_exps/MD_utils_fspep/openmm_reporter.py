import simtk.openmm.app as app
import simtk.openmm as omm
import simtk.unit as u 
from smartredis import Client

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
    def __init__(self, dataset_name, reportInterval):
        self._reportInterval = reportInterval
        self._client = Client(cluster=True)
        self._dataset_name = dataset_name
        self._out = np.empty(shape=(2,0))
        print(os.environ["SSDB"])

    def __del__(self):
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
        self._client.put_tensor(self._dataset_name, self._out)
        # TODO: move aggregation to here?