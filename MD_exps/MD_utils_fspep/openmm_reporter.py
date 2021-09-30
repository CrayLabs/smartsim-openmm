import time
import openmm.unit as u 
from smartredis import Client, Dataset
from smartredis.util import Dtypes

import numpy as np 
import h5py 
import os

from openmm.app import DCDFile
import io

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
    """The SmartSim contact map reporter will
    store the contact maps internally (every report interval).
    When the reporter is deleted (and thus the simulation is over),
    contact maps are put on the database, preprocessed and 
    the resulting samples (which can be used to train the CVAE)
    are concatenated to the existing ones.
    The concatenation is needed to maintain
    the ordering, which is crucial for the Outlier Search stage

    The argument `output_path` is used to track where files are written
    by other reporters. It could indicate a real filesystem path,
    or a virtual file, stored in the database, which can be written
    to the given path. Together with a timestamp, the `output_path`
    is stored at the end of the report.

    The implementation is based on the original ContactMapReporter
    as available in the DeepDriveMD legacy repository.

    :param reportInterval: the report interval
    :param output_path: destination path for other reporter files
    """
    def __init__(self, reportInterval, output_path):
        self._reportInterval = reportInterval
        self._output_path = output_path
        self._client = Client(address=None, cluster=bool(int(os.getenv("SS_CLUSTER", False))))
        dataset_name = os.getenv("SSKEYOUT")
        self._client.use_tensor_ensemble_prefix(False)
        self._dataset_prefix = "{"+dataset_name+"}."
        if self._client.key_exists(dataset_name):
           self._dataset = self._client.get_dataset(dataset_name)
           self._append = True
        else:
           self._dataset = Dataset(dataset_name)
           self._append = False
        self._out = None
        self._timestamp = str(time.time())

    def __del__(self):
        out = np.transpose(self._out).copy().astype(np.float32)
        traj_length = int(out.shape[1])
        batch_key = self._dataset_prefix+"batch"
        preproc_all_key = self._dataset_prefix+"preproc"
        if not self._append:
            preproc_batch_key = self._dataset_prefix+"preproc_0"
            self._client.put_tensor(batch_key, out)
            self._client.run_script("cvae_script",
                                    "cm_to_cvae",
                                    [batch_key],
                                    [preproc_batch_key])
            
            # preproc is the concatenated version used by outlier search
            self._client.copy_tensor(preproc_batch_key, preproc_all_key)
        else:
            md_iter = len(self._dataset.get_meta_strings("timestamps"))
            preproc_batch_key = self._dataset_prefix+f"preproc_{md_iter}"
            intermediate_key = self._dataset_prefix+"interm"
            self._client.delete_tensor(batch_key)
            self._client.put_tensor(batch_key, out)

            self._client.run_script("cvae_script",
                                    "cm_to_cvae",
                                    [batch_key],
                                    [preproc_batch_key])

            # preproc is the concatenated version used by outlier search                      
            self._client.run_script("cvae_script",
                                    "concatenate",
                                    [preproc_all_key, preproc_batch_key],
                                    [intermediate_key])
            self._client.delete_tensor(preproc_all_key)
            self._client.rename_tensor(intermediate_key, preproc_all_key)

        self._dataset.add_meta_scalar("cm_lengths", traj_length)

        print(f"Destroying reporter, final size of contact map: {out.shape}")
    
        self._dataset.add_meta_string("timestamps", self._timestamp)
        self._dataset.add_meta_string("paths", self._output_path)
        self._client.put_dataset(self._dataset)

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



class SmartSimDCDReporter(object):
    """SmartSimDCDReporter outputs a series of frames from a Simulation to a byte stream.

    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.

    This reporter is identical to openmm.app.DCDReporter, but allows to pass a byte
    stream instead of a filename as ``file``.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        """Create a DCDReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            mode = 'r+b'
        else:
            mode = 'wb'
        if isinstance(file, io.BytesIO):
            self._out = file
        else:
            self._out = open(file, mode)
        self._dcd = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        if self._dcd is None:
            self._dcd = DCDFile(
                self._out, simulation.topology, simulation.integrator.getStepSize(),
                simulation.currentStep, self._reportInterval, self._append
            )
        self._dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())

    def __del__(self):
        if not isinstance(self._out, io.BytesIO):
            self._out.close()