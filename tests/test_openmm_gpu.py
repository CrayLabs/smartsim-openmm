import openmm.app as app
import openmm as omm
import openmm.unit as u

import parmed as pmd
import random
import os

def openmm_simulate_amber_fs_pep(pdb_file, GPU_index=0,
        report_time=10*u.picoseconds, sim_time=10*u.nanoseconds,output_path='.'):
    """
    Start and run an OpenMM NVT simulation with Langevin integrator at 2 fs 
    time step and 300 K. The cutoff distance for nonbonded interactions were 
    set at 1.2 nm and LJ switch distance at 1.0 nm, which commonly used with
    Charmm force field. Long-range nonbonded interactions were handled with PME.  

    Parameters
    ----------
    pdb_file : coordinates file (.gro, .pdb, ...)
        This is the molecule configuration file contains all the atom position
        and PBC (periodic boundary condition) box in the system. 
   
    check_point : None or check point file to load 
        
    GPU_index : Int or Str 
        The device # of GPU to use for running the simulation. Use Strings, '0,1'
        for example, to use more than 1 GPU
  
    output_traj : the trajectory file (.dcd)
        This is the file stores all the coordinates information of the MD 
        simulation results. 
  
    output_log : the log file (.log) 
        This file stores the MD simulation status, such as steps, time, potential
        energy, temperature, speed, etc.
 
    output_cm : the h5 file contains contact map information

    report_time : 10 ps
        The program writes its information to the output every 10 ps by default 

    sim_time : 10 ns
        The timespan of the simulation trajectory
    """

    pdb = pmd.read_PDB(pdb_file)
    forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, 
            nonbondedCutoff=1.0*u.nanometer, constraints=app.HBonds)

    dt = 0.002*u.picoseconds
    integrator = omm.LangevinIntegrator(300*u.kelvin, 91.0/u.picosecond, dt)
    integrator.setConstraintTolerance(0.00001)

    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        properties = {'DeviceIndex': str(GPU_index), 'CudaPrecision': 'mixed'}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        properties = {'DeviceIndex': str(GPU_index)}

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setPositions(random.choice(pdb.get_coordinates())/10) #parmed \AA to OpenMM nm

    # equilibrate
    simulation.minimizeEnergy() 
    simulation.context.setVelocitiesToTemperature(300*u.kelvin, random.randint(1, 10000))
    simulation.step(int(100*u.picoseconds / (2*u.femtoseconds)))

    report_freq = int(report_time/dt)
    simulation.reporters.append(app.StateDataReporter('output_log.txt',
                                report_freq, step=True, time=True, speed=True,
                                potentialEnergy=True, temperature=True, totalEnergy=True))
    nsteps = int(sim_time/dt)
    simulation.step(nsteps)


if __name__ == '__main__': 
    openmm_simulate_amber_fs_pep('/lus/theta-fs0/projects/datascience/arigazzi/smartsim-dev/smartsim-openmm/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                                GPU_index="0",
                                report_time=50*u.picoseconds,
                                sim_time=float(5)*u.nanoseconds)