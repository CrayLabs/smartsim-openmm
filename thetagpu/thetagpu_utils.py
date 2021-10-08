import os


def generate_rankfiles(md_nodes: list, ml_nodes: list, processes_per_node: int, base_path: str):
    """Generate rank files for openmpi process binding. We assume there are 128 cpus per host and
    distribute them over processes. Each process is launched with a different call to `mpirun`,
    thus it is always rank 0.

    The path to the directory containing the rank files is returned.

    :param md_nodes: list of hosts on which MD simulations will run
    :type md_nodes: list[str]
    :param ml_nodes: list of hosts on which ML trainings will run
    :type ml_nodes: list[str]
    :param processes_per_node: number of processes to run on each host
    :type processes_per_node: int
    :returns: path to directory containing rank files
    :rtype: str 
    """

    rankfile_dir = os.path.join(base_path, "rankfiles")
    os.makedirs(rankfile_dir, exist_ok=True)

    num_md = len(md_nodes)*processes_per_node
    for md_idx in range(num_md):
        filename = os.path.join(rankfile_dir, f"md_ranks_{md_idx}")
        first_slot = 128//processes_per_node * (md_idx%processes_per_node)
        last_slot = 128//processes_per_node * ((md_idx%processes_per_node)+1) - 1
        with open(filename, 'w') as rankfile:
            rankfile.write(f"rank 0={md_nodes[md_idx//processes_per_node]} slot={first_slot}-{last_slot}")
    
    num_ml = len(ml_nodes)*processes_per_node
    for ml_idx in range(num_ml):
        filename = os.path.join(rankfile_dir, f"ml_ranks_{ml_idx}")
        first_slot = 128//processes_per_node * (ml_idx%processes_per_node)
        last_slot = 128//processes_per_node * ((ml_idx%processes_per_node)+1) - 1
        with open(filename, 'w') as rankfile:
            rankfile.write(f"rank 0={ml_nodes[ml_idx//processes_per_node]} slot={first_slot}-{last_slot}")

    return rankfile_dir

def assign_hosts(db_node_count: int, md_node_count: int, ml_node_count: int):
    nodefile_name = os.getenv('COBALT_NODEFILE')
    with open(nodefile_name, 'r') as nodefile:
        hosts = [node.strip("\n") for node in nodefile.readlines()]

    base_host = 0
    db_hosts = hosts[base_host:db_node_count]
    base_host += db_node_count
    md_hosts = hosts[base_host:base_host+md_node_count]
    base_host += md_node_count
    ml_hosts = hosts[base_host:base_host+ml_node_count]
    base_host += ml_node_count
    outlier_hosts = hosts[-1]

    return db_hosts, md_hosts, ml_hosts, outlier_hosts