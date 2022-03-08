from smartredis import Client, Dataset
from smartredis.error import RedisReplyError
from io import StringIO, BytesIO
import os
import time

def put_text_file(filename, client: Client, overwrite=False):
    """Read a text file and store it as dataset on the orchestrator.

    A Dataset will be created, with the file ame as key. If
    `overwrite` is set to ``True``, the Dataset is deleted if it
    already exists and replaced with the new file.
    If `overwrite` is set to ``False``, a ``IOError`` is raised

    :param filename: File to put on dataset
    :type filename: str
    :param client: Orchestrator client
    :type client: SmartRedis.Client
    :param overwrite: Whether the dataset should be 
                        overwritten if it already exists
    :type overwrite: bool
    """

    if client.dataset_exists(filename):
        if overwrite:
            client.delete_dataset(filename)
        else:
            raise IOError(f"File {filename} already exists in database.")
    
    dataset = Dataset(filename)

    with open(filename, 'r') as file:
        for line in file:
            dataset.add_meta_string("content", line)
    
    # dataset.add_meta_string("path", os.path.dirname(os.path.abspath(filename)))
    client.put_dataset(dataset)


def put_strings_as_file(filename, strings, client: Client, overwrite=False):
    """Put a list of strings on database, giving it attributes as a file.

    :param filename: Name of file 
    :param strings: Strings to store on the dataset
    :param client: Client to orchestrator
    :param overwrite: [description]. Defaults to False.

    """
    # file_basename = os.path.basename(filename)

    if client.dataset_exists(filename):
        if overwrite:
            client.delete_dataset(filename)
        else:
            raise IOError(f"File {filename} already exists in database.")
    
    dataset = Dataset(filename)

    for line in strings:
        dataset.add_meta_string("content", line)
    
    # if filename == file_basename:
    #     path_to_file = os.path.abspath(os.curdir)
    # else:
    #     path_to_file = os.path.dirname(filename)

    # dataset.add_meta_string("path", path_to_file)
    client.put_dataset(dataset)


def get_text_file(filename, client: Client):
    """Get text file from Orchestrator.

    The file name is used as a key in the
    database. If the key does not exists, a ``IOError`` is
    raised.

    :param filename: Name of file 
    :type filename: str
    :param client: Client to Orchestrator
    :type client: SmartRedis.Client
    :returns: Content of text file
    :rtype: list[str]
    """
    # file_basename = os.path.basename(filename)
    attempts = 5
    while attempts>0:
        try:
            return client.get_dataset(filename).get_meta_strings("content")
        except RedisReplyError:
            attempts -= 1
            time.sleep(5)
    
    raise IOError(f"File {filename} does not exist in database.")


def get_text_stream(filename, client: Client):
    """Get text file from Orchestrator.

    The file name is used as a key in the
    database. If the key does not exists, a ``IOError`` is
    raised.

    :param filename: Name of file 
    :type filename: str
    :param client: Client to Orchestrator
    :type client: SmartRedis.Client
    :returns: Content of text file
    :rtype: list[str]
    """
    file_content = get_text_file(filename, client)
    return StringIO(initial_value="\n".join(file_content))


def save_text_file(filename, client: Client, exist_ok=True, path=None):
    """Store a text file contained in dataset to file system.
    
    If `path` is ``None``, `filename` is used as  destination to store the file content.
    If `path` is not ``None``, the file will be stored as 
    ``os.path.join(path, os.path.basename(filename))``

    The value of `filename` is used as key to get the dataset from
    the database. If the key does not exist, a `IOError` is raised.

    :param filename: Name of file to write. 
    :type filename: str
    :param client: Client to orchestrator
    :type client: SmartRedis.Client
    :param exists_ok: Whether the path to the file must be created if it
                      already exists. Defaults to ``True``.
    :type exist_ok: bool
    :param path: File system path on which the file must be saved. If set
                 to ``None``, the path will be inferred from `filename`.
    """

    if not client.dataset_exists(filename):
        raise IOError(f"File {filename} does not exist on database.")

    dataset = client.get_dataset(filename)

    file_basename = os.path.basename(filename)
    if path is None:
        path = os.path.dirname(filename)

    filename = os.path.join(path, file_basename)

    os.makedirs(path, exist_ok=exist_ok)

    with open(filename, 'w') as file:
        for line in dataset.get_meta_strings("content"):
            file.write(line)


def put_bytes_as_file(filename, content, client, overwrite):
    """Put bytes on the DB -- WARNING: this is an experimental feature

    The binary file is converted to a string using ``latin-1`` encoder
    which provides a 1:1 bytes:char mapping.

    Args:
        filename ([type]): [description]
        :param content: The content of the binary file to store
        :type content: bytes
        client ([type]): [description]
    """

    str_content = content.decode('latin1')
    put_strings_as_file(filename, [str_content], client, overwrite)


def get_binary_file(filename, client):
    """Get the content of a stored binary file -- WARNING this is an experimental feature

    Args:
        filename ([type]): [description]
        client ([type]): [description]
    """

    str_content = get_text_file(filename, client)
    byte_content = str_content[0].encode('latin1')

    return byte_content


def get_binary_stream(filename, client):

    return BytesIO(initial_bytes=get_binary_file(filename, client))


def save_binary_file(filename, client: Client, exist_ok=True, path=None):
    """Store a text file contained in dataset to file system.
    
    If `path` is ``None``, `filename` is used as  destination to store the file content.
    If `path` is not ``None``, the file will be stored as 
    ``os.path.join(path, os.path.basename(filename))``

    The value of `filename` is used as key to get the dataset from
    the database. If the key does not exist, a `IOError` is raised.

    :param filename: Name of file to write. 
    :type filename: str
    :param client: Client to orchestrator
    :type client: SmartRedis.Client
    :param exists_ok: Whether the path to the file must be created if it
                      already exists. Defaults to ``True``.
    :type exist_ok: bool
    :param path: File system path on which the file must be saved. If set
                 to ``None``, the path will be inferred from `filename`.
    """

    if not client.dataset_exists(filename):
        raise IOError(f"File {filename} does not exist on database.")

    dataset = client.get_dataset(filename)

    file_basename = os.path.basename(filename)
    if path is None:
        path = os.path.dirname(filename)

    filename = os.path.join(path, file_basename)

    os.makedirs(path, exist_ok=exist_ok)

    with open(filename, 'wb') as file:
        for line in dataset.get_meta_strings("content"):
            file.write(line.encode('latin1'))
