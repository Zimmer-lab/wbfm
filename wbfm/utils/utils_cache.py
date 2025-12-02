import logging
import os
from typing import Callable
from tables.exceptions import HDF5ExtError


def cache_to_disk(cache_filename: str,
                  load_from_disk: Callable, save_to_disk: Callable):
    """
    Outer function that takes the cache filename as an argument, and returns a decorator that takes a function as an
    argument, and returns a function.

    Note that this is a simplistic cache, and does not check whether the arguments to the function have changed. It simply
    checks whether the cache file exists, and if so, loads the data from disk. If not, it runs the function and saves the
    output to disk.

    Parameters
    ----------
    cache_filename
    save_to_disk
    load_from_disk

    Returns
    -------

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_filename):
                logging.info(f'Loading cached data from {cache_filename}, ignoring args and kwargs:'
                             f'args={args}, kwargs={kwargs}')
                return load_from_disk(cache_filename)
            else:
                logging.info(f'Cache file {cache_filename} does not exist. Running function and saving output to disk.'
                             f'args={args}, kwargs={kwargs}')
                output = func(*args, **kwargs)
                save_to_disk(output, cache_filename)
        return wrapper
    return decorator


def cache_to_disk_class(cache_filename_method: str,
                        func_load_from_disk: Callable,
                        func_save_to_disk: Callable,
                        **cache_kwargs):
    """
    Similar to cache_to_disk, but for class methods.

    The cache file is not passed directly, but is instead passed as a class attribute. This is because the cache file
    name is not known until the class is instantiated.

    Example usage:
    @cache_to_disk_class('cache_filename', load_from_disk=pd.read_csv, save_to_disk=pd.to_csv)
    def my_func(self, *args, **kwargs):
        ...

    Additional kwargs in the decorator call will be passed to cache_filename_method

    Parameters
    ----------
    cache_filename_method: the name of the class method that returns the cache filename
    func_save_to_disk: the function to use to save the data to disk
        Signature: func_save_to_disk(cache_filename: str, data: Any)
    func_load_from_disk: the function to use to load the data from disk
        Signature: func_load_from_disk(cache_filename: str) -> Any
    cache_kwargs

    Returns
    -------

    """
    def decorator(func):
        def wrapper(self):
            cache_filename = getattr(self, cache_filename_method)(**cache_kwargs)
            if cache_filename is not None and os.path.exists(cache_filename):
                logging.info(f'Loading cached data from {cache_filename}')
                return func_load_from_disk(cache_filename)
            else:
                logging.info(f'Cache file {cache_filename} does not exist. Running function and saving output to disk.')
                output = func(self)
                try:
                    func_save_to_disk(cache_filename, output)
                except (PermissionError, AttributeError, TypeError, HDF5ExtError):
                    logging.warning(f'Could not save cache file {cache_filename} to disk. '
                                    f'Permission denied. Continuing without saving.')
                return output

        # For later debugging or introspection
        wrapper._decorator_args = dict(cache_filename_method=cache_filename_method,
                                       func_load_from_disk=func_load_from_disk,
                                       func_save_to_disk=func_save_to_disk,
                                       cache_kwargs=cache_kwargs)
        return wrapper
    return decorator
