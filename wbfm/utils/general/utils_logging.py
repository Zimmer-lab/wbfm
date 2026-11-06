import logging
import sys
from pathlib import Path

from wbfm.utils.general.utils_filenames import get_sequential_filename


def setup_logger_object(log_filename, actually_set_up_file=False):
    # Log to a file and the console
    # https://docs.python.org/3/howto/logging-cookbook.html
    log_name = Path(log_filename).stem

    # Main log object
    logger = logging.getLogger(f'{log_name}')
    logger.setLevel(logging.DEBUG)

    # Set up file handler
    try:
        if actually_set_up_file:
            log_filename = get_sequential_filename(log_filename, verbose=0)
            logger.info(f"Setting up log at: {log_filename}")
            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    except (PermissionError, OSError):
        # Assume we are reading someone else's project, so we shouldn't need a log
        # (OSError covers read-only filesystems, errno 30, not just PermissionError)
        pass

    # Check if the console logger has already been set up, and don't duplicate
    if all([isinstance(h, logging.FileHandler) for h in logger.handlers]):
        ch = _get_console_handler()
        logger.addHandler(ch)
    else:
        # Assume it has already been set up, so only set up the file handler
        pass

    logger.propagate = False
    # logger.info(f"Set up logger with name: {log_name}")

    return logger


def setup_root_logger(log_filename):
    # Same as setup_logger_object, but sets the global logger instead of returning an object
    log_filename = get_sequential_filename(log_filename)
    print(f"Setting up root log at: {log_filename}")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_filename,
                        filemode='w')

    ch = _get_console_handler()

    # Add the handlers to the root logger
    logger = logging.getLogger('')
    logger.addHandler(ch)

    logger.info("Set up logger")
    logging.info("From main logging")
    logger.debug("Debug")
    logger.warning("warn")

    # From tutorial
    # logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    # logging.debug('This message should go to the log file')
    # logging.info('So should this')
    # logging.warning('And this, too')

    return logger


def _get_console_handler():
    # Set up console handler (simplified compared to the file)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    ch.setFormatter(formatter)
    return ch


if __name__ == "__main__":
    setup_root_logger('/home/charles/dlc_stacks/gfp-ZIM2319_worm7-SHORT/log/test.log')

    setup_logger_object('/home/charles/dlc_stacks/gfp-ZIM2319_worm7-SHORT/log/test_obj.log')
