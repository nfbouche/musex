import dataset
import logging
import yaml

__all__ = ('load_yaml_config', 'load_db')


def load_yaml_config(filename):
    """Load a YAML config file, with string substitution."""
    with open(filename, 'r') as f:
        conftext = f.read()
        conf = yaml.load(conftext)
        return yaml.load(conftext.format(**conf))


def load_db(filename, verbose=False, **kwargs):
    """Open a sqlite database with dataset."""
    if not verbose:
        dataset.persistence.database.log.addHandler(logging.NullHandler())
    return dataset.connect('sqlite:///{}'.format(filename), **kwargs)
