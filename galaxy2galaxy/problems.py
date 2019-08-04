"""Access G2G Problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from galaxy2galaxy.data_generators import all_problems
from galaxy2galaxy.utils import registry

def problem(name):
  return registry.problem(name)

def available():
  return sorted(registry.list_problems())

# Import problem modules
_modules = list(all_problems.MODULES)

all_problems.import_modules(_modules)
