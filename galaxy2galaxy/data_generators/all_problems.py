"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import six
from six.moves import range  # pylint: disable=redefined-builtin

MODULES = [
    "galaxy2galaxy.data_generators.hsc"
]
# Modules that depend on galsim, only include if available
try:
  import galsim
  MODULES += ["galaxy2galaxy.data_generators.cosmos"]
except:
  print("Could not import GalSim, excluding some data generators")

ALL_MODULES = list(MODULES)


def _is_import_err_msg(err_str, module):
  parts = module.split(".")
  suffixes = [".".join(parts[i:]) for i in range(len(parts))]
  return err_str in (
      ["No module named %s" % suffix for suffix in suffixes] +
      ["No module named '%s'" % suffix for suffix in suffixes])


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  log_all = True  # pylint: disable=unused-variable
  err_msg = "G2G: skipped importing {num_missing} data_generators modules."
  print(err_msg.format(num_missing=len(errors)))
  for module, err in errors:
    err_str = str(err)
    if not _is_import_err_msg(err_str, module):
      print("From module %s" % module)
      raise err
    if log_all:
      print("Did not import module: %s; Cause: %s" % (module, err_str))


def import_modules(modules):
  errors = []
  for module in modules:
    try:
      importlib.import_module(module)
    except ImportError as error:
      errors.append((module, error))
  _handle_errors(errors)
