from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.utils.registry import *

# Adds a subsection to the registries to store specific G2G problems
registry.Registries.g2g_problems = registry.Registry("g2g_problems", validator=registry._problem_name_validator, on_set=registry._on_problem_set)
registry.Registries.g2g_hparams = registry.Registry("g2g_hparams", value_transformer=registry._hparams_value_transformer)

# Defines decorator
register_problem = lambda x: registry.register_problem(registry.Registries.g2g_problems.register(x))
register_hparams = lambda x: registry.register_hparams(registry.Registries.g2g_hparams.register(x))

# Overrides registry queries
list_g2g_problems = lambda: sorted(Registries.g2g_problems)
list_problems = list_g2g_problems
list_all_problems = list_base_problems

list_g2g_hparams = lambda: sorted(Registries.g2g_hparams)
#list_hparams = list_g2g_hparams
#list_all_hparams = registry.list_hparams
