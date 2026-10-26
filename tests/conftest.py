import importlib.util
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_circular", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_num_threads(2)
