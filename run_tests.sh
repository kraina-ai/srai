source /home/shoaib/scratch/venv/bin/activate
# pip install .
# pip install --ignore-installed six
# python -m pip install --ignore-installed six
# python -m pip install --ignore-installed pytest
# python -m pip install --ignore-installed torch
# python -m pip install --ignore-installed lightning_fabric
# python -m pip install srai[torch]
PYTHONPATH=. pytest tests/embedders/geovex/test_embedder.py::test_embedder_save_load