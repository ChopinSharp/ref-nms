export PYTHONPATH=$PYTHONPATH:$PWD
python tools/build_vocab.py && python tools/build_refdb.py && python tools/build_ctxdb.py