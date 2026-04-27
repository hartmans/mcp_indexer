# Implementation Status
* lance_indexer/plugins/base.py is fairly solid. If you think you want to change that file  seek human approval; describe the problem and a proposed fix.
* architecture.md states the direction we are going in. Adding new sections is fine; revise sections only with approval. If you run into areas where the architecture is problematic, then ask for feedback.
* Files commited to git are current files that largely follow the current architecture.

* The rest of the code is notional and is probably not aligned with the architecture or base plugin specs. Use the rest of the code to identify challenges you will face but assume the rest of the code will be rewritten.


* Use ~/venv/bin/python as your python

* Tests are pytest, use monkeypatch and build your own mocks rather than using unittest facilities.
