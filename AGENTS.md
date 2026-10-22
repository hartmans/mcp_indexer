# Implementation Status
* lance_indexer/plugins/base.py is fairly solid. If you think you want to change that file  seek human approval; describe the problem and a proposed fix.
* architecture.md states the direction we are going in. Adding new sections is fine; revise sections only with approval. If you run into areas where the architecture is problematic, then ask for feedback.
* Files commited to git are current files that largely follow the current architecture.

* The rest of the code is notional and is probably not aligned with the architecture or base plugin specs. Use the rest of the code to identify challenges you will face but assume the rest of the code will be rewritten.


* Use ~/venv/bin/python as your python

* Tests are pytest, use monkeypatch and build your own mocks rather than using unittest facilities.

# Testing
* The slow real-conversion Docling tests in tests/test_docling.py are marked `long_cpu` (each takes ~17-32s: PDF OCR on CPU). For a fast run, exclude them with `pytest -m "not long_cpu"`; the unmarked Docling infrastructure tests still run.

# Documentation placement

`architecture.md` describes high-level interfaces, abstractions, component
responsibilities, and architectural direction. Write connected prose organized
into useful sections, rather than bullet lists. Include contracts that other
components depend on and invariants that should survive optimization or
replacement of an implementation.

When deciding whether something belongs in architecture, ask: Would changing it
require changes elsewhere? Could there be more than one implementation or instance?
What should stay the same if we optimize or replace it? Must someone understand
it to add features or extend another component? Affirmative answers, especially
to more than one question, indicate architectural material.

`README.md` is the current home for user-facing documentation, including the
configuration schema, defaults, setup, commands, operation, and user-visible API
behavior. Configuration schema belongs there even when it is also useful to
developers. Propose additional user-facing documents when the material warrants
them, with links from the README.

Docstrings and nearby code comments describe detailed method contracts, local
algorithms, storage mechanics, and implementation-specific choices that do not
need to be understood across components. Keep those details close to the code
rather than turning the architecture document into an implementation reference.

Design plans are temporary working documents. Before retiring a plan, integrate
its lasting architectural decisions into `architecture.md`, its usage and
configuration details into user-facing documentation, and its local implementation
contracts into docstrings or comments. Maintained documentation must stand on its
own after the plan is deleted; implementation history and completed task lists
need not be carried forward.
