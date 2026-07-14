"""Keep the test process hermetic against operator configuration.

``tokenclaw.server`` calls ``load_dotenv()`` at import time. python-dotenv walks
up from the working directory, so on an operator machine it finds the live
orchestration ``.env`` (managed mode, recommendation server URL, shadow
keep-thinking, family opt-outs) and loads it into ``os.environ`` — from that
point every test inherits production behavior and fails or passes depending on
what the operator is currently running. Import the module here, before any test
module can, then strip every TOKENCLAW_*/AGENTFLOW_* variable so tests start
from the documented defaults. Tests that need managed behavior set their own
variables explicitly.
"""

from __future__ import annotations

import os

import tokenclaw.server  # noqa: F401  (triggers the one-time load_dotenv())

for _key in list(os.environ):
    if _key.startswith(("TOKENCLAW_", "AGENTFLOW_")):
        del os.environ[_key]
del _key
