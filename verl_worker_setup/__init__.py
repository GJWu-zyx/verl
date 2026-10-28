# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ray startup hooks must not import device runtimes before accelerator assignment."""

import os
import sys


# TransferQueue 0.1.8 eagerly imports the optional Mooncake native binding even
# when the configured storage backend is SimpleStorage.  The Mooncake wheel in
# the Ascend runtime corrupts the glibc heap during interpreter teardown, so a
# successful training step otherwise exits with SIGABRT (134).  Make the
# optional bindings unavailable before TransferQueue and checkpoint engines
# import them. Users
# that actually configure Mooncake can opt back in explicitly after installing
# a compatible wheel.
def configure_transfer_queue_worker() -> None:
    """Apply the optional native-backend guard before standalone TQ actors import."""
    if os.getenv("VERL_TRANSFER_QUEUE_ENABLE_MOONCAKE", "0").lower() not in {"1", "true", "yes"}:
        sys.modules.setdefault("mooncake.store", None)
        sys.modules.setdefault("mooncake.engine", None)
