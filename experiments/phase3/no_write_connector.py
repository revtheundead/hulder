"""The D1 run's B no-op arm: vLLM 0.30.0's ExampleHiddenStatesConnector without its disk write.

The hidden states are still cached during the forward pass by the extract_hidden_states drafter,
so the arm costs what B costs before anything reads them: the cache-only layer's memory, the
drafter's forward and the extra decode token. Loaded by `vllm serve` through
kv_connector_module_path, with use_synchronization_lock off so no lock files are made either.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)


class NoWriteHiddenStatesConnector(ExampleHiddenStatesConnector):
    def request_finished(self, request, block_ids):
        # The stock connector queues a save here and holds the blocks until it's copied out.
        # Free them at once and return nothing to the client instead.
        self._request_filenames.pop(request.request_id, None)
        return False, None

    def get_finished(self, finished_req_ids):
        # Nothing is ever sent, so no request is reported as done sending. The stock method reports
        # every finished request, which the scheduler then looks up after it has already freed it.
        return None, None
