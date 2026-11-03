# SPDX-License-Identifier: Apache-2.0
"""CPU-only, fail-closed proposal metadata; never changes tokens or scheduling."""
from copy import deepcopy
from uuid import uuid4


class ProposalTracker:
    def __init__(self):
        self.worker = uuid4().hex
        self.generation = self.sequence = 0
        self.requests = {}
        self.pending = {}
        self.staged = {}
        self.matched = {}

    def remove(self, request_id):
        for mapping in (self.requests, self.pending, self.staged, self.matched):
            mapping.pop(request_id, None)

    def start(self, request_id):
        self.remove(request_id)
        self.generation += 1
        self.requests[request_id] = self.generation

    def publish(self, request_id, start, tokens, *, unsupported=False):
        self.pending.pop(request_id, None)
        if (unsupported or request_id not in self.requests or start < 0
                or len(tokens) != 2 or any(type(t) is not int or t < 0 for t in tokens)):
            return None
        self.sequence += 1
        value = dict(worker=self.worker, generation=self.requests[request_id],
                     request_id=request_id, sequence=self.sequence, target_start=start,
                     tokens=list(tokens), positions=[start+1, start+2])
        self.pending[request_id] = deepcopy(value)
        return value

    def begin_step(self, echoed, *, unsupported=False):
        # Claim once BEFORE preparation can fail. Unscheduled/stale proposals are
        # discarded too: safe demand-only fallback for unsupported pipelining.
        self.staged, self.matched = {}, {}
        pending, self.pending = self.pending, {}
        if unsupported:
            return
        for request_id, envelope in echoed.items():
            if (envelope == pending.get(request_id) and envelope is not None
                    and self.requests.get(request_id) == envelope['generation']):
                self.staged[request_id] = deepcopy(envelope)

    def verify(self, request_id, tokens, positions, *, unsupported=False):
        envelope = self.staged.pop(request_id, None)
        if (unsupported or envelope is None
                or positions != [envelope['target_start'], *envelope['positions']]
                or len(tokens) != 3 or tokens[1:] != envelope['tokens']):
            return None
        self.matched[request_id] = envelope
        return deepcopy(envelope)


def clear_identity_on_error(function):
    from functools import wraps
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception:
            tracker = getattr(self, '_residency_proposals', None)
            if tracker is not None:
                tracker.pending.clear()
                tracker.staged.clear()
                tracker.matched.clear()
            handler = getattr(self, 'draft_tokens_handler', None)
            if handler is not None and hasattr(handler, 'proposal_identities'):
                handler.proposal_identities = {}
            raise
    return wrapped
