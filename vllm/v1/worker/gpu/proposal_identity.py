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
        self.events = []

    def remove(self, request_id):
        if hasattr(self, 'observer'):
            self.observer.discard('request_removed', request_id)
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
            if hasattr(self, 'observer'):
                self.observer.discard('unsupported_step')
            return
        for request_id, envelope in echoed.items():
            if (envelope == pending.get(request_id) and envelope is not None
                    and self.requests.get(request_id) == envelope['generation']):
                self.staged[request_id] = deepcopy(envelope)
        if hasattr(self, 'observer'):
            self.observer.retain(list(self.staged.values()))

    def verify(self, request_id, tokens, positions, *, unsupported=False):
        envelope = self.staged.pop(request_id, None)
        if len(self.events) < 128:
            self.events.append(dict(request_id=request_id, envelope=deepcopy(envelope),
                actual_tokens=list(tokens), actual_positions=list(positions),
                matched=not unsupported and envelope is not None
                and positions == [envelope['target_start'], *envelope['positions']]
                and len(tokens) == 3 and tokens[1:] == envelope['tokens']))
        if (unsupported or envelope is None
                or positions != [envelope['target_start'], *envelope['positions']]
                or len(tokens) != 3 or tokens[1:] != envelope['tokens']):
            if hasattr(self, 'observer'):
                self.observer.discard('verification_mismatch', request_id)
            return None
        if hasattr(self, 'observer') and not getattr(self.observer, 'guided', False):
            self.observer.consume(envelope)  # Guided mode consumes at runner pass start.
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
                if hasattr(tracker, 'observer'):
                    tracker.observer.discard('execution_error')
                tracker.pending.clear()
                tracker.staged.clear()
                tracker.matched.clear()
            model = getattr(self, 'model', None)
            execution = getattr(model, '_residency_execution', None)
            if execution is not None and execution.guided:
                execution.abort_forward('runner_exception')
            speculator = getattr(self, 'speculator', None)
            if speculator is not None and hasattr(speculator, '_residency_features'):
                speculator._residency_features = None
            handler = getattr(self, 'draft_tokens_handler', None)
            if handler is not None and hasattr(handler, 'proposal_identities'):
                handler.proposal_identities = {}
            raise
    return wrapped
