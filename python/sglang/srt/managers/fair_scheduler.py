# A deficit-tracking fair scheduler mixin based on the paper "Locality-aware
# Fair Scheduling in LLM Serving": https://arxiv.org/html/2501.14312v1
#
# This scheduler works by tracking per-client usage in tokens, and overridding
# the admission logic for the queue to allow only clients with remaining "quota"
# to be scheduled.
#
# This scheduler is compatible with any other scheduling policy, such as LPM or
# priority-based scheduling, and combining it with LPM yields the DLPM (deficit
# longest-prefix-match) scheduling approach from the paper.
#
# Yandex notes:
#
# This lives in a separate file to avoid complex rebasing issues, leaving only a
# few integration points in the main sglang code:
#
# 1. Scheduler must be changed to use _get_admission_iterator.
# 2. Scheduler must create and store DLPM scheduler when DLPM is in use.
# 3. Metrics mixin must call out to DLPM scheduler for collection.

import heapq
import time
import logging

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DeficitClient:
    """Tracks deficit state for a single client."""

    deficit: int = 0
    last_seen: float = 0
    token_delta: int = 0

    def seen(self) -> None:
        self.last_seen = time.perf_counter()

    def consume_tokens(self, tokens: int) -> None:
        self.deficit -= tokens
        self.token_delta += tokens
        self.seen()

    def refill(self, quantum: int) -> None:
        self.deficit += quantum
        self.seen()

    def get_token_delta(self) -> int:
        d = self.token_delta
        self.token_delta = 0
        return d


class DeficitMetrics:
    """
    Tracks fair scheduling metrics, such as the number of known/active clients
    and the token usage for top clients.
    """

    def __init__(self, export_top_clients):
        self.num_clients: int = 0
        self.refills_needed: int = 0
        self.active_clients: int = 0
        self.needs_flush: bool = False
        self.metrics_initialised: bool = False
        self.export_top_clients: bool = export_top_clients

        if export_top_clients:
            self.top_clients: List[Tuple[str, int]] = []
            self.known_clients = set()

    def initialise_metrics(self, labels):
        if self.metrics_initialised:
            return
        self.metrics_initialised = True

        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Gauge

        self.num_clients_metric = Gauge(
            name="sglang:deficit_clients_total",
            documentation="The total number of active clients being tracked by the fair scheduler at the moment.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.active_clients_metric = Gauge(
            name="sglang:deficit_active_clients",
            documentation="The number of active clients with positive token consumption in the last minute.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.refills_needed_metric = Gauge(
            name="sglang:deficit_refills_needed",
            documentation="The most deficit refills needed before progress could be made.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        if self.export_top_clients:
            self.top_clients_metric = Gauge(
                name="sglang:top_deficit_clients_tokens",
                documentation="The tokens consumed by top fair scheduler clients in the last minute.",
                labelnames=list(labels.keys()) + ["client"],
                multiprocess_mode="mostrecent",
            )

    def flush(self, labels):
        self.initialise_metrics(labels)

        self.num_clients_metric.labels(**labels).set(self.num_clients)
        self.refills_needed_metric.labels(**labels).set(self.refills_needed)
        self.active_clients_metric.labels(**labels).set(self.active_clients)

        if self.export_top_clients and self.needs_flush:
            self.needs_flush = False

            # Zero out metrics for clients that are no longer in top 10
            current_top_clients = {client_id for client_id, _ in self.top_clients}
            stale_clients = self.known_clients - current_top_clients
            for client_id in stale_clients:
                client_labels = {**labels, "client": client_id}
                self.top_clients_metric.labels(**client_labels).set(0)
                self.known_clients.remove(client_id)

            # Set metrics for current top clients
            for client_id, tokens_consumed in self.top_clients:
                client_labels = {**labels, "client": client_id}
                self.top_clients_metric.labels(**client_labels).set(tokens_consumed)
                self.known_clients.add(client_id)


class FairScheduler:
    """
    Deficit-tracking based fair scheduler. See package comment for details.
    """

    def __init__(
        self,
        prefill_token_cost,
        refill_quantum,
        export_top_clients,
    ):
        self.prefill_token_cost = prefill_token_cost
        self.refill_quantum = refill_quantum
        self.export_top_clients = export_top_clients

        self.clients: Dict[str, DeficitClient] = defaultdict(DeficitClient)
        self.last_metrics_update: float = time.perf_counter()
        self.metrics = DeficitMetrics(export_top_clients)

    def admission_iterator(self, waiting_queue: List[Req]):
        """
        Deficit-based admission iterator that with refill logic.

        Yields requests from clients with positive deficits. If no clients can be admitted
        in a full pass through the queue, refills all client deficits and continues.
        """
        remaining_requests = waiting_queue.copy()

        # track the maximum number of refills we had to do before any client
        # could progress, giving a sort of "fairness pressure" signal
        refills_needed = 0
        current_refill_streak = 0

        try:
            while remaining_requests:
                # Make a copy for iteration since we might modify remaining_requests
                current_pass_requests = remaining_requests.copy()
                admitted_any = False

                for req in current_pass_requests:
                    client_id = req.session_id or "<anonymous>"

                    # Only admit requests from clients with positive deficits
                    client = self.clients[client_id]
                    if client.deficit > 0:
                        if current_refill_streak > 0:
                            refills_needed = max(refills_needed, current_refill_streak)
                            current_refill_streak = 0

                        yield req
                        remaining_requests.remove(req)
                        admitted_any = True

                # If no requests could be admitted, refill ALL deficits
                # (including for backlogged clients not currently in the queue).
                if not admitted_any and remaining_requests:
                    for client in self.clients.values():
                        if client.deficit <= 0:
                            client.refill(self.refill_quantum)
                    current_refill_streak += 1

        finally:
            self.metrics.refills_needed = refills_needed

    def cleanup_inactive_clients(self):
        """Remove client state for clients that haven't been seen in an hour."""
        current_time = time.perf_counter()
        inactive_clients = [
            client_id
            for client_id, client in self.clients.items()
            if current_time - client.last_seen > 3600
        ]

        for client_id in inactive_clients:
            del self.clients[client_id]

        if inactive_clients:
            logger.info(
                f"Cleaned up {len(inactive_clients)} inactive deficit clients: {inactive_clients}"
            )

    def acknowledge_admission(self, req: Req):
        """
        Called by the scheduler when a request is actually admitted into a batch.
        """

        client_id = req.session_id or "<anonymous>"
        # Subtract extend tokens from client's deficit (with prefill cost multiplier)
        cost = int(req.extend_input_len * self.prefill_token_cost)
        self.clients[client_id].consume_tokens(cost)

    def account_decode_tokens(self, req: Req, tokens: int):
        """
        Called by the scheduler when output tokens are produced to account for them.
        """
        client_id = req.session_id or "<anonymous>"
        self.clients[client_id].consume_tokens(tokens)

    def update_metrics(self, labels):
        """Update fair scheduler related metrics."""
        self.metrics.num_clients = len(self.clients)

        # Only update clients metric once per minute
        current_time = time.perf_counter()
        if current_time - self.last_metrics_update >= 60:
            self.last_metrics_update = current_time

            client_tuples = []
            active_client_count = 0
            for client_id, client in self.clients.items():
                delta = client.get_token_delta()
                if delta > 0:
                    client_tuples.append((client_id, delta))
                    active_client_count += 1

            self.metrics.active_clients = active_client_count

            if self.export_top_clients:
                self.metrics.top_clients = heapq.nlargest(
                    10, client_tuples, key=lambda c: c[1]
                )
                self.metrics.needs_flush = True

        self.metrics.flush(labels)
