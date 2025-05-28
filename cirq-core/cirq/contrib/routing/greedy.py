# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from typing import Callable, cast, Iterable, Sequence, TYPE_CHECKING

import networkx as nx
import numpy as np

import cirq.contrib.acquaintance as cca
from cirq import circuits, ops, value, CircuitOperation
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph

if TYPE_CHECKING:
    import cirq

SWAP = cca.SwapPermutationGate()
QidPair = tuple[ops.Qid, ops.Qid]


def route_circuit_greedily(
    circuit: circuits.Circuit, device_graph: nx.Graph, **kwargs
) -> SwapNetwork:
    """Greedily routes a circuit on a given device.

    Alternates between heuristically picking a few SWAPs to change the mapping
    and applying all logical operations possible given the new mapping, until
    all logical operations have been applied.

    The SWAP selection heuristic is as follows. In every iteration, the
    remaining two-qubit gates are partitioned into time slices. (See
    utils.get_time_slices for details.) For each set of candidate SWAPs, the new
    mapping is computed. For each time slice and every two-qubit gate therein,
    the distance of the two logical qubits in the device graph under the new
    mapping is calculated. A candidate set 'S' of SWAPs is taken out of
    consideration if for some other set 'T' there is a time slice such that all
    of_the distances for 'T' are at most those for 'S' (and they are not all
    equal).

    If more than one candidate remains, the size of the set of SWAPs considered
    is increased by one and the process is repeated. If after considering SWAP
    sets of size up to 'max_search_radius', more than one candidate remains,
    then the pairs of qubits in the first time slice are considered, and those
    farthest away under the current mapping are brought together using SWAPs
    using a shortest path in the device graph.

    `CircuitOperation`s within the input circuit are handled recursively. The
    router will route the sub-circuit contained within a `CircuitOperation` by
    making a recursive call to `route_circuit_greedily`. The `initial_mapping`
    for this sub-circuit is determined by the current mapping of its logical
    qubits (as defined by the `CircuitOperation`'s `qubits` argument) to
    physical qubits in the parent router's state. Any SWAPs performed during
    the routing of the sub-circuit will update this mapping, and these changes
    will persist for subsequent operations in the parent circuit.

    Args:
        circuit: The circuit to route. This can be a `cirq.Circuit` which may
            contain `cirq.CircuitOperation`s.
        device_graph: The device's graph, in which each vertex is a qubit
            and each edge indicates the ability to do an operation on those
            qubits.
        **kwargs: Further keyword args, including
            max_search_radius: The maximum number of disjoint device edges to
                consider routing on.
            max_num_empty_steps: The maximum number of swap sets to apply
                without allowing a new logical operation to be performed.
            initial_mapping: The initial mapping of physical to logical qubits
                to use. Defaults to a greedy initialization.
            can_reorder: A predicate that determines if two operations may be
                reordered.
            random_state: Random state or random state seed.
    """

    router = _GreedyRouter(circuit, device_graph, **kwargs)
    router.route()

    swap_network = router.swap_network
    swap_network.circuit = circuits.Circuit(swap_network.circuit.all_operations())
    return swap_network


class _GreedyRouter:
    """Keeps track of the state of a greedy circuit routing procedure."""

    def __init__(
        self,
        circuit,
        device_graph: nx.Graph,
        *,
        max_search_radius: int = 1,
        max_num_empty_steps: int = 5,
        initial_mapping: dict[ops.Qid, ops.Qid] | None = None,
        can_reorder: Callable[[ops.Operation, ops.Operation], bool] = lambda op1, op2: not set(
            op1.qubits
        )
        & set(op2.qubits),
        random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ):

        self.prng = value.parse_random_state(random_state)

        self.device_graph = device_graph
        self.physical_distances: dict[QidPair, int] = {
            (a, b): d
            for a, neighbor_distances in nx.shortest_path_length(device_graph)
            for b, d in neighbor_distances.items()
        }

        self.top_level_operations: list[ops.Operation] = list(circuit.all_operations())
        self.logical_qubits = list(circuit.all_qubits()) # All qubits in the original circuit
        self.physical_qubits = list(self.device_graph.nodes)
        self.edge_sets: dict[int, list[Sequence[QidPair]]] = {}
        self.physical_ops: list[ops.Operation] = []

        # remaining_dag will be built dynamically for segments of non-CircuitOperations in route()
        self.remaining_dag = circuitdag.CircuitDag(can_reorder=can_reorder) # Initialize as empty
        self.can_reorder = can_reorder # Store for later DAG creation

        self.set_initial_mapping(initial_mapping)

        if max_search_radius < 1:
            raise ValueError('max_search_radius must be a positive integer.')
        self.max_search_radius = max_search_radius

        if max_num_empty_steps < 1:
            raise ValueError('max_num_empty_steps must be a positive integer.')
        self.max_num_empty_steps = max_num_empty_steps

    def get_edge_sets(self, edge_set_size: int) -> Iterable[Sequence[QidPair]]:
        """Returns matchings of the device graph of a given size."""
        if edge_set_size not in self.edge_sets:
            self.edge_sets[edge_set_size] = [
                cast(Sequence[QidPair], edge_set)
                for edge_set in itertools.combinations(self.device_graph.edges, edge_set_size)
                if all(set(e).isdisjoint(f) for e, f in itertools.combinations(edge_set, 2))
            ]
        return self.edge_sets[edge_set_size]

    def log_to_phys(self, *qubits: cirq.Qid) -> Iterable[ops.Qid]:
        """Returns an iterator over the physical qubits mapped to by the given
        logical qubits."""
        return (self._log_to_phys[q] for q in qubits)

    def phys_to_log(self, *qubits: cirq.Qid) -> Iterable[ops.Qid | None]:
        """Returns an iterator over the logical qubits that map to the given
        physical qubits."""
        return (self._phys_to_log[q] for q in qubits)

    def apply_swap(self, *physical_edges: QidPair):
        """Applies SWAP on the given edges."""
        self.update_mapping(*physical_edges)
        self.physical_ops += [SWAP(*e) for e in physical_edges]

    def update_mapping(self, *physical_edges: QidPair):
        """Updates the mapping in accordance with SWAPs on the given physical
        edges."""
        for physical_edge in physical_edges:
            old_logical_edge = tuple(self.phys_to_log(*physical_edge))
            new_logical_edge = old_logical_edge[::-1]
            for p, l in zip(physical_edge, new_logical_edge):
                self._phys_to_log[p] = l
                if l is not None:
                    self._log_to_phys[l] = p

    def set_initial_mapping(self, initial_mapping: dict[ops.Qid, ops.Qid] | None = None):
        """Sets the internal state according to an initial mapping.

        Args:
            initial_mapping: The mapping to use. If not given, one is found
                greedily.
        """

        if initial_mapping is None:
            time_slices = get_time_slices(self.remaining_dag)
            if not time_slices:
                initial_mapping = dict(zip(self.device_graph, self.logical_qubits))
            else:
                logical_graph = time_slices[0]
                logical_graph.add_nodes_from(self.logical_qubits)
                initial_mapping = get_initial_mapping(logical_graph, self.device_graph, self.prng)
        self.initial_mapping = initial_mapping
        self._phys_to_log = {q: initial_mapping.get(q) for q in self.physical_qubits}
        self._log_to_phys = {l: p for p, l in self._phys_to_log.items() if l is not None}
        self._assert_mapping_consistency()

    def _assert_mapping_consistency(self):
        # All physical qubits must be in _phys_to_log.
        assert sorted(self._phys_to_log.keys()) == sorted(self.physical_qubits)

        # For every logical qubit mapped in _log_to_phys:
        # 1. It must be one of the circuit's logical_qubits.
        # 2. Its corresponding physical qubit in _phys_to_log must map back to it.
        for lq, pq in self._log_to_phys.items():
            assert lq in self.logical_qubits, f"Mapped logical qubit {lq} not in overall circuit qubits."
            assert self._phys_to_log[pq] == lq, f"Inconsistent mapping for {lq} and {pq}."

        # For every physical qubit mapped in _phys_to_log (to a non-None logical qubit):
        # 1. The logical qubit must be one of the circuit's logical_qubits.
        # 2. It must be correctly registered in _log_to_phys.
        # (This also checks that no two physical qubits map to the same logical qubit)
        mapped_lq_count = {}
        for pq, lq in self._phys_to_log.items():
            if lq is not None:
                assert lq in self.logical_qubits, f"Phys qubit {pq} mapped to unknown logical qubit {lq}."
                assert self._log_to_phys[lq] == pq, f"Inconsistent mapping for {lq} and {pq} (from phys_to_log)."
                mapped_lq_count[lq] = mapped_lq_count.get(lq, 0) + 1
        
        for lq, count in mapped_lq_count.items():
            assert count == 1, f"Logical qubit {lq} is mapped from multiple physical qubits."


    def acts_on_nonadjacent_qubits(self, op: ops.Operation) -> bool:
        # Check if all qubits involved in the operation are currently mapped.
        if not all(q in self._log_to_phys for q in op.qubits):
            return True  # Not all qubits are mapped, so cannot be placed.
        # Single-qubit operations are fine if the qubit is mapped.
        if len(op.qubits) == 1:
            return False
        # Multi-qubit operations require mapped qubits to be adjacent on the device.
        return tuple(self.log_to_phys(*op.qubits)) not in self.device_graph.edges

    def apply_possible_ops(self) -> int:
        """Applies all logical operations possible given the current mapping from self.remaining_dag."""
        if not self.remaining_dag or not self.remaining_dag.nodes():
            return 0

        applied_count = 0
        # findall_nodes_until_blocked respects DAG dependencies and blocker conditions.
        # The is_blocker (self.acts_on_nonadjacent_qubits) checks for mappability and adjacency.
        nodes_to_process = list(
            self.remaining_dag.findall_nodes_until_blocked(self.acts_on_nonadjacent_qubits)
        )

        for node in nodes_to_process:
            logical_op = node.val
            # At this point, acts_on_nonadjacent_qubits(logical_op) should be False for these nodes.
            # This means all qubits are mapped and adjacent if multi-qubit.
            
            # Double check mapping and adjacency before applying (defensive)
            if self.acts_on_nonadjacent_qubits(logical_op):
                # This case should ideally not be reached if findall_nodes_until_blocked works as expected.
                # It might indicate a concurrent modification or a subtle issue in logic.
                continue

            physical_op = logical_op.with_qubits(*self.log_to_phys(*logical_op.qubits))
            # Ensure the physical operation is valid on the device (already checked by acts_on_nonadjacent_qubits for 2-qubit)
            assert len(physical_op.qubits) < 2 or physical_op.qubits in self.device_graph.edges
            
            self.physical_ops.append(physical_op)
            self.remaining_dag.remove_node(node)
            applied_count += 1
        
        return applied_count

    @property
    def swap_network(self) -> SwapNetwork:
        return SwapNetwork(circuits.Circuit(self.physical_ops), self.initial_mapping)

    def distance(self, edge: QidPair) -> int:
        """The distance between the physical qubits mapped to by a pair of
        logical qubits."""
        return self.physical_distances[cast(QidPair, tuple(self.log_to_phys(*edge)))]

    def swap_along_path(self, path: tuple[ops.Qid]):
        """Adds SWAPs to move a logical qubit along a specified path."""
        for i in range(len(path) - 1):
            self.apply_swap(cast(QidPair, path[i : i + 2]))

    def bring_farthest_pair_together(self, pairs: Sequence[QidPair]):
        """Adds SWAPs to bring the farthest-apart pair of logical qubits
        together."""
        distances = [self.distance(pair) for pair in pairs]
        assert distances
        max_distance = min(distances)
        farthest_pairs = [pair for pair, d in zip(pairs, distances) if d == max_distance]
        choice = self.prng.choice(len(farthest_pairs))
        farthest_pair = farthest_pairs[choice]
        edge = self.log_to_phys(*farthest_pair)
        shortest_path = nx.shortest_path(self.device_graph, *edge)
        assert len(shortest_path) - 1 == max_distance
        midpoint = max_distance // 2
        self.swap_along_path(shortest_path[:midpoint])
        self.swap_along_path(shortest_path[midpoint:])

    def get_distance_vector(self, logical_edges: Iterable[QidPair], swaps: Sequence[QidPair]):
        """Gets distances between physical qubits mapped to by given logical
        edges, after specified SWAPs are applied."""
        self.update_mapping(*swaps)
        distance_vector = np.array([self.distance(e) for e in logical_edges])
        self.update_mapping(*swaps)
        return distance_vector

    def apply_next_swaps(self, require_frontier_adjacency: bool = False):
        """Applies a few SWAPs to get the mapping closer to one in which the
        next logical gates can be applied.

        See route_circuit_greedily for more details.
        """

        time_slices = get_time_slices(self.remaining_dag)

        if require_frontier_adjacency:
            frontier_edges = sorted(time_slices[0].edges)
            self.bring_farthest_pair_together(frontier_edges)
            return

        for k in range(1, self.max_search_radius + 1):
            candidate_swap_sets = list(self.get_edge_sets(k))
            for time_slice in time_slices:
                edges = sorted(time_slice.edges)
                distance_vectors = list(
                    self.get_distance_vector(edges, swap_set) for swap_set in candidate_swap_sets
                )
                dominated_indices = _get_dominated_indices(distance_vectors)
                candidate_swap_sets = [
                    S for i, S in enumerate(candidate_swap_sets) if i not in dominated_indices
                ]
                if len(candidate_swap_sets) == 1:
                    self.apply_swap(*candidate_swap_sets[0])
                    if list(
                        self.remaining_dag.findall_nodes_until_blocked(
                            self.acts_on_nonadjacent_qubits
                        )
                    ):
                        return
                    else:
                        break

        self.apply_next_swaps(True)

    def route(self):
        op_iterator = iter(self.top_level_operations)
        current_op = None # To store a lookahead operation if needed
        empty_steps_remaining = self.max_num_empty_steps

        while True:
            if self.remaining_dag and self.remaining_dag.nodes():
                # Try to apply ops from the current non-CircuitOperation DAG segment
                n_applied_ops = self.apply_possible_ops()
                if n_applied_ops:
                    empty_steps_remaining = self.max_num_empty_steps
                    # Continue processing this segment in the next iteration
                    continue 
                else:
                    # No operations applied, need to perform swaps
                    self.apply_next_swaps(not empty_steps_remaining)
                    empty_steps_remaining -= 1
                    if empty_steps_remaining < 0:
                        # Stall detected
                        # TODO: Implement a more robust stall recovery or error mechanism
                        # For now, if apply_next_swaps was forced (empty_steps_remaining was 0),
                        # and still no ops could be applied, the circuit might be unroutable
                        # or stuck in a local minimum.
                        # We will let it try to fetch next segment if DAG is exhausted by swaps.
                        pass
                    # Continue to try processing this segment (or what's left of it)
                    continue

            # If remaining_dag is empty or exhausted, fetch the next operation(s)
            if current_op is None:
                current_op = next(op_iterator, None)

            if current_op is None: # All top-level operations processed
                if not (self.remaining_dag and self.remaining_dag.nodes()):
                    # And the last segment's DAG is also empty/done
                    break 
                else:
                    # Still processing the last segment's DAG, loop back
                    continue


            if isinstance(current_op, CircuitOperation):
                sub_circuit_op = current_op
                current_op = None # Consume this CircuitOperation

                sub_circuit = sub_circuit_op.circuit
                sub_qubits = sub_circuit_op.qubits

                sub_initial_mapping: dict[ops.Qid, ops.Qid] = {}
                for lq in sub_qubits:
                    if lq not in self._log_to_phys:
                        # This implies a logical qubit for the sub-circuit is not currently mapped.
                        # This is a complex scenario. The greedy router expects an initial mapping
                        # for all qubits it's supposed to route.
                        # One strategy could be to try to find a place for unmapped qubits first.
                        # For now, this is an error condition.
                        raise ValueError(
                            f"Logical qubit {lq} in CircuitOperation {sub_circuit_op} "
                            "is not mapped to a physical qubit. Cannot route sub-circuit."
                        )
                    sub_initial_mapping[lq] = self._log_to_phys[lq]
                
                # Ensure all physical qubits in sub_initial_mapping are distinct
                if len(set(sub_initial_mapping.values())) != len(sub_initial_mapping):
                    raise ValueError(
                        f"Duplicate physical qubits in initial mapping for CircuitOperation {sub_circuit_op}."
                    )


                # Recursively call route_circuit_greedily for the sub-circuit
                # Note: random_state (self.prng) is passed to ensure determinism if needed.
                # can_reorder could be self.can_reorder or a specific one for sub-circuits.
                sub_swap_network = route_circuit_greedily(
                    circuit=sub_circuit,
                    device_graph=self.device_graph, # Use the full device graph
                    initial_mapping=sub_initial_mapping,
                    max_search_radius=self.max_search_radius,
                    max_num_empty_steps=self.max_num_empty_steps,
                    can_reorder=self.can_reorder, 
                    random_state=self.prng,
                )

                self.physical_ops.extend(sub_swap_network.circuit.all_operations())

                # Update the main mapping based on the swaps in the sub_swap_network
                # The swaps in sub_swap_network are on physical qubits.
                # These physical qubits were initially mapped according to sub_initial_mapping.
                # We need to apply these physical swaps to whatever logical qubits were on them.
                temp_phys_to_log = self._phys_to_log.copy()
                for op_in_sub_swap_net in sub_swap_network.circuit.all_operations():
                    if op_in_sub_swap_net.gate == SWAP:
                        p1, p2 = op_in_sub_swap_net.qubits # Physical qubits that were swapped
                        
                        # The logical qubits on p1 and p2 might not be part of the sub-circuit's
                        # explicit logical qubits if the sub-circuit was small and swaps affected
                        # adjacent physical qubits holding other main-circuit logical qubits.
                        l1_on_p1 = temp_phys_to_log.get(p1)
                        l2_on_p2 = temp_phys_to_log.get(p2)
                        
                        temp_phys_to_log[p1] = l2_on_p2
                        temp_phys_to_log[p2] = l1_on_p1
                
                self._phys_to_log = temp_phys_to_log
                self._log_to_phys = {
                    l: p for p, l in self._phys_to_log.items() if l is not None
                }
                
                self._assert_mapping_consistency()
                empty_steps_remaining = self.max_num_empty_steps # Activity occurred
                # Loop back to potentially process more ops or next segment
                continue

            else: # current_op is a regular operation
                # Collect a segment of regular (non-CircuitOperation) ops
                current_regular_ops_segment: list[ops.Operation] = []
                if current_op is not None: # Add the op we peeked or started with
                     current_regular_ops_segment.append(current_op)
                     current_op = None # Consume it

                # Greedily fetch more non-CircuitOperations
                while True:
                    op_lookahead = next(op_iterator, None)
                    if op_lookahead is None or isinstance(op_lookahead, CircuitOperation):
                        current_op = op_lookahead # Save for next main loop iteration
                        break
                    current_regular_ops_segment.append(op_lookahead)

                if current_regular_ops_segment:
                    self.remaining_dag = circuitdag.CircuitDag.from_ops(
                        *current_regular_ops_segment, can_reorder=self.can_reorder
                    )
                    # Logical qubits for this segment are already part of self.logical_qubits.
                    # Mapping for them should exist or be created by placement logic (currently error if not mapped).
                    empty_steps_remaining = self.max_num_empty_steps
                    # Loop back to process this new DAG
                    continue
        
        # Final assertion
        assert ops_are_consistent_with_device_graph(self.physical_ops, self.device_graph)


def _get_dominated_indices(vectors: list[np.ndarray]) -> set[int]:
    """Get the indices of vectors that are element-wise at least some other
    vector.
    """
    dominated_indices = set()
    for i, v in enumerate(vectors):
        for w in vectors[:i] + vectors[i + 1 :]:
            if all(v >= w):
                dominated_indices.add(i)
                break
    return dominated_indices
