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

from multiprocessing import Process

import pytest
import networkx as nx

import cirq
from cirq import CircuitOperation, FrozenCircuit, ops, NamedQubit, LineQubit
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily, SWAP
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import ops_are_consistent_with_device_graph


# Helper to get all non-SWAP operations from a SwapNetwork or list of ops
def _get_non_swap_operations(operations: cirq.OP_TREE) -> list[cirq.Operation]:
    if isinstance(operations, SwapNetwork):
        operations = operations.circuit.all_operations()
    return [op for op in operations if not isinstance(op.gate, cca.SwapPermutationGate)]


# Helper to get all operations from a SwapNetwork
def _get_operations_from_swap_network(swap_network: SwapNetwork) -> list[cirq.Operation]:
    return list(swap_network.circuit.all_operations())


def test_bad_args() -> None:
    """Test zero valued arguments in greedy router."""
    circuit = cirq.testing.random_circuit(4, 2, 0.5, random_state=5)
    device_graph = ccr.get_grid_device_graph(3, 2)
    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_search_radius=0)

    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_num_empty_steps=0)


def create_circuit_and_device():
    """Construct a small circuit and a device with line connectivity
    to test the greedy router. This instance hangs router in Cirq 8.2.
    """
    num_qubits = 6
    gate_domain = {ops.CNOT: 2}
    circuit = cirq.testing.random_circuit(num_qubits, 15, 0.5, gate_domain, random_state=37)
    device_graph = ccr.get_linear_device_graph(num_qubits)
    return circuit, device_graph


def create_hanging_routing_instance(circuit, device_graph):
    """Create a test problem instance."""
    route_circuit_greedily(  # pragma: no cover
        circuit, device_graph, max_search_radius=2, random_state=1
    )


def test_router_hanging() -> None:
    """Run a separate process and check if greedy router hits timeout (20s)."""
    circuit, device_graph = create_circuit_and_device()
    process = Process(target=create_hanging_routing_instance, args=[circuit, device_graph])
    process.start()
    process.join(timeout=20)
    try:
        assert not process.is_alive(), "Greedy router timeout"
    finally:
        process.terminate()


class TestGreedyRouterRecursive:
    @pytest.fixture(scope="class")
    def line_device_3q(self):
        # p0 -- p1 -- p2
        phys_q = LineQubit.range(3)
        graph = nx.Graph()
        graph.add_edges_from([(phys_q[0], phys_q[1]), (phys_q[1], phys_q[2])])
        return phys_q, graph

    @pytest.fixture(scope="class")
    def line_device_4q(self):
        # p0 -- p1 -- p2 -- p3
        phys_q = LineQubit.range(4)
        graph = nx.Graph()
        graph.add_edges_from([(phys_q[i], phys_q[i+1]) for i in range(3)])
        return phys_q, graph


    def test_basic_recursive_no_swaps(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq = NamedQubit("main_lq0")
        sub_lq = NamedQubit("sub_lq0")

        sub_circuit = FrozenCircuit(ops.H(sub_lq))
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_circuit, qubits=(main_lq,))
        )
        initial_mapping = {main_lq: phys_q[0]}

        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)

        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        assert len(routed_ops) == 1
        assert routed_ops[0].gate == ops.H
        assert routed_ops[0].qubits == (phys_q[0],)

    def test_basic_recursive_with_swaps(self, line_device_3q):
        phys_q, device_graph = line_device_3q # p0-p1-p2
        main_lq = NamedQubit.range(2, prefix="main_lq") # main_lq0, main_lq1
        sub_lq = NamedQubit.range(2, prefix="sub_lq")   # sub_lq0, sub_lq1

        # Sub-circuit expects CNOT(sub_lq0, sub_lq1)
        # Main circuit maps this to CNOT(main_lq0, main_lq1)
        # Initial mapping places main_lq0 -> p0, main_lq1 -> p2. Needs SWAP.
        sub_circuit = FrozenCircuit(ops.CNOT(sub_lq[0], sub_lq[1]))
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_circuit, qubits=(main_lq[0], main_lq[1]))
        )
        initial_mapping = {main_lq[0]: phys_q[0], main_lq[1]: phys_q[2]}

        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping, max_search_radius=1
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)

        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 1
        assert non_swap_ops[0].gate == ops.CNOT

        # Verify the CNOT is on adjacent physical qubits
        cnot_op = non_swap_ops[0]
        assert device_graph.has_edge(*cnot_op.qubits)
        assert len(routed_ops) >= 2 # At least 1 CNOT and 1 SWAP

        # Verify final logical qubit positions
        current_map = {v: k for k, v in initial_mapping.items()} # phys -> log
        for op in routed_ops:
            if op.gate == SWAP: # Using the SWAP from greedy.py
                p1, p2 = op.qubits
                lq1, lq2 = current_map.get(p1), current_map.get(p2)
                if lq1 is not None: current_map[p2] = lq1
                else: current_map.pop(p2, None)
                if lq2 is not None: current_map[p1] = lq2
                else: current_map.pop(p1, None)
        
        final_log_to_phys = {v:k for k,v in current_map.items()}
        assert final_log_to_phys[main_lq[0]] == cnot_op.qubits[0]
        assert final_log_to_phys[main_lq[1]] == cnot_op.qubits[1]


    def test_nested_circuit_operation(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        # Logical qubits for different levels
        main_lq_ = NamedQubit.range(2, prefix="main_lq") # main_lq0, main_lq1
        mid_lq_ = NamedQubit.range(2, prefix="mid_lq")   # mid_lq0, mid_lq1 (used by sub_c1)
        inner_lq_ = NamedQubit("inner_lq0")              # used by sub_c2

        # sub_c2: Innermost circuit
        sub_c2 = FrozenCircuit(ops.Y(inner_lq_))
        # sub_c1: Middle circuit, contains CircuitOperation(sub_c2)
        # sub_c1 effectively performs X(mid_lq0) then Y(mid_lq1)
        sub_c1 = FrozenCircuit(
            ops.X(mid_lq_[0]),
            CircuitOperation(sub_c2, qubits=(mid_lq_[1],))
        )
        # Main circuit: contains CircuitOperation(sub_c1)
        # Main circuit effectively performs X(main_lq0), Y(main_lq1), then H(main_lq0)
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_c1, qubits=(main_lq_[0], main_lq_[1])),
            ops.H(main_lq_[0])
        )
        initial_mapping = {main_lq_[0]: phys_q[0], main_lq_[1]: phys_q[1]}

        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        
        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 3 # X, Y, H
        # Expected: X(p0), Y(p1), H(p0) because no swaps needed with this mapping
        assert non_swap_ops[0].gate == ops.X and non_swap_ops[0].qubits == (phys_q[0],)
        assert non_swap_ops[1].gate == ops.Y and non_swap_ops[1].qubits == (phys_q[1],)
        assert non_swap_ops[2].gate == ops.H and non_swap_ops[2].qubits == (phys_q[0],)
        assert len(routed_ops) == 3


    def test_multiple_circuit_operations_sequential(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq_ = NamedQubit.range(2, prefix="main_lq") # main_lq0, main_lq1
        sub1_lq_ = NamedQubit("sub1_lq0")
        sub2_lq_ = NamedQubit("sub2_lq0")

        sub_c1 = FrozenCircuit(ops.H(sub1_lq_))
        sub_c2 = FrozenCircuit(ops.Z(sub2_lq_))
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_c1, qubits=(main_lq_[0],)),
            CircuitOperation(sub_c2, qubits=(main_lq_[1],))
        )
        initial_mapping = {main_lq_[0]: phys_q[0], main_lq_[1]: phys_q[1]}
        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 2
        assert non_swap_ops[0].gate == ops.H and non_swap_ops[0].qubits == (phys_q[0],)
        assert non_swap_ops[1].gate == ops.Z and non_swap_ops[1].qubits == (phys_q[1],)
        assert len(routed_ops) == 2


    def test_multiple_circuit_operations_interspersed(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq_ = NamedQubit.range(3, prefix="main_lq") # m0,m1,m2
        sub_lq_ = NamedQubit("sub_lq0")

        sub_c1 = FrozenCircuit(ops.H(sub_lq_))
        main_circuit = cirq.Circuit(
            ops.X(main_lq_[0]),
            CircuitOperation(sub_c1, qubits=(main_lq_[1],)),
            ops.Y(main_lq_[2])
        )
        initial_mapping = {main_lq_[i]: phys_q[i] for i in range(3)}
        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 3
        assert non_swap_ops[0].gate == ops.X and non_swap_ops[0].qubits == (phys_q[0],)
        assert non_swap_ops[1].gate == ops.H and non_swap_ops[1].qubits == (phys_q[1],)
        assert non_swap_ops[2].gate == ops.Y and non_swap_ops[2].qubits == (phys_q[2],)
        assert len(routed_ops) == 3

    def test_qubit_mapping_different_subsets_and_reuse(self, line_device_4q):
        phys_q, device_graph = line_device_4q # p0-p1-p2-p3
        main_lq_ = NamedQubit.range(4, prefix="main_lq") # m0,m1,m2,m3
        sub_log_ = NamedQubit.range(2, prefix="sub_log") # sub_log0, sub_log1

        sub_circuit = FrozenCircuit(ops.CZ(sub_log_[0], sub_log_[1]))
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_circuit, qubits=(main_lq_[0], main_lq_[1])), # CZ(m0,m1)
            ops.X(main_lq_[3]), # X(m3)
            CircuitOperation(sub_circuit, qubits=(main_lq_[2], main_lq_[3])), # CZ(m2,m3)
            # Reuse main_lq0, main_lq1. Their physical positions should be the same as before
            # if no swaps occurred involving them.
            CircuitOperation(sub_circuit, qubits=(main_lq_[0], main_lq_[1]))  # CZ(m0,m1)
        )
        initial_mapping = {main_lq_[i]: phys_q[i] for i in range(4)}
        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        non_swap_ops = _get_non_swap_operations(routed_ops)
        
        assert len(non_swap_ops) == 4 # 3 CZs, 1 X
        # Expected physical ops if no swaps: CZ(p0,p1), X(p3), CZ(p2,p3), CZ(p0,p1)
        assert non_swap_ops[0].gate == ops.CZ and non_swap_ops[0].qubits == (phys_q[0], phys_q[1])
        assert non_swap_ops[1].gate == ops.X and non_swap_ops[1].qubits == (phys_q[3],)
        assert non_swap_ops[2].gate == ops.CZ and non_swap_ops[2].qubits == (phys_q[2], phys_q[3])
        assert non_swap_ops[3].gate == ops.CZ and non_swap_ops[3].qubits == (phys_q[0], phys_q[1])
        assert len(routed_ops) == 4


    def test_device_graph_constraints_force_sub_swap(self, line_device_3q):
        phys_q, device_graph = line_device_3q # p0-p1-p2
        main_lq_ = NamedQubit.range(2, prefix="main_lq")
        sub_lq_ = NamedQubit.range(2, prefix="sub_lq")

        # Sub-circuit needs CNOT(sub0, sub1)
        # Main maps this to CNOT(main0, main1)
        # Initial mapping for main0->p0, main1->p2.
        # The sub-router should perform a SWAP to make CNOT(p0,p2) possible.
        sub_circuit = FrozenCircuit(ops.CNOT(sub_lq_[0], sub_lq_[1]))
        main_circuit = cirq.Circuit(
            CircuitOperation(sub_circuit, qubits=(main_lq_[0], main_lq_[1]))
        )
        initial_mapping = {main_lq_[0]: phys_q[0], main_lq_[1]: phys_q[2]}

        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping, max_search_radius=1
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        
        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 1
        assert non_swap_ops[0].gate == ops.CNOT
        assert device_graph.has_edge(*non_swap_ops[0].qubits) # CNOT is on adjacent phys qubits
        assert len(routed_ops) > len(non_swap_ops) # At least one SWAP must have occurred


    def test_empty_circuit_operation(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq0 = NamedQubit("main_lq0")

        main_circuit = cirq.Circuit(
            CircuitOperation(FrozenCircuit(), qubits=()), # Empty CircuitOp
            ops.X(main_lq0)
        )
        initial_mapping = {main_lq0: phys_q[0]}
        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        assert len(routed_ops) == 1
        assert routed_ops[0].gate == ops.X and routed_ops[0].qubits == (phys_q[0],)

    def test_circuit_operation_single_gate(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq_ = NamedQubit.range(2, prefix="main_lq")
        sub_lq_ = NamedQubit("sub_lq0")

        sub_circuit = FrozenCircuit(ops.H(sub_lq_))
        main_circuit = cirq.Circuit(
            ops.X(main_lq_[0]),
            CircuitOperation(sub_circuit, qubits=(main_lq_[1],)),
            ops.Y(main_lq_[0])
        )
        initial_mapping = {main_lq_[0]: phys_q[0], main_lq_[1]: phys_q[1]}
        swap_network = route_circuit_greedily(
            main_circuit, device_graph, initial_mapping=initial_mapping
        )
        routed_ops = _get_operations_from_swap_network(swap_network)
        assert ops_are_consistent_with_device_graph(routed_ops, device_graph)
        non_swap_ops = _get_non_swap_operations(routed_ops)
        assert len(non_swap_ops) == 3
        assert non_swap_ops[0].gate == ops.X and non_swap_ops[0].qubits == (phys_q[0],)
        assert non_swap_ops[1].gate == ops.H and non_swap_ops[1].qubits == (phys_q[1],)
        assert non_swap_ops[2].gate == ops.Y and non_swap_ops[2].qubits == (phys_q[0],)
        assert len(routed_ops) == 3

    def test_circuit_op_unmapped_qubits_raises_error(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq_ = NamedQubit.range(2, prefix="main_lq") # m0, m1
        sub_lq_ = NamedQubit("sub_lq0")

        sub_circuit = FrozenCircuit(ops.H(sub_lq_))
        # COp uses main_lq1, but it's not in initial_mapping
        main_circuit = cirq.Circuit(CircuitOperation(sub_circuit, qubits=(main_lq_[1],)))
        initial_mapping = {main_lq_[0]: phys_q[0]}

        with pytest.raises(ValueError, match="is not mapped to a physical qubit"):
            route_circuit_greedily(main_circuit, device_graph, initial_mapping=initial_mapping)

    def test_circuit_op_duplicate_physical_qubits_in_sub_mapping_raises_error(self, line_device_3q):
        phys_q, device_graph = line_device_3q
        main_lq_ = NamedQubit.range(2, prefix="main_lq") # m0, m1
        sub_lq_ = NamedQubit.range(2, prefix="sub_lq")   # s0, s1

        sub_circuit = FrozenCircuit(ops.CNOT(sub_lq_[0], sub_lq_[1]))
        # COp maps its two logical qubits (via main_lq0, main_lq1) to the *same* physical qubit p0
        main_circuit = cirq.Circuit(CircuitOperation(sub_circuit, qubits=(main_lq_[0], main_lq_[1])))
        initial_mapping = {main_lq_[0]: phys_q[0], main_lq_[1]: phys_q[0]}

        with pytest.raises(ValueError, match="Duplicate physical qubits in initial mapping"):
            route_circuit_greedily(main_circuit, device_graph, initial_mapping=initial_mapping)
