from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import torch.fx as fx
from typing import Dict, Any


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        self.fwd_nodes: List[fx.Node] = []
        self.bwd_nodes: List[fx.Node] = []
        self.node_type: Dict[fx.Node, NodeType] = {}

        # last forward use: the last fwd-region node that consumes this activation
        self.act_last_fwd_use: Dict[fx.Node, fx.Node] = {}
        # first backward use: the first bwd-region node that consumes this activation
        self.act_first_bwd_use: Dict[fx.Node, fx.Node] = {}

        # node -> list of runtimes (ms) across profile iterations
        self.node_runtimes: Dict[fx.Node, List[float]] = {}
        # node -> list of memory deltas (bytes) across profile iterations
        self.node_memory_deltas: Dict[fx.Node, List[int]] = {}

        # Averaged stats (filled by aggregate_stats)
        self.avg_runtimes: Dict[fx.Node, float] = {}
        self.avg_memory_deltas: Dict[fx.Node, float] = {}

        sep_node: Optional[fx.Node] = None
        sep_bwd_node: Optional[fx.Node] = None
        param_nodes: set = set()
        grad_nodes: set = set()

        for node in self.module.graph.nodes:
            if (node.op == OP.CALL_FUNCTION and
                    node.target == torch.ops.separator.sep.default):
                sep_node = node
            if (node.op == OP.CALL_FUNCTION and
                    node.target == torch.ops.separator.sep_backward.default):
                sep_bwd_node = node
            if (node.op == OP.CALL_FUNCTION and
                    node.target == torch.ops.aten._fused_adam.default):
                # args[0] = list of param nodes, args[1] = list of grad nodes
                for p in node.args[0]:
                    if isinstance(p, fx.Node):
                        param_nodes.add(p)
                for g in node.args[1]:
                    if isinstance(g, fx.Node):
                        grad_nodes.add(g)

        assert sep_node is not None, "Could not find sep node in graph"
        assert sep_bwd_node is not None, "Could not find sep_backward node in graph"

        # Pass 2: assign regions and classify node types
        region = "fwd"
        for node in self.module.graph.nodes:
            if node == sep_node:
                region = "inter"
            elif node == sep_bwd_node:
                region = "bwd"

            if region == "fwd" and node != sep_node:
                self.fwd_nodes.append(node)
            elif region == "bwd" and node != sep_bwd_node:
                self.bwd_nodes.append(node)

            # Classify
            if node.op == OP.PLACEHOLDER:
                if node in param_nodes:
                    self.node_type[node] = NodeType.PARAM
                elif node in grad_nodes:
                    self.node_type[node] = NodeType.GRAD
                else:
                    self.node_type[node] = NodeType.OTHER
            elif region == "fwd" and node != sep_node:
                self.node_type[node] = NodeType.ACT
            else:
                self.node_type[node] = NodeType.OTHER

        # Pass 3: record activation last-fwd and first-bwd uses
        all_nodes: List[fx.Node] = list(self.module.graph.nodes)
        node_index: Dict[fx.Node, int] = {n: i for i, n in enumerate(all_nodes)}

        fwd_set = set(self.fwd_nodes)
        bwd_set = set(self.bwd_nodes)

        for node in self.fwd_nodes:
            if self.node_type[node] != NodeType.ACT:
                continue

            # Find last use still in fwd region
            last_fwd: Optional[fx.Node] = None
            first_bwd: Optional[fx.Node] = None

            for user in node.users:
                if user in fwd_set:
                    if last_fwd is None or node_index[user] > node_index[last_fwd]:
                        last_fwd = user
                if user in bwd_set:
                    if first_bwd is None or node_index[user] < node_index[first_bwd]:
                        first_bwd = user

            if first_bwd is not None:
                self.act_last_fwd_use[node] = last_fwd
                self.act_first_bwd_use[node] = first_bwd
            else:
                # Produced in fwd but never used in bwd — not an activation of interest
                self.node_type[node] = NodeType.OTHER

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        if n.op in (OP.PLACEHOLDER, OP.OUTPUT):
            return super().run_node(n)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        mem_before = torch.cuda.memory_allocated()
        start_event.record()
        result = super().run_node(n)
        end_event.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        elapsed_ms = start_event.elapsed_time(end_event)
        mem_delta = mem_after - mem_before

        if n not in self.node_runtimes:
            self.node_runtimes[n] = []
            self.node_memory_deltas[n] = []

        self.node_runtimes[n].append(elapsed_ms)
        self.node_memory_deltas[n].append(mem_delta)

        return result

    def aggregate_stats(self) -> None:
        for node, times in self.node_runtimes.items():
            self.avg_runtimes[node] = sum(times) / len(times)
        for node, deltas in self.node_memory_deltas.items():
            self.avg_memory_deltas[node] = sum(deltas) / len(deltas)

    def print_stats(self) -> None:
        print(f"\n{'='*90}")
        print(f"{'Node':<40} {'Type':<10} {'Avg Time (ms)':>15} {'Mem Delta (MB)':>15}")
        print(f"{'='*90}")

        for node in self.module.graph.nodes:
            if node.op in (OP.PLACEHOLDER, OP.OUTPUT):
                continue
            
            ntype = self.node_type.get(node, NodeType.OTHER).name
            avg_t = self.avg_runtimes.get(node, 0.0)
            avg_m = self.avg_memory_deltas.get(node, 0.0) / (1024 ** 2)
            print(f"{node.name:<40} {ntype:<10} {avg_t:>15.4f} {avg_m:>15.4f}")

        print(f"\n--- Activations (fwd->bwd lifetime) ---")
        print(f"{'Activation':<40} {'Last Fwd Use':<40} {'First Bwd Use':<40}")
        print(f"{'-'*120}")
        for act_node, first_bwd in self.act_first_bwd_use.items():
            last_fwd = self.act_last_fwd_use.get(act_node, None)
            last_fwd_name = last_fwd.name if last_fwd else "N/A"
            print(f"{act_node.name:<40} {last_fwd_name:<40} {first_bwd.name:<40}")

    def reset_stats(self) -> None:
        self.node_runtimes.clear()
        self.node_memory_deltas.clear()
        self.avg_runtimes.clear()
        self.avg_memory_deltas.clear()
