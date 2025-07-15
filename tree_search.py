import itertools
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from multiprocessing import Pool
from threading import Condition

import greedy
from exploration import GreedyExploration, LengthExploration
from graph_types import Edge, EdgeAttribute, TypedMultiGraph
from modification import AllModifications, FastModifications, ModificationManger
from objective import find_path, find_path_and_length_fast, find_path_fast, route_difference
from tree import MonteCarloNode, Node
from utility import Instance, Modification, Solution, UserModel


def run_tree_search(graph: TypedMultiGraph, foil_route: Sequence[Edge], user_model: UserModel, delta: float,
                    max_iter: int, time_limit: float, heuristics: bool = True):
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    # exploration = MonteCarloExploration(100000)
    exploration = LengthExploration()
    modification_manager = FastModifications(graph, user_model.minimum_width,
                                             user_model.maximum_height, user_model.path_preference)
    root_route = find_path(graph, (), user_model, begin, end)
    assert root_route is not None
    distance = route_difference(foil_route, root_route, graph)
    root_node = Node(None, (), 0, max(distance - delta, 0), distance, root_route)
    exploration.add_node(root_node)
    best_solution = Solution((), root_node.objective)
    seen_nodes: set[frozenset[tuple[Edge, EdgeAttribute]]] = set()
    history = [(0.0, *root_node.objective)]
    start_time = time.time()
    i = 0
    while exploration.has_next():
        node = exploration.pop_next()
        current_route = node.route
        assert current_route is not None
        encoding = [change for n in node.iterate_to_root() for change in n.changes]
        if best_solution.objective[0] == 0 and node.changes_number >= best_solution.objective[1] - 1:
            continue
        print(i, '{:.2f}'.format(time.time() - start_time),
              exploration.pending, node.objective, best_solution.objective)
        modifications = modification_manager.get_modifications(encoding, current_route, foil_route)
        children: list[Node] = []
        for modification in list(modifications):
            new_changes = tuple(itertools.chain(encoding, modification))
            if tuple(modification) in node.children:
                continue
            change_set = frozenset(new_changes)
            if change_set in seen_nodes:
                continue
            seen_nodes.add(change_set)
            new_path, foil_len, fact_len = find_path_and_length_fast(graph, user_model, foil_route,
                                                                     new_changes, begin, end)
            if new_path is None:
                continue
            distance = route_difference(foil_route, new_path, graph)
            new_node = Node(node, modification, len(new_changes), max(distance - delta, 0), distance, new_path,
                            foil_len, fact_len)
            children.append(new_node)
            if new_node.objective < best_solution.objective:
                best_solution = Solution(new_node.encoding, new_node.objective)
                history.append((time.time() - start_time, *new_node.objective))
                # ls_encoding, ls_objective = local_search.run_local_descend(
                #     graph, foil_route, new_path, new_node.objective, user_model, delta, new_changes)
                # best_solution = Solution(ls_encoding, ls_objective)
                # history.append((time.time() - start_time, *ls_objective))
            if heuristics and distance == 0:
                reduced = greedy.try_reduce(graph, foil_route, user_model, delta, new_changes, 0)
                if reduced.objective < best_solution.objective:
                    best_solution = reduced
                    history.append((time.time() - start_time, *reduced.objective))
            exploration.add_node(new_node)
        if heuristics and children:
            best_child = min(children, key=lambda it: it.objective)
            fixed = greedy.fix_first(modification_manager, graph, foil_route, user_model, delta,
                                     best_child, best_solution.objective[1] if best_solution.objective[0] == 0 else 128)
            new_nodes = itertools.takewhile(lambda it, new_node=best_child: it != new_node, fixed.iterate_to_root())
            for new_node in new_nodes:
                if new_node.objective < best_solution.objective:
                    best_solution = Solution(new_node.encoding, new_node.objective)
                    history.append((time.time() - start_time, *new_node.objective))
                if new_node.route_distance_violation == 0:
                    reduced = greedy.try_reduce(graph, foil_route, user_model, delta, new_node.encoding, 0)
                    if reduced.objective < best_solution.objective:
                        best_solution = reduced
                        history.append((time.time() - start_time, *reduced.objective))
            exploration.add_nodes(new_nodes)
        # node.set_processed()
        i += 1
        if i > max_iter or time.time() - start_time > time_limit:
            break
    return best_solution, history


def run_parallel_tree_search(graph: TypedMultiGraph, foil_route: Sequence[Edge], user_model: UserModel, delta: float,
                             max_iter: int, time_limit: float):
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    exploration = GreedyExploration()
    modification_manager = FastModifications(graph, user_model.minimum_width,
                                             user_model.maximum_height, user_model.path_preference)
    root_route = find_path_fast(graph, user_model, (), begin, end)
    assert root_route is not None
    distance = route_difference(foil_route, root_route, graph)
    root_node = Node(None, (), 0, max(distance - delta, 0), distance, root_route)
    exploration.add_node(root_node)
    best_solution = _Ref(Solution((), root_node.objective))
    seen_nodes: set[frozenset[tuple[Edge, EdgeAttribute]]] = set()
    history = [(0.0, *root_node.objective)]
    queue_length = _Ref(0)
    condition = Condition()
    start_time = time.time()
    i = 0
    with Pool(20, _global_init) as pool:
        while exploration.has_next() or queue_length.value > 0:
            with condition:
                while not exploration.has_next() or queue_length.value >= 20:
                    condition.wait()
                node = exploration.pop_next()
            if best_solution.value.objective[0] == 0 and node.changes_number >= best_solution.value.objective[1] - 1:
                continue
            print(i, '{:.2f}'.format(time.time() - start_time),
                  exploration.pending, node.objective, best_solution.value.objective, queue_length.value)
            queue_length.value += 1
            pool.apply_async(
                _process_node,
                (graph,
                 foil_route,
                 user_model,
                 delta,
                 modification_manager,
                 seen_nodes,
                 node),
                callback=lambda result,
                node=node: _handle_result(
                    Instance(
                        graph,
                        tuple(foil_route),
                        user_model,
                        delta),
                    exploration,
                    modification_manager,
                    best_solution,
                    history,
                    start_time,
                    node,
                    result, seen_nodes,
                    queue_length,
                    condition),
                error_callback=lambda e: print(e))
            i += 1
            if i > max_iter or time.time() - start_time > time_limit:
                break
    return best_solution.value, history


class _Ref[T]:
    def __init__(self, value: T) -> None:
        self.value = value


def _global_init():
    global global_seen
    global_seen = set()  # type: ignore


def _handle_result(instance: Instance, exploration: GreedyExploration, modification_manager: ModificationManger,
                   best_solution: _Ref[Solution], history: list[tuple[float, float, int, float]], start_time: float,
                   node: Node, result: tuple[Iterable[Node], Solution | None], seen_nodes: set[frozenset[tuple[Edge, EdgeAttribute]]],
                   queue_length: _Ref[int], condition: Condition):
    children, solution = result
    if solution is not None and solution.objective < best_solution.value.objective:
        best_solution.value = solution
        history.append((time.time() - start_time, *solution.objective))
    for new_node in children:
        new_node.parent = node
        if new_node.objective < best_solution.value.objective:
            best_solution.value = Solution(new_node.changes, new_node.objective)
            history.append((time.time() - start_time, *new_node.objective))
    children = [child for child in children if frozenset(child.encoding) not in seen_nodes]
    for child in children:
        seen_nodes.add(frozenset(child.encoding))
    # best_child = min(children, key=lambda it: it.objective)
    # fixed = greedy.fix_first(modification_manager, *instance,
    #                          best_child, 24)
    # new_nodes = itertools.takewhile(lambda it, new_node=best_child: it != new_node, fixed.iterate_to_root())
    # for new_node in new_nodes:
    #     if new_node.objective < best_solution.value.objective:
    #         best_solution.value = Solution(new_node.encoding, new_node.objective)
    #         history.append((time.time() - start_time, *new_node.objective))
    #     if new_node.route_distance_violation == 0:
    #         reduced = greedy.try_reduce(*instance, new_node.encoding, 0)
    #         if reduced.objective < best_solution.value.objective:
    #             best_solution.value = reduced
    #             history.append((time.time() - start_time, *reduced.objective))
    node.set_processed()
    with condition:
        queue_length.value -= 1
        exploration.add_nodes(children)
        condition.notify()


def _process_node(graph: TypedMultiGraph, foil_route: Sequence[Edge], user_model: UserModel, delta: float,
                  modification_manager: ModificationManger, seen_nodes: Iterable[frozenset[tuple[Edge, EdgeAttribute]]],
                  node: Node):
    global global_seen
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    current_route = node.route
    assert current_route is not None
    encoding = [change for n in node.iterate_to_root() for change in n.changes]
    modifications = modification_manager.get_modifications(encoding, current_route, foil_route)
    children: list[Node] = []
    for modification in modifications:
        new_changes = tuple(itertools.chain(encoding, modification))
        if tuple(modification) in node.children:
            continue
        change_set = frozenset(new_changes)
        if change_set in seen_nodes or change_set in global_seen:
            continue
        global_seen.add(change_set)  # type: ignore
        new_path = find_path_fast(graph, user_model, new_changes, begin, end)
        if new_path is None:
            continue
        distance = route_difference(foil_route, new_path, graph)
        new_node = Node(None, modification, len(new_changes), max(distance - delta, 0), distance, new_path)
        children.append(new_node)
    best_solution = None
    if children:
        best_child = min(children, key=lambda it: it.objective)
        fixed = greedy.fix_first(modification_manager, graph, foil_route, user_model, delta,
                                 best_child, 128)
        new_nodes = itertools.takewhile(lambda it, new_node=best_child: it != new_node, fixed.iterate_to_root())
        for new_node in new_nodes:
            if new_node.route_distance_violation == 0:
                reduced = greedy.try_reduce(graph, foil_route, user_model, delta, new_node.encoding, 0)
                if best_solution is None or reduced.objective < best_solution.objective:
                    best_solution = reduced
    return children, best_solution


def run_monte_carlo(instance: Instance, time_limit: float):
    graph, foil_route, user_model, delta = instance
    begin = foil_route[0][0]
    end = foil_route[-1][1]
    modification_manager = FastModifications(graph, user_model.minimum_width,
                                             user_model.maximum_height, user_model.path_preference)
    root_route = find_path(graph, (), user_model, begin, end)
    assert root_route is not None
    root_node = MonteCarloNode(None, (), 0, max(route_difference(foil_route, root_route, graph) - delta, 0), root_route)
    best_solution = Solution((), root_node.objective)
    seen_nodes: dict[frozenset[tuple[Edge, EdgeAttribute]], MonteCarloNode] = {}
    _process(root_node, modification_manager, foil_route, seen_nodes)
    history = [(0.0, *root_node.objective)]
    start_time = time.time()
    last_print = 0.0
    i = 0
    while time.time() - start_time < time_limit:
        i += 1
        candidate = _traverse(root_node, best_solution.objective[1] - 1 if best_solution.objective[0] == 0 else 32)
        if candidate is None:
            break
        node, modification = candidate
        encoding = tuple(itertools.chain(node.encoding, modification))
        if time.time() - last_print > 0.5:
            print(i, '{:.2f}'.format(time.time() - start_time), node.objective, best_solution.objective)
            last_print = time.time()
        new_path = find_path(graph, encoding, user_model, begin, end)
        if new_path is None:
            _process_infeasible(node, modification)
            continue
        distance = route_difference(foil_route, new_path, graph)
        new_node = node.make_child(modification, len(encoding), max(distance - delta, 0), new_path)
        new_node = _process2(instance, new_node, modification_manager, foil_route, seen_nodes)
        if new_node.objective < best_solution.objective:
            best_solution = Solution(new_node.encoding, new_node.objective)
            history.append((time.time() - start_time, *new_node.objective))
    return best_solution, history


def _process(node: MonteCarloNode, manager: ModificationManger, foil_route: Iterable[Edge],
             seen: Mapping[frozenset[Modification], MonteCarloNode]):
    objective = 1 / (node.route_distance_violation * 1000 + node.changes_number)
    parent = node.parent
    if parent is not None:
        parent.children[node.changes].node = node
    for tmp_node in node.iterate_to_root():
        parent = tmp_node.parent
        if parent is not None:
            arc = parent.children[tmp_node.changes]
            arc.objective += objective
            arc.used += 1
    _expand(node, manager, foil_route, seen)


def _process2(instance: Instance, node: MonteCarloNode, manager: ModificationManger, foil_route: Iterable[Edge],
              seen: MutableMapping[frozenset[Modification], MonteCarloNode]):
    if node.route_distance_violation != 0:
        fixed = greedy.fix_first(manager, *instance, node, 32)
    else:
        fixed = node
    objective = 1 / (fixed.route_distance_violation * 1000 + fixed.changes_number)
    for tmp_node in fixed.iterate_to_root():
        parent = tmp_node.parent
        if parent is not None:
            arc = parent.children[tmp_node.changes]
            arc.objective += objective
            arc.used += 1
    new_nodes = itertools.takewhile(lambda it: it != node, fixed.iterate_to_root())
    for tmp_node in itertools.chain(new_nodes, (node,)):
        _expand(tmp_node, manager, foil_route, seen)
        seen[frozenset(tmp_node.encoding)] = tmp_node
    if not fixed.feasible:
        modification = fixed.changes
        parent = fixed.parent
        assert parent is not None
        del parent.children[modification]
    if fixed.route_distance_violation == 0:
        parent = fixed.parent
        assert parent is not None
        parent_parent = parent.parent
        if parent_parent is not None:
            modification = parent.changes
            del parent_parent.children[modification]
    return min(fixed.iterate_to_root(), key=lambda it: it.objective)


def _expand(node: MonteCarloNode, manager: ModificationManger, foil_route: Iterable[Edge],
            seen: Mapping[frozenset[Modification], MonteCarloNode]):
    encoding = tuple(node.encoding)
    route = node.route
    if route is None:
        return
    for modification in manager.get_modifications(encoding, route, foil_route):
        modification_tuple = tuple(modification)
        if modification_tuple not in node.children and frozenset(itertools.chain(encoding, modification)) not in seen:
            node.children[modification_tuple] = MonteCarloNode.Child()


def _process_infeasible(node: MonteCarloNode, modification: tuple[Modification, ...]):
    del node.children[modification]


def _traverse(node: MonteCarloNode, limit: int) -> tuple[MonteCarloNode, tuple[tuple[Edge, EdgeAttribute], ...]] | None:
    children = node.children
    if not children or limit == 0:
        return None
    # weights = [child.obj / child.use for child in children]
    weights = [child.mean_objective for child in children.values()]
    # weights = [math.exp(math.log(www) / 0.02) for www in weights]
    # s = sum(weights)
    # for i in range(len(weights)):
    # weights[i] /= s
    # used = [child.tr for child in children]
    # ss = sum(used)
    # coef = 0 if self.u < 20 else 0.12
    # for i in range(len(weights)):
    #     weights[i] += math.sqrt(coef * math.log(ss) / used[i])
    # modification, choice = random.choices(tuple(children.items()), weights)[0]

    modification, choice = list(children.items())[max(range(len(children)), key=lambda i: weights[i])]
    child_node = choice.node
    if child_node is None:
        return node, modification
    result = _traverse(child_node, limit - 1)
    if result is None:
        del children[modification]
        # for n in node.iterate_to_root():
        #     if n.children:
        #         n.obj = max(it.obj for it in n.children)
        return _traverse(node, limit)
    return result
