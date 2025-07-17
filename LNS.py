import itertools
import math
import multiprocessing
import random
from collections.abc import Iterable
from copy import copy
from typing import Callable

from LNS_modifications import HeuristicModifications
from graph_types import Edge
from objective import compute_objective, find_path_fast, route_difference, compare_objectives
from utility import Instance, Modification, Solution, Timer, FullInstanceData
from validation.validator import validate_solution

find_path = find_path_fast


def make_foil_feasible(instance: Instance):
    graph = instance.graph.inner
    foil_route = instance.foil_route
    user_model = instance.user_model
    changes = []
    for edge in foil_route:
        if graph[edge[0]][edge[1]][edge[2]]['obstacle_free_width_float'] is None:
            continue
        if graph[edge[0]][edge[1]][edge[2]]['obstacle_free_width_float'] < user_model.minimum_width:
            changes.append(((edge[0], edge[1], edge[2]), 'obstacle_free_width_float'))
        if 'curb_height_max' not in graph[edge[0]][edge[1]][0].keys():
            continue
        if graph[edge[0]][edge[1]][edge[2]]['curb_height_max'] is None:
            continue
        if graph[edge[0]][edge[1]][edge[2]]['curb_height_max'] > user_model.maximum_height:
            changes.append(((edge[0], edge[1], edge[2]), 'curb_height_max'))
    return changes


def cancel_l_add_r(changes: Iterable[Modification], fixed_changes: Iterable[Modification],
                   l: int, r: int, instance: Instance, modification_manager: HeuristicModifications, timer: Timer):
    graph, foil_route, user_model, _ = instance
    begin, end = instance.endpoints
    to_cancel = []
    for change in changes:
        if change not in fixed_changes:
            to_cancel.append(change)
    to_cancel = random.sample(to_cancel, k=min(len(to_cancel), l))
    new_changes = tuple(change for change in changes if change not in to_cancel)
    new_route = find_path(graph, user_model, begin, end, new_changes)
    if new_route is None:
        for change in fixed_changes:
            if change not in new_changes:
                new_changes += change
        new_route = find_path(graph, user_model, begin, end, new_changes)
        if new_route is None:
            new_changes = fixed_changes
            new_route = find_path(graph, user_model, begin, end, new_changes)
            objective = compute_objective(instance, new_changes, new_route)
            return Solution(new_changes, objective, new_route), to_cancel
    objective = compute_objective(instance, new_changes, new_route)
    invalid_mods: set[Modification] = set()
    i = 0
    while i < r:
        if timer.out():
            return Solution(new_changes, objective, new_route), to_cancel
        modifications = list(modification_manager.get_modifications(
            new_changes, new_route, foil_route, False
        ))
        candidates = [m for m in modifications if m not in invalid_mods]
        if not candidates:
            break

        modification = random.choice(candidates)
        tmp_changes = tuple(itertools.chain(new_changes, modification))
        tmp_route = find_path(graph, user_model, begin, end, tmp_changes)
        if tmp_route is None:
            # record this mod as invalid so we won't retry it
            invalid_mods.add(modification)
            continue
        objective = compute_objective(instance, tmp_changes, tmp_route)
        i += 1
        new_changes = tmp_changes
        new_route = tmp_route
        if objective[0] == 0:
            break
    return Solution(new_changes, objective, new_route), to_cancel


def get_ca_neighbours(changes: Iterable[Modification], fixed_changes: Iterable[Modification], l: int, r: int,
                      instance: Instance, modification_manager: HeuristicModifications, n_size: int, timer: Timer):
    neighbors = []
    for _ in range(n_size):
        if timer.out():
            break
        neighbors.append(cancel_l_add_r(changes, fixed_changes, l, r, instance, modification_manager, timer))
    return neighbors


def greedy_repair(current_candidate: Solution, fixed_changes: Iterable[Modification],
                  instance: Instance, modification_manager: HeuristicModifications, timer: Timer, n_range=range(3, 10)):
    tmp = []
    if current_candidate.objective[0] == 0:
        return current_candidate
    for change in fixed_changes:
        if change not in current_candidate.encoding:
            tmp.append(change)
    current_candidate.encoding += tuple(tmp)
    i = 1
    min_candidate = copy(current_candidate)
    while current_candidate.objective[0] != 0:
        if timer.out():
            if max(n_range) == 1:
                return min_candidate
            else:
                return current_candidate
        n_size = random.sample(n_range, k=1)[0]
        neighbors = get_ca_neighbours(current_candidate.encoding, fixed_changes, 0, 1, instance,
                                      modification_manager, n_size, timer)
        if len(neighbors) == 0:
            return current_candidate
        best_neighbor = min(neighbors, key=lambda x: x[0].objective)[0]
        current_candidate = copy(best_neighbor)
        if max(n_range) == 1:
            if current_candidate.objective < min_candidate.objective:
                min_candidate = copy(current_candidate)
        i += 1
        if i > 1000:
            return min_candidate
    return current_candidate


def find_commons(history: dict[Modification, tuple[Edge, ...]], foil_route: tuple[Edge, ...]):
    if not history:
        return list(foil_route), []

    # Calculate common_foil: edges in foil_route not present in any history route
    common_foil = list(foil_route)
    for route in history.values():
        common_foil = [edge for edge in common_foil if edge not in route]

    # Calculate common_current: edges in the first history route (not in foil_route) and present in all other routes
    first_route = next(iter(history.values()))
    common_current = [edge for edge in first_route if edge not in foil_route]
    for route in history.values():
        common_current = [edge for edge in common_current if edge in route]

    return common_foil, common_current


def add_new_scheme_changes(changes: Iterable[Modification], common_foil: list[Edge],
                           instance: Instance, modification_manager: HeuristicModifications):
    graph, foil_route, user_model, _ = instance
    begin, end = instance.endpoints
    n_mods = random.sample(range(1, 5), k=1)[0]
    modifications = list(modification_manager.get_modifications(changes, [], common_foil, False, true_foil=foil_route))
    modifications = random.sample(modifications, k=min(len(modifications), n_mods))
    for modification in modifications:
        changes = tuple(itertools.chain(changes, modification))
    route = find_path(graph, user_model, begin, end, changes)
    objective = compute_objective(instance, changes, route)
    return Solution(changes, objective, route)


def add_new_scheme_constraint(changes: Iterable[Modification], common_current: list[Edge],
                              instance: Instance, modification_manager: HeuristicModifications):
    graph, foil_route, user_model, _ = instance
    begin, end = instance.endpoints
    n_mods = random.sample(range(1, 5), k=1)[0]
    modifications = list(
        modification_manager.get_modifications(changes, common_current, foil_route, False, true_foil=foil_route))
    modifications = random.sample(modifications, k=min(len(modifications), n_mods))
    for modification in modifications:
        changes = tuple(itertools.chain(changes, modification))
    route = find_path(graph, user_model, begin, end, changes)
    objective = compute_objective(instance, changes, route)
    return Solution(changes, objective, route)


def new_scheme(current_candidate: Solution, fixed_changes: Iterable[Modification], instance: Instance,
               modification_manager: HeuristicModifications, timer: Timer, h=0.5):
    foil_route = instance.foil_route
    d = random.randint(3, 6)
    history: dict[Modification, tuple[Edge, ...]] = dict()
    deleted = []
    for _ in range(d):
        got_neighbor = get_ca_neighbours(current_candidate.encoding, fixed_changes, 1, 0, instance,
                                         modification_manager, 1, timer)
        if len(got_neighbor) == 0:
            return current_candidate
        neighbor: Solution = got_neighbor[0][0]
        deleted_change: list[Modification] = got_neighbor[0][1]
        if len(deleted_change) == 0:
            continue
        history[deleted_change[0]] = neighbor.route
        deleted.append(deleted_change[0])
    if len(history) == 0:
        return current_candidate
    changes = tuple(change for change in current_candidate.encoding if change not in deleted)
    common_foil, common_current = find_commons(history, foil_route)
    if random.random() > h:
        return add_new_scheme_changes(changes, common_foil,
                                      instance, modification_manager)
    else:
        return add_new_scheme_constraint(changes, common_current,
                                         instance, modification_manager)


def destruction(current_candidate: Solution, fixed_changes: Iterable[Modification],
                instance: Instance, modification_manager: HeuristicModifications, timer: Timer, d=1):
    l = min(d, len(current_candidate.encoding))
    neighbors = get_ca_neighbours(current_candidate.encoding, fixed_changes, l, 0, instance,
                                  modification_manager, 1, timer)
    if len(neighbors) == 0:
        return current_candidate
    current_candidate = min(neighbors, key=lambda x: x[0].objective)[0]
    return current_candidate


def clean_up2(current_candidate: Solution, instance: Instance, timer: Timer):
    graph, foil_route, user_model, delta = instance
    begin, end = instance.endpoints
    useless = []
    for change in current_candidate.encoding:
        if timer.out():
            return current_candidate
        to_cancel = change
        new_changes = tuple(x for x in current_candidate.encoding if x != to_cancel and x not in useless)
        new_route = find_path(graph, user_model, begin, end, new_changes)
        if new_route is None:
            continue
        distance = route_difference(foil_route, new_route, graph)
        if distance <= delta:
            useless.append(change)
    current_candidate.encoding = tuple(x for x in current_candidate.encoding if x not in useless)
    current_candidate.route = find_path(graph, user_model, begin, end, current_candidate.encoding)
    current_candidate.objective = compute_objective(instance, current_candidate.encoding, current_candidate.route)
    return current_candidate


def clean_up(current_candidate: Solution, fixed_changes: Iterable[Modification], instance: Instance, timer: Timer):
    graph, foil_route, user_model, delta = instance
    begin, end = instance.endpoints
    useless = []
    for change in current_candidate.encoding:
        if timer.out():
            return current_candidate
        if change in fixed_changes:
            continue
        to_cancel = change
        new_changes = tuple(x for x in current_candidate.encoding if x != to_cancel)
        new_route = find_path(graph, user_model, begin, end, new_changes)
        if new_route is None:
            continue
        distance = route_difference(foil_route, new_route, graph)
        if distance <= delta:
            useless.append(change)
    pruned_encoding = tuple(x for x in current_candidate.encoding if x not in useless)
    final_route = find_path(graph, user_model, begin, end, pruned_encoding)
    if final_route is None:
        return current_candidate
    current_candidate.encoding = pruned_encoding
    current_candidate.route = final_route
    current_candidate.objective = compute_objective(instance, pruned_encoding, final_route)
    return current_candidate


def update_best(current_candidate: Solution, best_candidate: Solution | None, full_instance_data: FullInstanceData,
                timer: Timer, apply_validate: bool = True):
    if best_candidate is None:
        if apply_validate:
            is_obj_correct, correct_obj, correct_route = validate_solution(current_candidate, full_instance_data)
            if not is_obj_correct:
                current_candidate.objective = correct_obj
                current_candidate.route = correct_route
        best_candidate = copy(current_candidate)
        timer.update_time_to_best()
    elif compare_objectives(current_candidate.objective, best_candidate.objective) < 0:
        if apply_validate:
            is_obj_correct, correct_obj, correct_route = validate_solution(current_candidate, full_instance_data)
            if is_obj_correct:
                best_candidate = copy(current_candidate)
                timer.update_time_to_best()
            else:
                current_candidate.objective = correct_obj
                current_candidate.route = correct_route
                if correct_obj < best_candidate.objective:
                    best_candidate = copy(current_candidate)
                    timer.update_time_to_best()
        else:
            best_candidate = copy(current_candidate)
            timer.update_time_to_best()
    return best_candidate


def make_candidate(instance: Instance, changes: Iterable[Modification]):
    begin, end = instance.endpoints
    route = find_path(instance.graph, instance.user_model, begin, end, changes)
    return Solution(changes, compute_objective(instance, changes, route), route)


def prohibit_adjacent_edges(
        instance: Instance,
        mod_mgr: HeuristicModifications
) -> tuple[Modification, ...]:
    """
    Return all (edge, attribute) modifications that would 'forbid' an initially-feasible
    edge whose *start* node is in the foil_route and whose *end* node is not.
    """
    graph = instance.graph.inner
    foil_route = instance.foil_route
    foil_nodes = {u for u, v, _ in foil_route} | {v for u, v, _ in foil_route}

    adjacent = []
    for edge in graph.edges(keys=True):
        u, v, _ = edge
        if (u in foil_nodes) and not (v in foil_nodes):
            for attr in mod_mgr.forbid(edge):
                adjacent.append((edge, attr))

    return tuple(adjacent)


def _repair_and_update(
        current: Solution,
        best: Solution,
        fixed_changes: Iterable[Modification],
        full_instance_data: FullInstanceData,
        mod_mgr: HeuristicModifications,
        timer: Timer,
) -> tuple[Solution, Solution]:
    best = update_best(current, best, full_instance_data, timer)
    if timer.out():
        return current, best
    current = greedy_repair(current, fixed_changes, full_instance_data.instance, mod_mgr, timer)
    best = update_best(current, best, full_instance_data, timer)
    if timer.out():
        return current, best
    return current, best


def clean_up_construct(current_candidate: Solution, best_candidate: Solution, fixed_changes: Iterable[Modification],
                       full_instance_data: FullInstanceData, modification_manager: HeuristicModifications,
                       timer: Timer):
    current_candidate = clean_up(current_candidate, [], full_instance_data.instance, timer)
    return _repair_and_update(current_candidate, best_candidate, fixed_changes, full_instance_data,
                              modification_manager, timer)


def clean_up2_construct(current_candidate: Solution, best_candidate: Solution, fixed_changes: Iterable[Modification],
                        full_instance_data: FullInstanceData, modification_manager: HeuristicModifications,
                        timer: Timer):
    current_candidate = clean_up2(current_candidate, full_instance_data.instance, timer)
    return _repair_and_update(current_candidate, best_candidate, fixed_changes, full_instance_data,
                              modification_manager, timer)


def pop_based_construct(current_candidate: Solution, best_candidate: Solution, fixed_changes: Iterable[Modification],
                        full_instance_data: FullInstanceData, modification_manager: HeuristicModifications,
                        timer: Timer,
                        pop_based_threshold: float):
    current_candidate = new_scheme(current_candidate, fixed_changes, full_instance_data.instance, modification_manager,
                                   timer,
                                   h=pop_based_threshold)
    return _repair_and_update(current_candidate, best_candidate, fixed_changes, full_instance_data,
                              modification_manager, timer)


def destruct_construct(current_candidate: Solution, best_candidate: Solution, fixed_changes: Iterable[Modification],
                       full_instance_data: FullInstanceData, modification_manager: HeuristicModifications, timer: Timer,
                       pop_based_threshold: float = None):
    x = max(0.1, random.random() * 0.3)
    d = math.ceil(x * len(current_candidate.encoding))
    current_candidate = destruction(current_candidate, [], full_instance_data.instance, modification_manager, timer,
                                    d=d)
    return _repair_and_update(current_candidate, best_candidate, fixed_changes, full_instance_data,
                              modification_manager, timer)


def ls_destruct_construct(current_candidate: Solution, best_candidate: Solution, fixed_changes: Iterable[Modification],
                          full_instance_data: FullInstanceData, modification_manager: HeuristicModifications,
                          timer: Timer,
                          pop_based_threshold: float = None):
    x = max(0.1, random.random() * 0.3)
    d = math.ceil(x * len(current_candidate.encoding))
    current_candidate = ls_destruction(current_candidate, full_instance_data.instance, modification_manager, timer,
                                       destruct=d)
    return _repair_and_update(current_candidate, best_candidate, fixed_changes, full_instance_data,
                              modification_manager, timer)


def ls_destruction(current_candidate: Solution, instance: Instance, modification_manager: HeuristicModifications,
                   timer: Timer, destruct=1):
    d_in = current_candidate.objective[1]
    while current_candidate.objective[1] > max(0, d_in - destruct):
        n_size = random.sample(range(3, 10), k=1)[0]
        if timer.out():
            return current_candidate
        neighbors = get_ca_neighbours(current_candidate.encoding, [], 1, 0, instance,
                                      modification_manager, n_size, timer)
        if len(neighbors) == 0:
            return current_candidate
        current_candidate = max(neighbors, key=lambda x: x[0].objective)[0]
    return current_candidate

def fallback(best_candidate: Solution, fixed_changes: tuple[Modification],
             full_instance_data: FullInstanceData, modification_manager: HeuristicModifications, timer: Timer):
    if best_candidate is None or best_candidate.objective[0] > 0:
        # fallback: build a “backup” from the prohibit_adjacent_edges operators
        instance = full_instance_data.instance
        adjacent = prohibit_adjacent_edges(instance, modification_manager) + fixed_changes
        backup_candidate = make_candidate(instance, adjacent)
        return update_best(backup_candidate, best_candidate, full_instance_data, timer)
    return best_candidate

def LNS(full_instance_data: FullInstanceData, timer: Timer, destroy_operators: list[Callable],
        preprocess_operators: list[Callable],
        global_best_seed: Solution | None, incumbent_seed: Solution | None, local_best_seed: Solution | None,
        pop_based_threshold=None,
        verbose: bool = False):
    instance = full_instance_data.instance
    user_model = instance.user_model
    if pop_based_threshold is None:
        if pop_based_construct in destroy_operators:
            pop_based_threshold = random.sample([0.0, 0.5, 1.0], k=1)[0]
    modification_manager = HeuristicModifications(instance.graph, user_model.minimum_width,
                                                  user_model.maximum_height, user_model.path_preference)
    fixed_changes = tuple(make_foil_feasible(instance))
    if incumbent_seed is not None:
        current_candidate = copy(incumbent_seed)
    else:
        current_candidate = make_candidate(instance, fixed_changes)
    current_candidate = greedy_repair(current_candidate, fixed_changes, instance, modification_manager, timer,
                                      n_range=range(1, 2))
    if global_best_seed is not None:
        common_changes = tuple(change for change in current_candidate.encoding if change in global_best_seed.encoding)
        current_candidate = make_candidate(instance, common_changes)
        current_candidate = greedy_repair(current_candidate, fixed_changes, instance, modification_manager, timer,
                                          n_range=range(1, 2))
    if local_best_seed is not None:
        best_candidate = update_best(current_candidate, local_best_seed, full_instance_data, timer)
    else:
        best_candidate = update_best(current_candidate, None, full_instance_data, timer)
    while not timer.out():
        if verbose and timer.should_log(5.0):
            elapsed = timer.elapsed()
            proc = multiprocessing.current_process()
            best = best_candidate.objective
            print(f"[{proc.name}-{proc.pid}] {elapsed:6.1f}s │ Best Obj: {best}")
        preprocess_op = random.sample(preprocess_operators, k=1)[0]
        destroy_op = random.sample(destroy_operators, k=1)[0]
        current_candidate, best_candidate = preprocess_op(copy(current_candidate), best_candidate,
                                                          fixed_changes, full_instance_data,
                                                          modification_manager,
                                                          timer)
        if timer.out():
            break
        current_candidate, best_candidate = destroy_op(copy(current_candidate), best_candidate,
                                                       fixed_changes, full_instance_data,
                                                       modification_manager,
                                                       timer,
                                                       pop_based_threshold=pop_based_threshold)
    best_candidate = fallback(best_candidate, fixed_changes, full_instance_data, modification_manager, timer)
    return {'candidate': copy(best_candidate), 'current': copy(current_candidate),
            'time_to_best': timer.get_time_to_best()}
