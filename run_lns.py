import random
import time
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Optional, Any, Iterable

import LNS
import data
from LNS import (
    clean_up2_construct,
    pop_based_construct,
    destruct_construct,
    ls_destruct_construct,
    clean_up_construct,
)
from utility import Timer, FullInstanceData, Solution, calculate_segment_time_limit


def run_LNS(args):
    full_instance_data, destroy_ops, preprocess_ops, h, in_best, in_curr, prev_best, timer = args
    return LNS.LNS(
        full_instance_data,
        timer,
        destroy_ops,
        preprocess_ops,
        in_best,
        in_curr,
        prev_best,
        pop_based_threshold=h,
    )


def prepare_input(full_instance_data: FullInstanceData, operators, thresholds: list[Optional[float]],
                  best_candidate: Optional[Solution],
                  previous_results: list[dict[str, Any]], jobs1: int,
                  jobs2: int, timer: Timer):
    inputs = []
    for j in range(jobs1 + jobs2):
        ops = operators[j]
        h = thresholds[j]
        prev = previous_results[j]
        in_best = best_candidate if (j < jobs1 and best_candidate.objective[0] == 0) else prev["candidate"]
        in_curr = prev["current"]
        prev_best = prev["candidate"]
        new_timer = deepcopy(timer)
        new_timer.set_time_to_best(prev["time_to_best"])
        inputs.append((full_instance_data, ops[0], ops[1], h, in_best, in_curr, prev_best, new_timer))
    return inputs


def get_best_candidate(results: Iterable[dict[str, Any]]):
    return min(
        (res["candidate"] for res in results),
        key=lambda sol: sol.objective,
        default=None,
    )


def run_multistart_LNS(full_instance_data: FullInstanceData, segment_time_limit: float):
    timer = Timer(segment_time_limit)
    n_workers = cpu_count() or 1
    jobs1 = n_workers // 2
    jobs2 = n_workers - jobs1

    destroyers = [ls_destruct_construct, destruct_construct, pop_based_construct]
    preprocessors = [clean_up_construct, clean_up2_construct]
    operators = [
                    [random.sample(destroyers, k=3), random.sample(preprocessors, k=2)]
                    for _ in range(jobs1)
                ] + [
                    [random.sample(destroyers, k=random.randint(1, 3)),
                     random.sample(preprocessors, k=random.randint(1, 2))]
                    for _ in range(jobs2)
                ]
    thresholds = [None] * (jobs1 + jobs2)

    with Pool(processes=n_workers) as pool:
        # Stage 0: start from scratch
        args0 = [
            (full_instance_data, ops[0], ops[1], None, None, None, None, timer)
            for ops in operators
        ]
        res = pool.map(run_LNS, args0)

        # Stage 1 & 2: refine around the best found so far
        for i in range(2):
            best = get_best_candidate(res)
            timer.postpone((segment_time_limit * 3 - timer.elapsed()) / (2 - i))
            args = prepare_input(full_instance_data, operators, thresholds, best, res, jobs1, jobs2, timer)
            res = pool.map(run_LNS, args)

    best_entry = min(res, key=lambda r: r['candidate'].objective)
    return best_entry['candidate'], best_entry['time_to_best']


def main():
    total_time_limit = 30.0
    segment_time_limit = calculate_segment_time_limit(total_time_limit)
    res_list = []
    wall_clocks = []
    times = []

    for instance_id in range(1, 6):
        print(f"→ Running instance {instance_id} …")
        t0 = time.monotonic()
        full_instance_data = data.read_instance2(4, instance_id)
        sol, time_to_best = run_multistart_LNS(full_instance_data, segment_time_limit)
        dur = time.monotonic() - t0
        print(f"    • Best objective={sol.objective}   (wall-clock {dur:.1f}s, time_to_best {time_to_best:.2f}s)")
        res_list.append(sol)
        wall_clocks.append(dur)
        times.append(time_to_best)

    print("\nSummary:")
    for idx, (sol, dur, time_to_best) in enumerate(zip(res_list, wall_clocks, times), start=1):
        print(f"Instance {idx}: objective={sol.objective} — wall-clock {dur:.1f}s, time_to_best {time_to_best:.2f}s")


if __name__ == "__main__":
    main()
