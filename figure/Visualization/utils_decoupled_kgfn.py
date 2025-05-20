from typing import List

import numpy as np
import torch

# from decoupled_kg.models.decoupled_gp_network import (
#     GaussianProcessNetwork,
#     fit_gp_network,
# )


def read_result(
    problem: str, cost: str, cost_axes: str, trial_list: str, algo_list: List[str]
):
    results = {key: None for key in algo_list}
    for algo in algo_list:
        results[algo] = {}
        best_obs = []
        best_post = []
        obj_at_best = []
        no_trial = len(trial_list)
        for trial in trial_list:
            res = torch.load(
                f"../results/{problem}_{cost}/{algo}/trial_{trial}.pt",weights_only=False
            )
            results[algo][f"trial_{trial}"] = {
                "best_obs": res["best_obs_vals"],
                "best_post": res["best_post_means"],
                "runtime": res["runtimes"],
                "best_design": res["best_design_post_mean"],
            }
            if algo in ["KGFN"]:
                results[algo][f"trial_{trial}"]["node_count"] = res["node_eval_counts"]
            if "obj_at_best_designs" in res.keys():
                results[algo][f"trial_{trial}"]["obj_at_best_design"] = res[
                    "obj_at_best_designs"
                ]
            cost_list = res["cumulative_costs"]
            cost_list[0] = 0
            best_obs_temp = []
            best_post_temp = []
            obj_at_best_temp = []
            obj_max = -torch.inf
            for j in range(len(cost_list) - 1):
                cost_increment = int(cost_list[j + 1] - cost_list[j])
                for _ in range(0, cost_increment):
                    best_obs_temp.append(res["best_obs_vals"][j])
                    best_post_temp.append(res["best_post_means"][j])
                    if "obj_at_best_designs" in res.keys():
                        obj_max = res["obj_at_best_designs"][j]
                        obj_at_best_temp.append(obj_max)
            while len(best_obs_temp) < len(cost_axes):
                best_obs_temp.append(res["best_obs_vals"][-1])
                best_post_temp.append(res["best_post_means"][-1])
                if "obj_at_best_designs" in res.keys():
                    obj_max = res["obj_at_best_designs"][-1]
                    obj_at_best_temp.append(obj_max)
            best_obs.append(best_obs_temp)
            best_post.append(best_post_temp)
            obj_at_best.append(obj_at_best_temp)
        mean_obs = [
            np.mean([best_obs[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        std_obs = [
            np.std([best_obs[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        up_obs = [
            mean_obs[i] + (2 * std_obs[i] / np.sqrt(no_trial))
            for i in range(len(mean_obs))
        ]
        lo_obs = [
            mean_obs[i] - (2 * std_obs[i] / np.sqrt(no_trial))
            for i in range(len(mean_obs))
        ]
        mean_post = [
            np.mean([best_post[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        std_post = [
            np.std([best_post[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        up_post = [
            mean_post[i] + (2 * std_post[i] / np.sqrt(no_trial))
            for i in range(len(mean_post))
        ]
        lo_post = [
            mean_post[i] - (2 * std_post[i] / np.sqrt(no_trial))
            for i in range(len(mean_post))
        ]
        mean_obj_at_best = [
            np.mean([obj_at_best[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        std_obj_at_best = [
            np.std([obj_at_best[j][i] for j in range(0, no_trial)])
            for i in range(0, len(cost_axes))
        ]
        up_obj_at_best = [
            mean_obj_at_best[i] + (2 * std_obj_at_best[i] / np.sqrt(no_trial))
            for i in range(len(mean_obj_at_best))
        ]
        lo_obj_at_best = [
            mean_obj_at_best[i] - (2 * std_obj_at_best[i] / np.sqrt(no_trial))
            for i in range(len(mean_obj_at_best))
        ]
        results[algo]["mean_obs"] = mean_obs
        results[algo]["std_obs"] = std_obs
        results[algo]["up_obs"] = up_obs
        results[algo]["lo_obs"] = lo_obs
        results[algo]["mean_post"] = mean_post
        results[algo]["std_post"] = std_post
        results[algo]["up_post"] = up_post
        results[algo]["lo_post"] = lo_post
        results[algo]["mean_obj_at_best"] = mean_obj_at_best
        results[algo]["std_obj_at_best"] = std_obj_at_best
        results[algo]["up_obj_at_best"] = up_obj_at_best
        results[algo]["lo_obj_at_best"] = lo_obj_at_best
    return results
