import os
from collections import defaultdict

from lib.graph_utils import compute_residual_problem

from ..config import TOPOLOGIES_DIR
from ..constants import NUM_CORES
from ..partitioning.pop import (
    BaselineSplitter,
    GenericSplitter,
    RandomSplitter,
    RandomSplitter2,
    SmartSplitter,
)
from ..runtime_utils import parallelized_rt
from .abstract_formulation import Objective
from .path_formulation import PathFormulation

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "path-form")


class POP(PathFormulation):
    @classmethod
    def new_total_flow(
        cls,
        num_subproblems,
        split_method,
        split_fraction,
        algo_cls,
        num_paths=4,
        edge_disjoint=True,
        dist_metric="inv-cap",
        out=None,
    ):
        return cls(
            objective=Objective.TOTAL_FLOW,
            num_subproblems=num_subproblems,
            split_method=split_method,
            split_fraction=split_fraction,
            algo_cls=algo_cls,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_min_max_link_util(
        cls,
        num_subproblems,
        split_method,
        split_fraction,
        algo_cls,
        num_paths=4,
        edge_disjoint=True,
        dist_metric="inv-cap",
        out=None,
    ):
        return cls(
            objective=Objective.MIN_MAX_LINK_UTIL,
            num_subproblems=num_subproblems,
            split_method=split_method,
            split_fraction=split_fraction,
            algo_cls=algo_cls,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    @classmethod
    def new_max_concurrent_flow(
        cls,
        num_subproblems,
        split_method,
        split_fraction,
        algo_cls,
        num_paths=4,
        edge_disjoint=True,
        dist_metric="inv-cap",
        out=None,
    ):
        return cls(
            objective=Objective.MAX_CONCURRENT_FLOW,
            num_subproblems=num_subproblems,
            split_method=split_method,
            split_fraction=split_fraction,
            algo_cls=algo_cls,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=False,
            VERBOSE=False,
            out=out,
        )

    def __init__(
        self,
        *,
        objective,
        num_subproblems,
        split_method,
        split_fraction,
        algo_cls,
        num_paths=4,
        edge_disjoint=True,
        dist_metric="inv-cap",
        DEBUG=False,
        VERBOSE=False,
        out=None,
        **addl_kwargs,
    ):
        super().__init__(
            objective=objective,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=DEBUG,
            VERBOSE=VERBOSE,
            out=out,
        )
        self._num_subproblems = num_subproblems
        self._split_method = split_method
        self._split_fraction = split_fraction
        self._algo_cls = algo_cls
        self._addl_kwargs = addl_kwargs

    def split_problems(self, problem, num_subproblems):
        splitter = None
        if self._split_method == "skewed":
            splitter = BaselineSplitter(num_subproblems)
        elif self._split_method == "random":
            splitter = RandomSplitter(num_subproblems, self._split_fraction)
        elif self._split_method == "random2":
            splitter = RandomSplitter2(num_subproblems, self._split_fraction)
        elif self._split_method in ["tailored", "means", "covs", "cluster"]:
            if self._split_method == "tailored":
                paths_dict = PathFormulation.read_paths_from_disk_or_compute(
                    problem, self._num_paths, self.edge_disjoint, self.dist_metric
                )
                splitter = SmartSplitter(num_subproblems, paths_dict)
            else:
                pf_original = PathFormulation.get_pf_for_obj(
                    self._objective,
                    self._num_paths,
                    edge_disjoint=self.edge_disjoint,
                    dist_metric=self.dist_metric,
                )
                splitter = GenericSplitter(
                    num_subproblems,
                    pf_original,
                    self._split_method,
                    self._split_fraction,
                )
        else:
            raise Exception("Invalid split_method {}".format(self._split_method))

        return splitter.split(problem)

    ###############################
    # Override superclass methods #
    ###############################

    def solve(self, problem):
        self._problem = problem
        # List of subproblems that have not been solved yet. Each time, we solve a subproblem,
        # we'll remove an index from it
        unsolved_subproblem_indices = list(range(self._num_subproblems))
        # Initialize all the PF objects for each subproblem index. Even if the subproblem changes
        # from one iteration to the next of the outer while loop, we'll keep the same PF object

        self._algos = [
            self._algo_cls(
                objective=self._objective,
                num_paths=self._num_paths,
                DEBUG=self.DEBUG,
                VERBOSE=self.VERBOSE,
                **self._addl_kwargs,
            )
            for _ in range(self._num_subproblems)
        ]
        self._paths_dict = self.get_paths(problem)
        # Initialize this to be a list of Nones; each time we solve a subproblem, we'll replace None
        # with the solved subproblem
        self._subproblem_list = [None for i in range(self._num_subproblems)]
        unsolved_subproblems = self.split_problems(problem, self._num_subproblems)
        leftover_capacities = defaultdict(float)

        self.iter = 0
        while len(unsolved_subproblem_indices) > 0:
            self._print("WHILE LOOP, ITER {}".format(self.iter))
            num_subproblems_in_iter = len(unsolved_subproblem_indices)
            subproblems_to_remove = []
            for i in unsolved_subproblem_indices:
                self._print("SUBPROBLEM {}, ITER {}".format(i, self.iter))
                subproblem = unsolved_subproblems[i]
                print(
                    "SUBPROBLEM {}, total demand: {}".format(i, subproblem.total_demand)
                )
                algo = self._algos[i]
                algo._paths_dict = self._paths_dict
                obj_val = algo.solve(
                    subproblem,
                    # Force Gurobi to use a single thread
                    num_threads=max(NUM_CORES // num_subproblems_in_iter, 1),
                )
                if obj_val is not None:
                    # If the subproblem was solved, then we'll replace the None in the list with
                    # the solved subproblem
                    self._subproblem_list[i] = subproblem
                    # We also queue the index for removal from the list of unsolved subproblem indices
                    subproblems_to_remove.append(i)
                    # Finally, we compute the residual by subproblem subtracting the solved subproblem
                    # from it
                    residual_subproblem = compute_residual_problem(
                        subproblem, algo.sol_dict
                    )
                    for u, v, cap in residual_subproblem.G.edges.data("capacity"):
                        leftover_capacities[(u, v)] += cap

                else:
                    self._print(
                        "SUBPROBLEM {}, ITER {} is infeasible".format(i, self.iter)
                    )
            for i in subproblems_to_remove:
                unsolved_subproblem_indices.remove(i)
            self.iter += 1
            if len(unsolved_subproblem_indices) == 0:
                break
            # Add the leftover capacities to the remaining subproblems
            for (u, v), extra_cap in leftover_capacities.items():
                extra_cap_per_remaining_subproblem = extra_cap / len(
                    unsolved_subproblem_indices
                )
                for i in unsolved_subproblem_indices:
                    unsolved_subproblems[i].G[u][v][
                        "capacity"
                    ] += extra_cap_per_remaining_subproblem

        assert len(unsolved_subproblem_indices) == 0
        assert len([p for p in self._subproblem_list if p is None]) == 0

    @property
    def sol_dict(self):
        if not hasattr(self, "_sol_dict"):
            sol_dicts = [pf.sol_dict for pf in self._algos]
            merged_sol_dict = defaultdict(list)
            for sol_dict in sol_dicts:
                for (_, (src, target, _)), flow_list in sol_dict.items():
                    merged_sol_dict[(src, target)] += flow_list
            self._sol_dict = {
                commod_key: merged_sol_dict[(commod_key[-1][0], commod_key[-1][1])]
                for commod_key in self.problem.commodity_list
            }

        return self._sol_dict

    @property
    def sol_mat(self):
        raise NotImplementedError(
            "sol_mat needs to be implemented in the subclass: {}".format(self.__class__)
        )

    def runtime_est(self, num_threads):
        return parallelized_rt([pf.runtime for pf in self._algos], num_threads)

    @property
    def runtime(self):
        return sum([pf.runtime for pf in self._algos])
