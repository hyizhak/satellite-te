import os
from itertools import chain
from collections import defaultdict

from gurobipy import GRB, Model, quicksum

from lib.algorithms.abstract_formulation import Objective

from ..config import TOPOLOGIES_DIR
from ..lp_solver import LpSolver
from .path_formulation import PathFormulation

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "path-form")


class TEAVAR(PathFormulation):
    # failure_scenarios: [[(u_i, v_i),...(u_j, v_j)], ...[(u_k, v_k), ...(u_l, v_l)]], where (u_i, v_i) is an edge that failed in scenario s
    # failure_probs: [s_0, ..., s_n], where s_i is the probability of the i-th failure scenario
    def __init__(
        self,
        *,
        availability,
        failure_scenarios,
        failure_probs,
        num_paths,
        edge_disjoint=True,
        dist_metric="inv-cap",
        DEBUG=False,
        VERBOSE=False,
        out=None,
        objective=None,  # this argument has to be here so that it matches the signature of the superclass
    ):
        super().__init__(
            objective=Objective.TEAVAR,
            num_paths=num_paths,
            edge_disjoint=edge_disjoint,
            dist_metric=dist_metric,
            DEBUG=DEBUG,
            VERBOSE=VERBOSE,
            out=out,
        )
        assert len(failure_scenarios) == len(failure_probs)
        self._availability = availability
        self._failure_scenarios = failure_scenarios
        self._failure_probs = failure_probs

    def pre_solve(self, problem=None):
        edge_to_paths, num_paths = super().pre_solve(problem)
        self.failed_paths_per_scenario = []
        # [defaultdict_1(p_0: 1, …, p_n: 1), …, defaultdict_n(p_0: 1, …, p_n: # 1)]
        # where defaultdict_i is for the i-th failure scenario, p_j is for
        # the j-th path id
        for failure_scenario in self._failure_scenarios:
            failed_path_ids = list(
                chain.from_iterable(
                    edge_to_paths[edge]
                    for edge in failure_scenario
                    if edge in edge_to_paths
                )
            )
            failed_path_ids_dict = defaultdict(
                # 1 indicates the path is valid in this scenario,
                # 0 indicates that a link in the path has failed
                lambda: 1,
                [(p, 0) for p in failed_path_ids],
            )
            self.failed_paths_per_scenario.append(failed_path_ids_dict)
        return edge_to_paths, num_paths

    def _construct_path_lp(self, G, edge_to_paths, num_total_paths, sat_flows=[]):
        failure_probs = self._failure_probs
        failure_scenarios = self.failed_paths_per_scenario
        beta = self._availability
        num_scenarios = len(failure_scenarios)
        num_commodities = len(self.commodities)

        # Taken from page 5 of http://teavar.csail.mit.edu/paper.pdf
        m = Model("TEAVAR")
        # Create variables
        path_vars = m.addVars(num_total_paths, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
        # TEAVAR-specific variables
        alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="a")
        scenario_vars = m.addVars(num_scenarios, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
        scenario_commodity_vars = m.addVars(
            num_scenarios, num_commodities, vtype=GRB.CONTINUOUS, lb=0.0, name="sf"
        )
        m.update()
        weighted_losses = [
            fp * scenario_vars[s] for (fp, s) in zip(failure_probs, scenario_vars)
        ]
        m.setObjective(
            alpha + (1 / (1 - beta)) * quicksum(weighted_losses), GRB.MINIMIZE
        )

        self._demand_constrs = []
        # Add scenario constraints
        for s in scenario_vars:
            scenario_var = scenario_vars[s]
            for k, d_k, path_ids in self.commodities:
                # Demand constraints
                self._demand_constrs.append(
                    m.addConstr(quicksum(path_vars[p] for p in path_ids) <= d_k)
                )
                flow_in_scenario_vars = [
                    failure_scenarios[s][p] * path_vars[p] for p in path_ids
                ]
                commodity_loss = 1 - quicksum(flow_in_scenario_vars) / d_k
                m.addConstr(scenario_commodity_vars[s, k] >= commodity_loss)
                m.addConstr(scenario_var + alpha >= scenario_commodity_vars[s, k])

        # Add edge capacity constraints
        for u, v, c_e in G.edges.data("capacity"):
            if (u, v) in edge_to_paths:
                paths = edge_to_paths[(u, v)]
                constr_vars = [path_vars[p] for p in paths]
                m.addConstr(quicksum(constr_vars) <= c_e)

        if self.DEBUG:
            m.write("teavar_debug.lp")
        return LpSolver(m, None, self.DEBUG, self.VERBOSE, self.out)

    @property
    def obj_val(self):
        if not hasattr(self, "_obj_val"):
            self._obj_val = self._solver.obj_val
        return self._obj_val
