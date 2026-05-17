from __future__ import annotations
from planning.pddl import ActionSchema, State, Objects, get_applicable_actions
from planning.pddl import ActionSchema, State, Objects


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    actual = state
    used_actions = 0
    actions = get_applicable_actions(state, domain, objects)
    
    while not goal.issubset(actual):
        unsatisfied = goal - actual
        most_fluents_met = 0
        best = None
        
        for action in actions:
            fluents_met = len(action.add_list & unsatisfied)
            if fluents_met > most_fluents_met:
                most_fluents_met = fluents_met
                best = action
        
        if best is None:
            return float("inf")
        
        actual = actual | best.add_list        
        used_actions +=1
    return used_actions
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    actual = state
    costo = 0

    while not goal.issubset(actual):
        faltantes = goal - actual
        acciones = get_applicable_actions(actual, domain, objects)
        mejor = None
        mejor_ganancia = 0

        for accion in acciones:
            ganancia = len(accion.add_list & faltantes)
            if ganancia > mejor_ganancia:
                mejor = accion
                mejor_ganancia = ganancia

        if mejor is None:
            return float("inf")
        actual = frozenset(actual | mejor.add_list)
        costo += 1

    return costo
    ### End of your code ###
