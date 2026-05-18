from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier: Queue = Queue()
    frontier.push((start, []))
    visited: set[State] = {start}

    while not frontier.isEmpty():
        state, plan = frontier.pop()
        for next_state, action, _cost in problem.getSuccessors(state):
            if next_state in visited:
                continue
            new_plan = plan + [action]
            if problem.isGoalState(next_state):
                return new_plan
            visited.add(next_state)
            frontier.push((next_state, new_plan))
    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###

    if not (action.add_list & goal_set):
        return None

    if action.del_list & goal_set:
        return None
    

    new_goal = frozenset((goal_set - action.add_list) | action.precond_pos)
    
    if action.precond_neg & new_goal:
        return None

    return new_goal

    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    goal = problem.goal
    start = problem.getStartState()

    frontier = PriorityQueue()
    frontier.push((goal, []), len(goal))
    explored = set()
    all_actions = get_all_groundings(problem.domain, problem.objects)

    static_predicates = {"MedicalPost", "Adjacent", "Pickable"}

    while not frontier.isEmpty():
        current_goal, rev_plan = frontier.pop()
        
        if current_goal in explored:
            continue
        explored.add(current_goal)
        
        if current_goal.issubset(start):
            return list(reversed(rev_plan))

        unsatisfied = current_goal - start

        for action in all_actions:
            if action.add_list.isdisjoint(unsatisfied):
                continue

            new_goal = regress(current_goal, action)
            if new_goal is None:
                continue

            dead_end = any(
                fluent[0] in static_predicates and fluent not in start
                for fluent in new_goal
            )
            if dead_end:
                continue

            priority = len(new_goal - start)  # fluents aún no satisfechos
            if new_goal not in explored:
                frontier.push((new_goal, rev_plan + [action]), priority)

    return []

    ### End of your code ###


# ---------------------------------------------------------------------------
# REGISTRO DE USO DE IA (Política del Taller 4)
# ---------------------------------------------------------------------------
# Se utilizó un agente de IA durante la depuración de backwardSearch porque
# la versión inicial no terminaba (divergía en mediumRescue/cornerRescue).
#
# (1) VERSIÓN INICIAL (autoría propia, antes de IA):
#
#   def backwardSearch(problem):
#       goal = problem.goal
#       start = problem.getStartState()
#       frontier = Queue()
#       frontier.push((goal, []))
#       explored = set()
#       all_actions = get_all_groundings(problem.domain, problem.objects)
#       while not frontier.isEmpty():
#           current_goal, rev_plan = frontier.pop()
#           if current_goal in explored:
#               continue
#           explored.add(current_goal)
#           if current_goal.issubset(start):
#               return list(reversed(rev_plan))
#           for action in all_actions:
#               new_goal = regress(current_goal, action)
#               if new_goal is None:
#                   continue
#               if new_goal not in explored:
#                   frontier.push((new_goal, rev_plan + [action]))
#       return []
#
#   Problema observado: el algoritmo no terminaba. Los subobjetivos generados
#   contenían fluentes estáticos como ("MedicalPost", celda) para celdas que
#   no eran puestos médicos, generando ramas infinitas que nunca podrían
#   satisfacerse por el estado inicial.
#
# (2) PROMPT USADO:
#   "Mi backwardSearch para PDDL no termina en cornerRescue. Genera subobjetivos
#    que requieren MedicalPost en celdas que no lo son. ¿Cómo podo subobjetivos
#    que contienen predicados estáticos imposibles, y cómo priorizo la frontera
#    para no expandir en BFS puro?"
#
# (3) VERSIÓN FINAL (arriba): se aplicaron tres ajustes sugeridos por la IA y
#   validados por el grupo:
#     - Poda de predicados estáticos (MedicalPost, Adjacent, Pickable) que no
#       están en el estado inicial.
#     - Filtro de relevancia: action.add_list.isdisjoint(unsatisfied) antes de
#       llamar a regress, para evitar trabajo inútil.
#     - PriorityQueue con prioridad = |new_goal - start| (cantidad de fluentes
#       aún no satisfechos), para priorizar subobjetivos más cercanos al inicial.
#
# El resto del archivo (forwardBFS, regress, aStarPlanner) es de autoría propia
# sin asistencia de IA.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    start = problem.getStartState()

    frontier: PriorityQueue = PriorityQueue()
    frontier.push((start, [], 0), heuristic(start, problem.goal, problem.domain, problem.objects))

    best_cost: dict[State, int] = {start: 0}

    while not frontier.isEmpty():
        state, plan, cost = frontier.pop()
        if problem.isGoalState(state):
            return plan
        if cost > best_cost[state]:
            continue

        for next_state, action, step_cost in problem.getSuccessors(state):
            new_cost = cost + step_cost
            if next_state not in best_cost or new_cost < best_cost[next_state]:
                best_cost[next_state] = new_cost
                priority = new_cost + heuristic(
                    next_state,
                    problem.goal,
                    problem.domain,
                    problem.objects,
                )
                frontier.push((next_state, plan + [action], new_cost), priority)
    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
