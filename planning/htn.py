from __future__ import annotations

from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable
from planning.utils import Queue
from planning.domain import MOVE, PICKUP, PUTDOWN, RESCUE, SETUP_SUPPLIES


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """True if action is a grounded primitive Action, False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing the top-level HLA. At each step,
    find the first non-primitive step in the plan and replace it with one of
    its refinements. Continue until the plan is fully primitive AND, when
    executed from the initial state, reaches a goal state.
    """
    ### Your code here ###
    if not hlas:
        return []
    frontier: Queue = Queue()
    frontier.push([hlas[0]])
    while not frontier.isEmpty():
        plan = frontier.pop()
        if is_plan_primitive(plan):
            state = problem.getStartState()
            valid = True
            for action in plan:
                if not is_applicable(state, action):
                    valid = False
                    break
                state = apply_action(state, action)
            if valid and problem.isGoalState(state):
                return plan
            continue

        for i, step in enumerate(plan):
            if not is_primitive(step):
                for refinement in step.refinements:
                    new_plan = plan[:i] + refinement + plan[i + 1:]
                    frontier.push(new_plan)
                break

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Grid-level pathfinding (helper for Navigate refinements)
# ---------------------------------------------------------------------------


def _shortest_path(layout, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """BFS over grid cells. Returns the list of cells [start, ..., goal]."""
    if start == goal:
        return [start]
    walls = layout.walls
    w, h = layout.width, layout.height
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not walls[nx][ny]:
                nxt = (nx, ny)
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == goal:
                    return new_path
                visited.add(nxt)
                queue.append((nxt, new_path))
    return [start]


# ---------------------------------------------------------------------------
# Punto 5a – HLA constructors with concrete refinements
# ---------------------------------------------------------------------------


def _build_navigate(layout, from_cell, to_cell) -> HLA:
    """
    Navigate(from, to) refinement: the sequence of grounded Move actions that
    walks the shortest path between the two cells.
    """
    path = _shortest_path(layout, from_cell, to_cell)
    refinement: list = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        move = MOVE.ground({"r": "robot", "from_cell": a, "to_cell": b})
        refinement.append(move)
    hla = HLA(f"Navigate({from_cell}->{to_cell})")
    hla.refinements.append(refinement)
    return hla


def _build_prepare_supplies(layout, robot_start, supplies_name, supplies_loc, medical_post) -> HLA:
    """
    PrepareSupplies(s, m) refinement:
        Navigate(robot_start -> supplies_loc),
        PickUp(robot, s, supplies_loc),
        Navigate(supplies_loc -> medical_post),
        SetupSupplies(robot, s, medical_post)
    """
    nav1 = _build_navigate(layout, robot_start, supplies_loc)
    pickup = PICKUP.ground({"r": "robot", "obj": supplies_name, "loc": supplies_loc})
    nav2 = _build_navigate(layout, supplies_loc, medical_post)
    setup = SETUP_SUPPLIES.ground({"r": "robot", "s": supplies_name, "loc": medical_post})
    hla = HLA(f"PrepareSupplies({supplies_name}@{medical_post})")
    hla.refinements.append([nav1, pickup, nav2, setup])
    return hla


def _build_extract_patient(layout, robot_start, patient_name, patient_loc, medical_post) -> HLA:
    """
    ExtractPatient(p, m) refinement:
        Navigate(robot_start -> patient_loc),
        PickUp(robot, p, patient_loc),
        Navigate(patient_loc -> medical_post),
        PutDown(robot, p, medical_post),
        Rescue(robot, p, medical_post)
    """
    nav1 = _build_navigate(layout, robot_start, patient_loc)
    pickup = PICKUP.ground({"r": "robot", "obj": patient_name, "loc": patient_loc})
    nav2 = _build_navigate(layout, patient_loc, medical_post)
    putdown = PUTDOWN.ground({"r": "robot", "obj": patient_name, "loc": medical_post})
    rescue = RESCUE.ground({"r": "robot", "p": patient_name, "loc": medical_post})
    hla = HLA(f"ExtractPatient({patient_name}@{medical_post})")
    hla.refinements.append([nav1, pickup, nav2, putdown, rescue])
    return hla


def _build_full_mission(
    layout,
    robot_start,
    supplies_name,
    supplies_loc,
    patient_name,
    patient_loc,
    medical_post,
    with_supplies: bool = True,
) -> HLA:
    """
    FullRescueMission(s, p, m) refinement: composition of two HLAs.
        [PrepareSupplies(s, m), ExtractPatient(p, m)]
    When with_supplies is False (additional patients in multi-rescue, where
    SuppliesReady(m) already holds), the PrepareSupplies HLA is omitted.
    """
    steps: list = []
    if with_supplies:
        prepare = _build_prepare_supplies(
            layout, robot_start, supplies_name, supplies_loc, medical_post
        )
        steps.append(prepare)
        next_start = medical_post  # robot ends at m after SetupSupplies
    else:
        next_start = robot_start
    extract = _build_extract_patient(
        layout, next_start, patient_name, patient_loc, medical_post
    )
    steps.append(extract)
    hla = HLA(f"FullRescueMission({patient_name})")
    hla.refinements.append(steps)
    return hla


# ---------------------------------------------------------------------------
# Punto 5a + 5b – build_htn_hierarchy
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    Single-patient (SimpleRescueProblem):
        top = FullRescueMission(s_0, p_0, m_0)
              -> [PrepareSupplies, ExtractPatient]

    Multi-patient (MultiRescueProblem):
        top = MultiRescue
              -> [FullRescueMission(p_0), FullRescueMission(p_1), ...]
        Only the first mission performs PrepareSupplies; subsequent ones
        skip it because SuppliesReady(m) is already true.

    The Navigate HLA chains a shortest-path sequence of Move actions between
    adjacent cells (computed by BFS on the grid).
    """
    ### Your code here ###
    layout = problem.layout
    objects = problem.objects
    patients = objects.get("patients", [])
    supplies = objects.get("supplies", [])
    medical_posts = objects.get("medical_posts", [])

    if not patients or not medical_posts:
        return []

    m = medical_posts[0]
    robot_start = layout.robot_position

    if len(patients) == 1:
        s_name = supplies[0] if supplies else None
        s_loc = layout.supplies[0] if layout.supplies else None
        p_name = patients[0]
        p_loc = layout.patients[0]
        full = _build_full_mission(
            layout, robot_start, s_name, s_loc, p_name, p_loc, m, with_supplies=True
        )
        return [full]

    # Multi-rescue
    missions: list = []
    current_start = robot_start
    s_name = supplies[0] if supplies else None
    s_loc = layout.supplies[0] if layout.supplies else None
    for i, p_name in enumerate(patients):
        p_loc = layout.patients[i]
        with_supplies = (i == 0 and s_name is not None)
        mission = _build_full_mission(
            layout,
            current_start,
            s_name,
            s_loc,
            p_name,
            p_loc,
            m,
            with_supplies=with_supplies,
        )
        missions.append(mission)
        current_start = m  # robot ends at m after each mission

    top = HLA("MultiRescue")
    top.refinements.append(missions)
    return [top]
    ### End of your code ###
