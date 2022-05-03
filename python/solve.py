"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
import random
from point import Point
import pulp

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve_greedy(instance: Instance) -> Solution:
    towers = []
    solution = Solution(instance,towers)
    solution.towers = towers
    solution.instance = instance
    uncovered = instance.cities
    final_remove_list = []
    while not solution.valid():
        bestPoint = Point(0,0)
        maxCount = 0
        for x in range(0,instance.grid_side_length):
            for y in range(0, instance.grid_side_length):
                count = 0
                point = Point(x,y)
                if point in towers:
                    continue
                remove_list = []
                for city in uncovered:
                    if Point.distance_obj(point,city) <= instance.coverage_radius:
                        remove_list.append(city)
                        count+=1
                if count > maxCount:
                    maxCount = count
                    bestPoint = point
                    final_remove_list = remove_list
        towers.append(bestPoint)
        for city in final_remove_list:
            uncovered.remove(city)
    return Solution(instance=instance, towers=towers)

def solve_random(instance: Instance) -> Solution:
    towers = []
    solution = Solution(instance,towers)
    solution.towers = towers
    solution.instance = instance
    uncovered = instance.cities.copy()
    while not solution.valid():
        point = Point(random.randint(0,instance.grid_side_length-1),random.randint(0,instance.grid_side_length-1))
        goodPoint = False
        if point in towers:
            continue
        for city in uncovered:
            if Point.distance_obj(point,city) <= instance.coverage_radius:
                goodPoint = True
                break
        if goodPoint == True:
            towers.append(point)
            for city in uncovered:
                if Point.distance_obj(point,city) <= instance.coverage_radius:
                    uncovered.remove(city)
    return Solution(instance=instance, towers=towers)

def solve_LPnum(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    prob += pulp.lpSum(vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LPmin(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            minDist = 1000000
            for city in instance.cities:
                dist = Point.distance_obj(Point(x,y), city).value
                if dist <= minDist and dist <= instance.coverage_radius:
                    minDist = dist
            if minDist == 1000000 or minDist == 0:
                weights.append(5)
            else:
                weights.append(1/minDist)
    prob += pulp.lpDot(weights, vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LP3(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            maxDist = Point.distance_obj(Point(x,y), instance.cities[0])
            for city in instance.cities:
                dist = Point.distance_obj(Point(x,y), city)
                if dist <= instance.coverage_radius:
                    maxDist = dist
            weights.append(maxDist.value)
    prob += pulp.lpDot(weights, vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LPmax(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMaximize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            maxDist = Point.distance_obj(Point(x,y), instance.cities[0])
            for city in instance.cities:
                dist = Point.distance_obj(Point(x,y), city)
                if dist >= maxDist and dist <= instance.coverage_radius:
                    maxDist = dist
            weights.append(-maxDist.value)
    prob += pulp.lpDot(weights, vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LP2(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            maxDist = Point.distance_obj(Point(x,y), instance.cities[0])
            for city in instance.cities:
                dist = Point.distance_obj(Point(x,y), city)
                if dist >= instance.coverage_radius:
                    maxDist = dist
            weights.append(maxDist.value)
    prob += pulp.lpDot(weights, vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LP4(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            maxDist = Point.distance_obj(Point(x,y), instance.cities[0])
            for city in instance.cities:
                dist = Point.distance_obj(Point(x,y), city)
                if dist >= maxDist:
                    maxDist = dist
            weights.append(maxDist.value)
    prob += pulp.lpDot(weights, vars.values())
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LP5(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for coord in vars.keys():
        sum = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                dist = Point.distance_obj(Point(x,y), Point(coord[0],coord[1]))
                if dist <= instance.penalty_radius:
                    sum.append(vars[(x,y)])
        weights.append(pulp.lpSum(sum))
    prob += pulp.lpSum(weights)
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

def solve_LP6(instance: Instance) -> Solution:
    var_keys = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            var_keys.append((x,y))
    vars = pulp.LpVariable.dicts("Tower", var_keys, cat="Binary")
    prob = pulp.LpProblem("myProblem", pulp.LpMinimize)
    coverageRadius = instance.coverage_radius
    for city in instance.cities:
        constraint = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                if Point.distance_obj(Point(x,y),city) <= coverageRadius:
                    constraint.append(vars[(x,y)])
        prob += pulp.lpSum(constraint) >= 1
    weights = []
    for coord in vars.keys():
        sum = []
        for x in range(0,instance.grid_side_length):
            for y in range(0,instance.grid_side_length):
                dist = Point.distance_obj(Point(x,y), Point(coord[0],coord[1]))
                if dist <= instance.penalty_radius:
                    sum.append(vars[(x,y)])
        weights.append(pulp.lpSum(sum))
    prob += pulp.lpSum(weights)
    prob.solve()
    towers = []
    for x in range(0,instance.grid_side_length):
        for y in range(0,instance.grid_side_length):
            if pulp.value(vars[(x,y)]) >= 1:
                towers.append(Point(x,y))
    return Solution(instance=instance,towers=towers)

SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy,
    "random": solve_random,
    "minlp": solve_LPmin,
    "num-tow": solve_LPnum,
    "maxlp": solve_LPmax,
    "lp3": solve_LP3,
    "lp2": solve_LP2,
    "lp4": solve_LP4,
    "lp5": solve_LP5,
    "lp6": solve_LP6
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
