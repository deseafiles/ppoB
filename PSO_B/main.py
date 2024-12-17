from PsoTwoVar import PsoTwoVar
# from dijkstra import Dijkstra
import random


def run_dijkstra():
    graph = {
        'V1': {'V2': 4, 'V3': 6, 'V4': 2},
        'V2': {'V3': 3, 'V5': 3},
        'V3': {'V6': 2, 'V7': 1},
        'V4': {'V3': 2, 'V7': 5},
        'V5': {'V6': 2, 'V8': 3},
        'V6': {'V8': 3},
        'V7': {'V8': 3},
        'V8': {}
    }

    dijkstra = Dijkstra(graph)
    dijkstra.route('V1', 'V8')
    dijkstra.display_result()


def run_pso_1_var_a():
    print("PSO NOMOR 1 a.")
    x_1_var = [0, 1, 2]
    v_1_var = [0, 0, 0]
    r_1_var = [0.5, 0.5]
    c_1_var = [0.5, 1]
    w_1_var = 1

    pso_1_var = PsoOneVar(x_1_var, v_1_var, r_1_var, c_1_var, w_1_var)
    pso_1_var.iter(3)
    # pso_1_var.plot_iteration(3)


def run_pso_1_var_b():
    print("PSO NOMOR 1 b.")
    array_x0 = [random.randint(-5, 5) for i in range(10)]

    for i, x0 in enumerate(array_x0):
        c = [0.5, 1]
        w = 1
        v = [0, 0, 0]

        r1, r2 = random.random(), random.random()
        print(
            f"Perulangan ke-{i + 1} dengan nilai x0 = {x0}, r1 = {r1}, r2 = {r2}")
        r = [r1, r2]
        x = [x0, 1, 2]
        pso = PsoOneVar(x, v, c, r, w)
        pso.plot_iteration(100)


def run_pso_2_var_a():
    print("PSO NOMOR 2 a.")
    x_2_var = [1, 1, 2]
    y_2_var = [1, -1, 1]
    v_2_var = [0, 0, 0]
    c_2_var = [1, 0.5]
    r_2_var = [1, 1]
    w_2_var = 1

    pso_2_var = PsoTwoVar(x_2_var, y_2_var, v_2_var, c_2_var, r_2_var, w_2_var)
    pso_2_var.iter(5)
    pso_2_var.plot_iteration(3)


def run_pso_2_var_b():
    print("PSO NOMOR 2 b.")
    array_x0_2_var = [random.randint(-10, 10) for i in range(10)]
    array_y0_2_var = [random.randint(-10, 10) for i in range(10)]

    for i, (x0, y0) in enumerate(zip(array_x0_2_var, array_y0_2_var)):
        c = [1, 0.5]
        w = 1
        v = [0, 0, 0]
        r1, r2 = random.random(), random.random()
        print(
            f"Perulangan ke-{i + 1} dengan nilai x0 = {x0}, y0 = {y0}, r1 = {r1}, r2 = {r2}")
        r = [r1, r2]
        x = [x0, 1, 2]
        y = [y0, 1, -1]
        pso_2_var_x_y = PsoTwoVar(x, y, v, c, r, w)
        pso_2_var_x_y.plot_iteration(25)


run_pso_2_var_a()