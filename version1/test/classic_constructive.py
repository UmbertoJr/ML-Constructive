import numpy as np


class EdgeInsertion:
    @staticmethod
    def check_if_available(n1, n2, sol):
        return True if (bool(n1 != n2) and len(sol[str(n1)]) < 2 and len(sol[str(n2)]) < 2) else False

    @staticmethod
    def innerLoopTracker(edge_to_append, sol):
        n1, n2 = edge_to_append
        if len(sol[str(n1)]) == 0:
            return True
        if len(sol[str(n2)]) == 0:
            return True
        cur_city = sol[str(n1)][0]
        partial_tour = [n1, cur_city]
        while True:
            if len(sol[str(cur_city)]) == 2:
                for i in sol[str(cur_city)]:
                    if i not in partial_tour:
                        cur_city = i
                        partial_tour.append(cur_city)
                        if cur_city == n2:
                            return False
            else:
                return True


    @staticmethod
    def create_solution(start_sol, sol, n):
        assert len(start_sol) == 2, "too many cities with just one link"
        end = False
        n1, n2 = start_sol
        from_city = n2
        sol_list = [n1, n2]
        while not end:
            for node_connected in sol[str(from_city)]:
                if node_connected not in sol_list:
                    from_city = node_connected
                    sol_list.append(node_connected)
                if len(sol_list) == n:
                    end = True
        return sol_list

    @staticmethod
    def remove_hub(solution, hub_node):
        list_to_rem = solution[str(hub_node)]
        for node in list_to_rem:
            solution[str(node)].remove(hub_node)
        solution[str(hub_node)] = []
        return solution, len(list_to_rem)

    @staticmethod
    def get_free_nodes(solution):
        free = []
        for key in solution.keys():
            if len(solution[key]) < 2:
                free.append(int(key))
        return free

    @staticmethod
    def find_hub(dist_matrix):
        return np.argmin(np.sum(dist_matrix, axis=0))

    @staticmethod
    def compute_savings(dist_matrix, hub_node):
        n = dist_matrix.shape[0]
        A = np.tile(dist_matrix[hub_node], n).reshape((n, n))
        return A + np.transpose(A) - dist_matrix


class MultiFragment(EdgeInsertion):

    @staticmethod
    def mf(dist_matrix, solution=None, pre=False, inside=0):
        if solution is None:
            solution = {}
        mat = np.copy(dist_matrix)
        mat = np.triu(mat)
        mat[mat == 0] = 100000
        num_cit = dist_matrix.shape[0]
        start_list = [i for i in range(num_cit)]
        solution = {str(i): [] for i in range(num_cit)}
        for el in np.argsort(mat.flatten()):
            node1, node2 = el // num_cit, el % num_cit
            possible_edge = [node1, node2]

            if MultiFragment.check_if_available(node1, node2,
                                                solution):
                if MultiFragment.innerLoopTracker(possible_edge, solution):
                    solution[str(node1)].append(node2)
                    solution[str(node2)].append(node1)
                    if len(solution[str(node1)]) == 2:
                        start_list.remove(node1)
                    if len(solution[str(node2)]) == 2:
                        start_list.remove(node2)
                    inside += 1
                    if inside == num_cit - 1:
                        solution = MultiFragment.create_solution(start_list, solution, num_cit)
                        return solution


class ClarkeWright(EdgeInsertion):

    @staticmethod
    def cw(dist_matrix, solution=None, inside=0):
        if solution is None:
            solution = {}
        hub_node = ClarkeWright.find_hub(dist_matrix)
        savings_mat = ClarkeWright.compute_savings(dist_matrix, hub_node)
        mat = np.triu(savings_mat, 1)
        num_cit = mat.shape[0]
        start_list = [i for i in range(num_cit)]
        if len(solution) == 0:
            solution = {str(i): [] for i in range(num_cit)}
        else:
            solution, rem_number = ClarkeWright.remove_hub(solution, hub_node)
            inside -= rem_number
            for node in [int(n) for n in solution.keys() if len(solution[n]) == 2]:
                start_list.remove(node)
        for el in np.argsort(mat.flatten())[::-1]:
            node1, node2 = el // num_cit, el % num_cit
            possible_edge = [node1, node2]
            if hub_node not in possible_edge:
                if ClarkeWright.check_if_available(node1, node2, solution):
                    if ClarkeWright.innerLoopTracker(possible_edge, solution):
                        solution[str(node1)].append(node2)
                        solution[str(node2)].append(node1)
                        if len(solution[str(node1)]) == 2:
                            start_list.remove(node1)
                        if len(solution[str(node2)]) == 2:
                            start_list.remove(node2)
                        inside += 1
                        if inside == num_cit - 2:
                            free_list = ClarkeWright.get_free_nodes(solution)
                            if hub_node in free_list:
                                free_list.remove(hub_node)
                            node1 = free_list[0]
                            node2 = free_list[1]
                            solution[str(hub_node)] = [node1]
                            solution[str(node1)].append(hub_node)
                            start_list.remove(node1)
                            solution_list = ClarkeWright.create_solution(start_list, solution, num_cit)
                            return solution_list


class FarthestInsertion:

    @staticmethod
    def solve(dist_matrix, pos):
        mat = dist_matrix
        num_cities = mat.shape[0]
        free_cities = set(i for i in range(num_cities))
        i, j = np.unravel_index(mat.argmax(), mat.shape)
        solution_list = [i, j]
        free_cities.remove(i)
        free_cities.remove(j)
        while len(solution_list) != num_cities:
            next_node = FarthestInsertion.find(solution_list, free_cities, mat)
            index_insertion = FarthestInsertion.where(solution_list, next_node, mat)
            solution_list.insert(index_insertion, next_node)
            free_cities.remove(next_node)
        return solution_list


    @staticmethod
    def find(sol_list, free_, mat):
        row_index = np.array(sol_list)
        col_index = list(free_)
        small_mat = mat[row_index[:, None], col_index]
        sum_dist = np.sum(small_mat, axis=0)
        j = np.argmax(sum_dist)
        return col_index[j]

    @staticmethod
    def where(sol_list, next_node, mat):
        savings = []
        min_val = 10000000000
        ind_min = None
        for it_, i, j in zip(range(len(sol_list)), sol_list, np.roll(sol_list, 1)):
            val = mat[j, next_node] + mat[i, next_node] - mat[i, j]
            savings.append(val)
            if val < min_val:
                min_val = val
                ind_min = it_
        return ind_min


