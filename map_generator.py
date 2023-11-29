import random
import numpy as np

def generate_random_map(size=20):
    grid_map = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if random.random() < 0.2:
                grid_map[i][j] = 1
            if i == 0 or i == size-1 or j == 0 or j == size-1:
                grid_map[i][j] = 1
    return grid_map

def is_map_solvable(grid_map, start, goal):
    
    class Node:
        def __init__(self, position, parent=None):
            self.position = position
            self.parent = parent
            self.g = 0
            self.h = 0
            self.f = 0

        def __eq__(self, other):
            return self.position == other.position

    def astar(start, goal):
        open_set = []
        closed_set = []

        start_node = Node(start)
        goal_node = Node(goal)

        open_set.append(start_node)

        def get_neighbors(node):
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (node.position[0] + dx, node.position[1] + dy)
                if 0 <= new_pos[0] < len(grid_map) and 0 <= new_pos[1] < len(grid_map[0]) and grid_map[new_pos[0]][new_pos[1]] != 1:
                    neighbors.append(Node(new_pos))
            return neighbors

        while open_set:
            current_node = min(open_set, key=lambda x: x.f)
            open_set.remove(current_node)
            closed_set.append(current_node)

            if current_node == goal_node:
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return True, path[::-1]  # Path found, return path in reverse order

            neighbors = get_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                neighbor.g = current_node.g + 1
                neighbor.h = abs(neighbor.position[0] - goal_node.position[0]) + abs(neighbor.position[1] - goal_node.position[1])
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    open_set.append(neighbor)
                    neighbor.parent = current_node

        return False, []  # No path found

    if start is None or goal is None:
        return False

    return astar(start, goal)