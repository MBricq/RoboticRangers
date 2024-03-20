import numpy as np
import vision

def bestPath(came_from, current):
    
    '''
    This function returns the best path from the start to the goal node.
    
    :param came_from: A list of nodes that shows the path from the start node to the goal node.
    :param current: The goal node.
    
    :return: A list of nodes that shows the best path from the start node to the goal node.
    '''
    
    path = [int(current)]
    
    while current != 0:
        current = int(came_from[current])
        path.append(current)
    
    path.reverse()
    
    return path

def global_path(node_matrix):
    
    '''
    This function finds the best path from the start node to the goal node.
    
    :param node_matrix: A matrix that shows the cost of moving from one node to another.
    
    :return: A list of nodes that shows the best path from the start node to the goal node.
    '''
    
    open_set = [0] # Always start from the first node
    closed_set = []
    
    came_from = np.zeros(len(node_matrix))
    cost_so_far = np.inf * np.ones(len(node_matrix))
    cost_so_far[0] = 0
        
    while len(open_set) > 0:
        # Find the node in open_set with the lowest cost
        best_node = open_set[0]
        for node in open_set:
            if cost_so_far[node] < cost_so_far[best_node]:
                best_node = node
        
        # Remove the best node from open_set
        open_set.remove(best_node)
        
        # Add the best node to closed_set
        closed_set.append(best_node)
        
        # If the goal is reached, return the bestPath
        if best_node == len(node_matrix)-1:
            return bestPath(came_from, best_node)
        
        # Update the cost of the neighbors of the best node
        for neighbor in range(len(node_matrix)):
            if node_matrix[best_node][neighbor] != -1 and neighbor not in closed_set:
                new_cost = cost_so_far[best_node] + node_matrix[best_node][neighbor]
                if new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = best_node
                    
                    if neighbor not in open_set:
                        open_set.append(neighbor)
                        
def generate_map(frame):
    """From the first frame, extract the map and generate the global path for the robot to follow.

    Args:
        frame: Image of the first frame

    Returns:
        positions_vertex: List of the positions of the vertices of the map
        path: List of the nodes that form the global path
        current_goal_idx: Index of the current goal in the path
        current_goal: Position of the current goal
        robot_pos: Position of the robot
        orientation: Orientation of the robot
        m_per_pixel: Conversion factor from pixel to meter
    """
    
    # Extract the map
    vertex_list, adj_matrix, _, m_per_pixel = vision.extract_map(frame)
    positions_vertex = np.array(vertex_list) * m_per_pixel
    path = global_path(adj_matrix)
    
    current_goal_idx = 1
    current_goal = positions_vertex[path[current_goal_idx]]

    # Find the robot position at start
    robot_pos, orientation, _ = vision.detect_robot(frame)
    robot_pos = robot_pos * m_per_pixel
    
    return positions_vertex, path, current_goal_idx, current_goal, robot_pos, orientation, m_per_pixel