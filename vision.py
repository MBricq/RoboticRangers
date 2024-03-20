import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants for the image processing
ROBOT_MARKER_ID = 3
GOAL_MARKER_ID = 4

THRESHOLD = 50

DISTANCE_CORNER_0_TO_1 = 0.049 # m
ROBOT_RADIUS = 0.06 # m
DISTANCE_TEST = 0.005 # m

OFFSET_CORNERS_PX = 11


def extract_map(img: np.ndarray, display_imgs: bool = False):
    """
    Extracts the map from the image. The graph returns will contain the nodes and the edges between corners of obstacles and
    goal position. The goal position is the center of the goal marker.
    The goal node will be the last node of the graph.
    Afterwards, at each step, will need to detect position of the robot and update the graph accordingly.
    
    :param img: Image to extract the map from.
    :param display: If True, will display the image with the map extracted.
    :return:(vertex_box, adj_matrix, binarized_img, m_per_pixel)
        vertex_box: Array of shape (n, 2) containing the coordinates of the vertices of the map.
        adj_matrix: Adjacency matrix of the graph. The value at (i, j) is the distance between the i-th and j-th vertex.
            If the value is -1, it means the two vertices are not connected.
        binarized_img: Binarized image of the map.
        m_per_pixel: Ratio of meters per pixel.
    """
    
    # First detect the edges of the map
    # Choose which dictionary to use
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect the markers
    marker_corners, marker_ids, _ = detector.detectMarkers(cv_frame)
    
    if type(marker_ids) == type(None):
        raise Exception("No markers have been detected")
    
    if GOAL_MARKER_ID not in marker_ids:
        raise Exception("Goal marker has not been detected")
    
    if ROBOT_MARKER_ID not in marker_ids:
        raise Exception("Robot marker has not been detected")
    
    # Extract the corners of the goal marker
    goal_index = np.where(marker_ids == GOAL_MARKER_ID)[0][0]
    goal_corners = marker_corners[goal_index][0]
    goal_pos = np.mean(goal_corners, axis=0)

    # Extract robot position
    idx_marker = np.where(marker_ids == ROBOT_MARKER_ID)[0][0]
    robot_corners = marker_corners[idx_marker][0]
    center_pos = np.mean(robot_corners, axis=0)
    
    # Compute the pixel to meter ratio
    m_per_pixel = DISTANCE_CORNER_0_TO_1 / np.linalg.norm(goal_corners[0] - goal_corners[1])
    
    robot_radius_img = int(ROBOT_RADIUS / m_per_pixel)
    if robot_radius_img % 2 == 0:
        robot_radius_img += 1
    
    # Image processing to extract the map
    
    # First remove the markers from the image
    cv_no_artag = cv2.fillPoly(cv_frame, np.int32([goal_corners]), (255, 255, 255))
    cv_no_artag = cv2.fillPoly(cv_no_artag, np.int32([robot_corners]), (255, 255, 255))
    
    
    # Binarize the image
    # Convert to grayscale
    arena_grey = cv2.cvtColor(cv_no_artag, cv2.COLOR_RGB2GRAY)

    # Apply threshold
    _, objects_mask = cv2.threshold(arena_grey, THRESHOLD, 255, cv2.THRESH_BINARY)

    half_robot_radius_img = int(robot_radius_img / 2) + 1
    
    kernel_remove = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (half_robot_radius_img, half_robot_radius_img))
    kernel_rob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_radius_img, robot_radius_img))
    kernel_offset_corner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OFFSET_CORNERS_PX, OFFSET_CORNERS_PX))

    # These two next steps are to remove any small black artifacts, such as shadows
    only_objects = cv2.dilate(objects_mask, kernel_remove, iterations=1)
    dilated_objects = cv2.erode(only_objects, kernel_remove, iterations=2)

    # Dilate the objects so that the robot will not pass above them
    dilated_objects = cv2.erode(dilated_objects, kernel_rob, iterations=1)
    
    # Add a small offset to the corners of the objects
    dilated_objects_cnt = cv2.erode(dilated_objects, kernel_offset_corner, iterations=1)
    
    # Reverse black and white
    dilated_objects = 255 - dilated_objects
    dilated_objects_cnt = 255 - dilated_objects_cnt
    
    # Find the contours of the dilated image
    contours, _ = cv2.findContours(dilated_objects_cnt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    vertex_box = np.array([center_pos])

    for cnt in contours:
        
        # Compute the bounding box of the contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Keep the points inside the image
        box = box[np.where((box[:, 0] >= 0) & (box[:, 0] < img.shape[1]) & (box[:, 1] >= 0) & (box[:, 1] < img.shape[0]))[0]]
        
        if vertex_box.size == 0:
            vertex_box = box
        else:
            vertex_box = np.concatenate((vertex_box, box), axis=0)

    # Add the goal's center
    vertex_box = np.concatenate((vertex_box, [goal_pos]), axis=0)

    # Use the vertex_box to build a visibility graph
    # The representation of the graph is an adjacency matrix for which the value at (i, j) is the distance between the i-th and j-th vertex
    # If the value is -1, it means the two vertices are not connected
    # Main diagonal is filled with -1 (a vertex is not connected to itself)
    # The graph is undirected, so the matrix is symetric
    
    distance_test_px = int(DISTANCE_TEST / m_per_pixel)

    # Initialize the graph with -1
    adj_matrix = -1 * np.ones((vertex_box.shape[0], vertex_box.shape[0]))

    # Iterate over the vertices
    for i, vertex in enumerate(vertex_box):
        # Compute the distance between the vertex and all the others
        distances = np.linalg.norm(vertex_box - vertex, axis=1)
        
        # Need to make sure the link is not above an obstacle
        for j, neigh in enumerate(vertex_box):
            if i == j:
                continue
            
            # Compute the number of points to check
            num_points = int(distances[j] / distance_test_px)
            
            # Check if there is an obstacle between the two vertices
            line = np.linspace(vertex, neigh, num=num_points) 

            # Check if any point of the line is in the obstacles
            if np.sum(dilated_objects[np.int32(line[:, 1]), np.int32(line[:, 0])] == 255) > 0:
                # There is an obstacle
                distances[j] = -1
        
        
        # Set the distance to itself to -1
        distances[i] = -1
        # Set the corresponding values in the graph
        adj_matrix[i] = distances

    if display_imgs:
        figure, axes = plt.subplots(1, 3, figsize=(20, 8))
        axes[0].imshow(dilated_objects, cmap='gray')
        axes[0].set_title("Binarized image of the map with dilated objects (step 3)")
        axes[0].set_xlabel("x (px)")
        axes[0].set_ylabel("y (px)")
        #axes[0].axis('off')
        
        # Create image to plot voxels on
        img_graph1 = img.copy()
        
        for i, vertex in enumerate(vertex_box):
            color = (255, 0, 0)
            if i == 0:
                color = (0, 255, 0)
            elif i == vertex_box.shape[0] - 1:
                color = (255, 255, 0)
            cv2.circle(img_graph1, (int(vertex[0]), int(vertex[1])), 30, color, -1)
        
        axes[1].imshow(img_graph1)
        axes[1].set_title("Vertices of the map (step 4)")
        axes[1].set_xlabel("x (px)")
        axes[1].set_ylabel("y (px)")
        #axes[1].axis('off')
        
        # Create a copy of the image to plot the graph on
        img_graph2 = img_graph1.copy()
        
        for i, vertex in enumerate(vertex_box):
            for j, distance in enumerate(adj_matrix[i]):
                if distance != -1:
                    cv2.line(img_graph2, (int(vertex[0]), int(vertex[1])), (int(vertex_box[j][0]), int(vertex_box[j][1])), (255, 0, 0), 2)
                    
        axes[2].imshow(img_graph2)
        axes[2].set_title("Graph of the map (step 5)")
        axes[2].set_xlabel("x (px)")
        axes[2].set_ylabel("y (px)")
        #axes[2].axis('off')
                
        plt.show()
            
    return vertex_box, adj_matrix, dilated_objects, m_per_pixel



def detect_robot(frame):
    """
    Detects the robot in the image.
    
    :param frame: Image to detect the robot in.
    
    :return: (robot_pos, angle) or (None,None) if the robot has not been detected.
        robot_pos: Position of the robot in the image.
        angle: Angle of the robot in the image.
    """
    
    # Choose which dictionary to use
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the markers
    marker_corners, marker_ids, _ = detector.detectMarkers(cv_frame)
    
    if type(marker_ids) == type(None):
        return np.array([-1, -1]), 0, False
    
    if GOAL_MARKER_ID not in marker_ids:
        goal_detected = False
    else:
        goal_detected = True
    
    if ROBOT_MARKER_ID not in marker_ids:
        return np.array([-1, -1]), 0, goal_detected
    
    idx_marker = np.where(marker_ids == ROBOT_MARKER_ID)[0][0]
    robot_corners = marker_corners[idx_marker][0]
    robot_pos = np.mean(robot_corners, axis=0)
    
    main_axis = robot_corners[0] - robot_corners[3]
    # Compute the angle between the main axis and the x axis
    angle = np.arctan2(main_axis[1], main_axis[0])
    
    return robot_pos, angle, goal_detected


def update_frame(frame, xkk, Pkk, positions_vertex, path, current_goal, m_per_pixel):
    """ Use pyplot to plot the robot position and path in real time.

    Args:
        frame: The current frame
        xkk: Array containing: [x, y, theta, vl, vr]
        Pkk: Covariance matrix
        positions_vertex: Positions of all the vertices in the map in m
        path: The vertices in the path
        current_goal: Current index of the goal in the path
        m_per_pixel: Conversion factor from pixels to meters
        
    Returns:
        img: The updated frame
    """
        
    img = frame.copy()
    
    # Plot the robot position
    robot_pos = xkk[:2] / m_per_pixel
    cv2.circle(img, tuple(robot_pos.astype(int)), 15, (0, 0, 255), -1)
    
    # Add an ellipse to show variance
    x_var = Pkk[0, 0] / m_per_pixel
    y_var = Pkk[1, 1] / m_per_pixel
    cv2.ellipse(img, tuple(robot_pos.astype(int)), (int(x_var), int(y_var)), 0, 0, 360, (0, 255, 0), 2)
    
    # Orientation
    orientation = xkk[2]
    cv2.line(img, tuple(robot_pos.astype(int)), tuple((robot_pos + 50 * np.array([np.cos(orientation), np.sin(orientation)])).astype(int)), (0, 0, 255), 2)
    
    # Cone of incertainty
    angle_var = Pkk[2, 2]
    orientation_min = orientation - angle_var
    orientation_max = orientation + angle_var
    cv2.line(img, tuple(robot_pos.astype(int)), tuple((robot_pos + 50 * np.array([np.cos(orientation_min), np.sin(orientation_min)])).astype(int)), (0, 255, 0), 2)
    cv2.line(img, tuple(robot_pos.astype(int)), tuple((robot_pos + 50 * np.array([np.cos(orientation_max), np.sin(orientation_max)])).astype(int)), (0, 255, 0), 2)
    
    # Plot the path
    vertex_list = positions_vertex / m_per_pixel
    path_to_draw = vertex_list[path[current_goal:]]
    path_to_draw = np.concatenate((robot_pos.reshape(1,2), path_to_draw), axis=0)
    cv2.polylines(img, [path_to_draw.astype(int)], False, (255, 0, 0), 2)
    
    return img
    