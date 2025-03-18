# TCSS 435 A Fall
# Zachary Anderson

import numpy as np

# All positions in specified maze sizes to be referenced when finding probability
# values and storing the probability values in the correct position of the matrix
# and also using the probabilty value indexes and comparing it to these positions
# to be used as moves for the path when being created for the agent
start_array_3 = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
start_array_5 = [(1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3), (2,4), (2,5),
                 (3,1), (3,2), (3,3), (3,4), (3,5), (4,1), (4,2), (4,3), (4,4), (4,5),
                 (5,1), (5,2), (5,3), (5,4), (5,5)]
start_array_10 = [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10),
                  (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10),
                  (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10),
                  (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10),
                  (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10),
                  (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10),
                  (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10),
                  (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10),
                  (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8), (9,9), (9,10),
                  (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10)]
start_array_20 = [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15), (1,16), (1,17), (1,18), (1,19), (1,20),
                  (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11), (2,12), (2,13), (2,14), (2,15), (2,16), (2,17), (2,18), (2,19), (2,20),
                  (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (3,13), (3,14), (3,15), (3,16), (3,17), (3,18), (3,19), (3,20),
                  (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (4,11), (4,12), (4,13), (4,14), (4,15), (4,16), (4,17), (4,18), (4,19), (4,20),
                  (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12), (5,13), (5,14), (5,15), (5,16), (5,17), (5,18), (5,19), (5,20),
                  (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12), (6,13), (6,14), (6,15), (6,16), (6,17), (6,18), (6,19), (6,20),
                  (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15), (7,16), (7,17), (7,18), (7,19), (7,20),
                  (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12), (8,13), (8,14), (8,15), (8,16), (8,17), (8,18), (8,19), (8,20),
                  (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8), (9,9), (9,10), (9,11), (9,12), (9,13), (9,14), (9,15), (9,16), (9,17), (9,18), (9,19), (9,20),
                  (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10), (10,11), (10,12), (10,13), (10,14), (10,15), (10,16), (10,17), (10,18), (10,19), (10,20),
                  (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7), (11,8), (11,9), (11,10), (11,11), (11,12), (11,13), (11,14), (11,15), (11,16), (11,17), (11,18), (11,19), (11,20),
                  (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7), (12,8), (12,9), (12,10), (12,11), (12,12), (12,13), (12,14), (12,15), (12,16), (12,17), (12,18), (12,19), (12,20),
                  (13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7), (13,8), (13,9), (13,10), (13,11), (13,12), (13,13), (13,14), (13,15), (13,16), (13,17), (13,18), (13,19), (13,20),
                  (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7), (14,8), (14,9), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15), (14,16), (14,17), (14,18), (14,19), (14,20),
                  (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,7), (15,8), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15), (15,16), (15,17), (15,18), (15,19), (15,20),
                  (16,1), (16,2), (16,3), (16,4), (16,5), (16,6), (16,7), (16,8), (16,9), (16,10), (16,11), (16,12), (16,13), (16,14), (16,15), (16,16), (16,17), (16,18), (16,19), (16,20),
                  (17,1), (17,2), (17,3), (17,4), (17,5), (17,6), (17,7), (17,8), (17,9), (17,10), (17,11), (17,12), (17,13), (17,14), (17,15), (17,16), (17,17), (17,18), (17,19), (17,20),
                  (18,1), (18,2), (18,3), (18,4), (18,5), (18,6), (18,7), (18,8), (18,9), (18,10), (18,11), (18,12), (18,13), (18,14), (18,15), (18,16), (18,17), (18,18), (18,19), (18,20),
                  (19,1), (19,2), (19,3), (19,4), (19,5), (19,6), (19,7), (19,8), (19,9), (19,10), (19,11), (19,12), (19,13), (19,14), (19,15), (19,16), (19,17), (19,18), (19,19), (19,20),
                  (20,1), (20,2), (20,3), (20,4), (20,5), (20,6), (20,7), (20,8), (20,9), (20,10), (20,11), (20,12), (20,13), (20,14), (20,15), (20,16), (20,17), (20,18), (20,19), (20,20),]

start_matrix = []
final_path = {}
return_path = []
visited_nodes = []
agent_position = 0
goal_position = 0

# Creates a path from the final path array for the agent using backtracking
def possible_move(maze, postion):
    visited = postion
    min_value = 0
    position = 0
    moves = []

    # Navigates the maze map dictionary of dictionaries and adds possible moves
    # and checks to see if the moves are in the final path dictionary and if so
    # then add the position
    for move in maze.maze_map[visited]:
        if move == 'E' and maze.maze_map[visited]['E'] == 1:
            x, y = visited
            east = x + 0, y + 1
            if east not in visited_nodes and east in final_path:
                moves.append(east)

        if move == 'W' and maze.maze_map[visited]['W'] == 1:
            x, y = visited
            west = x + 0, y - 1
            if west not in visited_nodes and west in final_path:
                moves.append(west)

        if move == 'N' and maze.maze_map[visited]['N'] == 1:
            x, y = visited
            north = x - 1, y + 0
            if north not in visited_nodes and north in final_path:
                moves.append(north)

        if move == 'S' and maze.maze_map[visited]['S'] == 1:
            x, y = visited
            south = x + 1, y + 0
            if south not in visited_nodes and south in final_path:
                moves.append(south)

    # if there is more then one possible move for the agent to take then
    # the probability value is checked and since we are searching backwards
    # the move with the highest value is then selected
    if (len(moves) > 1):
        for each in moves:
            temp = final_path.get(each)
            if (temp > min_value):
                position = each
                min_value = temp
    elif (len(moves) == 1):
        position = moves.pop()

    return position

# Keeps track of all possible moves that the agent can take with each step of matrix multiplication
def create_path(array, maze_size):
    count = 0

    # Uses the count to check each index of start arrays built with all positions
    # and uses the array of probabilities of possible moves and adds the position
    # and probability value to a final path dictionary so it can be used to create
    # a path for the agent
    if (maze_size == 3):
        for i in array:
            if (i > 0):
                if (start_array_3[count] not in visited_nodes):
                    visited_nodes.append(start_array_3[count])
                    final_path.update({start_array_3[count]: i})
            count = count + 1
    elif (maze_size == 5):
        for i in array:
            if (i > 0):
                if (start_array_5[count] not in visited_nodes):
                    visited_nodes.append(start_array_5[count])
                    final_path.update({start_array_5[count]: i})
            count = count + 1
    elif (maze_size == 10):
        for i in array:
            if (i > 0):
                if (start_array_10[count] not in visited_nodes):
                    visited_nodes.append(start_array_10[count])
                    final_path.update({start_array_10[count]: i})
            count = count + 1
    elif (maze_size == 20):
        for i in array:
            if (i > 0):
                if (start_array_20[count] not in visited_nodes):
                    visited_nodes.append(start_array_20[count])
                    final_path.update({start_array_20[count]: i})
            count = count + 1

# Used to multiply matrices using numpy
def matrix_multiplication(matrix_1, matrix_2):
    result_matrix = np.matmul(matrix_1, matrix_2)

    return result_matrix

# Used to create initial matrix
def calculate_moves(maze, position, array):
    filler_array = []
    move_array = []
    visited = position
    ratio = 0
    value = 0

    # Navigates the maze map dictionary of dictionaries and adds possible moves
    # and keeps track of all the possible moves with value and divides the value
    # from 1 and creates the probability value
    for move in maze.maze_map[visited]:
        if move == 'E' and maze.maze_map[visited]['E'] == 1:
            x, y = visited
            east = x + 0, y + 1
            move_array.append(east)
            value = value + 1

        if move == 'W' and maze.maze_map[visited]['W'] == 1:
            x, y = visited
            west = x + 0, y - 1
            move_array.append(west)
            value = value + 1

        if move == 'N' and maze.maze_map[visited]['N'] == 1:
            x, y = visited
            north = x - 1, y + 0
            move_array.append(north)
            value = value + 1

        if move == 'S' and maze.maze_map[visited]['S'] == 1:
            x, y = visited
            south = x + 1, y + 0
            move_array.append(south)
            value = value + 1
        
    # Gives ration for moves
    ratio = 1/value

    # Sorts all moves to be in ascending order
    move_array.sort()

    # Checks for each spot in start array and stores the probability ratios for each move in start matrix
    check = move_array.pop(0)
    for i in array:
        if (check == i):
            filler_array.append(ratio)
            if (len(move_array) > 0):
                check = move_array.pop(0)
        else:
            filler_array.append(0.0)

    start_matrix.append(filler_array)

# Markov Model on maze
def Markov(maze, maze_size, agent_start, goal):
    previous_matrix = None
    count = 1
    n_step = 1
    position = goal
    val = 1
    
    # Creates first matrix based off of maze size and assigns a row based off agent
    # start position and assigns the index at which the goal will be in the row
    if (maze_size == 3):
        agent_position = start_array_3.index(agent_start)
        goal_position = start_array_3.index(goal)
        for index in start_array_3:
            calculate_moves(maze, index, start_array_3)
    elif (maze_size == 5):
        agent_position = start_array_5.index(agent_start)
        goal_position = start_array_5.index(goal)
        for index in start_array_5:
            calculate_moves(maze, index, start_array_5)
    elif (maze_size == 10):
        agent_position = start_array_10.index(agent_start)
        goal_position = start_array_10.index(goal)
        for index in start_array_10:
            calculate_moves(maze, index, start_array_10)
    elif (maze_size == 20):
        agent_position = start_array_20.index(agent_start)
        goal_position = start_array_20.index(goal)
        for index in start_array_20:
            calculate_moves(maze, index, start_array_20)

    # Displays matrix to consol and readme.txt
    text = open("Readme.txt", "a")
    text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
    # Using nested loops to display starting Matrix
    for row in start_matrix:
        for element in row:
            text.write(str(element) + " ")
            print(element, end=" ")
        text.write("\n")
        print()
    text.write("***INITIAL MATRIX*** " + "n_step: " + str(n_step) + "\n")
    text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
    print("goal state: ", start_matrix[agent_position][goal_position])
    print("n_step: ", n_step)
    text.close()

    # Declare matrice to be multiplied into start matrix
    multiplied_matrix_1 = start_matrix

    # Adds next move to path with each iteration, that is to be sent to agent
    create_path(multiplied_matrix_1[agent_position], maze_size)

    print("-----------------------------------------------------------------------------------------------------------------")

    # Does matrix mulitplication with the start matrix and the next mulitplied matrix
    # till the exact position of the goal state within the matrices has a value greater
    # than zero, which produces the optimal path for the agent
    while(multiplied_matrix_1[agent_position][goal_position] == 0.0):
        multiplied_matrix_1 = matrix_multiplication(start_matrix, multiplied_matrix_1)
        count = count + 1
        n_step = n_step + 1

        # Displays matrix every 10 steps till 100 steps or when goal is found before 100 steps then displays final matrix
        if (n_step <= 100):
            if (multiplied_matrix_1[agent_position][goal_position] != 0.0 or count == 10):

                # Displays matrix to consol and readme.txt
                text = open("Readme.txt", "a")
                text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
                # Using nested loops to display starting Matrix
                for row in multiplied_matrix_1:
                    for element in row:
                        text.write(str(element) + " ")
                        print(element, end=" ")
                    text.write("\n")
                    print()
                text.write("n_step: " + str(n_step) + "\n")
                text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
                print("goal state: ", multiplied_matrix_1[agent_position][goal_position])
                print("n_step: ", n_step)
                text.close()

                count = 0

                print("-----------------------------------------------------------------------------------------------------------------")

        # Displays matrix when 100 steps has already happened and the goal state is found
        if (multiplied_matrix_1[agent_position][goal_position] != 0.0 and n_step > 100):

            # Displays matrix to consol and readme.txt
            text = open("Readme.txt", "a")
            text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
            # Using nested loops to display starting Matrix
            for row in multiplied_matrix_1:
                for element in row:
                    text.write(str(element) + " ")
                    print(element, end=" ")
                text.write("\n")
                print()
            text.write("n_step: " + str(n_step) + "\n")
            text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
            print("goal state: ", multiplied_matrix_1[agent_position][goal_position])
            print("n_step: ", n_step)
            text.close()

            print("-----------------------------------------------------------------------------------------------------------------")

        # Adds next move to path with each iteration, that is to be sent to agent
        create_path(multiplied_matrix_1[agent_position], maze_size)

    # Overwrites agent start position with a high probability value so it is always selected for final move in backtracked path for agent
    final_path.update({agent_start: 1.0})

    # Clears visited nodes to be reused for selecting path
    visited_nodes.clear()

    # Adds goal state to return path for agent and to visited nodes
    return_path.append(position)
    visited_nodes.append(position)

    # Backtracks a path when goal state is found in matrice
    while (position != agent_start):

        position = possible_move(maze, position)
        return_path.append(position)
        visited_nodes.append(position)

    # Since backtracked for path, the path is backwards, so this reverses the path from start state to goal state
    return_path.reverse()

    # Removes first position since it is the agent start position and is not a valid move
    return_path.pop(0)

    # Used to check previous matrix to compare with later multiplied matrix
    previous_matrix = multiplied_matrix_1

    # Computes matrix multiplication until the the current matrix matches the matrix
    # two steps before because there are two steady states that alternate back and forth
    while(True):
        multiplied_matrix_1 = matrix_multiplication(previous_matrix, multiplied_matrix_1)
        val = val + 1

        # Stores whether the two matrices are equal with True or False
        are_equal = np.allclose(previous_matrix, multiplied_matrix_1)

        # Stores the matrix every two steps to compare with current matrix
        if (val == 3):
            previous_matrix = multiplied_matrix_1
            val = 1

        # If steady state is found, display matrix to consol and readme.txt
        if (are_equal):

            # Displays matrix to consol and readme.txt
            text = open("Readme.txt", "a")
            text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
            # Using nested loops to display starting Matrix
            for row in multiplied_matrix_1:
                for element in row:
                    text.write(str(element) + " ")
                    print(element, end=" ")
                text.write("\n")
                print()
            text.write("***STEADY STATE FOUND***" + "\n")
            text.write("-----------------------------------------------------------------------------------------------------------------" + "\n")
            print("***STEADY STATE FOUND***")
            text.close()

            print("-----------------------------------------------------------------------------------------------------------------")

            break

    return return_path
