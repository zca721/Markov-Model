#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TCSS 435 A Fall
# Zachary Anderson
#
#
#
# Assigment 3 Markov Model

import pymaze as maze
import random, sys
from Markov import Markov

text = open("Readme.txt", "a")

# Arguments passed for creating maze size and what loop percent to use
print("Name of Python script:", sys.argv[0])
print("Size of maze 20 or 10 or 5 or 3:", int(sys.argv[1]))
print("Loop percent 0 or 50:", sys.argv[2])

algorithm = sys.argv[0]
maze_size = int(sys.argv[1])
loop_percent = int(sys.argv[2])
text.write("******************************" + "\n")
text.write(algorithm + " " + str(maze_size) + " " + str(loop_percent) + ": " + "\n")
text.write("******************************" + "\n")
text.close()

# Randomly selects a position for the agent start state and goal state based off maze size
if maze_size == 20:
    m = maze.maze(20, 20)
    # All (x,y) cooridinates for Agents and maze goal
    agent_x = random.randint(1,20)
    agent_y = random.randint(1,20)
    goal_x = random.randint(1,20)
    goal_y = random.randint(1,20)
    agent_start = agent_x, agent_y
    goal = goal_x, goal_y

    print("Agent start: ", agent_start)
    print("Goal: ", goal)

    # Makes sure agent MAX doesnt have same spawn point as Goal
    while agent_x == goal_x and agent_y == goal_y:
        print("Agent has the same spot as Goal")
        agent_x = random.randint(1,20)
        agent_y = random.randint(1,20)
        agent_start = agent_x, agent_y
        print ("Agent new start: ", agent_start)

    print("-----------------------------------------------------------------------------------------------------------------")
elif maze_size == 10:
    m = maze.maze(10, 10)
    # All (x,y) cooridinates for Agents and maze goal
    agent_x = random.randint(1,10)
    agent_y = random.randint(1,10)
    goal_x = random.randint(1,10)
    goal_y = random.randint(1,10)
    agent_start = agent_x, agent_y
    goal = goal_x, goal_y

    print("Agent start: ", agent_start)
    print("Goal: ", goal)

    # Makes sure agent doesnt have same spawn point as Goal
    while agent_x == goal_x and agent_y == goal_y:
        print("Agent has the same spot as Goal")
        agent_x = random.randint(1,10)
        agent_y = random.randint(1,10)
        agent_start = agent_x, agent_y
        print ("Agent new start: ", agent_start)

    print("-----------------------------------------------------------------------------------------------------------------")
elif maze_size == 5:
    m = maze.maze(5,5)
    # All (x,y) cooridinates for Agents and maze goal
    agent_x = random.randint(1,5)
    agent_y = random.randint(1,5)
    goal_x = random.randint(1,5)
    goal_y = random.randint(1,5)
    agent_start = agent_x, agent_y
    goal = goal_x, goal_y

    print("Agent start: ", agent_start)
    print("Goal: ", goal)

    # Makes sure agent MAX doesnt have same spawn point as Goal
    while agent_x == goal_x and agent_y == goal_y:
        print("Agent has the same spot as Goal")
        agent_x = random.randint(1,5)
        agent_y = random.randint(1,5)
        agent_start = agent_x, agent_y
        print ("Agent new start: ", agent_start)

    print("-----------------------------------------------------------------------------------------------------------------")
elif maze_size == 3:
    m = maze.maze(3,3)
    # All (x,y) cooridinates for Agents and maze goal
    agent_x = random.randint(1,3)
    agent_y = random.randint(1,3)
    goal_x = random.randint(1,3)
    goal_y = random.randint(1,3)
    agent_start = agent_x, agent_y
    goal = goal_x, goal_y

    print("Agent start: ", agent_start)
    print("Goal: ", goal)

    # Makes sure agent MAX doesnt have same spawn point as Goal
    while agent_x == goal_x and agent_y == goal_y:
        print("Agent has the same spot as Goal")
        agent_x = random.randint(1,3)
        agent_y = random.randint(1,3)
        agent_start = agent_x, agent_y
        print ("Agent new start: ", agent_start)

    print("-----------------------------------------------------------------------------------------------------------------")
else:
    raise Exception("Invalid input! Choose 20 or 10 or 5 or 3 only.")

# Builds maze based off of maze size and loop percent
if loop_percent == 0:
    m.CreateMaze(goal_x,goal_y, loopPercent=loop_percent, theme = maze.COLOR.dark)
elif loop_percent == 50:
    m.CreateMaze(goal_x,goal_y, loopPercent=loop_percent, theme = maze.COLOR.dark)
else:
    raise Exception("Invalid input! Choose 0 or 50 only.")

# Creates an agent
a = maze.agent(m, agent_x, agent_y, footprints=True, color=maze.COLOR.red)

# Does matrix multiplication of move possibilities and creates a path for the agent
path = Markov(m, maze_size, agent_start, goal)

m.tracePath({a:path})

m.run()