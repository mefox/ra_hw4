#!/usr/bin/env python
# TESTING GIT MERGE

PACKAGE_NAME = 'hw4'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy

import collections
import Queue
from random import choice

# OpenRAVE
import openravepy
#openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


curr_path = os.getcwd()
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata


#this sets up the OPENRAVE_DATA environment variable to include the files we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
  if openrave_data_path == '':
      os.environ['OPENRAVE_DATA'] = ordata_path_thispack
  else:
      datastr = str('%s:%s'%(ordata_path_thispack, openrave_data_path))
      os.environ['OPENRAVE_DATA'] = datastr

#set database file to be in this folder only
relative_ordatabase = '/database'
ordatabase_path_thispack = curr_path + relative_ordatabase
os.environ['OPENRAVE_DATABASE'] = ordatabase_path_thispack

#get rid of warnings
openravepy.RaveInitialize(True, openravepy.DebugLevel.Fatal)
openravepy.misc.InitOpenRAVELogging()
#################################################################
#################################################################
# We use the following constants for the step size on the line,
# the probability of choosing a random configuration and the 
# condition to determine if the goal is reached or not
#################################################################
#################################################################

#constant for max distance to move any joint in a discrete step
MAX_MOVE_AMOUNT = 0.1

#Constant for probability of choosing a random target
PROB_RAND_TARGET = 0.9

#Constant for distance of current state near goal to determine the termination of the algorithm
DIST_THRESH = 0.8

#constant for max distance to move any joint in a discrete step
MAX_MOVE_AMOUNT = 0.1


class RoboHandler:
  def __init__(self):
    self.openrave_init()
    self.problem_init()

    #self.run_problem_birrt()




  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW4 Viewer')
    self.env.Load('models/%s_birrt.env.xml' %PACKAGE_NAME)
    # time.sleep(3) # wait for viewer to initialize. May be helpful to uncomment
    self.robot = self.env.GetRobots()[0]

    #set right wam as active manipulator
    with self.env:
      self.robot.SetActiveManipulator('right_wam');
      self.manip = self.robot.GetActiveManipulator()

      #set active indices to be right arm only
      self.robot.SetActiveDOFs(self.manip.GetArmIndices() )
      self.end_effector = self.manip.GetEndEffector()

  #######################################################
  # problem specific initialization
  #######################################################
  def problem_init(self):
    self.target_kinbody = self.env.GetKinBody("target")
    print "initializing the problem"
    # create a grasping module
    self.gmodel = openravepy.databases.grasping.GraspingModel(self.robot, self.target_kinbody)
    

    print "loading grasps"
    # load grasps
    if not self.gmodel.load():
      self.gmodel.autogenerate()

    self.grasps = self.gmodel.grasps
    self.graspindices = self.gmodel.graspindices

    print "loading ik model"
    # load ikmodel
    self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
    if not self.ikmodel.load():
      self.ikmodel.autogenerate()

    # create taskmanip
    self.taskmanip = openravepy.interfaces.TaskManipulation(self.robot)

    print "moving the left arm away"  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )


  #######################################################
  # Harder search problem from last time - use an RRT to solve
  #######################################################
  def run_problem_birrt(self):
    self.robot.GetController().Reset()

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    #goals = self.get_goal_dofs(10,3)
    goals = np.array([[ 1.53442279, -1.11094749,  0.2       ,  1.89507469,  0.9253871 ,
        -0.27590187, -0.93353661],
       [ 1.08088326, -1.11094749, -0.2       ,  1.89507469, -1.15533182,
        -0.47627667,  1.40590175],
       [ 1.64865961, -1.08494965,  0.3       ,  1.89507469,  1.12567395,
        -0.42894989, -1.20064072],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846, -1.14447409],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188,  1.2076898 ,
        -0.55054165, -1.30156536],
       [ 1.80822781, -1.00617436,  0.5       ,  1.88331188,  1.23775906,
        -0.72454447, -1.40740396],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -0.73232284,
        -0.60044153,  0.9098408 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018,  0.68660509,
        -0.56962218, -0.85889052],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018,  0.83605503,
        -0.69762048, -1.08462636],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -0.72377213,
        -0.61047535,  0.90372445],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934,  0.67519725,
        -0.57794147, -0.84513898],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934,  0.82590826,
        -0.7053313 , -1.07222512],
       [ 3.62542331, -0.50373029, -0.1       ,  2.15372919, -0.90608947,
        -1.35422117,  1.22439759],
       [ 4.1163159 , -0.54152784, -0.2       ,  2.15372919, -0.82842861,
        -1.04081465,  0.94191546],
       [ 3.62542331, -0.50373029, -0.1       ,  2.15372919, -4.04768212,
         1.35422117, -1.91719506],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -0.53511583,
        -0.50586635,  0.66089972],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027,  0.44896204,
        -0.47742308, -0.55906299],
       [ 1.5684208 , -1.07995335,  0.3       ,  1.98216027,  0.68165593,
        -0.5789909 , -0.87398179],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188,  1.2076898 ,
        -0.55054165,  1.8400273 ],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846,  1.99711856],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188, -2.0210129 ,
         0.38546846, -1.14447409],
       [ 3.49661161, -0.34059995, -0.1       ,  1.38477553,  1.20833943,
         1.53448864, -0.39066223],
       [ 3.88076306, -0.36079555, -0.2       ,  1.38477553,  1.01389006,
         1.32684258, -0.28712797],
       [ 4.55120287, -0.42927425, -0.3       ,  1.38477553,  0.50597369,
         1.0068676 ,  0.07352285],
       [ 1.71823564, -1.04694097,  0.5       ,  2.01730926,  0.91767346,
        -0.80895727,  1.95274455],
       [ 1.60263915, -1.09602265,  0.4       ,  2.01730926,  0.81743246,
        -0.66449298,  2.13438883],
       [ 1.83615837, -0.98539873,  0.6       ,  2.01730926,  0.97511267,
        -0.96908448,  1.8045713 ],
       [ 1.60313817, -1.09414142,  0.4       ,  2.01536424,  0.81746904,
        -0.66473871, -1.0084334 ],
       [ 1.71902033, -1.04498968,  0.5       ,  2.01536424,  0.91747166,
        -0.8094239 , -1.19031272],
       [ 1.83728186, -0.98334683,  0.6       ,  2.01536424,  0.97461756,
        -0.96979975, -1.33875245]]) 
 
    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

    # get the trajectory!
    traj = self.birrt_to_goal(goals)
    #print traj

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

    print "TRAVERSING THE PATH NOW"
    time.sleep(5)
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()



  #######################################################
  # finds the arm configurations (in cspace) that correspond
  # to valid grasps
  # num_goal: number of grasps to consider
  # num_dofs_per_goal: number of IK solutions per grasp
  #######################################################
  def get_goal_dofs(self, num_goals=1, num_dofs_per_goal=1):
    validgrasps,validindices = self.gmodel.computeValidGrasps(returnnum=num_goals) 

    curr_IK = self.robot.GetActiveDOFValues()

    goal_dofs = np.array([])
    for grasp, graspindices in zip(validgrasps, validindices):
      Tgoal = self.gmodel.getGlobalGraspTransform(grasp, collisionfree=True)
      sols = self.manip.FindIKSolutions(Tgoal, openravepy.IkFilterOptions.CheckEnvCollisions)

      # magic that makes sols only the unique elements - sometimes there are multiple IKs
      sols = np.unique(sols.view([('',sols.dtype)]*sols.shape[1])).view(sols.dtype).reshape(-1,sols.shape[1]) 
      sols_scores = []
      for sol in sols:
        sols_scores.append( (sol, np.linalg.norm(sol-curr_IK)) )

      # sort by closest to current IK
      sols_scores.sort(key=lambda tup:tup[1])
      sols = np.array([x[0] for x in sols_scores])
      
      # sort randomly
      #sols = np.random.permutation(sols)

      #take up to num_dofs_per_goal
      last_ind = min(num_dofs_per_goal, sols.shape[0])
      goal_dofs = np.append(goal_dofs,sols[0:last_ind])

    goal_dofs = goal_dofs.reshape(goal_dofs.size/7, 7)

    return goal_dofs


  #TODO
  #######################################################
  # Bi-Directional RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def birrt_to_goal(self, goals):
    print 'Starting BiRRT Algorithm'
    goals = np.array(goals) #Bring the goals into an np array
    q_initial = self.robot.GetActiveDOFValues() # Initial state is a np array
    print "INITIAL POSITION", q_initial
    q_initial_tuple = self.convert_for_dict(q_initial) # This is the tuple for initial state
    q_nearest = q_initial # This is the initialization for nearest point in np array format

    thresh = DIST_THRESH # The threshold to determine if the goal is reached or not
    
    #Initialize the tree for the start node
    start_tree = [q_initial]
    #start_tree.append(q_initial_tuple)
    

    #Initialize the parent dictionary for the start node
    parent = {}
    parent[q_initial_tuple] = None # Dictionaay to store the start_parents


    #trees = np.array([]) #np array to store goal trees
    #parents = np.array() #np array to store goal parent dictionaries

    #Now take all the goals and make a tree to add to the array of goals and make a dictionary of child (key) parents (value) to add to the array of parents
    i = 0
    goal_trees=[]
    for goal in goals:
      goal_tree=[goal]
      goal_trees.append(goal_tree)       
        
      parent[self.convert_for_dict(goal)] = None
      #print "IIIIIII", i
      #goal_parents[i]= goal_parent
      i=i+1


    #Find the lower and upper limits of the joint angles
    lower, upper = self.robot.GetActiveDOFLimits() # Get the joint limits

    # Calculate the minimum distance between closest node of the tree and the closest goal. Return both indices
    # Important to note here that the INDEX is being returned
    #dist, closest_goal, closest_point = self.min_euclid_dist_many_to_many(goals, self.convert_from_dictkey(tree)) #Consider commenting?

    tree_not_connected = True
    print 'Completed initialization, lets make some trees!'
    config1 = q_initial
    config2 = q_initial
    while(tree_not_connected): # Keep checking if the tree has not already reached a nearest goal
      #First grow start tree
      q_target, q_nearest, min_dist = self.rrt_choose_target_from_start(start_tree, goal_trees, lower, upper) 
      success = self.rrt_extend(q_nearest, q_target, start_tree, parent, lower, upper)
      if min_dist<DIST_THRESH and success:
        tree_not_connected = False
        config1 = q_nearest
        config2 = q_target
        break

      #Grow all goal trees
      for tree in goal_trees:
        q_target, q_nearest, min_dist = self.rrt_choose_target_from_goal(start_tree,tree, lower, upper) # Function returns a randomly chosen configuration or a nearest goal to the tree
        success = self.rrt_extend(q_nearest, q_target, start_tree, parent, lower, upper)      
        if min_dist<DIST_THRESH and success:
          tree_not_connected = False
          config1 = q_nearest
          config2 = q_target
          break
 
    return self.backtrace(parent, config1, config2) # The backtrace function gives the path. Return the path
             
    #return None

  #######################################################
  # The choose target from start function either returns a random 
  # configuration or the closest randomly selected configuration 
  # from each goal tree. This configuration will be the target to which the 
  # tree tries to expand to. The function also returns the nearest configuration
  # on the start tree and the minimum distance by taking advantage of rrt_nearest
  #######################################################
  def rrt_choose_target_from_start(self, start_tree, goal_trees, lower, upper):
    p = PROB_RAND_TARGET # The probability of choosing a random configuration
    if (np.random.random_sample()<p):
      q_target = np.array(lower+np.random.rand(len(lower))*(upper-lower)) #Choose a random configuration
      min_dist, q_nearest, nearest_start_tree_index = self.rrt_nearest(start_tree, q_target)
	#print 'A random configuration is chosen'

    #Otherwise choose the closest of a random selection from each of the goal trees
    else:
      i=0
      q_targets =[]
      for tree in goal_trees:
         q_targets.append(choice(goal_trees[i]))

      min_dist, nearest_start_tree_index, nearest_goal_tree_index = self.min_euclid_dist_many_to_many(start_tree,q_targets)
      q_target = q_targets[nearest_goal_tree_index] #Choose the node from the goal_tree that is closest the start_tree
      q_nearest = start_tree[nearest_start_tree_index]
      # print q_target
    return q_target, q_nearest, min_dist

  #######################################################
  # The choose target from goal function either returns a random 
  # configuration or the closest randomly selected configuration 
  # from the start tree. This configuration will be the target to which the 
  # tree tries to expand to. The function also returns the nearest configuration
  # on the current goal tree and the minimum distance by taking advantage of rrt_nearest
  #######################################################
  def rrt_choose_target_from_goal(self, start_tree, goal_tree, lower, upper):
    p = PROB_RAND_TARGET # The probability of choosing a random configuration
    if (np.random.random_sample()<p):
	  q_target = np.array(lower+np.random.rand(len(lower))*(upper-lower)) #Choose a random configuration
	#print 'A random configuration is chosen'
    else:
       q_target=choice(start_tree)
    min_dist, q_nearest, nearest_goal_tree_index = self.rrt_nearest(goal_tree, q_target)
    return q_target, q_nearest, min_dist

  #######################################################
  #  The nearest function returns the point in the tree thus far that is 
  #  nearest to the target point and chosen in the previous function
  #######################################################
  def rrt_nearest(self, tree, q_target):
    min_dist, index_nearest = self.min_euclid_dist_one_to_many(q_target,tree) # Determine the nearest point
    q_nearest = tree[index_nearest] #Assign and then return the nearest point in the tree
    return min_dist, q_nearest, index_nearest

  ##################################################################################################
  #  The extend function extends the nearest node on the tree to the target in increments as long as
  #  it is allowed. Otherwise it extends until possible and then terminates
  ##################################################################################################
  def rrt_extend(self, q_nearest, q_target, tree, parent, lower, upper):
    direction_vector = q_target - q_nearest # Obtain the direction of the direct line from initial to goal state
#    print q_nearest
#    print q_target
    dist, index = self.min_euclid_dist_one_to_many(q_nearest, [q_target]) # The distance between the initial and final state
    steps = int(dist / MAX_MOVE_AMOUNT) # Determine how many steps it will take to reach the final state

    #Try to extend all the way to the target, terminate at step if not possible
    for count in range(steps):
      q_parent = q_nearest + (direction_vector)/steps*(count) # Calculate the parent to be each previous node
      #print q_parent
      q_add = q_nearest + (direction_vector)/steps*(count+1) # The next node is obtained by 'walking' on this direct line

      with self.env:
        self.robot.SetActiveDOFValues(q_add) # This is done for demo purposes. The robot will assume a configuration as it tests.
      reach_limit = self.robot.CheckSelfCollision() or self.env.CheckCollision(self.robot) or self.limitcheck(q_add, lower, upper)# Collision checker
      #print reach_limit

      if reach_limit:
        #print reach_limit
	return False # Terminate the function of a collision occurs. 

      tree.append(q_add) # Add the first element to the tree
      parent[self.convert_for_dict(q_add)] = q_parent # Update the parent dictionary	
    
    return True


  #######################################################
  # This function returns the trajectory that will move the robot
  #######################################################
  def backtrace(self, parent, config1, config2):
     path1 = []
     print "CONFIG 1",config1
     print "CONFIG 2",config2
     path1.append(config1)
     path2 = []
     path2.append(config2)
     #print 'Inside the backtrace function'
     #print path
     #print start
     while path1[-1] is not None:
       #print parent[path[-1]]
       path1.append(parent[self.convert_for_dict(path1[-1])])
       #print path

     while path2[-1] is not None:
       path2.append(parent[self.convert_for_dict(path2[-1])])

     path1.remove(None)
     path2.remove(None)
     
     path1.reverse()
     print "PATH 1", path1
     print "PATH 2", path2
     path = path1+path2
     print path
     traj = self.points_to_traj(path)
     return traj

  #######################################################
  #######################################################
   # Limit Check
   # if the current state reaches the limit, the function returns true
   # We are using this function to create a boundary for the search space in all the 7 dimensions
   # The robot will not be allowed to go beyond its DOF limits
   ########################################################  
  def limitcheck(self, state, lower, upper):
    true_num = sum(state>=lower)+sum(state<=upper)
    #print true_num
    return true_num < 14


  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    #return tuple(np.int_(item*100))
    #return tuple(item)
    return tuple(np.around(item,decimals = 3))

  def convert_from_dictkey(self, item):
    #return np.array(item)/100.
    #return np.array(item)
    return np.array(np.around(item,decimals = 3))


  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
    return traj




  #######################################################
  # minimum distance from config (singular) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND index
  #######################################################
  def min_euclid_dist_one_to_many(self, config, o_configs):
    dists = np.sum((config-o_configs)**2,axis=1)**(1./2)
    min_ind = np.argmin(dists)
    return dists[min_ind], min_ind


  #######################################################
  # minimum distance from configs (plural) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND indices into config and o_configs
  #######################################################
  def min_euclid_dist_many_to_many(self, configs, o_configs):
    dists = []
    inds = []
    for o_config in o_configs:
      [dist, ind] = self.min_euclid_dist_one_to_many(o_config, configs)
      dists.append(dist)
      inds.append(ind)
    min_ind_in_inds = np.argmin(dists)
    return dists[min_ind_in_inds], inds[min_ind_in_inds], min_ind_in_inds


  
  #######################################################
  # close the fingers when you get to the grasp position
  #######################################################
  def close_fingers(self):
    self.taskmanip.CloseFingers()
    self.robot.WaitForController(0) #ensures the robot isn't moving anymore
    #self.robot.Grab(target) #attaches object to robot, so moving the robot will move the object now




if __name__ == '__main__':
  robo = RoboHandler()
  print "calling "
  robo.run_problem_birrt()
  time.sleep(30) #to keep the openrave window open
  
