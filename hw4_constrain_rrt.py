#!/usr/bin/env python

PACKAGE_NAME = 'hw4'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy
from random import choice
import collections
import Queue

from openravepy import IkParameterizationType, IkParameterization, RaveCreateKinBody, raveLogInfo, raveLogWarn, IkFilterOptions

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


#constant for max distance to move any joint in a discrete step
MAX_MOVE_AMOUNT = 0.1

#Constant for the maximum +/- distance in z
ERROR = .01 

#Constant for probability of choosing a random target
PROB_RAND_TARGET = .9

#Constant for distance of current state near goal to determine the termination of the algorithm
DIST_THRESH = 0.8

class RoboHandler:
  def __init__(self):
    self.openrave_init()
    self.problem_init()

    #self.run_problem_constrain_birrt()




  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW4 Viewer')
    self.env.Load('models/%s_constrain_rrt.env.xml' %PACKAGE_NAME)
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

    # create a grasping module
    self.gmodel = openravepy.databases.grasping.GraspingModel(self.robot, self.target_kinbody)
    
    # load grasps
    if not self.gmodel.load():
      self.gmodel.autogenerate()

    self.grasps = self.gmodel.grasps
    self.graspindices = self.gmodel.graspindices

    # load ikmodel
    self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
    if not self.ikmodel.load():
      self.ikmodel.autogenerate()

    # create taskmanip
    self.taskmanip = openravepy.interfaces.TaskManipulation(self.robot)
  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )


  #######################################################
  # use a Constrained Bi-Directional RRT to solve
  #######################################################
  def run_problem_constrain_birrt(self):
    self.robot.GetController().Reset()
    #startconfig = np.array([ 4.54538305,  1.05544618,  0., -0.50389025, -3.14159265,  0.55155592, -2.97458672])
    startconfig = np.array([ 2.37599388,-0.32562851, 0.,  1.61876989,-3.14159265, 1.29314139, -0.80519756])

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    #goals = self.get_goal_dofs(10,3)
    goals = np.array([[ 2.3056527 , -0.6846652 ,  0.9       ,  1.88331188, -2.00747441,
         1.51061724,  1.39244389],
       [ 2.17083819, -0.78792566,  0.8       ,  1.88331188, -1.95347928,
         1.29737952,  1.48609958],
       [ 2.04534288, -0.87443137,  0.7       ,  1.88331188, -1.91940061,
         1.0969241 ,  1.56834924],
       [ 1.92513441, -0.94671347,  0.6       ,  1.88331188, -1.90216145,
         0.9064694 ,  1.64815832],
       [ 1.80822781, -1.00617436,  0.5       ,  1.88331188, -1.90383359,
         0.72454447,  1.7341887 ],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188, -1.93390285,
         0.55054165,  1.8400273 ],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188, -2.0210129 ,
         0.38546846,  1.99711856],
       [ 5.36700926,  0.74723102, -2.3       ,  1.88331188, -1.9731337 ,
         1.38427661,  1.44916799],
       [ 5.23832771,  0.84028589, -2.4       ,  1.88331188, -1.93147499,
         1.17896707,  1.53489493],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846, -1.14447409],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -3.8739155 ,
         0.60044153, -2.23175185],
       [ 2.16880663, -0.80864954,  0.8       ,  2.02311018, -2.14530038,
         1.37567029,  1.48036157],
       [ 2.03768854, -0.90096949,  0.7       ,  2.02311018, -2.14142654,
         1.18510744,  1.60819768],
       [ 1.91338002, -0.97735382,  0.6       ,  2.02311018, -2.16321053,
         1.0085189 ,  1.73830671],
       [ 1.7932191 , -1.03976983,  0.5       ,  2.02311018, -2.21449075,
         0.84548711,  1.88281559],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018, -2.30553762,
         0.69762048,  2.0569663 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018, -2.45498756,
         0.56962218,  2.28270213],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -0.73232284,
        -0.60044153,  0.9098408 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018,  0.68660509,
        -0.56962218, -0.85889052],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018,  0.83605503,
        -0.69762048, -1.08462636],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -3.86536478,
         0.61047535, -2.2378682 ],
       [ 2.17524582, -0.8034851 ,  0.8       ,  2.03233934, -2.15035065,
         1.38676951,  1.4845399 ],
       [ 2.04175346, -0.89711285,  0.7       ,  2.03233934, -2.14716777,
         1.19433304,  1.61476842],
       [ 1.91575237, -0.97434547,  0.6       ,  2.03233934, -2.17015174,
         1.01666957,  1.74689361],
       [ 1.79426556, -1.03734381,  0.5       ,  2.03233934, -2.22296312,
         0.85314889,  1.89334198],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934, -2.31568439,
         0.7053313 ,  2.06936754],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934, -2.46639541,
         0.57794147,  2.29645368],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -0.72377213,
        -0.61047535,  0.90372445],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934,  0.67519725,
        -0.57794147, -0.84513898],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934,  0.82590826,
        -0.7053313 , -1.07222512],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -3.67670848,
         0.50586635, -2.48069293],
       [ 2.2422838 , -0.73929453,  0.8       ,  1.98216027, -2.18885146,
         1.4282525 ,  1.40806164],
       [ 2.08923017, -0.84082763,  0.7       ,  1.98216027, -2.17107639,
         1.22303934,  1.55512312],
       [ 1.95014943, -0.92258716,  0.6       ,  1.98216027, -2.18362634,
         1.03654076,  1.69775671],
       [ 1.81876453, -0.9884749 ,  0.5       ,  1.98216027, -2.22804033,
         0.86594147,  1.85167077],
       [ 1.69208629, -1.04051597,  0.4       ,  1.98216027, -2.31387296,
         0.71194491,  2.03383432],
       [ 1.5684208 , -1.07995335,  0.3       ,  1.98216027, -2.45993672,
         0.5789909 ,  2.26761086],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027, -2.69263061,
         0.47742308,  2.58252966],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -0.53511583,
        -0.50586635,  0.66089972],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027,  0.44896204,
        -0.47742308, -0.55906299],
       [ 2.0687755 ,  0.2760535 , -0.5       , -0.7156698 , -3.16570512,
         1.53420622,  0.32545303],
       [ 5.18898933, -0.44935039, -0.3       ,  1.17672238, -3.34753525,
        -1.25173407, -2.67065332],
       [ 5.18898933, -0.44935039, -0.3       ,  1.17672238, -0.20594259,
         1.25173407,  0.47093934],
       [ 4.26617914, -0.33212585, -0.3       ,  1.17672238,  0.64725314,
         1.39568075, -0.16769878],
       [ 5.03751866, -0.25019943,  2.6       , -0.7156698 , -2.97097236,
         1.54134808,  0.22815683],
       [ 5.38829682, -0.30193089,  2.7       , -0.7156698 , -3.37919587,
         1.54818631,  0.43162252],
       [ 4.26617914, -0.33212585, -0.3       ,  1.17672238, -2.49433951,
        -1.39568075,  2.97389387],
       [ 2.0687755 ,  0.2760535 , -0.5       , -0.7156698 , -0.02411247,
        -1.53420622, -2.81613962],
       [ 5.38829682, -0.30193089,  2.7       , -0.7156698 , -0.23760322,
        -1.54818631, -2.70997013],
       [ 5.03751866, -0.25019943,  2.6       , -0.7156698 ,  0.17062029,
        -1.54134808, -2.91343582]])


    with self.env:
      self.robot.SetActiveDOFValues(startconfig)

    # get the trajectory!
    traj = self.constrain_birrt_to_goal(goals)

    with self.env:
      self.robot.SetActiveDOFValues(startconfig)

    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()


  #TODO
  #######################################################
  # Constrained Bi-Directional RRT
  # Keep the z value of the end effector where it is!
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def constrain_birrt_to_goal(self, goals):
    print 'Starting CBiRRT Algorithm'

    #The z value that the arm must remain within +/- 1cm
    z_val_orig = self.manip.GetTransform()[2,3]

    #goals = np.array(goals) #Bring the goals into an np array
    
    q_initial = self.robot.GetActiveDOFValues() # Initial state is a np array
    print "INITIAL POSITION", q_initial
    q_initial_tuple = self.convert_for_dict(q_initial) # This is the tuple for initial state
    q_nearest = q_initial # This is the initialization for nearest point in np array format
    thresh = DIST_THRESH # The threshold to determine if the goal is reached or not
    
    #Initialize the tree for the start node
    start_tree = [q_initial]

    #start_tree.append(q_initial_tuple)
    #Initialize the parent dictionary for the start node
    start_parent = {}
    start_parent[q_initial_tuple] = None # Dictionary to store the start_parents


    #trees = np.array([]) #np array to store goal trees
    #parents = np.array() #np array to store goal parent dictionaries

    #Now take all the goals and make a tree to add to the array of goals and make a dictionary of child (key) parents (value) to add to the array of parents
    goal_tree=[]
    goal_parent={}
    for goal in goals:    
      goal_parent[self.convert_for_dict(goal)] = None
      goal_tree.append(goal)
    #for keys in parent:
    #  print 'The first key value pairs are', keys, parent[keys]

    #Find the lower and upper limits of the joint angles
    lower, upper = self.robot.GetActiveDOFLimits() # Get the joint limits
    


    q_target, q_nearest, min_dist = self.rrt_choose_target_from_start(start_tree, goal_tree, lower, upper)
    q_target2, q_nearest2, min_dist2 = self.rrt_choose_target_from_start(start_tree, goal_tree, lower, upper)
    print 'Q Target is', q_target

    print '==============='

    print 'Q Target2 is', q_target2
    new_config = self.project_z_val_manip(q_target2, z_val_orig, lower, upper)
    print 'New config is', new_config
    traj = self.points_to_traj(np.array([[ 4.26617914, -0.33212585, -0.3       ,  1.17672238, -2.49433951,
        -1.39568075,  2.97389387],
       [ 2.0687755 ,  0.2760535 , -0.5       , -0.7156698 , -0.02411247,
        -1.53420622, -2.81613962],
       [ 5.38829682, -0.30193089,  2.7       , -0.7156698 , -0.23760322,
        -1.54818631, -2.70997013],
       [ 5.03751866, -0.25019943,  2.6       , -0.7156698 ,  0.17062029,
        -1.54134808, -2.91343582]]))
    
    print 'transform', self.manip.GetEndEffectorTransform()
    print 'z_val_orig', z_val_orig
    return traj


  #TODO
  #######################################################
  # projects onto the constraint of that z value
  #######################################################
  def project_z_val_manip(self, c, z_val, lower, upper):
    
    print 'passed c is' , c

    #Try to set the robot to the desired configuration
    with self.env:
      self.robot.SetActiveDOFValues(c) # Sets the robot's DOFs to the passed configuration to test
      reach_limit = self.robot.CheckSelfCollision() or self.env.CheckCollision(self.robot) or self.limitcheck(c, lower, upper) #This needs to be under the self.env block!
    print 'Reach limit is', reach_limit
    #If collisions or joint limits are reached, exit and return none
    if reach_limit:
      return None # Terminate the function of a collision occurs.

    #Get the transform of this configuration
    transform = self.manip.GetEndEffectorTransform()
    print 'The transform is', transform
    z_transform = transform[2,3]
    print 'The current z value is', z_transform

    #Now figure out how to change z value of transform
    #If the z_value is above the positive error, project down to the z+error plane
    if z_transform > z_val+ERROR:
       transform[2,3] = z_val + ERROR
       print 'projected to positive error plane'
    #If the z_value is below the negative error, project back up to the z-error plane
    elif z_transform < z_val-ERROR:
       transform[2,3] = z_val - ERROR
       print 'projected to negative error plane'
    #Otherwise, project back to the origin z_val
    else:
       transform[2,3] = z_val
       print 'projected to center'
    
    print 'Transform is now', transform
    new_config = self.manip.FindIKSolution(transform, IkFilterOptions.CheckEnvCollisions)
    #[0:2,3]
    
    return new_config

  #######################################################
  # The choose target from start function either returns a random 
  # configuration or the closest randomly selected configuration 
  # from the goal tree. This configuration will be the target to which the 
  # tree tries to expand to. The function also returns the nearest configuration
  # on the start tree and the minimum distance by taking advantage of rrt_nearest
  #######################################################
  def rrt_choose_target_from_start(self, start_tree, goal_tree, lower, upper):
    
    if (np.random.random_sample()<PROB_RAND_TARGET):
      q_target = np.array(lower+np.random.rand(len(lower))*(upper-lower)) #Choose a random configuration
      print 'RANDOM target from start to goal chosen'

    #Otherwise choose the closest of a random selection from each of the goal trees
    else:
      q_target = choice(goal_tree)
      print 'specific target from start to goal chosen', q_target

    min_dist, q_nearest, nearest_start_tree_index = self.rrt_nearest(start_tree, q_target)
    return q_target, q_nearest, min_dist


  #######################################################
  # The choose target from goal function either returns a random 
  # configuration or the closest randomly selected configuration 
  # from the start tree. This configuration will be the target to which the 
  # tree tries to expand to. 
  #######################################################
  def rrt_choose_target_from_goal(self, start_tree, goal_tree, lower, upper):
    
    if (np.random.random_sample()< PROB_RAND_TARGET):
      q_target = np.array(lower+np.random.rand(len(lower))*(upper-lower)) #Choose a random configuration
      #print 'random target from goal to start chosen'

    else:
      q_target=choice(start_tree)
      #print 'specific target from goal to start chosen', q_target

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
  def rrt_extend(self, q_nearest, q_target, tree, parent, lower, upper, z_val):

    direction_vector = q_target - q_nearest # Obtain the direction of the direct line from initial to goal state
    dist, index = self.min_euclid_dist_one_to_many(q_nearest, [q_target]) # The distance between the initial and final state
    #steps = int(dist / MAX_MOVE_AMOUNT) # Determine how many steps it will take to reach the final state

    #Find the projection of q_target
    q_target_proj = project_z_val_manip(self, q_target, z_val, lower, upper)
    
    #Find the projection of q_nearest
    q_nearest_proj = project_z_val_manip(q_nearest, z_val, lower, upper)
    #Try to extend all the way to the target, terminate at step if not possible
    
    dist, index = self.min_euclid_dist_one_to_many(q_nearest, [q_target_proj]) # The distance between the initial and final state
    #for count in range(steps):
(direction_vector/dist)*MAX_MOVE_AMOUNT
    while(dist > DIST_THRESH):
      direction_vector = q_target - q_add
      q_parent = q_nearest_proj + ((direction_vector)/steps)*(count) # Calculate the parent to be each previous node
      q_add = q_nearest + ((direction_vector)/steps)*(count+1) # The next node is obtained by 'walking' on this direct line
      with self.env:
        self.robot.SetActiveDOFValues(q_add) # This is done for demo purposes. The robot will assume a configuration as it tests.
        reach_limit = self.robot.CheckSelfCollision() or self.env.CheckCollision(self.robot) or self.limitcheck(q_add, lower, upper) #This needs to be under the self.env block!

      if reach_limit:
        return None # Terminate the function of a collision occurs. 
      #print "TREE, Q_ADD", tree, q_add

      isPresent = False
      for node in tree:
        if self.convert_for_dict(node) == self.convert_for_dict(q_add):
          isPresent = True
      if not isPresent:
        #print "================ADDED AN ELEMENT****************************************"
        tree.append(q_add) # Add the first element to the tree
        parent[self.convert_for_dict(q_add)] = q_parent # Update the parent dictionary	
    
    return


  #######################################################
  # This function returns the trajectory that will move the robot
  #######################################################
  def backtrace(self, start_parent, goal_parent, config1, config2):

     print "CONFIG 1",config1
     print "CONFIG 2",config2
     path1 = []
     path1.append(config1)
     path2 = []
     path2.append(config2)

     print "START PARENT=========================================", start_parent
     print "GOAL_PARENT---------------------------------------", goal_parent

     while not path1[-1] is None:
       print "BUILDING PATH 1", len(path1)
       node = start_parent[self.convert_for_dict(path1[-1])]
       path1.append(node)

     while not path2[-1] is None:
       print "BUILDING PATH 2", len(path2)
       node = goal_parent[self.convert_for_dict(path2[-1])]
       path2.append(node)

     path1.remove(None)
     path2.remove(None)
     
     path1.reverse()
     print "PATH 1", path1
     print "PATH 2", path2
     path = path1+path2
     #print path
     traj = self.points_to_traj(path)
     return traj

  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    #return tuple(np.int_(item*100))
    return tuple(item)

  def convert_from_dictkey(self, item):
    #return np.array(item)/100.
    return np.array(item)

  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
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
  robo.run_problem_constrain_birrt()
  time.sleep(5) #to keep the openrave window open
  
