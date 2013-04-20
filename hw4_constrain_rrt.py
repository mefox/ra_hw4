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
MAX_MOVE_AMOUNT = 0.05

#Constant for the maximum +/- distance in z
ERROR = .01 

#Constant for probability of choosing a random target
PROB_RAND_TARGET = .60

#Constant for distance of current state near goal to determine the termination of the algorithm
DIST_THRESH1 = 0.8
DIST_THRESH2 = 0.8

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

    print "TRAVERSING THE PATH NOW"
    time.sleep(5) 
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
    thresh = DIST_THRESH1 # The threshold to determine if the goal is reached or not
    
    #Initialize the tree for the start node
    start_tree = [q_initial]

    #Initialize the parent dictionary for the start node
    start_parent = {}
    start_parent[q_initial_tuple] = None # Dictionary to store the start_parents

    #Now take all the goals and make a tree to add to the array of goals and make a dictionary of child (key) parents (value) to add to the array of parents
    goal_tree=[]
    goal_parent={}
    for goal in goals:    
      goal_parent[self.convert_for_dict(goal)] = None
      goal_tree.append(goal)

    #Find the lower and upper limits of the joint angles
    lower, upper = self.robot.GetActiveDOFLimits() # Get the joint limits
    

    tree_not_connected = True
    print 'Completed initialization, lets make some trees!'
    dist_between_trees, idx1, idx2 = self.min_euclid_dist_many_to_many(start_tree, goal_tree)
    while(dist_between_trees>thresh): # Keep checking if the tree has not already reached a nearest goal   
      #print "DISTANCE Initial: ",dist_between_trees
      #First grow start tree
      q_target, q_nearest, min_dist = self.rrt_choose_target_from_start(start_tree, goal_tree, lower, upper) 
      #print "FOR START TREE - Q-TARGET, Q-NEAREST",q_target, q_nearest
      self.rrt_extend(q_nearest, q_target, start_tree, start_parent, lower, upper, z_val_orig)
      #print 'returned 1'
      dist_between_trees, idx1, idx2 = self.min_euclid_dist_many_to_many(start_tree, goal_tree)
      print "DISTANCE after start_tree extend: ",dist_between_trees      
      if (dist_between_trees < thresh):
         break
      
      #Calculate the target for the goal tree
      q_target, q_nearest, min_dist = self.rrt_choose_target_from_goal(start_tree,goal_tree, lower, upper) 
      
      #Extend the goal tree along the manifold until collision or limit error
      self.rrt_extend(q_nearest, q_target, goal_tree, goal_parent, lower, upper, z_val_orig) 
      #print 'returned 2'
      #Recalculate distance between trees so loop will exit
      dist_between_trees, idx1, idx2 = self.min_euclid_dist_many_to_many(start_tree, goal_tree) 

      print "DISTANCE after goal tree extend: ", dist_between_trees

    #Set the traceback nodes as the nodes that are closest to one another on each tree
    config1 = start_tree[idx1]
    config2 = goal_tree[idx2]
       
    return self.backtrace(start_parent, goal_parent, config1, config2)



  #TODO
  #######################################################
  # projects onto the constraint of that z value
  #######################################################
  def project_z_val_manip(self, c, z_val, lower, upper):
    #Initialize variables and constants before running gradient descent 
    q_current = c #initialize q_current
    transform = self.manip.GetEndEffectorTransform() #initialize the transform
    z_current = transform[2,3] #Initialize z_current
    gradient = [1,1,1,1,1,1,1] #Initialize the gradient so that the while loop is entered

    GRADIENT_THRESH = .001 #Set the gradient threshold parameter to make the z transform good
    DELTA = .5 #Initialize the delta value

    count = 0
    #Perform gradient descent until the z value is +/- 1cm of the desired z plane
    while((not (z_current < z_val+ERROR and z_current > z_val-ERROR) or (np.linalg.norm(gradient) > GRADIENT_THRESH)) and count < 9):
      count = count + 1 #make sure we don't get stuck here
      #print 'count', count
      #Try to set the robot's DOFs to the current gradient descent configuration
      with self.env:
        self.robot.SetActiveDOFValues(q_current) 
        reach_limit = self.robot.CheckSelfCollision() or self.env.CheckCollision(self.robot) or self.limitcheck(c, lower, upper) #We do an error check, but the extend function will take care of seeing if the result is actually good. 

      #For debugging: Informs whether the limit is reached or not, but as above, the extend function has the final say because some
      #steps along the way to the projected target might still be good and worth adding to the tree
      #if reach_limit:
        #print 'we reached limit when projecting'
      #else:
        #print 'did not reach limit'

      #Get the transform
      transform = self.manip.GetEndEffectorTransform()
      
      #Pull the z value out of the transform
      z_current = transform[2,3]
      
      #Use the last row of the spatial Jacobian to run the gradient descent
      #Only the z axis is constrained, but x, y and rotation are allowed to be free
      #The last row is used because this maps to the rate of change of the end effector
      j_spatial = self.manip.CalculateJacobian()
      j_spatial_last = j_spatial[-1]
      
      
      #Calculate the gradient by multiplifying the difference between the current z and desired z
      #by the last row of the spatial jacobian (factor of 2 is because of derivative chain rule)
      gradient = 2*(z_current - z_val)*j_spatial_last
      
      #Find the new q_current by decrementing it by the gradient times a fixed delta
      q_current = q_current - gradient*DELTA

      #Some value print statements for debugging
      #print 'transform', transform
      #print 'z_current', z_current
      #print 'j_spatial_last',j_spatial_last
      #print 'q current', q_current
      #print 'gradient', gradient
      #print 'linalg', np.linalg.norm(gradient)
    return q_current

  #######################################################
  # The choose target from start function either returns a random 
  # configuration or a randomly selected configuration 
  # from the goal tree. This configuration will be the target to which the 
  # start tree tries to extend to. The function also returns the nearest configuration
  # on the start tree and the minimum distance by taking advantage of rrt_nearest
  #######################################################
  def rrt_choose_target_from_start(self, start_tree, goal_tree, lower, upper):
    
    if (np.random.random_sample()<PROB_RAND_TARGET):
      q_target = np.array(lower+np.random.rand(len(lower))*(upper-lower)) #Choose a random configuration
      #print 'RANDOM target from start to goal chosen'

    #Otherwise choose the closest of a random selection from each of the goal trees
    else:
      q_target = choice(goal_tree)
      #print 'specific target from start to goal chosen', q_target

    min_dist, q_nearest, nearest_start_tree_index = self.rrt_nearest(start_tree, q_target)
    return q_target, q_nearest, min_dist


  #######################################################
  # The choose target from goal function either returns a random 
  # configuration or a randomly selected configuration 
  # from the start tree. This configuration will be the target to which the 
  # goal tree tries to extend to from its own nearest node by taking advantage
  # of rrt_nearest
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

    #Obtain the direction of the direct line from initial to goal state
    direction_vector = q_target - q_nearest 

    #Find the projection of q_target
    q_target_proj = self.project_z_val_manip(q_target, z_val, lower, upper)
    #print 'q target projection is:', q_target_proj

    #Find the projection of q_nearest
    q_nearest_proj = self.project_z_val_manip(q_nearest, z_val, lower, upper)
    #Try to extend all the way to the target, terminate at step if not possible
    #print 'q nearest projection is:', q_nearest_proj
    
    ##if q_nearest_proj or q_target_proj is None:
    ##  return
    dist, index = self.min_euclid_dist_one_to_many(q_nearest, [q_target_proj]) # The distance between the initial and final state
    #for count in range(steps):
    
    #Initialize the parent
    q_parent = q_nearest
    steps = 0
    while(dist > DIST_THRESH2 and steps < 40):
      steps = steps+1 #make sure we don't get stuck here if distance does not converge
      #Jump towards the target from the parent
      q_jump = q_parent + (direction_vector/dist)*MAX_MOVE_AMOUNT
      
      #Project the result and store inside q_add
      q_add = self.project_z_val_manip(q_jump, z_val, lower, upper)

      #Check to see if q_add is none and quit if so
      if q_add is None:
        return
      #q_nearest + ((direction_vector)/steps)*(count+1) # The next node is obtained by 'walking' on this direct line
      
      #Make sure the new node is not in collision or at limits. If still good, add to the tree
      with self.env:
        self.robot.SetActiveDOFValues(q_add) # This is done for demo purposes. The robot will assume a configuration as it tests.
        reach_limit = self.robot.CheckSelfCollision() or self.env.CheckCollision(self.robot) or self.limitcheck(q_add, lower, upper)

      if reach_limit:
        return None # Terminate the function of a collision occurs. 
      
      isPresent = False
      for node in tree:
        if self.convert_for_dict(node) == self.convert_for_dict(q_add):
          isPresent = True
      if not isPresent:
        tree.append(q_add) # Add the first element to the tree
        parent[self.convert_for_dict(q_add)] = q_parent # Update the parent dictionary

	  #Recalculate the direction vector
      direction_vector = q_target - q_add
      
      #Recalculate the distance between the current node and the q_target projection
      dist, index = self.min_euclid_dist_one_to_many(q_add, [q_target_proj])

      #For the next node, set the parent to the current node
      q_parent = q_add
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

     #print "START PARENT=========================================", start_parent
     #print "GOAL_PARENT---------------------------------------", goal_parent

     while not path1[-1] is None:
       #print "BUILDING PATH 1", len(path1)
       node = start_parent[self.convert_for_dict(path1[-1])]
       path1.append(node)

     while not path2[-1] is None:
       #print "BUILDING PATH 2", len(path2)
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

  #######################################################
  # Trajectory building function
  #######################################################
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
  
