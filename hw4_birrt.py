#!/usr/bin/env python
# TESTING GIT MERGE
//Blah
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

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

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
    return None

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
  #time.sleep(10000) #to keep the openrave window open
  
