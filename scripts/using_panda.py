import sys
import os
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path,'../build'))
import numpy as np
import libry as ry
import time
import cv2 as cv
from pyquaternion import Quaternion
import math
import matplotlib.pyplot as plt

def psi_to_quat(psi):
    return [math.cos(psi/2),0,0,math.sin(psi/2)]

def quat_to_psi(qd):
    return math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )
    
def get_z_angle_diff(frame1, frame2):
    q1 = Quaternion(frame1.getQuaternion())
    q = Quaternion(frame2.getQuaternion())

    qd = q.conjugate * q1

    # Calculate Euler angles from this difference quaternion
    # phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
    # theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )
    psi   = quat_to_psi(qd)
    
    return psi

def get_xy_position_diff(frame1,frame2):
    res = frame1.getPosition() - frame2.getPosition()
    return res[0], res[1]

def rad_to_deg(angle):
    return angle/(2*math.pi)*360

def start_direction(frame, code):
    qd = Quaternion(frame.getQuaternion())
    psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )
    
    R = np.array([[math.cos(psi), -math.sin(psi)],
                     [math.sin(psi), math.cos(psi)]])
    vel = 1
    dist = .22 
    offset = .1
    
    if code == 0:
        rel_start = np.array([-offset, -dist])
        direction = np.array([0, vel])
    elif code == 1:
        rel_start = np.array([0, -dist])
        direction = np.array([0, vel])
    elif code == 2:  
        rel_start = np.array([offset, -dist])
        direction = np.array([0, vel])
    elif code == 3:     
        rel_start = np.array([dist, -offset])
        direction = np.array([-vel, 0])
    elif code == 4:     
        rel_start = np.array([dist, 0])
        direction = np.array([-vel, 0])
    elif code == 5:     
        rel_start = np.array([dist, offset])
        direction = np.array([-vel, 0])
    elif code == 6:     
        rel_start = np.array([offset, dist])
        direction = np.array([0, -vel])
    elif code == 7:     
        rel_start = np.array([0, dist])
        direction = np.array([0, -vel])
    elif code == 8:     
        rel_start = np.array([-offset, dist])
        direction = np.array([0, -vel])
    elif code == 9:     
        rel_start = np.array([-dist, offset])
        direction = np.array([vel, 0])
    elif code == 10:   
        rel_start = np.array([-dist, 0])
        direction = np.array([vel, 0])
    elif code == 11: 
        rel_start = np.array([-dist, -offset])
        direction = np.array([vel, 0])
    
    pos = frame.getPosition()[:2] + R.dot(rel_start)
    direction = R.dot(direction)
    return list(pos) + [frame.getPosition()[2]] ,  list(direction) + [0.]
    

world_configuration = "scenarios/pushSimWorldComplete.g"

C = ry.Config()
C.addFile(world_configuration)
# C.getFrame("grapper_base").setContact(1)
C.addFrame("ball_marker")
C.getFrame("ball_marker").setColor([1.,0,0])
C.getFrame("ball_marker").setShape(ry.ST.sphere, [.03])
# C.getFrame("ball_marker").setPosition([0., .2, 1.])
C.getFrame("ball_marker").setPosition([0., .2, .7])

# C.getFrame("ball_marker").setContact(1)

box = C.getFrame("box")
box_start_state = box.getPosition()
box_t = C.getFrame("box_t")
ball = C.getFrame("ball")

S = C.simulation(ry.SimulatorEngine.bullet, True)


def solve_place_ball(komo, position, collision_box = True):

    # collsion avoidance
    if collision_box:
        komo.addObjective([0,1.], ry.FS.distance, ["box", "ball"], ry.OT.ineq, .5e2, [-.05])
        komo.addObjective([0,1.], ry.FS.distance, ["box", "grapper_hand"], ry.OT.ineq, .5e2, [-.05])

    komo.addObjective([0,1.], ry.FS.distance, ["table", "ball"], ry.OT.ineq, .5e2, [-.05])
    komo.addObjective([0,1.], ry.FS.distance, ["table", "grapper_hand"], ry.OT.ineq, .5e2, [-.05])

    # vertical alignement
    komo.addObjective([1.], ry.FS.scalarProductXZ, [ "grapper_hand", "box" ], ry.OT.eq, [.1e2], [1.])

    # position
    komo.addObjective([1.], ry.FS.positionRel, ["ball", "world"], ry.OT.eq, 1e2, position)

    # smoothness
    komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e2], order=1)

    komo.optimize()
    
    return

def executeSplineNeutral(path, t):
    S.setMoveto(path, t)
    while S.getTimeToMove() > 0.:
        time.sleep(tau)
        S.step([], tau, ry.ControlMode.spline)
        C.setJointState(S.get_q()) 


tau = .01
waypoints = 10
horizon_seconds = 2.

########################################
final_move = 1

start, direction = start_direction(box,final_move)

C.getFrame("ball_marker").setPosition(start)

komo = C.komo_path(1., waypoints, horizon_seconds, False)
solve_place_ball(komo, start)
komo.getReport()
komo.view(False, "motion")
# komo.view_play(True, 0.3)

# send the motion with spline

executeSplineNeutral(komo.getPath_qOrg(), 1)


#grab manipulator sensor readings from the simulation
q = S.get_q()

C.setJointState(q); # set your robot model to match the real q

########################################
position = np.array(start)+.1*np.array(direction)
C.getFrame("ball_marker").setPosition(position)

komo = C.komo_path(1., waypoints, horizon_seconds, False)
solve_place_ball(komo, np.array(start)+.1*np.array(direction), collision_box=False)
komo.getReport()
komo.view(False, "motion")
komo.view_play(True, 0.3)

# send the motion with spline

executeSplineNeutral(komo.getPath_qOrg(), 1)


#grab manipulator sensor readings from the simulation
q = S.get_q()

C.setJointState(q); # set your robot model to match the real q

########################################
final_move = 9

start, direction = start_direction(box,final_move)
C.getFrame("ball_marker").setPosition(start)


komo = C.komo_path(1., waypoints, horizon_seconds, False)
solve_place_ball(komo, start)
komo.getReport()
komo.view(False, "motion")
komo.view_play(True, 0.3)

# send the motion with spline

executeSplineNeutral(komo.getPath_qOrg(), 1)


#grab manipulator sensor readings from the simulation
q = S.get_q()

C.setJointState(q); # set your robot model to match the real q