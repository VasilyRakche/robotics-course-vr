import sys
import os
from unittest import result
from cv2 import rectangle 
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path,'../build'))
import numpy as np
import libry as ry
import matplotlib.pyplot as plt
import time

# configuration of different evaluation objects
EXEC_NO_VERBOSE = False
EXEC_COMPARISON = False

box_names = ["boxc", "boxcl", "boxr"]
box_labels = ["Cube", "Cylinder","Cuboid"]
box_name = "boxcl"

from pyquaternion import Quaternion
import math

from lib.agent import Agent
from lib.helper import plot

from collections import deque
import lib.logger
import torch


def solve_place_ball(komo, position, q_conf_ref_pose, collision_box = True, force_orientation = True):

    # collsion avoidance
    if collision_box:
        komo.addObjective([0,1.], ry.FS.distance, [box_name, "ball"], ry.OT.ineq, .5e2, [-.05])
        komo.addObjective([0,1.], ry.FS.distance, [box_name, "grapper_hand"], ry.OT.ineq, .5e2, [-.05])

    komo.addObjective([0,1.], ry.FS.distance, ["table", "ball"], ry.OT.ineq, .5e2, [-.05])
    komo.addObjective([0,1.], ry.FS.distance, ["table", "grapper_hand"], ry.OT.ineq, .5e2, [-.05])

    # vertical alignement
    if force_orientation:
        komo.addObjective([1.], ry.FS.scalarProductXZ, [ "grapper_hand", box_name ], ry.OT.eq, [.1e2], [1.])


    # position
    komo.addObjective([1.], ry.FS.positionRel, ["ball", "world"], ry.OT.eq, 1e3, position)

    # smoothness
    komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e2], order=1)

    if q_conf_ref_pose is not None:
        komo.addObjective([0.,1.], ry.FS.qItself, [], ry.OT.sos, [.1e1], q_conf_ref_pose, order=0)

    komo.optimize()

    # komo.view(True, "motion")
    # komo.view_play(True, 0.3)

    return komo.getConstraintViolations()



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

    vel = .6
    dist = .116
    distx = .12 + 0.022 + 0.019
    offset = .065   
    
    if code == 0:     
        rel_start = np.array([offset, dist])
        direction = np.array([0, -vel])
    elif code == 1:     
        rel_start = np.array([0, dist])
        direction = np.array([0, -vel])
    elif code == 2:     
        rel_start = np.array([-offset, dist])
        direction = np.array([0, -vel])
    elif code == 3:     
        rel_start = np.array([dist if box_name == "boxc" else distx, -offset])
        direction = np.array([-vel, 0])
    elif code == 4:     
        rel_start = np.array([dist if box_name == "boxc" else distx, 0])
        direction = np.array([-vel, 0])
    elif code == 5:     
        rel_start = np.array([dist if box_name == "boxc" else distx, offset])
        direction = np.array([-vel, 0])
    if code == 6:
        rel_start = np.array([-offset, -dist])
        direction = np.array([0, vel])
    elif code == 7:
        rel_start = np.array([0, -dist])
        direction = np.array([0, vel])
    elif code == 8:  
        rel_start = np.array([offset, -dist])
        direction = np.array([0, vel])
    elif code == 9:     
        rel_start = np.array([-dist if box_name == "boxc" else -distx, offset])
        direction = np.array([vel, 0])
    elif code == 10:   
        rel_start = np.array([-dist if box_name == "boxc" else -distx, 0])
        direction = np.array([vel, 0])
    elif code == 11: 
        rel_start = np.array([-dist if box_name == "boxc" else -distx, -offset])
        direction = np.array([vel, 0])


    # overwrite the rel_start and direction in case of cylinder
    if box_name == "boxcl":
        angle = code*2*math.pi/12
        if code == 2 or code == 5 or code ==8 or code == 11:
            direction *= 1.5
            angle -= 2*2*math.pi/36

        if code == 1 or code == 4 or code ==7 or code == 10:
            direction *= 1.1
            angle -= 2*math.pi/36

        x = dist*math.sin(angle)
        y = dist*math.cos(angle)
        rel_start = np.array([x, y])

    pos = frame.getPosition()[:2] + R.dot(rel_start)

    direction = R.dot(direction)
    return list(pos) + [frame.getPosition()[2]] ,  list(direction) + [0.]


class Game:

    def __init__(self, num_states = 3, world_configuration = os.path.join(file_path,"../scenarios/pushSimWorldComplete.g")):
        self.C = ry.Config()

        self.C.addFrame("ball_start_marker")
        self.ball_start_marker = self.C.getFrame("ball_start_marker")
        self.ball_start_marker.setColor([1.,0,0])
        self.ball_start_marker.setShape(ry.ST.sphere, [.022])

        self.C.addFrame("ball_end_marker")
        self.ball_end_marker = self.C.getFrame("ball_end_marker")
        self.ball_end_marker.setColor([.8,0,0])
        self.ball_end_marker.setShape(ry.ST.sphere, [.022])

        self.D = self.C.view()
        self.C.addFile(world_configuration)
        if EXEC_NO_VERBOSE:
            self.S_verbose = self.C.simulation(ry.SimulatorEngine.bullet, False)
        else: 
            self.S_verbose = self.C.simulation(ry.SimulatorEngine.bullet, True)

        self.tau = 0.02
        self.box = self.C.getFrame(box_name)
        for box in box_names:
            if not box==box_name:
                frame = self.C.getFrame(box)
                frame.setPosition([0.,0.,0.])

        self.box_t = self.C.getFrame("box_t")
        self.ball = self.C.getFrame("ball")
        self.r_max = .5
        self.disc_r = .3
        self.disc_angle = .5 # TODO: define meaningful angle
        self.state = num_states*[0]
        self.start_r = 0
        self.start_angle = 0
        self.score = 0
        self.r_target_hits = 0
        self.fail_to_move_the_box  = 0
        self.fail_counter_constraint_soft = 0
        self.panda_home_q = self.S_verbose.get_q()
        self.exec_fails = 0
        self.exec_sucesses = 0

        # random initialization 
        self.reset()
        
    def calculate_state(self):
        x_diff_box, y_diff_box = get_xy_position_diff(self.box_t, self.box)
        x_diff_ball, y_diff_ball = get_xy_position_diff(self.box, self.ball)
        z_angle_diff = get_z_angle_diff(self.box_t, self.box)
        
        self.state = [x_diff_box, y_diff_box, z_angle_diff, x_diff_ball, y_diff_ball]

    
    def executeSplineNeutral(self, path, t):
        self.S_verbose.setMoveto(path, t)
        while self.S_verbose.getTimeToMove() > 0.:
            if not EXEC_NO_VERBOSE:
                time.sleep(self.tau)
            self.S_verbose.step([], self.tau, ry.ControlMode.spline)
            self.C.setJointState(self.S_verbose.get_q()) 

    def box_position_feasable(self):
        q1 = Quaternion(self.box_t.getQuaternion())
        q = Quaternion( self.box.getQuaternion())
        qd = q.conjugate * q1
        phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
        theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )

        if abs(phi)>.3 or abs(theta)>.3:
            print("infiseable")
            return False

        return True

    def return_panda_home(self):
        waypoints = 15
        horizon_seconds = 3.

        komo = self.C.komo_path(1., waypoints, horizon_seconds, False)

        if self.fail_to_move_the_box <2:
            constraints = solve_place_ball(komo, self.box.getPosition() + np.array([0, 0, .15]), self.panda_home_q)
        elif self.fail_to_move_the_box <5:
            constraints = solve_place_ball(komo, self.box.getPosition() + np.array([0, 0, .5]), self.panda_home_q, force_orientation=False)
        else:
            return False
            raise ValueError("Could not resolve the situation")

        if constraints > 0.5:
            print("problem not solved: ", constraints)

        # execute
        self.executeSplineNeutral(komo.getPath_qOrg(), .2)

        # grab manipulator sensor readings from the simulation
        q = self.S_verbose.get_q()
        self.C.setJointState(q); # set your robot model to match the real q

        return True

    def calculate_execute_komo(self, target, collision_box = True):
        waypoints = 15
        horizon_seconds = 3.

        komo = self.C.komo_path(1., waypoints, horizon_seconds, False)

        constraints = solve_place_ball(komo, target, self.S_verbose.get_q(), collision_box = collision_box)

        if constraints > 0.15:
            self.fail_counter_constraint_soft += 1
            if self.fail_counter_constraint_soft > 4:
                if not self.return_panda_home():
                    self.reset()
                    print("Reseting state due to unresolved error")
                    self.fail_to_move_the_box  = 0
                    self.fail_counter_constraint_soft = 0
                    return False
        else:
            self.fail_counter_constraint_soft = 0

        if constraints > 0.3:
            self.fail_to_move_the_box+=1
            if not self.return_panda_home():
                self.reset()
                print("Reseting state due to unresolved error")
                self.fail_to_move_the_box  = 0
                self.fail_counter_constraint_soft = 0
            return False

        # execute
        self.executeSplineNeutral(komo.getPath_qOrg(), .3)

        # grab manipulator sensor readings from the simulation
        q = self.S_verbose.get_q()
        self.C.setJointState(q); # set your robot model to match the real q

        return True

    def step(self, action, show_simulation = False):
        reward = 0
        game_over = False
        sucess = False

        if not self.box_position_feasable():
            game_over = True
            self.exec_fails+=1
            return reward, game_over, self.score

        start, direction = start_direction(self.box,np.nonzero(action)[0][0])
        # self.ball.setPosition(start)

        self.ball_start_marker.setPosition(start)
        self.ball_end_marker.setPosition(np.array(start)+.1*np.array(direction))

        # go to start position
        motion_sucess = self.calculate_execute_komo(start)
        if not motion_sucess:
            return reward, game_over, self.score

        # go to end position
        motion_sucess = self.calculate_execute_komo(np.array(start)+.1*np.array(direction), collision_box = False)
        if not motion_sucess:
            return reward, game_over, self.score

        self.calculate_state()
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        dr = int(r / self.disc_r)
        dangle = int(self.state[2] / self.disc_angle)
        
        if abs(self.prev_r - r) < 0.00005:
            self.fail_to_move_the_box +=1
            self.return_panda_home()
            print("No movement of the box: ", self.prev_r - r)
            return reward, game_over, self.score

        if r >= self.r_max: 
            reward = -3.
            print("Distance too large: ", r)
            game_over = True        
            
        elif (dr - self.prev_dr) > 2 :
            game_over = True
            #TODO: dangle adding strategy
            reward = -1
            print("Negative reward for position distance increase: ", reward)
            self.prev_dr = dr
            
        elif r < 0.1:
            self.r_target_hits += 1
            # Define desired more precise positioning 
            if r < 0.01 or self.r_target_hits > 2:
                print("Scored position: ", r)
                sucess = True
                game_over = True
                # Introduce better rewards for more precise positioning 
                reward = 10 - abs(dangle) - r*5
            
        # penalize angle difference increase 
        if (abs(dangle) - abs(self.prev_dangle)) > 2 :
            game_over = True
            reward = -1
            print("Negative reward for angle increase: ", reward)
            self.prev_dangle = dangle
            
        self.score += reward

        # Bias for movements towards the goal
        # update discrete states:
        if abs(dangle) < abs(self.prev_dangle):
            reward+=.02
            self.prev_dangle = dangle

        if dr <  self.prev_dr:
            reward+=.02
            self.prev_dr = dr

        # reset variables
        if game_over:
            self.r_target_hits = 0

        self.prev_r = r

        self.fail_to_move_the_box  = 0 # reset fail counter after sucessful step

        if game_over and sucess:
            self.exec_sucesses+=1
        elif game_over:
            self.exec_fails+=1

        return reward, game_over, self.score
        
    def get_state(self):
        return self.state

    def get_exec_stats(self):
        return (self.exec_sucesses, self.exec_fails)

    def reset(self):
        #define the new state of the box to be somewhere around the target:
        reset_max_dist = .4
        state_validated = False
        while (not state_validated):
            init_radius = 0.075+reset_max_dist*np.random.rand()
            alpha = 2*math.pi*np.random.rand()
            x = init_radius*math.cos(alpha)
            y = init_radius*math.sin(alpha)

            new_state = np.array(self.box_t.getPosition())+np.array([x, y, 0])

            self.box.setQuaternion(psi_to_quat(2*math.pi*np.random.rand()))
            self.box.setPosition(new_state)

            # make sure the box doesnt colide with the link
            y1, J1 = self.C.evalFeature(ry.FS.distance, ["panda_coll0", box_name])
            y2, J2 = self.C.evalFeature(ry.FS.distance, ["ball", box_name])

            if (y1 < -0.1) and (y2 < -0.1):
                state_validated = True
      
        self.S_verbose.setState(self.C.getFrameState())
        
        self.calculate_state()
        
        r = (self.state[0]**2 + self.state[1]**2)**.5
        
        self.prev_r = r
        self.prev_dr = int(r / self.disc_r)
        self.prev_dangle = int(self.state[2] / self.disc_angle)
        
        self.score = 0
                

        return 

class Worker:
    def __init__(self, 
                    max_steps = 10_000_000,
                    # agent params
                    gamma = 0.97, 
                    learning_rate = 1e-3, 
                    # memory 
                    memory_len = 1_000, #10_000 in google code
                    batch_size = 64,

                    # network
                    layers_sizes = [5, 256, 64, 12],
                    
                ):

        self.layers_sizes = layers_sizes
        self.act_dim = self.layers_sizes[-1]
        self.sim_verbose = True
        self.max_steps = max_steps

        # Learning Agent 
        self.agent = Agent(
            gamma = gamma, 
            learning_rate = learning_rate, 
            memory_len = memory_len, 
            layers_sizes = self.layers_sizes,
            batch_size = batch_size
            )

        # Environment
        self.game = Game(num_states = self.layers_sizes[0])

    def evaluate_model(self, model_filename = 'model.pth'):
        self.agent.trainer.model.restore_from_saved(model_filename)
        # Model should be switched to evaluation model and back when training is desired: mode.train()
        self.agent.trainer.model.eval() 

        episode_rewards = []
        run_steps = []
        episodes = 0
        last_game_itr = 0

        l_episode_return = deque([], maxlen=30)

        for itr in range(self.max_steps):
            state = self.game.get_state()
            act = [0]*self.act_dim

            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.agent.model(state0)
            move = torch.argmax(prediction).item()
            act[move] = 1

            # perform move
            reward, done, score = self.game.step(act, True)

            episode_rewards.append(reward)

            if done:
                self.game.reset()

                # Print basic game statistics
                print("~~~~~~~~~~ Game ~~~~~~~~~~")
                run_steps += [itr - last_game_itr]
                print('Running steps:           \t', run_steps)

                # save to indicate how many iterations for start to finish of the game:
                last_game_itr = itr
                
                # Reward statistics
                episode_return = np.sum(episode_rewards)
                episodes +=1
                episode_rewards = []
                print('Reward:                  \t', episode_return)
                
                l_episode_return.append(episode_return)

                # Save the model if the preformance increased
                episode_return_mean = math.ceil(np.mean(l_episode_return)*10)/10.

                print('Mean reward:             \t', episode_return_mean)

                
                sucess, fails = self.game.get_exec_stats()
                print("Sucess times {}\nFail times {}".format(sucess,fails))

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

                # only return if comparison is being performed
                if sucess+fails == 20 and EXEC_COMPARISON:
                    return (sucess, np.array(run_steps), np.array(l_episode_return))

            
if __name__ == "__main__":

    if EXEC_COMPARISON:
        storage = dict()
        for box in box_names:
            results = []
            total_steps = []
            total_rewards = []
            box_name = box
            for i in range(10):
                result, steps, rewards = (0,0,0)
                if box_name == "boxc":
                    setup = Worker()
                    result, steps, rewards = setup.evaluate_model(model_filename = 'model_cube_box.pth')
                elif box_name == "boxr":
                    setup = Worker()
                    result, steps, rewards = setup.evaluate_model(model_filename = 'model_rectangle_box.pth')
                elif box_name == "boxcl":
                    setup = Worker()
                    result, steps, rewards = setup.evaluate_model(model_filename = 'model_cylinder.pth')
                results += [result]
                total_steps += [steps]
                total_rewards += [rewards]
            storage[box_name] = (np.array(results), np.array(total_steps).flatten(), np.array(total_rewards).flatten())


        fig, ax = plt.subplots()
        fig.suptitle('Sucess rate', fontsize=14, fontweight='bold')
        ax.set_title('10 batches of 20 experiments')
        ax.set_ylabel('sucess')
        ax.boxplot([ storage[box_names[0]][0],storage[box_names[1]][0],storage[box_names[2]][0] ], labels=box_labels)

        fig1, ax1 = plt.subplots()
        fig1.suptitle('Execution steps', fontsize=14, fontweight='bold')
        ax1.set_title('10 batches of 20 experiments')
        ax1.set_ylabel('steps')
        ax1.boxplot([ storage[box_names[0]][1],storage[box_names[1]][1],storage[box_names[2]][1] ], labels=box_labels)

        fig2, ax2 = plt.subplots()
        fig2.suptitle('Rewards received', fontsize=14, fontweight='bold')
        ax2.set_title('10 batches of 20 experiments')
        ax2.set_ylabel('reward')
        ax2.boxplot([ storage[box_names[0]][2],storage[box_names[1]][2],storage[box_names[2]][2] ], labels=box_labels)

        hey = input("Press Enter to continue...")
    else:
        if box_name == "boxc":
            setup = Worker()
            result, steps, rewards = setup.evaluate_model(model_filename = 'model_cube_box.pth')
        elif box_name == "boxr":
            setup = Worker()
            result, steps, rewards = setup.evaluate_model(model_filename = 'model_rectangle_box.pth')
        elif box_name == "boxcl":
            setup = Worker()
            result, steps, rewards = setup.evaluate_model(model_filename = 'model_cylinder.pth')
