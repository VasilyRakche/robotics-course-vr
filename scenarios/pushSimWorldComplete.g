world {}
table {X:<t(0 0 .6)>, shape: ssBox, size: [10. 10. .1 .05], friction: .9}

# Boxes 
box_t {X:<[-0.2, 0, .7, 1., 0., 0., 0.]>, shape: ssBox, size: [.15 .15 .1 .01], color [1 1 0 0.7]}
box_t_marker (box_t) {X:<[0., -0.035, .71, 1., 0., 0., 0.]>, shape: ssBox, size: [0.01 0.075 0.1 0.01], color [1 .5 0 0.7]}

box {X:<t(0.4 0. .7)>, shape: ssBox, size: [.15 .15 .1 .01], mass: 0.1}
box_marker (box) {X:<t(0.4 -0.035 .71)>, shape: ssBox, size: [0.01 0.075 0.1 0.01], color: [1 0 0 .5]}

### pandas

# Prefix: "L_"
#Include: '../rai-robotModels/scenarios/panda_fixGripper.g'
Include '../rai-robotModels/panda_convexGripper/panda_convexGripper.g'

# Prefix!
        
Edit panda_link0 (table) { Q:<t(-.4 -.4 .1) d(90 0 0 1)> }

### tool

# GRAPPERS 
# grapper_base (gripper) {X:<[.1, -0.4, .67, 1., 0., 0., 0.]>, shape: ssBox, size: [.3 .025 .025], color [1 1 1]}
grapper_base (gripper) {Q:<t(-0.15 0. 0.1)>, shape: ssBox, size: [.4 .03 .03 .01], color [1 1 1]}

# y: -0.4(x_base)+0.1(hand_lenght)-0.025(base_width)
# grapper_hand (grapper_base) {X:<[.25, -0.325, .67, 0.707, 0., 0., -0.707]>, shape: ssBox, size: [.2 .05 .05 .01], color [1 1 0.5]}
grapper_hand (grapper_base) {Q:<t(-0.3 -0.09 0.)>, shape: ssBox, size: [.2 .03 .03 .01], color [1 1 0.5]}

### ball
# ball (grapper_hand) {Q:<t(.25 -0.275 0)> shape: sphere, size: [.03]}
ball (grapper_hand) {Q:<t(-0.1 0 0)> shape: sphere, size: [.022]}






