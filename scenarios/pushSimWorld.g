world {}
table {X:<t(0 0 .6)>, shape: ssBox, size: [10. 10. .1 .05], friction: .9}

# Boxes 
box_t {X:<[0., 0., .7, 1., 0., 0., 0.]>, shape: ssBox, size: [.15 .15 .1 .01], color [1 1 0 0.7]}
box_t_marker (box_t) {X:<[0., -0.035, .71, 1., 0., 0., 0.]>, shape: ssBox, size: [0.01 0.075 0.1 0.01], color [1 .5 0 0.7]}

# box {X:<t(0.4 0. .7)>, shape: cylinder, size: [.05 .1 .1 .075], mass: 0.1}
box {X:<t(0.4 0. .7)>, shape: ssBox, size: [.15 .15 .1 .01], mass: 0.1}
# box {X:<t(0.4 0. .7)>, shape: ssBox, size: [.24 .15 .1 .01], mass: 0.1}

box_marker (box) {X:<t(0.4 -0.035 .71)>, shape: ssBox, size: [0.01 0.075 0.1 0.01], color: [1 0 0 .5]}

# box_t {X:<[0., 0., .7, 1., 0., 0., 0.]>, shape: ssBox, size: [.3 .3 .1 .01], color [1 1 0 0.7], mass: 0.1}
# box {X:<t(1.3 0. .7)>, shape: ssBox, size: [.3 .3 .1 .01], mass: 0.1}

# box_t_marker (box_t) {X:<[0., -0.07, .71, 1., 0., 0., 0.]>, shape: ssBox, size: [0.01 0.15 0.1 0.01], color [1 .5 0 0.7]}
# box_marker (box) {X:<t(1.3 -0.07 .71)>, shape: ssBox, size: [0.01 0.15 0.1 0.01], color: [1 0 0 .5]}

ball (world) {joint: trans3, Q:<t(.5 .1 .7)> shape: sphere, size: [.022]}