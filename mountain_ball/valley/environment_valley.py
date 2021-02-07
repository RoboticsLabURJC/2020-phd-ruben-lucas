"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

from time import sleep
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from sympy import *

class MountainBallEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    car_mass=4
    seconds_per_step=1

    prev_angle=0
    sum_acc=0
    granularity_level=1000

    DISCRETIZATION_POS=1
    DISCRETIZATION_SPEED=1

    # We have to create a "symbol" called x
    x = Symbol('x')
    #ramp
    #heigh_function=0.3*x +2
    #weird courves mountain
    #heigh_function=4*sin(0.3*x)**3+ 4.5
    #floor
    #heigh_function=0.00000000000000000001*x + 2
    #mountainCar environment
    heigh_function=granularity_level*sin(0.3/granularity_level*x-10.5)*4 + 4*granularity_level

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.world_function= lambdify(self.x, self.heigh_function)
        #Consider position is in metters
        self.min_position = -11*self.granularity_level
        self.max_position = 11*self.granularity_level

        #Consider that speed is metters per millisecond
        self.max_speed = 10000

        self.goal_velocity = goal_velocity

        self.force = 10/self.car_mass
        self.gravity = 9.8

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )


        self.goal_position = (10*self.granularity_level - self.observation_space.low[0])*self.DISCRETIZATION_POS
        self.goal_position = np.round(self.goal_position, 0)

        # Determine size of discretized state space INTERPOLATION
        self.num_states = (self.observation_space.high - self.observation_space.low)*\
                        np.array([self.DISCRETIZATION_POS, self.DISCRETIZATION_SPEED])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        print("num states")
        print(self.num_states)

        self.min_pos = (self.min_position - self.observation_space.low[0])*self.DISCRETIZATION_POS
        self.min_pos = np.round(self.min_pos, 0).astype(int)

        self.max_pos = (self.max_position - self.observation_space.low[0])*self.DISCRETIZATION_POS
        self.max_pos = np.round(self.max_pos, 0).astype(int)

        self.observation_space.low[0]=self.min_pos
        self.observation_space.high[0]=self.max_pos

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #TODO hacerla genérica en función de la pendiente en la posición del mundo creado
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        f_prime = lambdify(self.x, self.heigh_function.diff(self.x))

        position, velocity = self.state

        angle=math.atan(f_prime(position))
        print("angle in radians")
        print(angle)

        #accelaration per second
        self.acceleration=-self.gravity*math.sin(angle) + ((action - 1) * self.force)
        self.sum_acc=self.sum_acc+self.acceleration
        print("sum = " + str(self.sum_acc))
#        new_acceleration=-self.gravity*math.sin(angle)


#        sleep(0.5)
        #print("diff acceleration")
        #print(new_acceleration-self.acceleration)

#        if (velocity>0 and new_acceleration>0) or (velocity<0 and new_acceleration<0):
#            self.acceleration = new_acceleration
#            print("previo")
#        else:
#            print("actual")

        print("acceleration")
        print(self.acceleration)
        print("prev")
        velocity_prev=velocity
        position_prev= position
        print(position_prev)
        print(velocity_prev)
        print("action" + str(action))

        espacio_recorrido=(velocity*self.seconds_per_step)+((1/2)*self.acceleration*(self.seconds_per_step**2))
        position+=espacio_recorrido*math.cos(angle)

        velocity+=self.acceleration*self.seconds_per_step

        position = np.round(position, 0).astype(int)

        #velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        #position += velocity
        position = np.clip(position, self.min_pos, self.max_pos)

        print("post pos and vel preliminar")
        print(position)
        print(velocity)

        print("adjusted velocity")
        #TODO ñapa para cuando la velocidad inicial es 0
        t=0.1
        inner_velocity=velocity_prev
        for x in range(int(abs(position-position_prev))):
            print(x)
            print(inner_velocity)
            if(inner_velocity>1.0 or inner_velocity<-1.0):
                t=abs(1/inner_velocity)

            print(t)
            inner_velocity+=self.acceleration*t

            if(position>position_prev):
                inner_angle=math.atan(f_prime(position_prev+x))
            else:
                inner_angle=math.atan(f_prime(position_prev-x))

            #accelaration per second
            self.acceleration=-self.gravity*math.sin(inner_angle) + ((action - 1) * self.force)
            print("inner acceleration = " + str(self.acceleration))
            self.sum_acc=self.sum_acc+self.acceleration
            velocity=inner_velocity

        print("adjusted " + str(velocity))

        if (position == self.min_pos and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position
            #position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)

#        self.acceleration = new_acceleration
        print("------------------------------------")
        if((velocity_prev<0 and velocity>0) or (velocity_prev>0 and velocity<0)):
            print("total sum = " + str(self.sum_acc))
            self.sum_acc=0
        if(velocity_prev<0 and velocity>0):
            print("CHANGE FROM LEFT TO RIGHT")
        if(velocity_prev>0 and velocity<0):
            print("CHANGE FROM RIGHT TO LEFT")

        if((angle<0 and self.prev_angle>0) or (angle>0 and self.prev_angle<0)):
            print("total sum = " + str(self.sum_acc))
            print("CHANGE VALLEY PASSED!!!")
            self.sum_acc=0
        self.prev_angle=angle
        return np.array(self.state), reward, done, {}

    def reset(self):
        #TODO la posición inicial debe depender del rango de posiciones mostradas
        #en la ventana y se debe configurar como constante
        #self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.state = np.array([(self.max_pos-self.min_pos)/2, 0])
        return np.array(self.state)

    def _height(self, xs):
        return self.world_function(xs)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 30
        carheight = 15

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            print("min pos " + str(self.min_pos))
            print("max pos " + str(self.max_pos))

            xs = np.linspace(self.min_pos, self.max_pos, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_pos) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_pos) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        print("pooos " +str(pos))
        self.cartrans.set_translation(
            (pos-self.min_pos) * scale, self._height(pos) * scale
        )

        f_prime = lambdify(self.x, self.heigh_function.diff(self.x))
        self.cartrans.set_rotation(math.sin(f_prime(pos)))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
