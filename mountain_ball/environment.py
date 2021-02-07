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
    seconds_per_step=3
    force = 5
    initial_position_low=8000
    initial_position_high=9100
    initial_speed=0

    lower_position=-11000
    higher_position=11000
    max_speed = 10000

    goal_position=10000

    discretization_level=0

    # We have to create a "symbol" called x
    x = Symbol('x')
    #ramp
    #heigh_function=0.3*x +2
    #weird courves mountain
    #heigh_function=4*sin(0.3*x)**3+ 4.5
    #floor
    #heigh_function=0.00000000000000000001*x + 2
    #mountainCar environment
    heigh_function=sin(0.0003*x-10.5)*4000 + 4000

    gravity=9.8

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.force=self.force/self.car_mass
        self.world_function= lambdify(self.x, self.heigh_function)
        #Consider position is in metters
        self.min_position = self.lower_position
        self.max_position = self.higher_position

        #Consider that speed is metters per millisecond
        self.goal_velocity = goal_velocity

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


        self.goal_position = (self.goal_position - self.observation_space.low[0])
        self.goal_position = np.round(self.goal_position, self.discretization_level)

        self.min_pos = (self.min_position - self.observation_space.low[0])
        self.min_pos = np.round(self.min_pos, self.discretization_level).astype(int)

        self.max_pos = (self.max_position - self.observation_space.low[0])
        self.max_pos = np.round(self.max_pos, self.discretization_level).astype(int)

        self.observation_space.low[0]=self.min_pos
        self.observation_space.high[0]=self.max_pos

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        #derivative function in point to get the function slope
        f_prime = lambdify(self.x, self.heigh_function.diff(self.x))
        position, velocity = self.state
        #and so, its angle in radians
        angle=math.atan(f_prime(position))

        #accelaration per second
        self.acceleration=-self.gravity*math.sin(angle) + ((action - 1) * self.force)

        #if(sum_acc is not None):
        #    sum_acc=sum_acc+self.acceleration
        #else:
        #    sum_acc=0


        #just variables to store in memory the previous locaation, which will be used later to adjust the real velocities (physics)
        velocity_prev=velocity
        position_prev= position
        prev_angle=angle


        space_run=(velocity*self.seconds_per_step)+((1/2)*self.acceleration*(self.seconds_per_step**2))
        position+=space_run*math.cos(angle)

        velocity+=self.acceleration*self.seconds_per_step
        position = np.round(int(position), self.discretization_level).astype(int)

        #Velocity adjustment to make physics behaviour more precise.
        #We calculate the real velocity the car will have at the previous final point reached estimation applying the forces in each step between initial and final positions

        #TODO This is a gap fix to enable the usage of a simpler and faster physics equation (we consider no aceleration for this inner velocities adjustment)
        t=0.1
        #_______________

        #Note that discretization_level is used to enable the usage of decimes or centesimes instead of seconds
        inner_velocity=velocity_prev
        for x in range(int(abs(position-position_prev)*(10**self.discretization_level))):
            #print(x)
            #print(inner_velocity)
            if(inner_velocity>1.0 or inner_velocity<-1.0):
                t=abs(1/(inner_velocity*(10**self.discretization_level)))

#MORE SOPHISTICATED (but much slower performance) PHYSICS!!! However, if needed, the abs values in last elif must be reviewed and fixed
#        inner_velocity=velocity_prev
#        for x in range(int(abs(position-position_prev)*(10**self.discretization_level))):
            #print(x)
            #print(inner_velocity)
#            if(self.acceleration==0 and inner_velocity!=0.0):
#                t=abs(1/(inner_velocity*(10**self.discretization_level)))
#            elif(self.acceleration==0 and inner_velocity==0.0):
#                break
#            elif(self.acceleration!=0):
#                t=(sqrt(2*abs(self.acceleration)+(abs(inner_velocity)*(10**self.discretization_level))**2)-(abs(inner_velocity)*(10**self.discretization_level)))/abs(self.acceleration)
#______________________________!!!

            inner_velocity+=self.acceleration*t

            #if else needed to calculate the angle due to the abs method used in for conditions with the x position
            if(position>position_prev):
                inner_angle=math.atan(f_prime(position_prev+x))
            else:
                inner_angle=math.atan(f_prime(position_prev-x))

            self.acceleration=-self.gravity*math.sin(inner_angle) + ((action - 1) * self.force)
            #sum_acc useful to debug the physics precission
            #sum_acc=sum_acc+self.acceleration

        #final velocity adjustment
        velocity=inner_velocity


        #Once the vlelocity and position is calculated, we truncate them to be within the configured boundaries
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = np.clip(position, self.min_pos, self.max_pos)

        if (position == self.min_pos and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position
            #if we consider the velocity which I did not, the sentence below must be applied
            #position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = -1.0
        self.state = (position, velocity)

        #sum_acc and traces below useful to debug the physics precission:

        #if((velocity_prev<0 and velocity>0) or (velocity_prev>0 and velocity<0)):
        #    sum_acc=0
        #if(velocity_prev<0 and velocity>0):
        #    print("CHANGE FROM LEFT TO RIGHT")
        #if(velocity_prev>0 and velocity<0):
        #    print("CHANGE FROM RIGHT TO LEFT")

        #if((angle<0 and self.prev_angle>0) or (angle>0 and self.prev_angle<0)):
        #    print("total sum = " + str(sum_acc))
        #    print("CHANGE VALLEY PASSED!!!")
        #    sum_acc=0

        #_________________________________

        return np.array(self.state), reward, done, {}

    def reset(self):
        #TODO la posición inicial debe depender del rango de posiciones mostradas
        #en la ventana y se debe configurar como constante
        self.state = np.array([self.np_random.uniform(low=self.initial_position_low, high=self.initial_position_high), self.initial_speed])
        #self.state = np.array([(self.max_pos-self.min_pos)/2, 0])
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
