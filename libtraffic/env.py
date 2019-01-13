import gym
from gym import spaces
import enum
import random
import numpy as np


class Actions(enum.Enum):
    noAction = 0
    accelerate = 1
    decelerate = 2
    goLeftAction = 3
    goRightAction = 4


class Car:
    Length = 4

    def __init__(self, speed, pos_x, pos_y):
        self.speed = speed
        self.pos_x = pos_x
        self.pos_y = pos_y

    def overlaps(self, car, safety_dist=0):
        assert isinstance(car, Car)
        if self.pos_x != car.pos_x:
            return False
        # if other car is ahead of us significantly
        if self.pos_y - car.pos_y - self.Length > safety_dist:
            return False
        # if other car is behind us
        if car.pos_y - self.pos_y - self.Length > safety_dist:
            return False
        return True


class TrafficState:
    """
    State of the game
    """
    Safety_front = 4

    def __init__(self, width_lanes=7, height_cells=70, cars=20, history=0, init_speed_my=80, init_speed_others=65):
        self.width_lanes = width_lanes
        self.height_cells = height_cells
        self.cars_count = cars
        self.history_count = history
        self.init_speed = init_speed_others
        self.my_car = Car(init_speed_my, (width_lanes-1)//2, 2*height_cells//3)
        self.cars = self._make_cars_initial(cars)
        self.state = self._render_state(self.my_car, self.cars)

    def _make_cars_initial(self, count):
        assert isinstance(count, int)

        res = []
        others = [self.my_car]
        while len(res) < count:
            pos_x, pos_y = self._find_spot(self.width_lanes, self.height_cells, others)
            car = Car(self.init_speed, pos_x, pos_y)
            res.append(car)
            others.append(car)
        return res

    def _make_car_new(self):
        positions = []
        for y in [0, self.height_cells-Car.Length]:
            for x in range(self.width_lanes):
                positions.append((x, y))

        random.shuffle(positions)
        for x, y in positions:
            speed = self.init_speed + random.randrange(-20, 20)
            car = Car(speed, x, y)
            if any(map(lambda c: car.overlaps(c), self.cars)):
                continue
            return car
        return None

    @staticmethod
    def _find_spot(max_x, max_y, cars):
        while True:
            x = random.randrange(max_x)
            y = random.randrange(max_y - Car.Length)
            test_car = Car(0, x, y)
            if any(map(lambda c: test_car.overlaps(c), cars)):
                continue
            return x, y

    def _render_state(self, my_car, cars):
        """
        Returns grid of relative speeds
        :return:
        """
        res = np.zeros((self.width_lanes, self.height_cells), dtype=np.float32)
        for car in cars:
            dspeed = car.speed - my_car.speed
            res[car.pos_x, car.pos_y:(car.pos_y + Car.Length)] = dspeed
        return res


class EnvDeepTraffic(gym.Env):
    def __init__(self, lanes_side=3, patches_ahead=20, patches_behind=10):
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box()
