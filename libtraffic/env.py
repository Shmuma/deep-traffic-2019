import gym
from gym import spaces
import enum
import random
import collections
import numpy as np


class Actions(enum.Enum):
    noAction = 0
    accelerate = 1
    decelerate = 2
    goLeft = 3
    goRight = 4


# relative Y speed equals 1 pos item per 5 speed units per frame

class Car:
    Length = 4
    Cell = 10
    MaxSpeed = 80
    SpeedUnitsPerPos = 5

    def __init__(self, speed, cell_x, cell_y):
        self.speed = speed
        self.safe_speed = speed
        self.cell_x = cell_x
        self.pos_y = cell_y * self.Cell

    def __repr__(self):
        return f"Car(lane={self.cell_x}, cell_y={self.cell_y}, speed={self.speed}, safe_speed={self.safe_speed})"

    @property
    def cell_y(self):
        return self.pos_y // self.Cell

    @property
    def cell(self):
        return self.cell_x, self.cell_y

    def overlaps(self, car, safety_dist=0):
        assert isinstance(car, Car)
        if self.cell_x != car.cell_x:
            return False
        # if other car is ahead of us significantly
        if self.cell_y - car.cell_y - self.Length > safety_dist:
            return False
        # if other car is behind us
        if car.cell_y - self.cell_y - self.Length > safety_dist:
            return False
        return True

    def overlaps_range(self, min_cell_y, max_cell_y):
        if min_cell_y <= self.cell_y <= max_cell_y:
            return True
        if min_cell_y <= self.cell_y + Car.Length - 1 <= max_cell_y:
            return True
        return False

    def shift_forward(self, rel_speed):
        assert isinstance(rel_speed, int)
        # we're negating rel speed, as our Y coordinate is decreasing with moving forward
        self.pos_y -= rel_speed // Car.SpeedUnitsPerPos

    def is_inside(self, y_cells):
        return 0 <= self.cell_y <= (y_cells-self.Length)

    def accelerate(self, delta=1):
        new_speed = self.speed + delta
        if 0 < new_speed <= Car.MaxSpeed:
            self.speed = new_speed

    def state(self):
        return {
            'speed': self.speed,
            'sspeed': self.safe_speed,
            'c_x': self.cell_x,
            'c_y': self.cell_y,
            'p_y': self.pos_y
        }


class TrafficState:
    """
    State of the game
    """
    # Amount of cells we keep in front of us
    Safety_front = 4

    def __init__(self, width_lanes=7, height_cells=70, cars=20, history=0, init_speed_my=80, init_speed_others=65,
                 other_cars_action_prob=0.1, state_render_view=None):
        """
        Construct internal DeepTraffic model
        :param width_lanes: width of the road in lanes
        :param height_cells: height of the road in cells
        :param cars: how many other cars will be there
        :param history: how many history states to track
        :param init_speed_my: central car initial speed
        :param init_speed_others: inital speed of other cars
        :param other_cars_action_prob: probability of other cars to take random action
        :param state_render_view: if None, full field will be rendered, otherwise has to be a tuple
        (side_lanes, cells_before, cells_after) defining dimensions of state to be rendered
        """
        self.width_lanes = width_lanes
        self.height_cells = height_cells
        self.cars_count = cars
        self.init_speed = init_speed_others
        self.other_cars_action_prob = other_cars_action_prob
        self.state_render_view = state_render_view

        self.my_car = Car(init_speed_my, (width_lanes-1)//2, 2*height_cells//3)
        self.cars = self._make_cars_initial(cars)
        self._update_safe_speed(self.my_car, self.cars)
        self.state = self._render_state(self.my_car, self.cars)
        # history has more recent entries in front
        self.history = collections.deque(maxlen=history)
        self.actions_history = collections.deque(maxlen=history)
        # populate history
        for _ in range(history):
            self.tick()

    def _make_cars_initial(self, count):
        assert isinstance(count, int)

        res = []
        others = [self.my_car]
        while len(res) < count:
            cell_x, cell_y = self._find_spot(self.width_lanes, self.height_cells, others)
            car = Car(self.init_speed, cell_x, cell_y)
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
            speed = self.init_speed + random.randrange(-5, 5)
            car = Car(speed, x, y)
            if any(map(lambda c: car.overlaps(c, safety_dist=4), self.cars)):
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

    def _update_safe_speed(self, my_car, cars):
        """
        For each car including our own calculate safe speed, taking into account car in front of us
        """
        assert isinstance(my_car, Car)
        assert isinstance(cars, list)

        list_cars = cars + [my_car]
        list_cars.sort(key=lambda c: c.cell)
        prev_x = -1
        prev_car_ends = None
        prev_car_speed = None

        for car in list_cars:
            if prev_x != car.cell_x:
                prev_car_ends = None
                prev_car_speed = None
            y = car.cell_y
            prev_x = car.cell_x
            # no car ahead or distance is enougth to keep the speed
            if prev_car_ends is None or y - prev_car_ends > self.Safety_front:
                car.safe_speed = car.speed
            elif y - prev_car_ends == self.Safety_front:
                car.safe_speed = prev_car_speed
            else:
                car.safe_speed = prev_car_speed // 2
            prev_car_ends = y + Car.Length
            prev_car_speed = car.safe_speed

    def _iterate_car_render_cells(self, car, full_state=False):
        if full_state or self.state_render_view is None:
            for d in range(Car.Length):
                yield (car.cell_x, car.cell_y+d)
        else:
            # need to remap coords
            render_lanes, render_before, render_behind = self.state_render_view
            dx = car.cell_x - self.my_car.cell_x
            if abs(dx) <= render_lanes:
                map_x = dx + render_lanes
                for d in range(Car.Length):
                    dy = car.cell_y - self.my_car.cell_y + d
                    if dy < 0 and -dy <= render_before:
                        # remap and yield -- cell is before us
                        map_y = dy + render_before
                        yield map_x, map_y
                    elif 0 <= dy < render_behind:
                        # remap and yield -- cell is behind
                        map_y = dy + render_before
                        yield map_x, map_y

    def _state_shape(self, full_state=False):
        if full_state or self.state_render_view is None:
            return self.width_lanes, self.height_cells
        else:
            return self.state_render_view[0]*2+1, self.state_render_view[1]+self.state_render_view[2]

    def _render_state(self, my_car, cars, render_full=False):
        """
        Returns grid of relative speeds
        :return:
        """
        assert isinstance(my_car, Car)
        assert isinstance(cars, list)

        res = np.zeros(self._state_shape(full_state=render_full), dtype=np.float32)
        for car in cars:
            dspeed = car.safe_speed - my_car.safe_speed
            for x, y in self._iterate_car_render_cells(car, full_state=render_full):
                res[x, y] = dspeed
        return res

    def _render_occupancy(self, my_car, cars, render_full=False):
        res = np.zeros(self._state_shape(full_state=render_full), dtype=int)
        for x, y in self._iterate_car_render_cells(my_car, full_state=render_full):
            res[x, y] = 2
        for car in cars:
            for x, y in self._iterate_car_render_cells(car):
                res[x, y] = 1
        return res

    def _move_cars(self, my_car, cars):
        assert isinstance(my_car, Car)
        assert isinstance(cars, list)

        for car in cars:
            dspeed = car.safe_speed - my_car.safe_speed
            car.shift_forward(dspeed)

    def _apply_action(self, car, action, other_cars):
        assert isinstance(car, Car)
        assert isinstance(action, Actions)
        assert isinstance(other_cars, list)

        if action == Actions.noAction:
            return

        if action == Actions.accelerate:
            car.accelerate(1)
            return
        elif action == Actions.decelerate:
            car.accelerate(-1)
            return

        new_x = car.cell_x
        if action == Actions.goLeft:
            new_x -= 1
        else:
            new_x += 1
        # if new position is beyond the road, ignore it
        if new_x < 0 or new_x >= self.width_lanes:
            return
        # check the safety system
        min_y = car.cell_y - 6
        max_y = car.cell_y + car.Length + int(round((car.pos_y % 10)/10))
        for c in other_cars:
            if c.cell_x == new_x and c.overlaps_range(min_y, max_y):
                return
        car.cell_x = new_x

    def _random_action(self, cars):
        if random.random() > self.other_cars_action_prob:
            return
        actions = [Actions.accelerate, Actions.decelerate, Actions.goLeft, Actions.goRight]
        action = random.choice(actions)
        car = random.choice(cars)
#        print("%s will do %s" % (car, action))
        self._apply_action(car, action, cars + [self.my_car])

    def tick(self, action=Actions.noAction):
        """
        Move time one frame forward
        """
        # housekeep the history, more recent entries are in front
        self.history.appendleft(self.state)
        self.actions_history.appendleft(action)

        # apply action to my car
        self._apply_action(self.my_car, action, self.cars)

        # perform random actions on other cars
        self._random_action(self.cars)

        # update safe speed
        self._update_safe_speed(self.my_car, self.cars)

        # change car's positions
        self._move_cars(self.my_car, self.cars)

        # throw away cars which have driven away and add new cars instead of them
        self.cars = list(filter(lambda c: c.is_inside(self.height_cells), self.cars))
        while len(self.cars) < self.cars_count:
            new_car = self._make_car_new()
            if new_car is None:
                break
            self.cars.append(new_car)

        # render new state
        self.state = self._render_state(self.my_car, self.cars)

    def is_collision(self):
        for c in self.cars:
            if self.my_car.overlaps(c):
                return self.my_car, c
        for c1 in self.cars:
            for c2 in self.cars:
                if c1 is c2:
                    continue
                if c1.overlaps(c2):
                    return c1, c2
        return None

    def snapshot(self):
        res = [self.my_car.state()]
        for c in self.cars:
            res.append(c.state())
        return res

    def render_occupancy(self, full):
        assert isinstance(full, bool)
        return self._render_occupancy(self.my_car, self.cars, render_full=full)

    def render_state(self, full):
        assert isinstance(full, bool)
        return self._render_state(self.my_car, self.cars, render_full=full)


class DeepTraffic(gym.Env):
    def __init__(self, lanes_side=1, patches_ahead=20, patches_behind=10, history=3):
        self.state = None
        self.history_steps = history
        self.action_space = spaces.Discrete(len(Actions))
        self.lanes_side = lanes_side
        self.patches_ahead = patches_ahead
        self.patches_behind = patches_behind
        # observations stack are current state, history of states and history of one-hot encoded actions
        self.obs_shape = (history + 1 + len(Actions)*history, lanes_side*2 + 1, patches_ahead + patches_behind)
        self.observation_space = spaces.Box(low=-Car.MaxSpeed, high=Car.MaxSpeed,
                                            shape=self.obs_shape, dtype=np.float32)

    def reset(self):
        render_view = (self.lanes_side, self.patches_ahead, self.patches_behind)
        self.state = TrafficState(history=self.history_steps, state_render_view=render_view)
        result = self._render_state(self.state)
        return result

    def _render_state(self, state):
        res = np.zeros(self.obs_shape, dtype=np.float32)
        # current state
        res[0] = state.state
        # history
        ofs = 1
        for hist_state in state.history:
            res[ofs] = hist_state
            ofs += 1
        # actions history
        for action in state.actions_history:
            res[ofs + action.value] = 1.0
            ofs += len(Actions)
        return res

    def _reward(self):
        return (self.state.my_car.safe_speed - 60) / 20

    def step(self, action_index):
        action = Actions(action_index)
        self.state.tick(action)
        obs = self._render_state(self.state)
        r = self._reward()
        return obs, r, False, {}

    def current_speed(self):
        return self.state.my_car.safe_speed

    def close(self):
        pass
