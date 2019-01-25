import random
import unittest
import numpy as np
import pprint

from libtraffic import env


class TestCar(unittest.TestCase):
    def test_overlaps(self):
        c = env.Car(0, 0, 0)
        self.assertTrue(c.overlaps(c))
        self.assertTrue(c.overlaps(env.Car(0, 0, 1)))
        self.assertTrue(c.overlaps(env.Car(0, 0, 2)))
        self.assertTrue(c.overlaps(env.Car(0, 0, 3)))
        self.assertTrue(c.overlaps(env.Car(0, 0, 4)))
        self.assertFalse(c.overlaps(env.Car(0, 0, 5)))
        self.assertTrue(c.overlaps(env.Car(0, 0, 5), safety_dist=1))
        self.assertFalse(c.overlaps(env.Car(0, 1, 2)))

        c = env.Car(0, 3, 20)
        self.assertFalse(c.overlaps(env.Car(0, 3, 15)))
        self.assertTrue(c.overlaps(env.Car(0, 3, 16)))
        self.assertTrue(c.overlaps(env.Car(0, 3, 17)))
        self.assertTrue(c.overlaps(env.Car(0, 3, 18)))
        self.assertTrue(c.overlaps(env.Car(0, 3, 19)))
        self.assertTrue(c.overlaps(env.Car(0, 3, 24)))
        self.assertFalse(c.overlaps(env.Car(0, 3, 25)))

    def test_overlaps_range(self):
        c = env.Car(0, 0, 0)
        self.assertTrue(c.overlaps_range(0, 0))
        self.assertTrue(c.overlaps_range(0, 10))
        self.assertTrue(c.overlaps_range(-10, 0))
        self.assertFalse(c.overlaps_range(-10, -1))
        self.assertTrue(c.overlaps_range(1, 10))
        self.assertTrue(c.overlaps_range(3, 10))
        self.assertFalse(c.overlaps_range(4, 10))

    def test_is_inside(self):
        c = env.Car(0, 0, 2)
        self.assertTrue(c.is_inside(y_cells=6))
        self.assertFalse(c.is_inside(y_cells=5))
        self.assertFalse(env.Car(0, 0, -1).is_inside(y_cells=100))

    def test_shift(self):
        c = env.Car(0, 0, 1)
        c.shift_forward(rel_speed=env.Car.SpeedUnitsPerPos)
        self.assertEqual(c.pos_y, 9)
        self.assertEqual(c.cell_y, 0)


class TestTrafficState(unittest.TestCase):
    def test_init(self):
        ts = env.TrafficState()
        self.assertEqual(len(ts.cars), ts.cars_count)
        self.assertEqual(ts.my_car.cell_x, 3)
        self.assertEqual(ts.my_car.cell_y, 46)

    def test_render_state(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 10)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ]))

        c2 = env.Car(0, 1, 0)
        s = ts._render_state(my_car, [c1, c2])
        np.testing.assert_array_almost_equal(s, [
            [1,   1,  1,  1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1],
            [0.005, 0.005, 0.005, 0.005, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1,   1,  1,  1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

    def test_update_safe_speed(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        # nothing should change
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 10)
        ts._update_safe_speed(my_car, [c1])
        self.assertEqual(my_car.safe_speed, 5)
        self.assertEqual(c1.safe_speed, 15)
        # our car should be slowed down to the speed of upfront car
        my_car = env.Car(20, 1, 10)
        c1 = env.Car(10, 1, 2)
        ts._update_safe_speed(my_car, [c1])
        self.assertEqual(my_car.safe_speed, 10)
        self.assertEqual(c1.safe_speed, 10)

    def test_move_cars(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 10)
        # relative speed is 10, so c1 should move 1 cell forward
        ts._move_cars(my_car, [c1])
        self.assertEqual(c1.cell_y, 9)
        self.assertEqual(c1.pos_y, 98)
        self.assertEqual(my_car.cell_y, 10)
        c1 = env.Car(0, 0, 10)
        # relative speed is -5, so, it should keep the current cell (due to rounding)
        ts._move_cars(my_car, [c1])
        self.assertEqual(c1.cell_y, 10)
        self.assertEqual(c1.pos_y, 101)
        ts._move_cars(my_car, [c1])
        self.assertEqual(c1.cell_y, 10)
        self.assertEqual(c1.pos_y, 102)

    def test_apply_action(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 10)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

        ts._apply_action(c1, env.Actions.accelerate, [my_car])
        self.assertEqual(c1.speed, 16)
        ts._update_safe_speed(my_car, [c1])
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.011, 0.011, 0.011, 0.011, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

        # should be ignored - central car is on the way
        ts._apply_action(c1, env.Actions.goRight, [my_car])
        self.assertEqual(c1.cell_x, 0)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.011, 0.011, 0.011, 0.011, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

        # move central car to the right
        ts._apply_action(my_car, env.Actions.goRight, [c1])
        self.assertEqual(my_car.cell_x, 2)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.011, 0.011, 0.011, 0.011, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

        # should now change lanes
        ts._apply_action(c1, env.Actions.goRight, [my_car])
        self.assertEqual(c1.cell_x, 1)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_almost_equal(s, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.011, 0.011, 0.011, 0.011, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1],
        ])

    def test_safety_system(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 14)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_equal(s, [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0],
        ])

        # should be refused
        ts._apply_action(my_car, env.Actions.goLeft, [c1])
        self.assertEqual(my_car.cell, (1, 10))

        c1 = env.Car(15, 0, 15)
        # should move
        ts._apply_action(my_car, env.Actions.goLeft, [c1])
        self.assertEqual(my_car.cell, (0, 10))

        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_equal(s, [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0],
        ])

        ts._update_safe_speed(my_car, [c1])
        self.assertEqual(c1.safe_speed, 2)

        # check in front of the car
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 6)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_equal(s, [
            [0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,  0,  0,  0,  0, +0, +0, +0, +0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0],
        ])

        # should be refused
        ts._apply_action(my_car, env.Actions.goLeft, [c1])
        self.assertEqual(my_car.cell, (1, 10))

        # should be refused
        c1 = env.Car(15, 0, 1)
        ts._apply_action(my_car, env.Actions.goLeft, [c1])
        self.assertEqual(my_car.cell, (1, 10))
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_equal(s, [
            [0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0, 0, +0, +0, +0, +0, 0, 0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # should be accepted
        c1 = env.Car(15, 0, 0)
        ts._apply_action(my_car, env.Actions.goLeft, [c1])
        self.assertEqual(my_car.cell, (0, 10))

    def test_check_collisions(self):
        ts = env.TrafficState()
        # history = [ts.snapshot()]
        for _ in range(1000):
            c = ts.is_collision()
            if c is not None:
                print("Collision! %s with %s" % c)
                # with open("collision.txt", 'wt', encoding='utf-8') as fd:
                #     pprint.pprint(history, fd)
                assert False
            ts.tick()
#            history.append(ts.snapshot())

    def test_render(self):
        random.seed(10)
        ts = env.TrafficState(state_render_view=(1, 10, 5))
        self.assertEqual(ts.state.shape, (3, 15))
        np.testing.assert_array_almost_equal(ts.state, [
            [0.0325, 1, 1, 1, 1, 1, 1, 1, 1,   1,   1,   1,   1, 1, 1],
            [  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.04, 0.04, 0.04, 0.04, 1],
            [  1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0325, 0.0325, 0.0325, 0.0325, 1, 1],
        ])
        ts.tick(env.Actions.goLeft)
        np.testing.assert_array_almost_equal(ts.state, [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0325, 0.0325, 0.0325, 0.0325, 1],
            [0.0325, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.04, 0.04, 0.04, 0.04, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        for step in range(16):
            ts.tick(env.Actions.noAction)
        ts.tick(env.Actions.goLeft)
        ts.tick(env.Actions.goLeft)
        np.testing.assert_array_almost_equal(ts.state, [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.04, 0.04, 0.04, 0.04, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])


class TestDeepTraffic(unittest.TestCase):
    def test_basic(self):
        e = env.DeepTraffic(lanes_side=1, patches_ahead=20, patches_behind=10, history=0)
        r = e.reset()
        self.assertEqual(r.shape, (1, 3, 30))

    def test_hist(self):
        e = env.DeepTraffic(lanes_side=1, patches_ahead=20, patches_behind=10, history=3)
        r = e.reset()
        self.assertEqual(r.shape, (19, 3, 30))
        np.testing.assert_array_equal(r[4], 1.0)
        np.testing.assert_array_equal(r[5:9], 0.0)
        np.testing.assert_array_equal(r[9], 1.0)
        np.testing.assert_array_equal(r[10:14], 0.0)
        np.testing.assert_array_equal(r[14], 1.0)
        np.testing.assert_array_equal(r[15:19], 0.0)
