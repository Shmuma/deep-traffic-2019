import unittest
import numpy as np

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


class TestTrafficState(unittest.TestCase):
    def test_init(self):
        ts = env.TrafficState()
        self.assertEqual(len(ts.cars), ts.cars_count)
        self.assertEqual(ts.my_car.pos_x, 3)
        self.assertEqual(ts.my_car.pos_y, 46)

    def test_render_state(self):
        ts = env.TrafficState(width_lanes=3, height_cells=20, cars=0)
        my_car = env.Car(5, 1, 10)
        c1 = env.Car(15, 0, 10)
        s = ts._render_state(my_car, [c1])
        np.testing.assert_array_equal(s, [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  10, 10, 10, 10, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0],
        ])

        c2 = env.Car(0, 1, 0)
        s = ts._render_state(my_car, [c1, c2])
        np.testing.assert_array_equal(s, [
            [0,   0,  0,  0, 0, 0, 0, 0, 0, 0,  10, 10, 10, 10, 0, 0, 0, 0, 0, 0],
            [-5, -5, -5, -5, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,   0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0],
        ])
