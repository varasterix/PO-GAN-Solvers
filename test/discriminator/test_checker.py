import unittest
from src.discriminator import checker as c


class TestCheckerMethods(unittest.TestCase):
    def test_nb_cycles(self):
        self.assertTrue(c.nb_cycles([1, 2, 3, 0]) == 1)
        self.assertFalse(c.nb_cycles([0, 1, 2]) == 1)
        self.assertFalse(c.nb_cycles([1, 2, 3, 1]) == 1)
        self.assertFalse(c.nb_cycles([10, 2, 3, 0]) == 1)
        self.assertFalse(c.nb_cycles(["c", 2, 3, 0]) == 1)


if __name__ == '__main__':
    unittest.main()
