import unittest
import tdse


class TestAtom(unittest.TestCase):
    def test_mg(self):
        atom = tdse.atom.Mg

        self.assertEqual(atom.countOrbs, 5)

        self.assertEqual(atom.get_l(0), 0)
        self.assertEqual(atom.get_l(1), 0)
        self.assertEqual(atom.get_l(2), 0)
        self.assertEqual(atom.get_l(3), 1)
        self.assertEqual(atom.get_l(4), 1)

        self.assertEqual(atom.getCountElectrons(0), 2)
        self.assertEqual(atom.getCountElectrons(1), 2)
        self.assertEqual(atom.getCountElectrons(2), 2)
        self.assertEqual(atom.getCountElectrons(3), 2)
        self.assertEqual(atom.getCountElectrons(4), 4)

        self.assertEqual(atom.get_m(0), 0)
        self.assertEqual(atom.get_m(1), 0)
        self.assertEqual(atom.get_m(2), 0)
        self.assertEqual(atom.get_m(3), 0)
        self.assertEqual(atom.get_m(4), 1)

    def test_ne(self):
        atom = tdse.atom.Ne
        self.assertEqual(atom.countOrbs, 4)

        self.assertEqual(atom.ground_state.l, 1)
        self.assertEqual(atom.ground_state.m, 0)
        self.assertEqual(atom.ground_state.n, 0)

    def test_ar(self):
        atom = tdse.atom.Ar
        self.assertEqual(atom.countOrbs, 7)

        self.assertEqual(atom.ground_state.l, 1)
        self.assertEqual(atom.ground_state.m, 0)
        self.assertEqual(atom.ground_state.n, 1)

    def test_ar_sae(self):
        atom = tdse.atom.Ar_sae
        self.assertEqual(atom.ground_state.l, 1)
        self.assertEqual(atom.ground_state.m, 0)
        self.assertEqual(atom.ground_state.n, 1)
