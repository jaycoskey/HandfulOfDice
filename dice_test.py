#!/usr/bin/env python

from collections import Counter, OrderedDict
from dice_pmf import DicePmf, DiceUtil
from scipy.stats import multinomial as stats_multinomial
import sys
from typing import Dict
import unittest
from util import Util


class DiceTest(unittest.TestCase):
    def test_dice_infer_stepwise(self):
        dice = {4:1, 6:1, 8:1, 12:1, 20:1}
        dist = Counter(DiceUtil.get_dice_rolls(dice, 100_000))
        num_faces_avail = [4, 6, 8, 12, 20]
        prob_dist = Util.normalized_dict(dist)
        # print(f'INFO: dist={print_dist(dist)}')
        # print(f'INFO: prob_dist={print_dist(prob_dist)}')
        dpmf = DiceUtil.dice_infer_stepwise(prob_dist, num_faces_avail, init_dice=None, kind='energy')
        print(f'Actual dice:   {OrderedDict(dice)}')
        print(f'Inferred dice: {OrderedDict(dpmf.dice)}')

    def test_dice_pmf_2d6(self):
        FUNCNAME = sys._getframe().f_code.co_name
        dpmf = DicePmf({6:2})
        assert(dpmf.low == 2)
        assert(dpmf.high == 12)
        pmf = list(map(lambda x: int(36 * x), dpmf.pmf))
        assert(pmf) == [1,2,3,4,5,6,5,4,3,2,1]
        print(f'{FUNCNAME}: test passed')

    def test_dice_rolls_sample_2d6(self, dice:Dict={6:2}, num_rolls:int=36, do_debug_print=True, do_debug_plot=False):
        '''Test with num_rolls groups of rolls of the specified set of dice'''
        FUNCNAME = sys._getframe().f_code.co_name
        Util.enter(FUNCNAME)

        rolls = DiceUtil.get_dice_rolls(dice, num_rolls)
        cntr = Counter(rolls)
        hist = OrderedDict([(k, cntr[k]) for k in sorted(cntr.keys())])
        percentages = ['{}, {:.2f}%'.format(k, 100 * v / num_rolls) for k, v in hist.items()]
        if do_debug_print:
            print(list(hist.items()))
            print(percentages)
        if do_debug_plot:
            Util.plot_vhist(hist)
        Util.exit(FUNCNAME)
        return hist
 
    def test_dice_rolls_sample_5(self, n=3):
        FUNCNAME = sys._getframe().f_code.co_name
        Util.enter(FUNCNAME)

        print('Rolls of 1d4 + 1d6 + 1d8 + 1d12 + 1d20 (5-50): ', end='')
        print(', '.join(map(str, DiceUtil.get_dice_rolls({4:1, 6:1, 8:1, 12:1, 20:1}, 10))))
        Util.exit(FUNCNAME)
    
    def test_multinomial_check_total_2d6(self):
        FUNCNAME = sys._getframe().f_code.co_name
        Util.enter(FUNCNAME)
        DEFAULT_PLACES = 6

        multi = stats_multinomial(2, [1/6] * 6)
        prob_total = 0.0
        pmfs = []

        for x in range(1, 7):
            for y in range(1, x + 1):
                dist = [0] * 6
                dist[x - 1] += 1
                dist[y - 1] += 1
                pmf = multi.pmf(dist)
                pmfs.append(pmf)
                prob_total += pmf
        self.assertAlmostEqual(prob_total, 1.0, places=DEFAULT_PLACES)
        Util.exit(FUNCNAME)
 
    def test_print_platonic_dice_variances(self):
        FUNCNAME = sys._getframe().f_code.co_name
        Util.enter(FUNCNAME)

        platonic_dice_variances = []
        for num_sides in [4, 6, 8, 12, 20]:
            ROUNDOFF_CONST = 10000
            sides_range = range(1, num_sides + 1)
            mean = (1 + num_sides) / 2
            variance = sum([((k - mean)**2) for k in sides_range]) / num_sides
            variance = int(ROUNDOFF_CONST * variance) / ROUNDOFF_CONST
            platonic_dice_variances.append(variance)
        print(f'platonic_dice_variances={platonic_dice_variances}')
        Util.exit(FUNCNAME)


if __name__ == '__main__':
    unittest.main() 
