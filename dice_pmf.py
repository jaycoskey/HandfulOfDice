#!/usr/bin/env python

from collections import Counter
from dice_config import DiceConfig
from functools import partial
import numpy as np
from numpy.random import multinomial as rand_multinomial
from scipy.stats import energy_distance
from scipy.stats import entropy as kl_divergence
from scipy.stats import wasserstein_distance
import sys
from typing import Dict


class memoize_instance_method:
    def __init__(self, f):
        self.cache = {}
        self.func = f

    def __call__(self, self_dpmf, dice_tuple):
        if dice_tuple in self.cache:
            return self.cache[dice_tuple]
        else:
            value = self.func(self_dpmf, dice_tuple)
            self.cache[dice_tuple] = value
            return value
        
    def __get__(self, instance, owner):
        return partial(self.__call__, instance)


class DicePmf:
    @staticmethod
    def dice_infer(prob_dist, kind='wasserstein'):
        gap_func = DiceUtil.get_prob_gap_func(prob_dist, kind)
        best_gap = sys.maxsize
        best_dpmf = None
        print(f'd4 ranges from 0 to {DiceConfig.MAX_DICE_PER_TYPE}.')
        for d4 in range(0, DiceConfig.MAX_DICE_PER_TYPE + 1):
            print(f'd4={d4}: ', end='')
            for d6 in range(0, DiceConfig.MAX_DICE_PER_TYPE + 1):
                print('*', end='')
                for d8 in range(0, DiceConfig.MAX_DICE_PER_TYPE + 1):
                    print('.', end='')
                    for d12 in range(0, DiceConfig.MAX_DICE_PER_TYPE + 1):
                        for d20 in range(0, DiceConfig.MAX_DICE_PER_TYPE + 1):
                            if all(map(lambda x: x == 0, [d4, d6, d8, d12, d20])):
                                continue
                            dpmf = DicePmf({4: d4, 6: d6, 8: d8, 12: d12, 20: d20})
                            gap = gap_func(dpmf)
                            if gap < best_gap:
                                best_dpmf = dpmf
                                best_gap = gap
            print()
        return best_dpmf

    def __init__(self, dice, low=None, high=None, pmf=None):
        self.dice = dice
        self.low = low if low else sum(dice.values())
        self.high = high if high else sum([k * v for k, v in dice.items()])
        self.pmf = self._get_pmf(dice) if pmf is None else pmf

    def __str__(self):
        return f'<<DicePmf: dice={self.dice}, low={self.low}, high={self.high}, pmf={self.pmf}>>'

    def _convolve(self, num_faces):
        if num_faces in self.dice.keys():
            self.dice[num_faces] += 1
        else:
            self.dice[num_faces] = 1
        self.low += 1
        self.high += num_faces
        die_pmf = np.repeat(1/num_faces, num_faces)
        self.pmf = np.convolve(self.pmf, die_pmf, 'full')

    def _convolved(self, num_faces):
        other = self._copy()
        other._convolve(num_faces)
        return other

    def _copy(self):
        return DicePmf(self.dice, self.low, self.high, self.pmf)

    def _get_pmf(self, dice):
        dice_tuple = tuple(sorted(dice.items()))
        return self._impl_get_pmf(dice_tuple)

    @memoize_instance_method
    def _impl_get_pmf(self, dice_tuple):
        dice = {p[0]: p[1] for p in dice_tuple}
        is_base_case = (sum(dice.values()) == 1
                        and len([k for k in dice.values() if k > 0]) == 1
                       )
        if is_base_case:
            num_faces = [k for k, v in dice.items() if v > 0][0]
            if num_faces in DICE_DPMFS.keys():
                return DICE_DPMFS[num_faces].pmf
            else:
                raise ValueError('DicePmf: Invalid number of faces')
        else:  # Induction step. Recurse on first positive value.
            induction_k = min([k for k, v in dice.items() if v > 0])
            prev_dice = {k: (v - 1 if k == induction_k else v)
                         for k, v in dice.items()
                        }
            prev_dpmf = DicePmf(prev_dice)
            new_dpmf = prev_dpmf._convolved(induction_k)
            return new_dpmf.pmf

DICE_DPMFS = {
    4: DicePmf({4:1}, low=1, high=4, pmf=np.repeat(1/4, 4))
    , 6: DicePmf({6:1}, low=1, high=6, pmf=np.repeat(1/6, 6))
    , 8: DicePmf({8:1}, low=1, high=8, pmf=np.repeat(1/8, 8))
    , 12: DicePmf({12:1}, low=1, high=12, pmf=np.repeat(1/12, 12))
    , 20: DicePmf({20:1}, low=1, high=20, pmf=np.repeat(1/20, 20))
}


class DiceUtil:
    def dice_comparable_arrays(dist: Dict[int, int], dpmf: DicePmf):
        low = min(min(dist), dpmf.low)
        high = max(max(dist), dpmf.high)
        xs = list(range(low, high + 1))
    
        # Array of distribution values extending from the lowest index to the highest
        pk = np.asarray([dist.get(k, 0) for k in xs], dtype=float)
        
        # Array of PMF values extending from the lowest index to the highest
        pmf = dpmf.pmf
        if low < dpmf.low:
            pmf = np.concatenate([np.zeros(dpmf.low - low), pmf])
        if high > dpmf.high:
            pmf = np.concatenate([pmf, np.zeros(high - dpmf.high)])
    
        return (pk, pmf)
    
    def dice_func_energy(dist: Dict[int, int], dpmf: DicePmf):
        pk, pmf = DiceUtil.dice_comparable_arrays(dist, dpmf)
        return energy_distance(pk, pmf)
    
    def dice_func_entropy(dist: Dict[int, int], dpmf: DicePmf):
        """Calls scipy.stats.entropy (i.e., Kullman-Leibler divergence)"""
        pk, pmf = DiceUtil.dice_comparable_arrays(dist, dpmf)
        return kl_divergence(pk, pmf)
    
    def dice_func_entropy_conditioned(dist: Dict[int, int], dpmf: DicePmf):
        """Calls scipy.stats.entropy, but avoids result being inf"""
        def condition_pk(pk_raw):
            length = pk_raw.shape[0]
            idx_zeros = [k for k in range(length) if dist.get(k + dpmf.low, 0) == 0]
            min_val = min([pk_raw[k] for k in range(length) if k not in idx_zeros])
            extra = min_val * len(idx_zeros)
            scale = 1 / (1 + extra)
            pk = np.array([min_val if k in idx_zeros else scale * pk_raw[k]
                            for k in range(length)
                            ])
            return pk
    
        pk_raw, pmf = DiceUtil.dice_comparable_arrays(dist, dpmf)
        pk = condition_pk(pk_raw)
    
        return kl_divergence(pk, pmf)
    
    def dice_func_wasserstein(dist: Dict[int, int], dpmf: DicePmf):
        pk, pmf = DiceUtil.dice_comparable_arrays(dist, dpmf)
        return wasserstein_distance(pk, pmf)
    
    def dice_infer_stepwise(prob_dist, num_faces_avail, init_dice=None, kind='entropy'):
        VERBOSE = False
    
        def gap_func(dpmf):
            assert(type(dpmf).__name__ == 'DicePmf')
            gapfunc = DiceUtil.get_prob_gap_func(prob_dist, kind)
            return gapfunc(dpmf)
    
        if init_dice is None:
            nfaces = []
            for nf in num_faces_avail:
                nfaces.append([nf])
            #TODO: Could use itertools.product here
            #for nf1 in num_faces_avail:
            #    for nf2 in num_faces_avail:
            #        nfaces_popn.append([nf1, nf2])
            #for nf1 in num_faces_avail:
            #    for nf2 in num_faces_avail:
            #        for nf3 in num_faces_avail:
            #            nfaces_popn.append([nf1, nf2, nf3])
            #etc.
            init_dice = map(Counter, nfaces)
    
        if VERBOSE:
            print(f'VERBOSE: init_dice={init_dice}')
    
        dpmf_popn = map(lambda dice: DicePmf(dice), init_dice)
        best_dpmf = min(dpmf_popn, key=gap_func)
        best_gap = gap_func(best_dpmf)
        is_local_min_found = False
        round_limit = 50
        round_num = 0
        while not is_local_min_found and round_num < round_limit:
            round_num += 1
            print(f'========== Round #{round_num} ==========')
            dpmf_nbrs = [best_dpmf._convolved(num_faces)
                            for num_faces in num_faces_avail
                        ]
            if VERBOSE:
                print(f'VERBOSE: Neighbors of {best_dpmf.dice}:\n\t', end='')
                print('\n\t'.join(list(map(lambda x: str(x.dice), dpmf_nbrs))))
            best_nbr = min(dpmf_nbrs, key=gap_func)
            best_nbr_gap = gap_func(best_nbr)
            if best_nbr_gap < best_gap:
                best_dpmf = best_nbr
                if VERBOSE:
                    print(f'New best_dpmf.dice={best_dpmf.dice}: {best_nbr_gap:.4f} < {best_gap:.4f}')
                best_gap = best_nbr_gap
            else:
                is_local_min_found = True
        if VERBOSE:
            print(f'Found best local min of dice: {best_dpmf.dice}')
        return best_dpmf

    def dice_str(dice):
        return ' + '.join([f'{n}d{k}' for k, n in dice.items()])

    def get_dice_rolls(dice: Dict, num_rolls=1):
        return [DiceUtil.get_dice_roll(dice) for _ in range(num_rolls)]

    def get_dice_roll(dice: Dict):
        return sum([DiceUtil.get_die_rolls(num_faces, count) for num_faces, count in dice.items()])

    def get_die_rolls(num_faces: int, count: int):
        roll_sums = [ int(sum([(k+1)*row[k] for k in range(len(row))]))
                      for row in rand_multinomial(count, [1/num_faces] * num_faces, 1)
                    ]
        return roll_sums[0]

    def get_prob_gap_func(prob_dist, kind='entropy_conditioned'):
        if kind == 'energy':
            f = lambda dpmf: DiceUtil.dice_func_energy(prob_dist, dpmf)
        elif kind == 'entropy':
            f = lambda dpmf: DiceUtil.dice_func_entropy(prob_dist, dpmf)
        elif kind == 'entropy_conditioned':
            f = lambda dpmf: DiceUtil.dice_func_entropy_conditioned(prob_dist, dpmf)
        elif kind == 'wasserstein':
            f = lambda dpmf: DiceUtil.dice_func_wasserstein(prob_dist, dpmf)
        else:
            raise ValueError('dice_infer: Invalid "kind" argument')
        return f

    def print_dist(prob_dist):
        for k, v in sorted(prob_dist.items()):
            print(f'\t{k}: {v:.4f}')
        return ''
