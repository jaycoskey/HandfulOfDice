#!/usr/bin/env python


from numpy.linalg import norm as norm2
import sys


class Util:
    def enter(funcname):
        print(f'== Entering {funcname} ==')

    def exit(funcname):
        print(f'== Exiting {funcname} ==')

    def normalized_dict(d):
        dict_norm = norm2(list(d.values()))
        return {k: v/dict_norm for k, v in d.items()}

    def normalized_vec(vec):
        return vec / norm2(vec)

    def plot_vhist(self, hist, height=10):
        '''Print a text version of a vertically-oriented histogram'''
        FUNCNAME = sys._getframe().f_code.co_name
        Util.enter(FUNCNAME)

        max_val = max([v for k, v in hist.items()])
        for vert_div in reversed(list(range(height))):
            for k, v in hist.items():
                if v > (vert_div / height) * max_val:
                    print('*', end='')
                else:
                    print(' ', end='')
                if vert_div == 0 and v == 0:
                    print('_', end='')
            print()
        Util.exit(FUNCNAME)
