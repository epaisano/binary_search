#!/bin/python3

import random


def find_smallest_positive(xs):
    '''
    Assume that xs is a list of numbers sorted from LOWEST to HIGHEST.
    Find the index of the smallest positive number.
    If no such index exists, return `None`.

    >>> find_smallest_positive([-3, -2, -1, 0, 1, 2, 3])
    4
    >>> find_smallest_positive([1, 2, 3])
    0
    >>> find_smallest_positive([-3, -2, -1]) is None
    True
    '''
    if len(xs) == 0:
        return None

    if xs[0] < 0 and xs[-1] < 0:
        return None

    def go(left, right):
        if left == right:
            if xs[left] == 0:
                return None
            if xs[left] > 0:
                return left

        mid = (left + right) // 2
        if xs[mid] > 0:
            right = mid
        if xs[mid] < 0:
            left = mid + 1
        if xs[mid] == 0:
            return mid + 1
            left = mid + 1
        return go(left, right)

    return go(0, len(xs) - 1)


def bin_lowindex(xs, x):
    '''
    Assume xs is a list of numbers sorted from HIGHEST to LOWEST,
    and that x is a number.
    Find the lowest index with a value >= x.

    >>> bin_lowindex([5, 4, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0], 5)
    0
    >>> bin_lowindex([5, 4, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0], 0)
    8
    '''
    left = 0
    right = len(xs) - 1
    pos = -1

    while (left <= right):
        mid = (left + right) // 2
        if xs[mid] > x:
            left = mid + 1
        elif xs[mid] < x:
            right = mid - 1

        else:
            pos = mid
            right = mid - 1
    return pos


def bin_highindex(xs, x):
    '''
    Assume xs is a list of numbers sorted from HIGHEST to LOWEST,
    and that x is a number.
    Find the lowest index with a value >= x.

    >>> bin_highindex([5, 4, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0], 2)
    8
    >>> bin_highindex([5, 4, 3, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0], 0)
    12
    '''
    left = 0
    right = len(xs) - 1
    pos = -1
    while (left <= right):
        mid = (left + right) // 2

        if xs[mid] > x:
            left = mid + 1
        elif xs[mid] < x:
            right = mid - 1

        else:
            pos = mid
            left = mid + 1
    return pos


def count_repeats(xs, x):
    '''
    Assume that xs is a list of numbers sorted from HIGHEST to LOWEST,
    and that x is a number.
    Calculate the number of times that x occurs in xs.

    >>> count_repeats([5, 4, 3, 3, 3, 3, 3, 3, 3, 2, 1], 3)
    7
    >>> count_repeats([3, 2, 1], 4)
    0
    '''
    if x not in xs:
        return 0

    upper = bin_highindex(xs, x)
    lower = bin_lowindex(xs, x)
    if xs[0] == xs[-1]:
        return len(xs)
    if upper == lower:
        return 1
    if lower == 0:
        return upper - lower
    else:
        newupper = upper + 1
        return newupper - lower


def argmin(f, lo, hi, epsilon=1e-3):
    '''
    Assumes that f is an input function that takes a float as
    input and returns a float with a unique global minimum,
    and that lo and hi are both floats satisfying lo < hi.
    Returns a number that is within epsilon of the value that
    minimizes f(x) over the interval [lo,hi]

    >>> argmin(lambda x: (x-5)**2, -20, 20)
    5.000040370009773
    >>> argmin(lambda x: (x-5)**2, -20, 0)
    -0.00016935087808430278
    '''

    def find_min(lo, hi):

        if (hi - lo) < epsilon:
            return lo

        m1 = random.uniform(lo, ((hi + lo) / 2))
        m2 = random.uniform(((lo + hi) / 2), hi)
        xs = [f(lo), f(m1), f(m2), f(hi)]

        if min(xs) == xs[0]:
            return find_min(lo, m2)

        if min(xs) == xs[1]:
            return find_min(lo, m2)

        if min(xs) == xs[2]:
            return find_min(m1, hi)

        if min(xs) == xs[3]:
            return find_min(m1, hi)

        return find_min(lo, hi)
    return find_min(lo, hi)


######################################
# the functions below are extra credit
#####################################

def find_boundaries(f):
    '''
    Returns a tuple (lo,hi).
    If f is a convex function, then the minimum is
    guaranteed to be between lo and hi.
    This function is useful for initializing argmin.

    HINT:
    Begin with initial values lo=-1, hi=1.
    Let mid = (lo+hi)/2
    if f(lo) > f(mid):
        recurse with lo*=2
    elif f(hi) < f(mid):
        recurse with hi*=2
    else:
        you're done; return lo,hi
    '''

    def go(lo, hi):
        mid = (lo+hi)/2
        if f(lo) < f(mid):
            lo = lo*2
            return go(lo, hi)
        elif f(hi) < f(mid):
            hi = hi*2
            return go(lo,hi)

        else:
            return lo,hi
    return go(-1,1)


'''    
    lo = -1 
    hi = 1

    while (lo <= hi):
        mid = (lo + hi) / 2
        if (f(lo) < f(mid)):
            lo *= 2
        elif (f(hi) < f(mid)):
            hi *= 2

        else:
            return lo,hi
        return lo,hi
    return lo,hi
'''

def argmin_simple(f, epsilon=1e-3):
    '''
    This function is like argmin, but it internally
    uses the find_boundaries function so that
    you do not need to specify lo and hi.

    NOTE:
    There is nothing to implement for this function.
    If you implement the find_boundaries function correctly,
    then this function will work correctly too.
    '''
    lo, hi = find_boundaries(f)
    return argmin(f, lo, hi, epsilon)
