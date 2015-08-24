"""01-HW1: Python primer."""
import math
import numpy

def nth_prime(n):
    primes = [True]*1000000

    for x in numpy.arange(0,1000000,1):
        if x % 2 == 0 :
            primes[x] = False

    primes[1] = False
    primes[2] = True
    count = 0

    arr = numpy.arange(3,31623,2)
    for x in arr:
        if primes[x] :
            for i in numpy.arange(x*x, 1000000, x) :
                    primes[i] = False

    for idx,val in enumerate(primes):
        if val :
            count = count +1
            if count == n :
                return idx
    if n == 1 : return 2

def test_run():
    """Driver function called by Test Run."""
    print nth_prime(1)  # should print 2
    print nth_prime(22)


if __name__ == "__main__":
    test_run()
