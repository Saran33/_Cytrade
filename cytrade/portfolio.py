from sympy import symbols, Eq, solve


def get_target(target, value):
    '''Get the necessary order quantity to
    rebalance a position to a specific target.'''
    trgt = symbols('trgt')
    expr = trgt + value - target
    eq1 = Eq(expr, 0)
    sol = solve(eq1)
    return float(sol[0])

