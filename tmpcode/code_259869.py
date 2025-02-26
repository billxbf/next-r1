
try:
    from sympy import *

    from sympy import symbols, summation
    
    # Define the variable k
    k = symbols('k', integer=True, positive=True)
    
    # The expression corresponds to the sum:
    # (2^3 - 1^3) + (4^3 - 3^3) + ... + (18^3 - 17^3)
    # which can be written as sum for k = 1 to 9 of [(2k)^3 - (2k - 1)^3]
    expr = summation((2*k)**3 - (2*k - 1)**3, (k, 1, 9))
    
    print(expr)  # This will output: 3159
    
except Exception as e:
    print(e)
    print('FAIL')
