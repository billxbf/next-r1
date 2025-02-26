
try:
    from sympy import *

    import sympy as sp
    
    # Define the symbols with the condition they are positive
    x, y = sp.symbols('x y', positive=True)
    
    # Define the system of equations
    eq1 = sp.Eq(y**3, x**2)
    eq2 = sp.Eq((y - x)**2, 4*y**2)
    
    # Solve the system of equations
    solution = sp.solve((eq1, eq2), (x, y), dict=True)
    
    # From the solutions, pick the one where both x and y are positive.
    # We know that the valid solution comes from (y - x)^2 = 4y^2 implies x = 3y.
    sol = solution[0]  # Taking the first solution from the list
    x_val = sol[x]
    y_val = sol[y]
    
    # Compute x + y
    result = sp.simplify(x_val + y_val)
    print(result)
    
except Exception as e:
    print(e)
    print('FAIL')
