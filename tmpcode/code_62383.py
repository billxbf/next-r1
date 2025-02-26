
try:
    from sympy import *

    import sympy as sp
    
    # Define the slopes as sympy rationals
    m1 = sp.Rational(2)
    m2 = sp.Rational(1, 3)
    
    # Compute the absolute value of (m1 - m2) / (1 + m1*m2)
    ratio = abs((m1 - m2) / (1 + m1 * m2))
    
    # Compute the angle in radians using atan, then convert to degrees
    angle_rad = sp.atan(ratio)
    angle_deg = angle_rad * 180 / sp.pi
    
    # Convert to integer (since the answer is an integer)
    result = int(angle_deg)
    
    print(result)
    
except Exception as e:
    print(e)
    print('FAIL')
