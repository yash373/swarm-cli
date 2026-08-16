import math
import statistics
from typing import List, Tuple, Union

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

def add(a: Number, b: Number) -> Number:
    """
    Add two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The sum of the two numbers.
    """
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """
    Subtract two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The difference of the two numbers.
    """
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """
    Multiply two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The product of the two numbers.
    """
    return a * b


def divide(a: Number, b: Number) -> float:
    """
    Divide two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The quotient of the two numbers.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def modulo(a: Number, b: Number) -> Number:
    """
    Compute the remainder of a divided by b.
    Args:
        a: The dividend.
        b: The divisor.
    Returns:
        The remainder of a / b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a % b


def floor_divide(a: Number, b: Number) -> int:
    """
    Compute the floor (integer) division of a by b.
    Args:
        a: The dividend.
        b: The divisor.
    Returns:
        The largest integer less than or equal to a / b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a // b


def absolute_value(a: Number) -> Number:
    """
    Compute the absolute value of a number.
    Args:
        a: The number.
    Returns:
        The absolute value of a.
    """
    return abs(a)


def negate(a: Number) -> Number:
    """
    Negate a number.
    Args:
        a: The number.
    Returns:
        The negation of a.
    """
    return -a


def round_number(a: Number, digits: int = 0) -> Number:
    """
    Round a number to a given number of decimal places.
    Args:
        a: The number to round.
        digits: The number of decimal places (default 0).
    Returns:
        The rounded number.
    """
    return round(a, digits)


def floor(a: Number) -> int:
    """
    Round a number down to the nearest integer.
    Args:
        a: The number.
    Returns:
        The floor of a.
    """
    return math.floor(a)


def ceiling(a: Number) -> int:
    """
    Round a number up to the nearest integer.
    Args:
        a: The number.
    Returns:
        The ceiling of a.
    """
    return math.ceil(a)


def truncate(a: Number) -> int:
    """
    Truncate a number toward zero, discarding the fractional part.
    Args:
        a: The number.
    Returns:
        The truncated integer part of a.
    """
    return math.trunc(a)


def clamp(value: Number, minimum: Number, maximum: Number) -> Number:
    """
    Clamp a value between a minimum and maximum bound.
    Args:
        value: The value to clamp.
        minimum: The lower bound.
        maximum: The upper bound.
    Returns:
        The value clamped to the range [minimum, maximum].
    """
    if minimum > maximum:
        raise ValueError("minimum must be less than or equal to maximum.")
    return max(minimum, min(value, maximum))


def sign(a: Number) -> int:
    """
    Determine the sign of a number.
    Args:
        a: The number.
    Returns:
        -1 if a is negative, 0 if a is zero, 1 if a is positive.
    """
    return (a > 0) - (a < 0)


# ---------------------------------------------------------------------------
# Powers, roots, logarithms
# ---------------------------------------------------------------------------

def exponentiate(base: Number, exponent: Number) -> float:
    """
    Raise a base to the power of an exponent.
    Args:
        base: The base number.
        exponent: The exponent to raise the base to.
    Returns:
        The result of raising the base to the specified exponent.
    """
    return math.pow(base, exponent)


def square_root(value: Number) -> float:
    """
    Calculate the square root of a value.
    Args:
        value: The value to take the square root of.
    Returns:
        The square root of the value.
    """
    if value < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    return math.sqrt(value)


def cube_root(value: Number) -> float:
    """
    Calculate the cube root of a value.
    Args:
        value: The value to take the cube root of.
    Returns:
        The cube root of the value.
    """
    return math.copysign(abs(value) ** (1 / 3), value)


def nth_root(value: Number, n: Number) -> float:
    """
    Calculate the nth root of a value.
    Args:
        value: The value to take the root of.
        n: The degree of the root.
    Returns:
        The nth root of the value.
    """
    if n == 0:
        raise ValueError("Root degree cannot be zero.")
    if value < 0 and n % 2 == 0:
        raise ValueError("Cannot take an even root of a negative number.")
    return math.copysign(abs(value) ** (1 / n), value) if value < 0 else value ** (1 / n)


def logarithm(base: float, value: float) -> float:
    """
    Calculate the logarithm of a value with a given base.
    Args:
        base: The base of the logarithm.
        value: The value to calculate the logarithm for.
    Returns:
        The logarithm of the value with the specified base.
    """
    if base <= 0 or base == 1:
        raise ValueError("Base must be greater than 0 and not equal to 1.")
    if value <= 0:
        raise ValueError("Value must be greater than 0.")
    return math.log(value, base)


def natural_log(value: float) -> float:
    """
    Calculate the natural logarithm (base e) of a value.
    Args:
        value: The value to calculate the natural log for.
    Returns:
        The natural logarithm of the value.
    """
    if value <= 0:
        raise ValueError("Value must be greater than 0.")
    return math.log(value)


def log10(value: float) -> float:
    """
    Calculate the base-10 logarithm of a value.
    Args:
        value: The value to calculate the log for.
    Returns:
        The base-10 logarithm of the value.
    """
    if value <= 0:
        raise ValueError("Value must be greater than 0.")
    return math.log10(value)


def log2(value: float) -> float:
    """
    Calculate the base-2 logarithm of a value.
    Args:
        value: The value to calculate the log for.
    Returns:
        The base-2 logarithm of the value.
    """
    if value <= 0:
        raise ValueError("Value must be greater than 0.")
    return math.log2(value)


def exp(value: float) -> float:
    """
    Calculate e raised to the power of a value.
    Args:
        value: The exponent.
    Returns:
        e raised to the power of value.
    """
    return math.exp(value)


# ---------------------------------------------------------------------------
# Trigonometry (degrees in, degrees where inverse)
# ---------------------------------------------------------------------------

def sine(angle: float) -> float:
    """
    Calculate the sine of an angle in degrees.
    Args:
        angle: The angle in degrees.
    Returns:
        The sine of the angle.
    """
    return math.sin(math.radians(angle))


def cosine(angle: float) -> float:
    """
    Calculate the cosine of an angle in degrees.
    Args:
        angle: The angle in degrees.
    Returns:
        The cosine of the angle.
    """
    return math.cos(math.radians(angle))


def tangent(angle: float) -> float:
    """
    Calculate the tangent of an angle in degrees.
    Args:
        angle: The angle in degrees.
    Returns:
        The tangent of the angle.
    """
    return math.tan(math.radians(angle))


def arcsine(value: float) -> float:
    """
    Calculate the inverse sine (arcsine) of a value, returned in degrees.
    Args:
        value: The value, must be between -1 and 1 inclusive.
    Returns:
        The angle in degrees whose sine is value.
    """
    if not -1 <= value <= 1:
        raise ValueError("Value must be between -1 and 1.")
    return math.degrees(math.asin(value))


def arccosine(value: float) -> float:
    """
    Calculate the inverse cosine (arccosine) of a value, returned in degrees.
    Args:
        value: The value, must be between -1 and 1 inclusive.
    Returns:
        The angle in degrees whose cosine is value.
    """
    if not -1 <= value <= 1:
        raise ValueError("Value must be between -1 and 1.")
    return math.degrees(math.acos(value))


def arctangent(value: float) -> float:
    """
    Calculate the inverse tangent (arctangent) of a value, returned in degrees.
    Args:
        value: The value.
    Returns:
        The angle in degrees whose tangent is value.
    """
    return math.degrees(math.atan(value))


def arctangent2(y: float, x: float) -> float:
    """
    Calculate the angle (in degrees) between the positive x-axis and the
    point (x, y), handling all four quadrants correctly.
    Args:
        y: The y-coordinate.
        x: The x-coordinate.
    Returns:
        The angle in degrees, between -180 and 180.
    """
    return math.degrees(math.atan2(y, x))


def degrees_to_radians(angle: float) -> float:
    """
    Convert an angle from degrees to radians.
    Args:
        angle: The angle in degrees.
    Returns:
        The angle in radians.
    """
    return math.radians(angle)


def radians_to_degrees(angle: float) -> float:
    """
    Convert an angle from radians to degrees.
    Args:
        angle: The angle in radians.
    Returns:
        The angle in degrees.
    """
    return math.degrees(angle)


def hypotenuse(a: float, b: float) -> float:
    """
    Calculate the length of the hypotenuse of a right triangle given the
    two other side lengths.
    Args:
        a: The length of one side.
        b: The length of the other side.
    Returns:
        The length of the hypotenuse.
    """
    return math.hypot(a, b)


# ---------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.
    Args:
        n: The number to calculate the factorial of.
    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)


def gcd(a: int, b: int) -> int:
    """
    Calculate the greatest common divisor of two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The greatest common divisor of a and b.
    """
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """
    Calculate the least common multiple of two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The least common multiple of a and b.
    """
    return math.lcm(a, b)


def is_prime(n: int) -> bool:
    """
    Determine whether an integer is a prime number.
    Args:
        n: The integer to test.
    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_factors(n: int) -> List[int]:
    """
    Compute the prime factorization of a positive integer.
    Args:
        n: The integer to factorize.
    Returns:
        A list of prime factors of n, in ascending order (with repetition).
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def divisors(n: int) -> List[int]:
    """
    List all positive divisors of a positive integer.
    Args:
        n: The integer to find divisors of.
    Returns:
        A sorted list of all positive divisors of n.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    small, large = [], []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
    return sorted(small + large)


def is_perfect_square(n: int) -> bool:
    """
    Determine whether an integer is a perfect square.
    Args:
        n: The integer to test.
    Returns:
        True if n is a perfect square, False otherwise.
    """
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number (0-indexed, fib(0) = 0, fib(1) = 1).
    Args:
        n: The index of the Fibonacci number to calculate.
    Returns:
        The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def integer_square_root(n: int) -> int:
    """
    Calculate the integer square root (floor of the square root) of a
    non-negative integer.
    Args:
        n: The integer to take the square root of.
    Returns:
        The largest integer whose square is less than or equal to n.
    """
    if n < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    return math.isqrt(n)


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def permutations_count(n: int, r: int) -> int:
    """
    Calculate the number of ways to arrange r items out of n distinct items,
    where order matters.
    Args:
        n: The total number of items.
        r: The number of items to arrange.
    Returns:
        The number of permutations, nPr.
    """
    if r < 0 or n < 0 or r > n:
        raise ValueError("Require 0 <= r <= n.")
    return math.perm(n, r)


def combinations_count(n: int, r: int) -> int:
    """
    Calculate the number of ways to choose r items out of n distinct items,
    where order does not matter.
    Args:
        n: The total number of items.
        r: The number of items to choose.
    Returns:
        The number of combinations, nCr.
    """
    if r < 0 or n < 0 or r > n:
        raise ValueError("Require 0 <= r <= n.")
    return math.comb(n, r)


# ---------------------------------------------------------------------------
# Statistics (operate on lists of numbers)
# ---------------------------------------------------------------------------

def sum_list(values: List[Number]) -> Number:
    """
    Compute the sum of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The sum of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return sum(values)


def product_list(values: List[Number]) -> Number:
    """
    Compute the product of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The product of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return math.prod(values)


def mean(values: List[Number]) -> float:
    """
    Compute the arithmetic mean (average) of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The arithmetic mean of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return statistics.mean(values)


def median(values: List[Number]) -> float:
    """
    Compute the median of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The median of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return statistics.median(values)


def mode(values: List[Number]) -> Number:
    """
    Compute the mode (most common value) of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The mode of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return statistics.mode(values)


def variance(values: List[Number], population: bool = False) -> float:
    """
    Compute the variance of a list of numbers.
    Args:
        values: The list of numbers.
        population: If True, compute population variance; otherwise sample
            variance (default).
    Returns:
        The variance of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return statistics.pvariance(values) if population else statistics.variance(values)


def standard_deviation(values: List[Number], population: bool = False) -> float:
    """
    Compute the standard deviation of a list of numbers.
    Args:
        values: The list of numbers.
        population: If True, compute population standard deviation;
            otherwise sample standard deviation (default).
    Returns:
        The standard deviation of the values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return statistics.pstdev(values) if population else statistics.stdev(values)


def minimum(values: List[Number]) -> Number:
    """
    Find the minimum value in a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The smallest value in the list.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return min(values)


def maximum(values: List[Number]) -> Number:
    """
    Find the maximum value in a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The largest value in the list.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return max(values)


def range_of(values: List[Number]) -> Number:
    """
    Compute the range (max - min) of a list of numbers.
    Args:
        values: The list of numbers.
    Returns:
        The difference between the largest and smallest values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    return max(values) - min(values)


def percentage(part: Number, whole: Number) -> float:
    """
    Compute what percentage `part` is of `whole`.
    Args:
        part: The partial amount.
        whole: The total amount.
    Returns:
        The percentage that part represents of whole.
    """
    if whole == 0:
        raise ValueError("whole cannot be zero.")
    return (part / whole) * 100


def percentage_change(old_value: Number, new_value: Number) -> float:
    """
    Compute the percentage change from an old value to a new value.
    Args:
        old_value: The original value.
        new_value: The new value.
    Returns:
        The percentage change from old_value to new_value.
    """
    if old_value == 0:
        raise ValueError("old_value cannot be zero.")
    return ((new_value - old_value) / old_value) * 100


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------

def solve_linear_equation(a: float, b: float) -> float:
    """
    Solve a linear equation of the form a*x + b = 0 for x.
    Args:
        a: The coefficient of x.
        b: The constant term.
    Returns:
        The solution x = -b / a.
    """
    if a == 0:
        raise ValueError("a cannot be zero (not a linear equation in x).")
    return -b / a


def solve_quadratic_equation(a: float, b: float, c: float) -> Tuple[Union[float, complex], Union[float, complex]]:
    """
    Solve a quadratic equation of the form a*x^2 + b*x + c = 0.
    Args:
        a: The coefficient of x^2.
        b: The coefficient of x.
        c: The constant term.
    Returns:
        A tuple of the two roots (real or complex) of the equation.
    """
    if a == 0:
        raise ValueError("a cannot be zero (not a quadratic equation).")
    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-b + sqrt_disc) / (2 * a)
        root2 = (-b - sqrt_disc) / (2 * a)
    else:
        sqrt_disc = complex(0, math.sqrt(-discriminant))
        root1 = (-b + sqrt_disc) / (2 * a)
        root2 = (-b - sqrt_disc) / (2 * a)
    return root1, root2


def evaluate_polynomial(coefficients: List[Number], x: Number) -> Number:
    """
    Evaluate a polynomial at a given value of x.
    Args:
        coefficients: A list of coefficients, highest degree first
            (e.g. [1, -3, 2] represents x^2 - 3x + 2).
        x: The value at which to evaluate the polynomial.
    Returns:
        The value of the polynomial at x.
    """
    if not coefficients:
        raise ValueError("coefficients must not be empty.")
    result = 0
    for coeff in coefficients:
        result = result * x + coeff
    return result


def weighted_average(values: List[Number], weights: List[Number]) -> float:
    """
    Compute the weighted average of a list of values.
    Args:
        values: The list of values.
        weights: The list of weights corresponding to each value.
    Returns:
        The weighted average.
    """
    if not values or not weights:
        raise ValueError("values and weights must not be empty.")
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length.")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero.")
    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def circle_area(radius: float) -> float:
    """
    Calculate the area of a circle.
    Args:
        radius: The radius of the circle.
    Returns:
        The area of the circle.
    """
    if radius < 0:
        raise ValueError("radius cannot be negative.")
    return math.pi * radius ** 2


def circle_circumference(radius: float) -> float:
    """
    Calculate the circumference of a circle.
    Args:
        radius: The radius of the circle.
    Returns:
        The circumference of the circle.
    """
    if radius < 0:
        raise ValueError("radius cannot be negative.")
    return 2 * math.pi * radius


def rectangle_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.
    Args:
        length: The length of the rectangle.
        width: The width of the rectangle.
    Returns:
        The area of the rectangle.
    """
    if length < 0 or width < 0:
        raise ValueError("length and width cannot be negative.")
    return length * width


def triangle_area(base: float, height: float) -> float:
    """
    Calculate the area of a triangle given its base and height.
    Args:
        base: The length of the base.
        height: The height relative to the base.
    Returns:
        The area of the triangle.
    """
    if base < 0 or height < 0:
        raise ValueError("base and height cannot be negative.")
    return 0.5 * base * height


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """
    Calculate the area of a triangle given its three side lengths, using
    Heron's formula.
    Args:
        a: The length of the first side.
        b: The length of the second side.
        c: The length of the third side.
    Returns:
        The area of the triangle.
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Side lengths must be positive.")
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("The given sides do not form a valid triangle.")
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def distance_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.
    Args:
        x1: The x-coordinate of the first point.
        y1: The y-coordinate of the first point.
        x2: The x-coordinate of the second point.
        y2: The y-coordinate of the second point.
    Returns:
        The Euclidean distance between the two points.
    """
    return math.hypot(x2 - x1, y2 - y1)


def sphere_volume(radius: float) -> float:
    """
    Calculate the volume of a sphere.
    Args:
        radius: The radius of the sphere.
    Returns:
        The volume of the sphere.
    """
    if radius < 0:
        raise ValueError("radius cannot be negative.")
    return (4 / 3) * math.pi * radius ** 3


def sphere_surface_area(radius: float) -> float:
    """
    Calculate the surface area of a sphere.
    Args:
        radius: The radius of the sphere.
    Returns:
        The surface area of the sphere.
    """
    if radius < 0:
        raise ValueError("radius cannot be negative.")
    return 4 * math.pi * radius ** 2


def cylinder_volume(radius: float, height: float) -> float:
    """
    Calculate the volume of a right circular cylinder.
    Args:
        radius: The radius of the base.
        height: The height of the cylinder.
    Returns:
        The volume of the cylinder.
    """
    if radius < 0 or height < 0:
        raise ValueError("radius and height cannot be negative.")
    return math.pi * radius ** 2 * height


# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------

def simple_interest(principal: float, rate: float, time: float) -> float:
    """
    Calculate simple interest.
    Args:
        principal: The initial amount of money.
        rate: The annual interest rate, as a decimal (e.g. 0.05 for 5%).
        time: The time period in years.
    Returns:
        The simple interest earned.
    """
    if principal < 0 or time < 0:
        raise ValueError("principal and time cannot be negative.")
    return principal * rate * time


def compound_interest(principal: float, rate: float, times_compounded: int, time: float) -> float:
    """
    Calculate the final amount after compound interest is applied.
    Args:
        principal: The initial amount of money.
        rate: The annual interest rate, as a decimal (e.g. 0.05 for 5%).
        times_compounded: The number of times interest is compounded per year.
        time: The time period in years.
    Returns:
        The final amount after compound interest.
    """
    if principal < 0 or time < 0 or times_compounded <= 0:
        raise ValueError("principal and time cannot be negative, and times_compounded must be positive.")
    return principal * (1 + rate / times_compounded) ** (times_compounded * time)