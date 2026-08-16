from worker import Worker

worker1 = Worker("qwen3:4b")
worker1.respond(
    """
    A cylindrical water tank has a radius equal to the smallest prime factor of 8,190, and a height equal to the number of ways to arrange 3 items chosen from a set of 7 distinct items (order matters).
    Compute the tank's volume.
    The tank is being filled by a pump whose flow rate (liters/hour) equals the positive root of the quadratic equation 2x² - 11x - 63 = 0.
    Using that flow rate, calculate how many hours it takes to fill the tank (volume in liters — treat 1 unit³ = 1 liter).
    A technician records 5 different fill-time trial runs due to pressure fluctuations: [trial_hours, trial_hours × 1.1, trial_hours × 0.85, trial_hours × 1.05, trial_hours × 0.95] where trial_hours is your answer from step 3. Compute the mean, median, and sample standard deviation of these 5 trials.
    Finally, the tank must be insured. The insurance premium formula is: base premium of ₹5,000 compounded annually at a rate equal to arctangent(height/radius) / 100 (as a decimal rate), for a term of 3 years. Compute the final insured value.
    """
)


# print(dir(arithmetic))