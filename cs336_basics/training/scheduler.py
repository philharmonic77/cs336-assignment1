import math  

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    assert t >= 0
    assert alpha_max >= alpha_min
    assert T_c > T_w >= 0

    # Warm-up
    if T_w > 0 and t < T_w:
        return t / T_w * alpha_max
    # Cosine annealin
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    # Post-annealing
    else:
        return alpha_min
    
    



