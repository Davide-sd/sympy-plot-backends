from sympy import (
    Basic,
    Function,
    Number,
    frac,
    pi,
    sstr,
    S,
    Symbol,
    symbols,
    sin,
    cos,
    log,
)


def build_example():
    r, ro, ri = symbols("r, r_o, r_i")
    mdot, cp, hc = symbols(r"\dot{m}, c_p, h_c")
    alpha, k, L, z = symbols("alpha, k, L, z")
    Tin, Pave = symbols(r"T_{in}, P_{ave}")

    # Fuel temperature distribution along the channel
    # here, the only variable is z, everything else are parameters
    Tf = (
        Tin
        + (Pave * L * pi * (ro ** 2 - ri ** 2) / (2 * mdot * cp))
        * (1 - sin(alpha * (L / 2 - z)) / sin(alpha * L / 2))
        + (alpha * Pave * L / 2)
        * (cos(alpha * (L / 2 - z)) / sin(alpha * L / 2))
        * (
            (ro ** 2 - ri ** 2) / (2 * hc * ri)
            - (1 / (2 * k)) * ((r ** 2 - ri ** 2) / 2 + ro ** 2 * log(ri / r))
        )
    )
    # Fuel temperature distribution at the inner and outer walls
    Twi = Tf.subs(r, ri)
    Two = Tf.subs(r, ro)
    # Cooling fluid temperature
    Tp = Tin + (Pave * L / 2) * pi * (ro ** 2 - ri ** 2) / (mdot * cp) * (
        1 - sin(alpha * (L / 2 - z)) / sin(alpha * L / 2)
    )
    # Power
    P = (alpha * Pave * L / (2 * sin(alpha * L / 2))) * cos(alpha * (L / 2 - z))

    return [ro, ri, mdot, cp, hc, alpha, k, L, z, Tin, Pave, Twi, Two, Tp, P]
