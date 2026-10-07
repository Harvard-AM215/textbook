---
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 9: Optimal Racing Strategy

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harvard-AM215/textbook/blob/gh-pages/notebooks/09_optimal_racing.ipynb)
*Runs in the browser -- click the power icon above to start the kernel -- or in Colab, which has a full Python and a live network.*

> A runner has limited force and a limited energy supply, so the fastest pacing strategy depends on the distance: this simple model explains why short races favor an all-out sprint while longer races favor a quick start, a steady pace, and a brief slowdown at the finish.

:::{note} Running the code
The Python cells on this page run **in your browser**. Click the **power icon** at the top of the page to activate the kernel, then run — or edit — any cell. Packages (NumPy, SciPy, …) load automatically the first time you import them (a few seconds).
:::

## Motivation

In 1926, the physiologist A. V. Hill pointed out that running world records are also physiological data. In the historical record table we will use, race times range from about 5 seconds for 50 yards to 27 minutes for 10 000 metres, while average speed falls from about 10 m/s at 200 m to 6 m/s at 10 km. A runner cannot hold a sprint for a mile; the records quantify how much the pace must fall as the distance grows. Hill explained this trend with a simple energy budget: the runner maintains a constant speed chosen to exhaust the available energy exactly at the finish line. That argument accounts for the energy used, but not for how the runner gets up to speed or changes pace. A sprinter starts from rest and accelerates hard during the opening seconds. To describe that motion, we need dynamics. And to choose a strategy, we need to ask how the effort should be spread over the race: how hard should the runner start, when should they settle, and should they save anything for the finish?

Joseph Keller added the dynamics in 1973. His model couples Newton's second law to an energy balance. The runner controls the propulsive force $f(t)$, subject to two limits: the muscles can deliver at most a force $F$, and the body cannot spend energy it does not have. The task is to choose how that force varies over time so that the runner covers a given distance as quickly as possible. This is an **optimal control problem**: the unknown is a whole function, not a single number.

The predicted strategy depends on the distance. Below a critical distance, the runner pushes at full force from start to finish. Longer races generally have three phases: an opening sprint, a cruise at constant speed, and a brief slowdown at the end. Just beyond the critical distance, the cruise is absent. The model also lets us work backward from performance to physiology, as Hill proposed: fitting its four constants to the records estimates the force and energy supply needed to account for those performances. Because the records come from different runners, these are effective parameters for the record table rather than measurements of one athlete.

Chapter 8 modeled the largest of many random draws. This chapter studies a different kind of limit, the best performance allowed by a runner's force and energy supply, using optimization rather than probability. The worked examples follow the modeling loop. Example 1 **verifies** the solver against Keller's published calculation and checks the three-phase strategy with a numerical optimization that does not assume those phases. Example 2 **estimates** the four constants from the 1972 records and examines how well the data distinguish different parameter choices. Example 3 **validates** predictions made with the historical constants against later world records, then compares them with Olympic finals, where winning the race can matter more than minimizing the time.

## Setup & notation

All quantities are per unit mass of runner, which removes body mass from the problem. The runner's state at time $t$ is described by the speed $v(t)$ and the available energy store $E(t)$. The runner controls the propulsive force per unit mass $f(t)$. We call it force for short, but its units are m/s$^2$.

New symbols in this chapter:

- $D$, $T$: the race distance and the time taken, with $D = \int_0^T v\,dt$.
- $\tau$: the **damping time**. A runner who stops pushing slows down with time constant $\tau$; a runner who pushes with a constant force $f$ approaches the terminal speed $f\tau$.
- $F$: the **maximum force** per unit mass the muscles can deliver, so $0 \le f(t) \le F$. The lower bound says the runner never pulls backward; Keller's papers omit it and his solution never needs it.
- $\sigma$: the **resupply rate**, the energy per unit mass per second that breathing and circulation deliver over and above resting metabolism.
- $E_0$: the **initial energy store**, the energy available at the gun.
- $t_1$, $t_2$: the times at which the sprint ends and at which the energy store runs out.
- $T_c$, $D_c$: the **critical time** and **critical distance**, the longest race that can be run flat out from start to finish; $T^*$: the race time above which all three phases of the long-race solution are present.

Four assumptions in the model:

- **Linear resistance.** The resistive force per unit mass is $v/\tau$ with constant $\tau$. At running speeds most of the resistance is internal, in the limbs and tissues, and Keller reports that a speed-independent resistance fits the records clearly worse. Air drag, which grows like $v^2$, dominates in cycling (Exercise 7).
- **One energy pool.** A single energy store $E$ holds everything the runner can draw on beyond the resupply. This combines several physiological processes that do not behave identically in a real runner.
- **Constant resupply.** $\sigma$ does not depend on the speed or on how long the race has lasted.
- **No reaction time.** The model's clock starts when the runner starts moving; an official time starts at the gun. We use 0.15 s as a representative reaction-time correction in the modern fit, and vary it in Exercise 5.

A word on units before any calculation. Keller published $\sigma = 9.93$ cal/(kg s) and $E_0 = 575$ cal/kg, in calories, while the mechanical quantities are in metres and seconds. Energy per unit mass has the units of a speed squared, $m^2/s^2$ = J/kg, and $1$ cal $= 4.184$ J, so
$$
\sigma = 9.93 \times 4.184 = 41.5\ \mathrm{W/kg}, \qquad E_0 = 575 \times 4.184 = 2406\ \mathrm{J/kg}.
$$
The model mixes $\sigma$ with $\tau$ and $v$: the speed $\sqrt{\sigma\tau}$ that appears below is $6.09$ m/s in SI units and a meaningless $2.98$ if the calories are left in. **Convert $\sigma$ to W/kg and $E_0$ to J/kg before combining them with the mechanical quantities.**

## Core ideas

### The model as an optimal control problem

The race is described by five relations. The distance is the integral of the speed,
$$
D = \int_0^T v(t)\,dt .
$$
Newton's second law, with a resistance proportional to speed, gives the equation of motion,
$$
\frac{dv}{dt} + \frac{v}{\tau} = f(t), \qquad v(0) = 0 ,
$$
which says the force the runner exerts goes partly into accelerating and partly into overcoming resistance; a runner at rest who pushes with the constant force $F$ reaches the speed $F\tau(1 - e^{-t/\tau})$, approaching the terminal speed $F\tau$ with time constant $\tau$. The first limit on the control is the strength of the muscles,
$$
0 \le f(t) \le F .
$$
The second limit comes from energy. Running at speed $v$ with force $f$ does work at the rate $fv$ per unit mass, in this simplified energy budget. The store $E$ is drained at that rate and refilled at the constant rate $\sigma$:
$$
\frac{dE}{dt} = \sigma - f v, \qquad E(0) = E_0, \qquad E(t) \ge 0 .
$$
The inequality is the physiological constraint: the runner cannot spend energy that is not available. Indirectly it is a second restriction on $f$.

**The problem.** Given $\tau$, $F$, $\sigma$, $E_0$ and the distance $D$, choose $f(t)$ so that the runner finishes in the shortest possible time while obeying the motion and energy equations. The resulting $f(t)$ determines the speed $v(t)$ and the remaining energy $E(t)$.

Instead of minimizing the time at a fixed distance, we can maximize the distance at a fixed time. The two problems trace out the same curve of $T$ against $D$: if a shorter time were possible at some distance, then in the original time the runner could have gone farther, and the other way round. The fixed-time version is also convenient for the numerical solver used below.

### Eliminating the force and the energy

Keller's first move is to express everything through the speed alone. The equation of motion defines the force, $f = \dot v + v/\tau$, so the strength limit becomes a condition on $v$:
$$
0 \le \frac{dv}{dt} + \frac{v}{\tau} \le F .
$$
Substituting the same expression into the energy balance gives $\dot E = \sigma - (\dot v + v/\tau)\,v$, and the product on the right is an exact derivative plus a square: $(\dot v + v/\tau)\,v = \tfrac{d}{dt}\!\left(\tfrac{v^2}{2}\right) + \tfrac{v^2}{\tau}$. Integrating from $0$ to $t$ with $E(0) = E_0$ and $v(0) = 0$,
$$
E(t) = \underbrace{E_0 + \sigma t}_{\text{supplied}} \;-\; \underbrace{\frac{v^2(t)}{2}}_{\text{kinetic energy now}} \;-\; \underbrace{\frac{1}{\tau}\int_0^t v^2(s)\,ds}_{\text{dissipated}} .
$$
Read it as an account: what has been supplied, minus what is currently held as motion, minus everything that the resistance has dissipated. The constraint $E \ge 0$ is now a condition on the function $v$ alone,
$$
E_0 + \sigma t - \frac{v^2(t)}{2} - \frac{1}{\tau}\int_0^t v^2(s)\,ds \;\ge\; 0 ,
$$
and the problem reads: find $v(t)$ with $v(0) = 0$, obeying the strength condition and this energy condition, that maximizes $D$ for the given $T$. The force and the energy can be recovered afterwards from $v$.

### Short races: full force to the line

Start with a short race in which energy is plentiful. Pushing harder gives a larger acceleration, so full force gives the greatest speed at every instant. It is therefore the fastest strategy as long as the energy store can support it. With $f = F$, the speed from rest is
$$
v(t)=F\tau\left(1-e^{-t/\tau}\right).
$$
At $t=0$ this gives zero speed, as it must. The speed then rises rapidly and approaches the top speed $F\tau$. Substituting it into the energy account gives the energy left during an all-out sprint:
$$
E_{\text{dash}}(t)
=E_0+\sigma t-F^2\tau^2\left(\frac{t}{\tau}+e^{-t/\tau}-1\right).
$$
At $t=0$ the formula returns $E_0$. At long times, the store is being spent at the net rate $F^2\tau-\sigma$. With Keller's constants this rate is positive, so the store eventually reaches zero. The time when that happens is the **critical time** $T_c$.

If the race time satisfies $T \le T_c$, the runner can take $t_1 = T$: full force the whole way, with the store staying nonnegative until the finish. The distance covered is then
$$
D = F\tau^2\left(\frac{T}{\tau} + e^{-T/\tau} - 1\right), \qquad T \le T_c ,
$$
and the longest such race is the **critical distance** $D_c$, this formula evaluated at $T_c$. A units check catches the commonest transcription error: $F\tau^2$ has units ($m/s^2$)($s^2$) = m, as a distance must, while $F^2\tau^2$ has units of $m^2/s^2$ and would give a critical distance of several kilometres. Two limiting cases check the formula itself. For $T \ll \tau$ the bracket is $\tfrac12 (T/\tau)^2$ and $D \approx \tfrac12 F T^2$, the distance under constant acceleration $F$ from rest; for $T \gg \tau$ the bracket is $T/\tau - 1$ and $D \approx F\tau\,(T - \tau)$, top speed for the whole race less the time lost getting there.

With Keller's constants, $T_c = 27.67$ s and $D_c = 291$ m. This places the 100 m and 200 m in the all-out regime and the 400 m in the regime where pacing matters. **The critical distance tells you which races can be run entirely at full force; it does not tell you how far to sprint at the start of a longer race.**

### Long races: sprint, cruise, fade

For $T > T_c$ the runner cannot hold $f = F$ to the finish. A quick start is still useful, but spending the whole store early would leave too much distance to cover on resupply alone. Keller proposes a compromise: accelerate hard, settle to a pace that uses the store gradually, and let it empty shortly before the finish. The runner then continues on the resupply while slowing down.

These three phases are often called **arcs**, intervals with different constraints controlling the motion. The switching times $t_1$ and $t_2$ separate them:

| phase | interval | active constraint | behavior |
|---|---|---|---|
| sprint | $0 \le t \le t_1$ | $f = F$ | $v$ rises as $F\tau(1 - e^{-t/\tau})$ |
| cruise | $t_1 \le t \le t_2$ | neither limit is active | constant speed |
| fade | $t_2 \le t \le T$ | $E = 0$ | $v$ falls toward the sustainable speed |

During the fade, the runner uses the incoming energy as it arrives, keeping the store empty. Setting $E(t) = 0$ in the account above and differentiating,
$$
\sigma - \frac12\frac{d(v^2)}{dt} - \frac{v^2}{\tau} = 0, \qquad t_2 \le t \le T ,
$$
a linear equation for $v^2$ whose solution from the value at $t_2$ is
$$
v^2(t) = \sigma\tau + \left[v^2(t_2) - \sigma\tau\right] e^{-2(t - t_2)/\tau}, \qquad t_2 \le t \le T .
$$
The extra squared speed above $\sigma\tau$ decays exponentially. The limiting speed $\sqrt{\sigma\tau}$ is the one at which resistance consumes energy at exactly the resupply rate. With Keller's constants it is $6.09$ m/s. **A runner can sustain this speed without drawing on the store; a higher constant speed spends the store.** On the fade the force is $f = \sigma/v$ (Exercise 2), so the runner is still pushing. The slowdown can be substantial in speed even though it lasts only a small part of the race.

There is a short transition between an all-out dash and a race with a cruise. Just above $T_c$ the race is too short for a cruise: the runner sprints until the store empties at $t_1 = t_2 = T_c$ and then fades to the line. Keller shows this happens for $T_c \le T \le T^*$ with
$$
T^* = T_c + \tau\left\{\log 2 - \tfrac12 \log\!\left[1 - \frac{\sigma}{F^2\tau}\left(1 - e^{-T_c/\tau}\right)^{-2}\right]\right\},
$$
which with his constants is $T^* = 28.46$ s, a window of $0.8$ s or about $7$ m of distance beyond $D_c$. Above $T^*$ a cruise appears, at first at a speed indistinguishable from top speed, and only as the race lengthens does the cruise speed fall clearly below $F\tau$.

### Why a steady cruise saves energy

The energy account explains why the middle of the race should be steady. Resistance dissipates energy at the rate $v^2/\tau$. If you alternate between a little faster and a little slower, you cover the same distance as at the average speed, but the faster part costs more energy than the slower part saves. For two equal time intervals at speeds $\bar v+\delta$ and $\bar v-\delta$, the average squared speed is
$$
\frac{(\bar v+\delta)^2+(\bar v-\delta)^2}{2}
=\bar v^2+\delta^2.
$$
The extra term is positive whenever the speed varies. Away from the start and finish, smoothing out those variations reduces the energy spent on resistance. You can then use the saved energy to maintain a higher average speed.

The start is different because the runner must accelerate from rest. The finish is different because no energy needs to be saved for running beyond the line. Those boundary effects give the sprint and fade around the steady cruise. Keller's mathematical calculation makes this physical argument precise; we use its three-phase form here and check it numerically in Example 1. **Even pacing saves energy during the main part of the race, while the force limit and the final energy balance determine how that pace begins and ends.**

### The long-distance limit: Hill's constant-speed approximation

For $D \gg D_c$ the sprint and the fade occupy a vanishing fraction of the race. Neglecting them, and the kinetic energy held at the line, the energy account at $t = T$ reads $E_0 + \sigma T = v^2 T/\tau$ with $v = D/T$, so
$$
\left(\frac{D}{T}\right)^2 \approx \sigma\tau + \frac{\tau E_0}{T}, \qquad D \gg D_c .
$$
This recovers Hill's energy-budget argument: the square of the average speed is approximately a straight line in $1/T$, with intercept $\sigma\tau$ and slope $\tau E_0$. In a shorter race the initial store contributes strongly to the average power available. In a very long race its contribution per second becomes small, and the average speed approaches $\sqrt{\sigma\tau}$.

The approximation neglects the start and finish. With Keller's constants, the full solver gives a long-race slope about $1\%$ below $\tau E_0$ ([companion script](./code/09_optimal_racing/hill_asymptote.py)). The main physical conclusion survives: the store supports speed above the sustainable level, but its benefit is spread more thinly over a longer race.

### Reducing the solve to one number

Once we adopt the sprint–cruise–fade pattern, we only need to choose how long to sprint. A choice of $t_1$ sets the cruise speed,
$$
v_1=F\tau(1-e^{-t_1/\tau}).
$$
For a cruise above the sustainable speed, the store drains at a constant rate, so the energy balance tells us when it empties, $t_2$. The fade formula then tells us how far the runner travels in the time remaining. Adding the three distances gives $D(t_1)$ for a fixed race duration $T$.

The computer tries a range of sprint durations and refines the best one. A longer sprint does not always give a greater total distance: it raises the cruise speed but empties the store sooner. **The best sprint duration balances the distance gained at high speed against the distance lost after the store runs out.** To solve a race of specified distance, we vary $T$ until the best strategy covers that distance.

## Worked example 1: the optimal 400 m and the critical distance

Does a solver built on the reduction above agree with Keller's published calculation? This is a **verification** check on the implementation; it says nothing yet about runners. Start with the constants, in both unit systems.

```{code-cell} python
import numpy as np

CAL = 4.184                                  # joules per calorie
tau, F = 0.892, 12.2                         # s, m/s^2         (Keller 1973, Table 2)
sigma, E0 = 9.93 * CAL, 575.0 * CAL          # cal/(kg s), cal/kg -> W/kg, J/kg

print(f"sigma = {sigma:.1f} W/kg   E0 = {E0:.0f} J/kg   E0/sigma = {E0 / sigma:.0f} s of resupply")
print(f"top speed F tau = {F * tau:.2f} m/s   sustainable speed sqrt(sigma tau) = {np.sqrt(sigma * tau):.2f} m/s")
print(f"in calorie units sqrt(sigma tau) would read {np.sqrt(9.93 * tau):.2f}, which is not a speed")
print(f"F^2 tau = {F**2 * tau:.0f} W/kg > sigma, so the store does run out")

# sigma = 41.5 W/kg   E0 = 2406 J/kg   E0/sigma = 58 s of resupply
# top speed F tau = 10.88 m/s   sustainable speed sqrt(sigma tau) = 6.09 m/s
# in calorie units sqrt(sigma tau) would read 2.98, which is not a speed
# F^2 tau = 133 W/kg > sigma, so the store does run out
```

The initial store is worth $58$ s of resupply. Next the sprint formulas, the critical time by root-finding, and the window above it.

```{code-cell} python
from scipy.optimize import brentq

def v_dash(t):                               # speed under maximal force from rest
    return F * tau * (1 - np.exp(-t / tau))

def D_dash(T):                               # distance covered flat out by time T
    return F * tau**2 * (T / tau + np.exp(-T / tau) - 1)

def int_v2_dash(t):                          # integral of v_dash^2 from 0 to t
    e = np.exp(-t / tau)
    return F**2 * tau**2 * (t - 2 * tau * (1 - e) + 0.5 * tau * (1 - e**2))

def E_dash(t):                               # the store while sprinting flat out
    return E0 + sigma * t - v_dash(t)**2 / 2 - int_v2_dash(t) / tau

Tc = brentq(E_dash, 1e-3, 2000.0)
Dc = D_dash(Tc)
Tstar = Tc + tau * (np.log(2) - 0.5 * np.log(1 - sigma / (F**2 * tau) * (1 - np.exp(-Tc / tau))**-2))
print(f"Tc = {Tc:.2f} s   Dc = {Dc:.1f} m   T* = {Tstar:.2f} s")

# Tc = 27.67 s   Dc = 291.4 m   T* = 28.46 s
```

Every race up to 220 yards is a dash and the 400 m is not. Now the solver: the closed form for $t_2$, the closed form for the fade distance, the one-variable function $D(t_1)$, the grid-then-polish maximization, and the inversion from distance to time.

```{code-cell} python
from scipy.optimize import minimize_scalar

def t2_of_t1(t1, T):                         # E(t2) = 0 is linear in t2 on a constant cruise
    v1 = v_dash(t1)
    num = v1**2 / 2 + int_v2_dash(t1) / tau - E0 - v1**2 * t1 / tau
    return num / (sigma - v1**2 / tau)

def D_arc3(t1, t2, T):                       # fade distance, closed form
    a = sigma * tau
    b = v_dash(t1)**2 - a
    u0, us = np.sqrt(a + b), np.sqrt(a + b * np.exp(-2 * (T - t2) / tau))
    return tau * (u0 - us) + np.sqrt(a) * (tau * np.log((np.sqrt(a) + us) / (np.sqrt(a) + u0)) + T - t2)

def D_of_t1(t1, T):                          # distance in time T if the sprint ends at t1
    t1 = np.asarray(t1, dtype=float)
    v1 = v_dash(t1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(v1**2 > sigma * tau, t2_of_t1(t1, T), T)
    t2 = np.clip(t2, t1, T)
    return D_dash(t1) + v1 * (t2 - t1) + D_arc3(t1, t2, T)

def solve_race(T):                           # -> (D, t1, t2) for the optimal race of duration T
    if T <= Tc:
        return D_dash(T), T, T
    grid = np.linspace(1e-3, Tc - 1e-6, 60)
    Dg = D_of_t1(grid, T)
    i = int(np.argmax(Dg))
    res = minimize_scalar(lambda a: -float(D_of_t1(a, T)), method="bounded",
                          bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 59)]),
                          options=dict(xatol=1e-9))
    t1, D3 = float(res.x), -float(res.fun)
    t2 = float(t2_of_t1(t1, T))
    D_pin = float(D_dash(Tc) + D_arc3(Tc, Tc, T))    # sprint until empty, then fade
    return (D_pin, Tc, Tc) if (t2 < t1 or D_pin >= D3 - 1e-6) else (D3, t1, t2)

def T_of_D(D):                               # invert the monotone map D(T)
    return brentq(lambda T: solve_race(T)[0] - D, 0.1, 1.2 * D / 3 + 20, xtol=1e-5)

T400 = T_of_D(400.0)
D, t1, t2 = solve_race(T400)
print(f"400 m: T = {T400:.2f} s   sprint {t1:.2f} s   cruise at {v_dash(t1):.2f} m/s   fade {T400 - t2:.2f} s")
print("Keller's Table 1:  43.27 s   sprint 1.78 s                       fade 0.86 s")

# 400 m: T = 43.28 s   sprint 1.78 s   cruise at 9.41 m/s   fade 0.86 s
# Keller's Table 1:  43.27 s   sprint 1.78 s                       fade 0.86 s
```

The computed 400 m time differs from Keller's printed value by $0.01$ s, while the sprint and fade durations agree to the displayed precision. That is close agreement with the published calculation. Next we examine how the choice of sprint duration affects distance, then compare pacing across races.

```{code-cell} python
grid = np.linspace(1e-3, Tc - 1e-6, 2000)
Dg = D_of_t1(grid, T400)
near = grid[Dg > Dg.max() - 1.0]
print(f"peak {Dg.max():.2f} m at t1 = {grid[Dg.argmax()]:.2f} s; within 1 m of it only for t1 in [{near.min():.2f}, {near.max():.2f}] s")
print(f"plateau: for every t1 > 10 s, D is {Dg[grid > 10].min():.2f} to {Dg[grid > 10].max():.2f} m")

for D_race in (100, 400, 800, 1500, 5000, 10000):
    T = T_of_D(float(D_race))
    _, t1, t2 = solve_race(T)
    m, s = divmod(T, 60)
    print(f"{D_race:>6} m  T = {int(m):2d}:{s:05.2f}   sprint {t1:5.2f} s   cruise {v_dash(t1):5.2f} m/s   fade {T - t2:4.2f} s")

# peak 400.00 m at t1 = 1.79 s; within 1 m of it only for t1 in [1.76, 1.87] s
# plateau: for every t1 > 10 s, D is 388.90 to 388.91 m
#    100 m  T =  0:10.08   sprint 10.08 s   cruise 10.88 m/s   fade 0.00 s
#    400 m  T =  0:43.28   sprint  1.78 s   cruise  9.41 m/s   fade 0.86 s
#    800 m  T =  1:45.96   sprint  1.07 s   cruise  7.59 m/s   fade 1.08 s
#   1500 m  T =  3:39.47   sprint  0.89 s   cruise  6.85 m/s   fade 1.31 s
#   5000 m  T = 13:13.21   sprint  0.77 s   cruise  6.31 m/s   fade 1.82 s
#  10000 m  T = 26:54.28   sprint  0.75 s   cruise  6.20 m/s   fade 2.12 s
```

Every race starts with maximal force, but the longer races stop accelerating sooner. Their lower cruise speed is reached more quickly: in the model, the 10 km opening sprint lasts only three quarters of a second. The cruise speed approaches $\sqrt{\sigma\tau}=6.09$ m/s as the race lengthens. The 100 m row has no cruise; its listed speed is the speed at the finish.

The [companion script](./code/09_optimal_racing/table1.py) computes all 22 distances in Keller's historical record table. Its largest discrepancy from the observed records is $3.1\%$, at 6 miles, and the dashes agree within $2.1\%$. These are errors against data used to fit the constants. Agreement with Keller's calculation checks the implementation; agreement with his fitting data describes the fit. Later records provide a separate test.

### Checking the three phases without assuming them

The solver assumes the sprint–cruise–fade pattern. We can check that assumption by dividing the race into $N$ short time steps, allowing the force at every step to vary independently between $0$ and $F$, and asking an optimizer to maximize distance while keeping $E \ge 0$. Nothing in this second calculation tells the optimizer to produce three phases. If Keller's physical argument is right, the phases should appear on their own. The [companion script](./code/09_optimal_racing/direct_transcription.py) performs this check for the 400 m with $200$ and then $400$ time steps:

```
400 m with Keller's constants: T = 43.28 s; three-arc solution D = 400.00 m, t1 = 1.78 s, t2 = 42.42 s
N =  200: success = True, 1275 iterations, D = 399.99 m (-0.005), min E = 0.000,
          force leaves F at t = 1.84 s, store empties at t = 42.41 s
N =  400: success = True, 2218 iterations, D = 400.00 m (-0.001), min E = 0.000,
          force leaves F at t = 1.79 s, store empties at t = 42.41 s
```

The finer calculation comes within about a millimetre of the distance from the three-phase solver and finds nearly the same switching times. The improvement on refining the time grid is consistent with a discretization error. The force drops when the runner stops accelerating, remains constant during the cruise, then drops again when the store empties. During the fade it rises as the speed falls, because $f=\sigma/v$.

This is useful **verification**: two different numerical approaches find the same pacing strategy within the same model. It supports the physical argument for the three phases, but says nothing yet about whether real runners follow them.

## Worked example 2: fitting the four constants to the 1972 records

Now the other half of the modeling loop: given the record table, what values of $\tau$, $F$, $\sigma$ and $E_0$ make the model fit? The structure of the solution suggests a two-stage fit, and Keller used it. The eight dashes, from 50 yards to 220 yards, depend only on $\tau$ and $F$ through the dash formula, so we fit those two first. With $\tau$ and $F$ fixed, the fourteen longer records, from 400 m to 10 000 m, are then used to fit $\sigma$ and $E_0$, with the full three-phase solver inside every evaluation. The criterion is the sum of squared *relative* errors,
$$
\sum_{\text{races}} \left(\frac{T_{\text{record}} - T_{\text{model}}(D)}{T_{\text{record}}}\right)^2 ,
$$
because the times span $5$ s to $28$ min: squared errors measured in seconds would be dominated by the longest races. The data give distances, so each residual requires us to invert the model's distance–time relation by root-finding.

> **Aside: choosing what counts as a good fit.** Relative errors give comparable importance to the same percentage miss at different distances. This is a choice about the purpose of the model. Unlike the likelihood in Chapter 2, it does not by itself specify a probability model for the errors or give uncertainty intervals for the fitted constants.

Stage 1 runs in the browser; the 22 records are Keller's Table 1, with yards converted to metres.

```{code-cell} python
from scipy.optimize import least_squares

YD = 0.9144
records = [("50 yd", 50 * YD, 5.1), ("50 m", 50.0, 5.5), ("60 yd", 60 * YD, 5.9),
           ("60 m", 60.0, 6.5), ("100 yd", 100 * YD, 9.1), ("100 m", 100.0, 9.9),
           ("200 m", 200.0, 19.5), ("220 yd", 220 * YD, 19.5)]           # the eight dashes
D_rec = np.array([r[1] for r in records])
T_rec = np.array([r[2] for r in records])

def T_dash_of_D(D, tau_, F_):                # invert the dash formula for one race
    return brentq(lambda T: F_ * tau_**2 * (T / tau_ + np.exp(-T / tau_) - 1) - D, 1e-3, 100.0)

def residuals(p):                            # log-parameters keep tau and F positive
    tau_, F_ = np.exp(p)
    return (T_rec - np.array([T_dash_of_D(d, tau_, F_) for d in D_rec])) / T_rec

fit = least_squares(residuals, np.log([1.0, 10.0]))
tau_fit, F_fit = np.exp(fit.x)
print(f"stage 1: tau = {tau_fit:.3f} s   F = {F_fit:.2f} m/s^2   (Keller: 0.892 s, 12.2 m/s^2)")
print(f"RMS relative error over the eight dashes: {100 * np.sqrt(np.mean(fit.fun**2)):.1f}%")

# stage 1: tau = 0.893 s   F = 12.21 m/s^2   (Keller: 0.892 s, 12.2 m/s^2)
# RMS relative error over the eight dashes: 1.2%
```

Stage 2 nests a full race solve inside every residual, which takes a few seconds on a laptop and longer in the browser, so it lives in the [companion script](./code/09_optimal_racing/fit_records.py). It gives $\sigma = 10.03$ cal/(kg s) and $E_0 = 559$ cal/kg against Keller's $9.93$ and $575$, with a root-mean-square relative error of $2.0\%$ over the fourteen longer records. Then the consistency check: with all four constants in hand, recompute $D_c$. The fit gives $284$ m, between the longest race treated as a dash ($201$ m) and the shortest treated as a long race ($400$ m), so the split assumed in stage 1 is consistent with the constants it produced. Had $D_c$ landed at $150$ m, the whole fit would have to be redone with a different split.

How well do fourteen records determine two constants? The script maps the stage-2 loss on a grid around its minimum:

```
RMS relative error at the minimum 2.05%
RMS error within 0.1 points of the minimum: sigma 9.85 to 10.21 cal/(kg s),
    E0 532 to 586 cal/kg; corr(sigma, E0) inside = -0.66
Keller's published (sigma, E0) = (9.93, 575) with this tau and F: RMS error 2.09%,
    i.e. 0.04 points above the minimum
```

These ranges answer a practical question: how much can we change the constants while allowing the RMS error to rise by only $0.1$ percentage point? That tolerance is our choice; the ranges are not confidence intervals. They also hold the fitted $\tau$ and $F$ fixed.

The physical tradeoff is straightforward. The total energy supplied during a race is $E_0+\sigma T$. A larger resupply can compensate for a smaller initial store, especially over races of similar duration. The compensation is not exact across all distances, which is what allows a broad record table to distinguish the two effects. Keller's pair and the refit have similar overall errors even though some individual predictions differ by several seconds. **A good fit to finishing times does not mean that each effective parameter has been measured precisely.**

## Worked example 3: the 1972 constants against later records and Olympic finals

Keller fitted the best recorded performances available in 1972. They came from different athletes at different distances, so the constants summarize a collection of performances rather than the physiology of one runner. We now keep those constants fixed and compare their predictions with later data.

**Validation: later world records.** Record performances are a useful comparison because minimizing time is close to the model's objective. The following table uses men's record performances set between 1998 and 2020 at the seven Olympic distances. All postdate the fitting data; none is used to change the constants here.

```{code-cell} python
world_records = [(100.0, 9.58, "Bolt 2009"), (200.0, 19.19, "Bolt 2009"),
                 (400.0, 43.03, "van Niekerk 2016"), (800.0, 60 + 40.91, "Rudisha 2012"),
                 (1500.0, 3 * 60 + 26.00, "El Guerrouj 1998"),
                 (5000.0, 12 * 60 + 35.36, "Cheptegei 2020"),
                 (10000.0, 26 * 60 + 11.00, "Cheptegei 2020")]
errors = []
for D_race, T_data, who in world_records:
    T_th = T_of_D(D_race)
    err = 100 * (T_th - T_data) / T_data          # Keller's convention: theory minus data
    errors.append(err)
    print(f"{D_race:>6.0f} m   record {T_data:8.2f} s   theory {T_th:8.2f} s   {err:+5.1f}%   {who}")
print(f"median |error| {np.median(np.abs(errors)):.1f}%")

#    100 m   record     9.58 s   theory    10.08 s    +5.2%   Bolt 2009
#    200 m   record    19.19 s   theory    19.27 s    +0.4%   Bolt 2009
#    400 m   record    43.03 s   theory    43.28 s    +0.6%   van Niekerk 2016
#    800 m   record   100.91 s   theory   105.96 s    +5.0%   Rudisha 2012
#   1500 m   record   206.00 s   theory   219.47 s    +6.5%   El Guerrouj 1998
#   5000 m   record   755.36 s   theory   793.21 s    +5.0%   Cheptegei 2020
#  10000 m   record  1571.00 s   theory  1614.28 s    +2.8%   Cheptegei 2020
# median |error| 5.0%
```

Every discrepancy is positive: the model with its 1972 constants is slower than these later records. The gaps range from under $1\%$ at 200 m and 400 m to $6.5\%$ at 1500 m. The broad decrease of average speed with distance remains useful, but the historical constants do not predict modern performance equally well at every distance.

Reaction time does not explain away these gaps. Official times include the delay after the gun, while the model starts with movement. Subtracting that delay makes the observed running time shorter and the model's positive discrepancy larger. Exercise 5 explores how the assumed delay affects a refit.

**Comparison: Olympic finals.** Consider instead the winning times of the men's Olympic finals from 2008 to 2024, a total of 35 races. A championship rewards finishing first, so a slow tactical race can be a success for the winner. These times therefore test a different question from a record attempt. The [companion script](./code/09_optimal_racing/fit_records.py), run with `--olympic`, prints the comparisons:

```
median |error| 2.0%; most negative -4.6% (2016 1500 m, Matthew Centrowitz);
most positive +5.7% (2024 1500 m, Cole Hocker)
```

The smaller median error does not show that the model describes championship racing better. Some finals are slower than the records, bringing them closer to the older model's predictions. The largest discrepancies go both ways: the theory is $5.7\%$ slower than the 2024 1500 m winning time, but $4.6\%$ faster than the 2016 winning time of $3{:}50.00$. **Before interpreting a prediction error, check whether the observed event had the objective your model assumes.**

**Refitting the constants.** We can also ask what constants fit the fastest Olympic winning time at each distance. There is a trap in the two dashes. Under the model, a runner keeps accelerating, so the second 100 m of a 200 m dash must take less time than the first. Yet the Olympic bests are $9.63$ s and $19.30$ s: the 200 m takes slightly more than twice the 100 m time. No common pair of positive $\tau$ and $F$ can match both exactly.

If we fit anyway, the optimizer drives $\tau$ toward zero and $F$ upward, keeping their product near $10.4$ m/s. This describes nearly instantaneous acceleration. The small inconsistency between the two times leads to an implausible physical explanation. The model treats the two events as the same runner on the same straight course; the observations do not meet that idealization.

For an illustrative refit, we hold $\tau$ at its historical fitted value. This is an explicit modeling choice, not evidence that the damping time has stayed unchanged. The script subtracts an assumed $0.15$ s reaction time from each winning time, fits $F$ to the two dashes, and then fits $\sigma$ and $E_0$ to the longer events:

```
          constant    Keller 1973  Olympic bests
           tau (s)          0.892          0.892
         F (m/s^2)          12.20          12.66
  sigma (cal/kg s)           9.93          10.54
       E0 (cal/kg)            575            590
            Dc (m)            291            287
      E0/sigma (s)           57.9           56.0
```

Under these choices, the fitted force rises by about $4\%$ and the resupply by about $6\%$. The critical distance remains between 200 m and 400 m, so the model's distinction between all-out dashes and paced races survives. These parameter shifts summarize the faster times; they do not identify how much improvement came from training, equipment, or any particular physiological change. **Inspect the physical meaning of fitted parameters as well as the size of the residuals.**

The [companion plot](./code/09_optimal_racing/hill_asymptote.py) compares squared average speeds with Hill's line. It uses the same 1972 constants without another fit. It is a compact way to see the overall trend and the departures from it, while remembering that a championship winning time need not represent the fastest possible run.

## Intuition

> **Why start with full force?** The runner begins at rest and needs to reach the cruising speed. Full force gets there quickly. The energy cost starts at zero because $fv=0$ at rest, then rises as speed increases; the acceleration is not free. In a long race, the important choice is when to stop accelerating so that enough energy remains for the distance ahead.

> **Why does the model finish with a slowdown and not a kick?** Once the store is empty, the resupply cannot support the previous cruise speed. The runner keeps pushing, but slows down. The model's best strategy accepts that brief slowdown in exchange for a faster earlier pace. A real finishing kick may reflect tactics or energy processes missing from this simple model; it is a reason to examine the assumptions, not to treat the model as a coaching prescription.

> **Why does a longer race have a shorter sprint?** The sprint sets the cruise speed, and the cruise speed is what the store and the resupply can sustain over the whole race. A longer race needs a lower cruise speed, which is reached sooner; three quarters of a second of full force already puts a 10 km runner at the $6.2$ m/s they can hold.

> **Can two sprint times determine two constants?** In principle, yes. For races long compared with $\tau$, the dash formula gives $T\approx D/(F\tau)+\tau$. The slope determines the top speed $F\tau$, and the intercept represents the time lost accelerating. But a fitted intercept can be sensitive to timing and differences between events, or even come out negative, as in Example 3. More observations help you see whether one set of constants can plausibly describe them.

> **Why fit relative errors?** A $1\%$ error is $0.1$ s on a 100 m record and $16$ s on a 10 000 m record. Squared errors measured in seconds would be dominated by the longest races; squared relative errors treat a $1\%$ miss as a $1\%$ miss at every distance, which is appropriate when the aim is to fit performance across the full range of distances.

## Connections

- **Builds on:** Chapter 1, for verification against known results (Keller's calculation) as distinct from validation against data the model did not see (later records); Chapter 8, which models the largest of many random draws, while this chapter models a performance limit through constrained optimization.
- **Further directions:** optimal control for a formal treatment of the best pacing strategy; cycling or swimming, where the resistance law changes; hills, where the energy cost varies along the course; and racing against competitors, where minimizing time is not the only objective.

## Exercises

1. **Derivation.** Show that $(\dot v + v/\tau)\,v = \frac{d}{dt}\!\big(\tfrac12 v^2\big) + v^2/\tau$, and integrate to obtain the energy account $E(t) = E_0 + \sigma t - \tfrac12 v^2 - \tfrac1\tau\int_0^t v^2\,ds$. Then substitute the sprint speed $v = F\tau(1 - e^{-t/\tau})$ and carry out the integral to verify $E_{\text{dash}}(t) = E_0 + \sigma t - F^2\tau^2(t/\tau + e^{-t/\tau} - 1)$.

2. **Derivation.** On the fade, substitute $u = v^2$ into $\sigma - \tfrac12\dot u - u/\tau = 0$ and solve from $u(t_2)$ to obtain the fade formula. Then compute the force $f = \dot v + v/\tau$ on this arc and show that $f = \sigma/v$. Why must the cruise speed exceed $\sqrt{\sigma\tau}$ in any race that has a fade?

3. **Computational.** Compute $T^*$ from the formula and from the solver: for race times $T$ from $T_c$ to $30$ s, record the $t_1$ and $t_2$ that `solve_race` returns and find the first $T$ at which $t_1 < t_2$. How wide is the window in metres? Then plot the cruise speed against $D$ from $300$ m to $2000$ m.

4. **Computational.** Compute the speed deficit at the line, $v(t_2)-v(T)$, for the 400 m, 1500 m and 10 000 m. Imagine a competitor alongside the runner at $t_2$ who maintains the cruise speed until time $T$. How many extra metres would that competitor cover? Could they do it with the same empty store and the same resupply?

5. **Computational.** The $0.15$ s reaction-time correction in the modern refit was a choice. Sweep it from $0.10$ to $0.20$ s, holding $\tau$ fixed. Refit $F$ to the two Olympic dashes each time, then refit $\sigma$ and $E_0$ to the longer events using the same correction. Which conclusions of Worked Example 3 depend on the choice?

6. **Modeling judgment.** A coach reads "$D_c = 291$ m" and tells a 400 m runner to sprint flat out for the first 291 m and hang on. State precisely what $D_c$ means in the model, what the model's 400 m actually looks like (sprint length, cruise speed, fade), and what the coach's plan would do to the energy store. Then say what in the model's assumptions might make the model's own plan wrong for a real 400 m.

7. **Modeling judgment.** For cycling, air drag dominates, so replace $v/\tau$ in the equation of motion with $cv^2$. Redo the elimination of $f$ and $E$: what replaces the energy account? Without solving, argue whether the cruise is still at constant speed and whether there is still a sustainable speed with no store; then use $v(t)=\sqrt{F/c}\tanh(\sqrt{Fc}\,t)$ to find the dash distance and invert it for $T(D)$, assuming $F,c>0$.

:::{admonition} Solutions
:class: dropdown

**1.** $\frac{d}{dt}(\tfrac12 v^2) = v\dot v$, so $(\dot v + v/\tau)v = v\dot v + v^2/\tau$ as claimed. Integrating $\dot E = \sigma - v\dot v - v^2/\tau$ from $0$ to $t$ with $v(0) = 0$ gives the account. With $v = F\tau(1 - e^{-t/\tau})$, $v^2 = F^2\tau^2(1 - 2e^{-t/\tau} + e^{-2t/\tau})$ and $\int_0^t v^2 ds = F^2\tau^2\big[t - 2\tau(1 - e^{-t/\tau}) + \tfrac{\tau}{2}(1 - e^{-2t/\tau})\big]$. Adding $\tfrac12 v^2$ and dividing the integral by $\tau$, the $e^{-2t/\tau}$ terms cancel and the rest collects to $F^2\tau^2(t/\tau + e^{-t/\tau} - 1)$. At $t = 0$ this expenditure term vanishes, so $E = E_0$; for large $t$ it grows like $F^2\tau t$, the energy spent while running near top speed.

**2.** With $u = v^2$ the equation is $\dot u = 2\sigma - 2u/\tau$, linear with constant coefficients; the particular solution is $u = \sigma\tau$ and the homogeneous part decays as $e^{-2(t - t_2)/\tau}$, so $u(t) = \sigma\tau + [u(t_2) - \sigma\tau]e^{-2(t - t_2)/\tau}$. From $E = 0$ on the arc, $\dot E = 0$, so $fv = \sigma$ and $f = \sigma/v$. If the cruise speed were below $\sqrt{\sigma\tau}$ the store would be filling on the cruise, so it could not have emptied at $t_2$; a fade requires $v(t_2) > \sqrt{\sigma\tau}$, and then the bracket is positive and $v$ decays.

**3.** With Keller's constants the solver returns $t_1 = t_2 = T_c = 27.67$ s for $T_c \le T \le T^*$ and $t_1 < t_2$ beyond it, matching the formula's $T^* = 28.46$ s. The distance at $T^*$ is $298$ m, so the window is about $7$ m wide. The cruise speed is $10.9$ m/s just above $T^*$ and falls to $9.4$ m/s at 400 m, $7.6$ m/s at 800 m and $6.6$ m/s at 2000 m; see the [companion script](./code/09_optimal_racing/table1.py) for the table.

**4.** The speed deficits are $2.74$, $0.72$ and $0.11$ m/s for the 400 m, 1500 m and 10 000 m respectively. The hypothetical competitor gains
$$
\Delta D=v_1(T-t_2)-\int_{t_2}^T v(t)\,dt,
$$
which is $1.47$, $0.67$ and $0.18$ m at the original runner's finish time. The short fade limits the distance loss even when the speed loss is appreciable. But holding $v_1$ requires power $v_1^2/\tau>\sigma$, so a runner whose store is already empty cannot do it. The comparison requires extra energy. The [exercise calculations](./code/09_optimal_racing/exercise_checks.py) evaluate both the speed and distance deficits.

**5.** Removing more reaction time makes the inferred running times shorter, so the fitted force increases. Across the sweep, $F$ rises from $12.60$ to $12.71$ m/s$^2$, a change of less than $1\%$. Its increase over Keller's $12.2$ remains about $3.3$ to $4.2\%$. After refitting the energy constants, $D_c$ ranges from about $288$ to $285$ m and $E_0/\sigma$ from $55.8$ to $56.1$ s. The main conclusions survive this particular sensitivity check; the exact parameter values depend on the correction. The [exercise calculations](./code/09_optimal_racing/exercise_checks.py) run the sweep.

**6.** $D_c$ is the longest race the model runs at maximal force from start to finish without the store running out; it is a statement about which races are dashes, not about how far to sprint in a longer race. The model's 400 m is a $1.78$ s sprint, a cruise at $9.4$ m/s for forty seconds and a $0.86$ s fade. The coach's plan spends the store at the rate of a flat-out sprint for 291 m, which by construction empties it at $T_c = 27.7$ s with 109 m to go; from there the runner is on resupply alone and the speed decays toward $6.1$ m/s, which leaves the runner slower than the paced plan. What might make the model's own plan wrong: the single energy pool treats anaerobic reserves and oxygen debt as one store with one resupply rate, and a real 400 m is run far above the steady state the model's $\sigma$ describes; the model also has no reaction time, no bend and no competitors.

**7.** Eliminating force gives $fv=\frac{d}{dt}(\tfrac12v^2)+cv^3$, so
$$
E(t)=E_0+\sigma t-\frac12v^2(t)-c\int_0^t v^3(s)\,ds.
$$
Resistance now costs power $cv^3$. Speed variations at a fixed average speed increase that cost, so the same physical argument favors a steady cruise. Without a store, the sustainable speed satisfies $cv^3=\sigma$, giving $v=(\sigma/c)^{1/3}$, provided the available force can support it.

Integrating the supplied dash speed and solving for time gives
$$
D(T)=\frac1c\log\cosh(\sqrt{Fc}\,T),\qquad
T(D)=\frac{\operatorname{arcosh}(e^{cD})}{\sqrt{Fc}}.
$$
Here $\operatorname{arcosh}$ is the inverse hyperbolic cosine. There is a closed-form inverse in this case. For small $T$, expanding the distance formula gives $D\approx FT^2/2$, as expected for acceleration from rest; at long times the speed approaches $\sqrt{F/c}$. Changing the resistance law changes the sustainable and top speeds, while preserving the basic competition between acceleration and energy use.

:::

## Further reading

- Keller, J. B. (1973), "A theory of competitive running," *Physics Today* 26(9), 43–47 — the short introduction with the record table and the fitted constants; everything in this chapter's Worked Examples 1 and 2 is there in outline. [doi:10.1063/1.3128231](https://doi.org/10.1063/1.3128231) · [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Keller%20theory%20of%20competitive%20running%20Physics%20Today&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).
- Keller, J. B. (1974), "Optimal velocity in a race," *American Mathematical Monthly* 81(5), 474–480 — the full calculation: the elimination of $f$ and $E$, the variational argument, the matching conditions and the closed-form distance. [doi:10.1080/00029890.1974.11993589](https://doi.org/10.1080/00029890.1974.11993589) · [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Keller%20optimal%20velocity%20in%20a%20race&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).
- Lloyd, B. B. (1966), "The energetics of running: an analysis of world records," *Advancement of Science* 22, 515–530 — the energy-balance view between Hill and Keller, and the source of Keller's four shortest records. [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Lloyd%20energetics%20of%20running&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2).
- Hill, A. V. (1926), *Muscular Activity* (Williams & Wilkins) — the Herter Lectures in which Hill first read physiology off the record table. [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Hill%20Muscular%20Activity%20Herter%20lectures&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).
