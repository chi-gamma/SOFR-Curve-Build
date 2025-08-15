# 📈 SOFR Interest Rate Curve Construction
This repository implements and compares **three methods** for constructing a SOFR (Secured Overnight Financing Rate) interest rate curve using different forward curve smoothing and bootstrapping techniques. Each method aims to produce a curve that is arbitrage-free, realistic, and suitable for pricing and risk management.

**All methods use swap rates as the primary calibration instruments throughout the curve-building process.**

---

## 🔧 Methods Overview

### 1. **Piecewise Constant Forward Curve**

- Solves for a set of **piecewise constant**, **continuously compounded** forward rates.
- Uses a **multivariate Newton-Raphson** algorithm to solve the nonlinear pricing equations.
- Produces a curve that exactly reprices market instruments, but is **discontinuous** at maturity points.
- While fast and straightforward, the resulting forward curve has sharp **jumps in slope** and lacks realism.

---

### 2. **Smart Quadratic Iterative Curve**

- **Builds on Method 1** by applying **iteration** and **quadratic interpolation** to the piecewise constant forwards.
- Iteratively adjusts market rates to ensure instruments are repriced using the smoothed curve.
- Produces a **C⁰-continuous** forward curve that is smooth in value but not in slope.

#### ⚠ Limitations:
- Can introduce **non-economic artifacts**, such as **double humps** (adjacent local maxima/minima).
- The forward curve's **first derivative is discontinuous**, which can lead to unstable risks and sensitivities.

> 📌 *The smart quadratic method is reliable and widely used, but the resulting curves often exhibit “double humps,” which are artifacts not present in the real forward curve. These slope discontinuities introduce errors in pricing, risks, and hedges, though typically smaller than those from Method 1.*

---

### 3. **C¹ Continuous Forward Curve via Area-Preserving Quadratic Spline**

- Also **builds on Method 1**, but applies a more robust smoothing technique.
- Uses an **area-preserving quadratic spline** that enforces **C¹ continuity** (smooth slope transitions) by solving a **tridiagonal linear system** (via the Thomas algorithm).
- Iteratively adjusts market quotes to match repriced instrument values, just like Method 2.

#### ✅ Advantages over Method 2:
- **First-derivative continuity** eliminates slope jumps and ensures a **visually and numerically smooth** curve.
- **No double extrema** (no artificial humps or dips).
- Curve responds **smoothly to input data**, producing **stable risk and hedge outputs**.

---

#  📌 Instantaneous Forward Rate Modeling
Methods 2 (Smart Quadratic) and 3 (Area-Preserving Quadratic Spline) directly model the instantaneous forward rate, producing smoothly varying forward curves rather than piecewise constant (stepwise) forward rates as in Method 1.This smoothness is particularly beneficial for OIS curve construction, where the one-day forward rate closely approximates the instantaneous forward rate, making these methods more accurate and realistic for such applications.

---


📚 Reference
Hagan, P. S. (2019).
Building Curves Using Area Preserving Quadratic Splines.
Available at: https://papers.ssrn.com/abstract=3455395

This repository adapts the methodology from Hagan’s paper to construct C¹-continuous, arbitrage-free interest rate curves with practical advantages in pricing accuracy and hedge stability.
