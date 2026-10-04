#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace solverslib::detail
{
// Truncated power series with normalized coefficients: coefficient[k]=f^(k)/k!.
// Nesting series<T, N> allows forward differentiation of the expansion point
// independently of the curve parameter. No finite difference is used.
template <typename T, int N> struct taylor_series
{
    std::array<T, N + 1> coefficient{};
    taylor_series() = default;
    taylor_series(double value) { coefficient[0] = T(value); }

    friend taylor_series operator+(const taylor_series& a, const taylor_series& b)
    {
        taylor_series c;
        for (int i = 0; i <= N; ++i)
            c.coefficient[i] = a.coefficient[i] + b.coefficient[i];
        return c;
    }
    friend taylor_series operator-(const taylor_series& a, const taylor_series& b)
    {
        taylor_series c;
        for (int i = 0; i <= N; ++i)
            c.coefficient[i] = a.coefficient[i] - b.coefficient[i];
        return c;
    }
    friend taylor_series operator-(const taylor_series& a) { return taylor_series(0.) - a; }
    friend taylor_series operator+(const taylor_series& a) { return a; }
    friend taylor_series operator*(const taylor_series& a, const taylor_series& b)
    {
        taylor_series c;
        for (int i = 0; i <= N; ++i)
            for (int j = 0; j <= i; ++j)
                c.coefficient[i] += a.coefficient[j] * b.coefficient[i - j];
        return c;
    }
    friend taylor_series operator/(const taylor_series& a, const taylor_series& b)
    {
        taylor_series c;
        for (int i = 0; i <= N; ++i)
        {
            T value = a.coefficient[i];
            for (int j = 1; j <= i; ++j)
                value -= b.coefficient[j] * c.coefficient[i - j];
            c.coefficient[i] = value / b.coefficient[0];
        }
        return c;
    }
    taylor_series& operator+=(const taylor_series& b) { return *this = *this + b; }
    taylor_series& operator-=(const taylor_series& b) { return *this = *this - b; }
    taylor_series& operator*=(const taylor_series& b) { return *this = *this * b; }
    taylor_series& operator/=(const taylor_series& b) { return *this = *this / b; }
    friend bool    operator<(const taylor_series& a, const taylor_series& b)
    {
        return a.coefficient[0] < b.coefficient[0];
    }
    friend bool operator>(const taylor_series& a, const taylor_series& b) { return b < a; }
    friend bool operator<=(const taylor_series& a, const taylor_series& b)
    {
        return a.coefficient[0] <= b.coefficient[0];
    }
    friend bool operator>=(const taylor_series& a, const taylor_series& b)
    {
        return a.coefficient[0] >= b.coefficient[0];
    }
    friend bool operator==(const taylor_series& a, const taylor_series& b)
    {
        return a.coefficient[0] == b.coefficient[0];
    }
    friend bool operator!=(const taylor_series& a, const taylor_series& b) { return !(a == b); }
};

template <typename T, int N> taylor_series<T, N> exp(const taylor_series<T, N>& x)
{
    using std::exp;
    taylor_series<T, N> y;
    y.coefficient[0] = exp(x.coefficient[0]);
    for (int n = 1; n <= N; ++n)
    {
        for (int k = 1; k <= n; ++k)
            y.coefficient[n] += T(k) * x.coefficient[k] * y.coefficient[n - k];
        y.coefficient[n] /= T(n);
    }
    return y;
}

template <typename T, int N> taylor_series<T, N> log(const taylor_series<T, N>& x)
{
    using std::log;
    taylor_series<T, N> derivative;
    for (int k = 0; k < N; ++k)
        derivative.coefficient[k] = T(k + 1) * x.coefficient[k + 1];
    const auto          quotient = derivative / x;
    taylor_series<T, N> y;
    y.coefficient[0] = log(x.coefficient[0]);
    for (int k = 1; k <= N; ++k)
        y.coefficient[k] = quotient.coefficient[k - 1] / T(k);
    return y;
}

template <typename T, int N> taylor_series<T, N> sqrt(const taylor_series<T, N>& x)
{
    using std::sqrt;
    taylor_series<T, N> y;
    y.coefficient[0] = sqrt(x.coefficient[0]);
    for (int n = 1; n <= N; ++n)
    {
        T value = x.coefficient[n];
        for (int k = 1; k < n; ++k)
            value -= y.coefficient[k] * y.coefficient[n - k];
        y.coefficient[n] = value / (T(2) * y.coefficient[0]);
    }
    return y;
}

template <typename T, int N>
std::pair<taylor_series<T, N>, taylor_series<T, N>> sincos(const taylor_series<T, N>& x)
{
    using std::cos;
    using std::sin;
    taylor_series<T, N> s, c;
    s.coefficient[0] = sin(x.coefficient[0]);
    c.coefficient[0] = cos(x.coefficient[0]);
    for (int n = 1; n <= N; ++n)
    {
        for (int k = 1; k <= n; ++k)
        {
            s.coefficient[n] += T(k) * x.coefficient[k] * c.coefficient[n - k];
            c.coefficient[n] -= T(k) * x.coefficient[k] * s.coefficient[n - k];
        }
        s.coefficient[n] /= T(n);
        c.coefficient[n] /= T(n);
    }
    return {s, c};
}
template <typename T, int N> taylor_series<T, N> sin(const taylor_series<T, N>& x)
{
    return sincos(x).first;
}
template <typename T, int N> taylor_series<T, N> cos(const taylor_series<T, N>& x)
{
    return sincos(x).second;
}
template <typename T, int N> taylor_series<T, N> tan(const taylor_series<T, N>& x)
{
    const auto sc = sincos(x);
    return sc.first / sc.second;
}
template <typename T, int N> taylor_series<T, N> pow(taylor_series<T, N> x, int n)
{
    const bool   negative = n < 0;
    unsigned int exponent =
        negative ? 0u - static_cast<unsigned int>(n) : static_cast<unsigned int>(n);
    taylor_series<T, N> result(1.);
    while (exponent)
    {
        if (exponent & 1u)
            result *= x;
        exponent >>= 1u;
        if (exponent)
            x *= x;
    }
    return negative ? taylor_series<T, N>(1.) / result : result;
}
template <typename T, int N> taylor_series<T, N> pow(const taylor_series<T, N>& x, double n)
{
    if (std::isfinite(n) && n == std::trunc(n) && n >= std::numeric_limits<int>::min() &&
        n <= std::numeric_limits<int>::max())
        return pow(x, static_cast<int>(n));
    return exp(log(x) * taylor_series<T, N>(n));
}
template <typename T, int N> taylor_series<T, N> abs(const taylor_series<T, N>& x)
{
    return x < taylor_series<T, N>(0.) ? -x : x;
}
}  // namespace solverslib::detail
