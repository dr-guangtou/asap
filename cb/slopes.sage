mh, m1, ms, ms0, y, b, d = var("mh m1 ms ms0 y b d")

mhalo_as_a_function_of_ms = log(m1, 10) + b*log(ms/ms0, 10) + ((ms/ms0)^d) / (1 + (ms/ms0)^(-y)) - 1/2

d_log_mhalo_log_mstar = diff(mhalo_as_a_function_of_ms, ms)*(ms * log(10))
print(d_log_mhalo_log_mstar)

# And now get the numerical result!

dhc_tot = [5.23998141e+12, 1.09925660e+11, 1.12472176e+00, 1.73781619e-11, 1.25033110e-14]
dhc_cen = [4.35598327e+12, 7.35172463e+10, 4.92691009e-01, 2.26529034e-01, 2.85934692e+00]

for d in [dhc_cen, dhc_tot]:
    m1_, ms0_, b_, d_, y_ = d
    for ms_ in [13, 13.5, 14, 13.5, 15]:
        print(numerical_approx(d_log_mhalo_log_mstar.subs(ms=10^ms_, m1=m1_, ms0=ms0_, y=y_, d=d_, b=b_)))
    print("\n\n")
