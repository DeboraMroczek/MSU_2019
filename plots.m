wp = import("pathological_w.dat");
rhop = import("pathological_rho.dat");
wa = import("acceptable_w.dat");
rhoa = import("acceptable_rho.dat");

figure;
plot(wp,rhop)
hold on
plot(wa,whoa)