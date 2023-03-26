clear;

%% Model Estimation 1D Case
% part 1 - Gaussian
load("SYDE572/lab_2_data/lab2_1.mat");

a_est_mu = mean(a)
a_var = 0;
b_est_mu = mean(b)
b_var = 0;
a_size = size(a)
b_size = size(b)
a_true_sig = 1;
a_true_mu = 5;
b_lambda_true = 1;

for i = 1:a_size(2)
   a_var = a_var + (a(i) - a_est_mu)^2;
   b_var = b_var + (b(i) - b_est_mu)^2;
end

a_var = a_var/a_size(2)
b_var = b_var/b_size(2)

x = linspace(0, 10);
a_est_sig = sqrt(a_var);
b_est_sig = sqrt(b_var);
a_plot_est = 1/(a_est_sig*sqrt(2*pi))*exp(((-(x-a_est_mu).^2)/(2*a_est_sig.^2)));
a_plot_true = 1/(a_true_sig*sqrt(2*pi))*exp(((-(x-a_true_mu).^2)/(2*a_true_sig.^2)));
b_plot_est = 1/(b_est_sig*sqrt(2*pi))*exp(((-(x-b_est_mu).^2)/(2*b_est_sig.^2)));
b_plot_true = exp(-b_lambda_true * x);

figure(1)
plot(x, a_plot_est);
hold on
plot(x, a_plot_true);
hold on
plot(x, b_plot_est);
hold on
plot(x, b_plot_true);
legend("Estimated a", "True a", "Estimated b", "True b");
title("Parametric Estimation - Gaussian");

% part 2 - Exponential
x = 0:0.0001:10;
a_lambda_est = a_size(2)/sum(a)
b_lambda_est = b_size(2)/sum(b)
a_plot_est = exp(-a_lambda_est * x);
a_plot_true = 1/(a_true_sig*sqrt(2*pi))*exp(((-(x-a_true_mu).^2)/(2*a_true_sig.^2)));
b_plot_est = exp(-b_lambda_est * x);
b_plot_true = exp(-b_lambda_true * x);

figure(2)
plot(x, a_plot_est);
hold on
plot(x, a_plot_true);
hold on
plot(x, b_plot_est);
hold on
plot(x, b_plot_true);
legend("Estimated a", "True a", "Estimated b", "True b");
title("Parametric Estimation - Exponential");

% part 3 - Uniform Distribution
a_est_a = min(a);
b_est_a = max(a);
a_est_b = min(b);
b_est_b = max(b);
a_uniform = makedist('Uniform','lower',a_est_a,'upper',b_est_a);
b_uniform = makedist('Uniform','lower',a_est_b,'upper',b_est_b);
a_plot_est = pdf(a_uniform,x);
a_plot_true = 1/(a_true_sig*sqrt(2*pi))*exp(((-(x-a_true_mu).^2)/(2*a_true_sig.^2)));
b_plot_est = pdf(b_uniform,x);
b_plot_true = exp(-b_lambda_true * x);

figure(3)
plot(x, a_plot_est);
hold on
plot(x, a_plot_true);
hold on
plot(x, b_plot_est);
hold on
plot(x, b_plot_true);
legend("Estimated a", "True a", "Estimated b", "True b");
title("Parametric Estimation - Uniform");

% part 4 - Parzen Window
x = linspace(0, 10);
a_plot_true = 1/(a_true_sig*sqrt(2*pi))*exp(((-(x-a_true_mu).^2)/(2*a_true_sig.^2)));
b_plot_true = exp(-b_lambda_true * x);
% Gaussian Window with standard deviation of 0.1
a_plot_est = parzen(a, x, 0.1, a_size(2));
b_plot_est = parzen(b, x, 0.1, b_size(2));

figure(4)
plot(x, a_plot_est);
hold on
plot(x, a_plot_true);
hold on
plot(x, b_plot_est);
hold on
plot(x, b_plot_true);
legend("Estimated a", "True a", "Estimated b", "True b");
title("Parzen Window with Standard Deviation of 0.1");

% Gaussian Window with standard deviation of 0.4
a_plot_est = parzen(a, x, 0.4, a_size(2));
b_plot_est = parzen(b, x, 0.4, b_size(2));

figure(5)
plot(x, a_plot_est);
hold on
plot(x, a_plot_true);
hold on
plot(x, b_plot_est);
hold on
plot(x, b_plot_true);
legend("Estimated a", "True a", "Estimated b", "True b");
title("Parzen Window with Standard Deviation of 0.4");


function [ p_est ] = parzen(data, x, h, n_samples)
    p_est = zeros(size(x));

    for i=1:size(x,2)
        sum = 0;
        for j=1:size(data,2)
            sum = sum + normpdf(x(i), data(j), h);
        end
        p_est(i) = 1/n_samples * sum;
    end
end





