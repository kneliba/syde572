clear;

% import parzen function
addpath 'SYDE572/lab_2_code'

%% Model Estimation 1D Case
% part 1 - Gaussian
load("SYDE572/lab_2_data/lab2_1.mat");

a_est_mu = mean(a);
a_est_sig = std(a);
b_est_mu = mean(b);
b_est_sig = std(b);
a_size = size(a);
b_size = size(b);
a_true_sig = 1;
a_true_mu = 5;
b_lambda_true = 1;

x = linspace(0, 10);
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
a_lambda_est = a_size(2)/sum(a);
b_lambda_est = b_size(2)/sum(b);
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
a_plot_est = parzen1d(a, x, 0.1, a_size(2));
b_plot_est = parzen1d(b, x, 0.1, b_size(2));

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
a_plot_est = parzen1d(a, x, 0.4, a_size(2));
b_plot_est = parzen1d(b, x, 0.4, b_size(2));

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


%% Model Estimation 2D Case
load("SYDE572/lab_2_data/lab2_2.mat");
step_size = 1;
min_al = min(al); max_al = max(al);
min_bl = min(bl); max_bl = max(bl);
min_cl = min(cl); max_cl = max(cl);
min_xy = min([min_al; min_bl; min_cl]);
max_xy = max([max_al; max_bl; max_cl]);

al_est_mu = mean(al);
al_est_sig = cov(al);
bl_est_mu = mean(bl);
bl_est_sig = cov(bl);
cl_est_mu = mean(cl);
cl_est_sig = cov(cl);

[win_x, win_y] = meshgrid(-1000:1:1000);
win = reshape(mvnpdf([win_x(:) win_y(:)], 0, [400 0; 0 400]), [length(win_x), length(win_y)]);
res = [step_size, min_xy(1), min_xy(2), max_xy(1), max_xy(2)];

[al_est_pdf, X, Y] = parzen(al, res, win);
[bl_est_pdf, ~, ~] = parzen(bl, res, win);
[cl_est_pdf, ~, ~] = parzen(cl, res, win);
[YY, XX] = meshgrid(Y, X); % for contour plot

ml = zeros([length(X) length(Y)]);
pw = zeros([length(X) length(Y)]);

for i = 1:length(X)
    for j = 1:length(Y)
        point = [X(i), Y(j)];
        % part 1 - Parametric estimation
        p_al = mvnpdf(point, al_est_mu, al_est_sig);
        p_bl = mvnpdf(point, bl_est_mu, bl_est_sig);
        p_cl = mvnpdf(point, cl_est_mu, cl_est_sig);
        if (p_al > p_bl) && (p_al > p_cl)
            ml(i, j) = 1;
        elseif (p_bl > p_al) && (p_bl > p_cl)
            ml(i, j) = 0;
        else
            ml(i, j) = -1;
        end
        % part 2 - Nonparametric estimation
        if (al_est_pdf(j,i) > bl_est_pdf(j,i)) && (al_est_pdf(j,i) > cl_est_pdf(j,i))
            pw(i, j) = 1;
        elseif (bl_est_pdf(j,i) > al_est_pdf(j,i)) && (bl_est_pdf(j,i) > cl_est_pdf(j,i))
            pw(i, j) = 0;
        else
            pw(i, j) = -1;
        end
    end
end

figure(6)
hold on;
scatter(al(:,1), al(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(bl(:,1), bl(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
scatter(cl(:,1), cl(:,2), "MarkerEdgeColor", [0.4660 0.6740 0.1880]);
contour(XX, YY, ml, 2, 'Color', 'm', 'LineWidth', 3);
title('Parametric Estimation - Class Boundaries','FontSize', 14);
xlabel('x1','FontSize', 14);
ylabel('x2','FontSize', 14);

figure(7)
hold on;
scatter(al(:,1), al(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(bl(:,1), bl(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
scatter(cl(:,1), cl(:,2), "MarkerEdgeColor", [0.4660 0.6740 0.1880]);
contour(XX, YY, pw, 2, 'Color', 'm', 'LineWidth', 3);
title('Nonparametric Estimation - Class Boundaries','FontSize', 14);
xlabel('x1','FontSize', 14);
ylabel('x2','FontSize', 14);


%% Sequencial Discriminants
load("SYDE572/lab_2_data/lab2_3.mat");

max_ab = max([max(a); max(b)]);
min_ab = min([min(a); min(b)]);
[X, Y] = meshgrid(min_ab(1):1:max_ab(1), min_ab(2):1:max_ab(2));

% Deliverable 1
for classifer = 1:3
    [p_a, p_b, n_aBj, n_bAj] = train(a, b, 10e5);

    seq = zeros(size(X));
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            point = [X(i, j), Y(i, j)];
            seq(i, j) = predict(point, p_a, p_b, n_aBj, n_bAj);
        end
    end
    % plot
    figure(7+classifer)
    hold on;
    scatter(a(:,1), a(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
    scatter(b(:,1), b(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
    contour(X, Y, seq, 2, 'Color', 'm', 'LineWidth', 3);
    title(sprintf('Sequencial Discriminant %d - Class Boundaries', classifer),'FontSize', 14);
    xlabel('x1','FontSize', 14);
    ylabel('x2','FontSize', 14);
end

% Deliverable 3
all_errors = [];
for J = 1:5
    errs = zeros(1, 20);
    for trial = 1:20
        [p_a, p_b, n_aBj, n_bAj] = train(a, b, J);
        errs(trial) = eval(a, b, p_a, p_b, n_aBj, n_bAj);
    end
    all_errors = [all_errors; errs];
end

figure(11)
hold on;
plot(1:5, max(all_errors, [], 2), 'Color', 'r', 'LineWidth', 3)
plot(1:5, min(all_errors, [], 2), 'Color', 'g', 'LineWidth', 3)
plot(1:5, mean(all_errors, 2), 'Color', 'b', 'LineWidth', 3)
plot(1:5, std(all_errors, 0, 2), 'Color', 'c', 'LineWidth', 3)
legend('Max Error', 'Min Error', 'Mean of Error', 'Std of Error');
title('Sequencial Discriminant Error Rates','FontSize', 14);
xlabel('J','FontSize', 14);
ylabel('error rate','FontSize', 14);

% train
function [p_a, p_b, n_aBj, n_bAj] = train(a, b, J)
    % for saving MED
    p_a = []; p_b = [];
    n_aBj = []; n_bAj = [];
    % 1.
    for j = 1:J
        while 1
            % 2.
            point_a = a(randi(size(a, 1)), :);
            point_b = b(randi(size(b, 1)), :);
            % 3. & 4.
            n_aB = 0; n_bA = 0;
            for i = 1:size(a, 1)
                point = a(i, :);
                if norm(point-point_a) > norm(point-point_b)
                    n_aB = n_aB + 1;
                end
            end
            for i = 1:size(b, 1)
                point = b(i, :);
                if norm(point-point_b) > norm(point-point_a)
                    n_bA = n_bA + 1;
                end
            end
            % 5.
            if n_aB == 0 || n_bA == 0
                break
            end
        end
        % 6.
        p_a = [p_a; point_a];
        p_b = [p_b; point_b];
        n_aBj = [n_aBj; n_aB];
        n_bAj = [n_bAj; n_bA];
        % 7.
        deleterow = false([size(b, 1), 1]);
        if n_aB == 0
            for i = 1:size(b, 1)
                point = b(i, :);
                if norm(point-point_b) < norm(point-point_a)
                    deleterow(i) = true;
                end
            end
        end
        b(deleterow, :) = [];
        % 8.
        deleterow = false([size(a, 1), 1]);
        if n_bA == 0
            for i = 1:size(a, 1)
                point = a(i, :);
                if norm(point-point_a) < norm(point-point_b)
                    deleterow(i) = true;
                end
            end
        end
        a(deleterow, :) = [];
        % 9.
        if isempty(a) || isempty(b)
            break
        end
    end
end

% predict, -1: class A, 1: class B
function [res] = predict(point, p_a, p_b, n_aBj, n_bAj)
    % 1. & 4.
    for j = 1:size(p_a, 1)
        point_a = p_a(j, :);
        point_b = p_b(j, :);
        % 2.
        if norm(point-point_b) < norm(point-point_a) && n_aBj(j) == 0
            res = 1;
            return
        end
        % 3.
        if norm(point-point_a) < norm(point-point_b) && n_bAj(j) == 0
            res = -1;
            return
        end
    end
    res = 0; % point could not be classified
end

% eval
function [err] = eval(a, b, p_a, p_b, n_aBj, n_bAj)
    err = 0;
    for i = 1:size(a, 1)
        if predict(a(i, :), p_a, p_b, n_aBj, n_bAj) ~= -1
            err = err + 1;
        end
    end
    for i = 1:size(b, 1)
        if predict(b(i, :), p_a, p_b, n_aBj, n_bAj) ~= 1
            err = err + 1;
        end
    end
    err = err / (size(a, 1) + size(b, 1));
end

function [ p_est ] = parzen1d(data, x, h, n_samples)
    p_est = zeros(size(x));

    for i=1:size(x,2)
        sum = 0;
        for j=1:size(data,2)
            sum = sum + normpdf(x(i), data(j), h);
        end
        p_est(i) = 1/n_samples * sum;
    end
end






