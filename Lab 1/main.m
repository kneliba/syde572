clear all;
close all;

%Case 1
n_a= 200;
n_b= 200;

u_a = [5, 10];
u_b = [10, 15];

cov_a = [8, 0; 0,4];
cov_b = [8, 0; 0, 4];

%Case 2
n_c = 100; 
n_d = 200; 
n_e = 150; 

u_c = [5, 10];
u_d = [15, 10];
u_e = [10, 5];

cov_c = [8, 4; 4, 40];
cov_d = [8, 0; 0, 8];
cov_e = [10, -5; -5, 20];

%% Part 2 Generating Clusters
[eig_vecs_a, eig_vals_a] = eig(cov_a); 
[eig_vecs_b, eig_vals_b] = eig(cov_b); 
[eig_vecs_c, eig_vals_c] = eig(cov_c); 
[eig_vecs_d, eig_vals_d] = eig(cov_d); 
[eig_vecs_e, eig_vals_e] = eig(cov_e); 

cluster_A = generate_clusters(n_a, u_a, cov_a);
cluster_B = generate_clusters(n_b, u_b, cov_b);
cluster_C = generate_clusters(n_c, u_c, cov_c);
cluster_D = generate_clusters(n_d, u_d, cov_d);
cluster_E = generate_clusters(n_e, u_e, cov_e);

% Case 1
theta_a = atan(eig_vecs_a(2,2)/eig_vecs_a(2,1));
theta_b = atan(eig_vecs_b(2,2)/eig_vecs_b(2,1));

figure(1)
hold on;
plot_ellipse(u_a(1),u_a(2),theta_a,sqrt(eig_vals_a(2,2)),sqrt(eig_vals_a(1,1)),'r')
plot_ellipse(u_b(1),u_b(2),theta_b,sqrt(eig_vals_b(2,2)),sqrt(eig_vals_b(1,1)),'b')

scatter(cluster_A(:,1), cluster_A(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_B(:,1), cluster_B(:,2), "MarkerEdgeColor", "#0072BD");

plot(u_a(1), u_a(2), 'or', 'MarkerFaceColor','r');
plot(u_b(1), u_b(2), 'ob', 'MarkerFaceColor','b');

% Case 2
theta_c = atan(eig_vecs_c(2,2)/eig_vecs_c(2,1));
theta_d = atan(eig_vecs_d(2,2)/eig_vecs_d(2,1));
theta_e = atan(eig_vecs_e(2,2)/eig_vecs_e(2,1));

figure(2)
hold on;
scatter(cluster_C(:,1), cluster_C(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_D(:,1), cluster_D(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
scatter(cluster_E(:,1), cluster_E(:,2), "MarkerEdgeColor",[0.4660 0.6740 0.1880]);

plot_ellipse(u_c(1),u_c(2),theta_c,sqrt(eig_vals_c(2,2)),sqrt(eig_vals_c(1,1)),'r')
plot_ellipse(u_d(1),u_d(2),theta_d,sqrt(eig_vals_d(2,2)),sqrt(eig_vals_d(1,1)),'b')
plot_ellipse(u_e(1),u_e(2),theta_e,sqrt(eig_vals_e(2,2)),sqrt(eig_vals_e(1,1)),'g')

plot(u_c(1), u_c(2), 'or', 'MarkerFaceColor','r');
plot(u_d(1), u_d(2), 'ob', 'MarkerFaceColor','b');
plot(u_e(1), u_e(2), 'og', 'MarkerFaceColor','g');

%% Part 3: Classifiers
grid_count = 1000;
epsilon = 0.01;
x = linspace(min([min(cluster_A(:, 1)), min(cluster_B(:, 1))]), max([max(cluster_A(:, 1)), max(cluster_B(:, 1))]), grid_count);
y = linspace(min([min(cluster_A(:, 2)), max(cluster_B(:, 2))]), max([max(cluster_A(:, 2)), max(cluster_B(:, 2))]), grid_count);
[X, Y] = meshgrid(x, y);

% Case 1 MED, GED & MAP
figure(3)
hold on;
plot_ellipse(u_a(1),u_a(2),theta_a,sqrt(eig_vals_a(2,2)),sqrt(eig_vals_a(1,1)),'r')
plot_ellipse(u_b(1),u_b(2),theta_b,sqrt(eig_vals_b(2,2)),sqrt(eig_vals_b(1,1)),'b')

scatter(cluster_A(:,1), cluster_A(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_B(:,1), cluster_B(:,2), "MarkerEdgeColor", "#0072BD");

plot(u_a(1), u_a(2), 'or', 'MarkerFaceColor','r');
plot(u_b(1), u_b(2), 'ob', 'MarkerFaceColor','b');

med_line_x = []; med_line_y = [];
ged_line_x = []; ged_line_y = [];
map_line_x = []; map_line_y = [];

for i = 1:grid_count
    for j = 1:grid_count
        point = [X(i, j), Y(i, j)];

        % MED
        d_a = sqrt((point - u_a) * (point - u_a).');
        d_b = sqrt((point - u_b) * (point - u_b).');
        if abs(d_a - d_b) < epsilon
            med_line_x = [med_line_x point(1)];
            med_line_y = [med_line_y point(2)];
        end

        % GED
        d_a = sqrt((point - u_a) * inv(cov_a) * (point - u_a).');
        d_b = sqrt((point - u_b) * inv(cov_b) * (point - u_b).');
        if abs(d_a - d_b) < epsilon
            ged_line_x = [ged_line_x point(1)];
            ged_line_y = [ged_line_y point(2)];
        end

        % MAP
        if abs(d_b^2 - d_a^2 - 2*log(n_b/n_a) - log(det(cov_a) / det(cov_b))) < epsilon
            map_line_x = [map_line_x point(1)];
            map_line_y = [map_line_y point(2)];
        end
    end
end

plot(med_line_x, med_line_y, '-k', 'LineWidth', 3);
plot(ged_line_x, ged_line_y, '-r', 'LineWidth', 3);
plot(map_line_x, map_line_y, '-m', 'LineWidth', 3);


% Case 1 NN & 5NN
figure(4)
hold on;
plot_ellipse(u_a(1),u_a(2),theta_a,sqrt(eig_vals_a(2,2)),sqrt(eig_vals_a(1,1)),'r')
plot_ellipse(u_b(1),u_b(2),theta_b,sqrt(eig_vals_b(2,2)),sqrt(eig_vals_b(1,1)),'b')

scatter(cluster_A(:,1), cluster_A(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_B(:,1), cluster_B(:,2), "MarkerEdgeColor", "#0072BD");

plot(u_a(1), u_a(2), 'or', 'MarkerFaceColor','r');
plot(u_b(1), u_b(2), 'ob', 'MarkerFaceColor','b');

nn_line_x = []; nn_line_y = [];
knn_line_x = []; knn_line_y = [];

for i = 1:grid_count
    for j = 1:grid_count
        point = [X(i, j), Y(i, j)];
        d_A = sqrt(diag((cluster_A - point) * (cluster_A - point).'));
        d_B = sqrt(diag((cluster_B - point) * (cluster_B - point).'));
        
        % NN
        if abs(min(d_A) - min(d_B)) < epsilon
            nn_line_x = [nn_line_x point(1)];
            nn_line_y = [nn_line_y point(2)];
        end

        % 5NN
        if abs(mean(mink(d_A, 5)) - mean(mink(d_B, 5))) < epsilon
            knn_line_x = [knn_line_x point(1)];
            knn_line_y = [knn_line_y point(2)];
        end
    end
end

plot(nn_line_x, nn_line_y, '.k', 'LineWidth', 3);
plot(knn_line_x, knn_line_y, '.r', 'LineWidth', 3);



x = linspace(min([min(cluster_C(:, 1)), min(cluster_D(:, 1)), min(cluster_E(:, 1))]), max([max(cluster_D(:, 1)), max(cluster_B(:, 1)), max(cluster_E(:, 1))]), grid_count);
y = linspace(min([min(cluster_C(:, 2)), min(cluster_D(:, 2)), min(cluster_E(:, 2))]), max([max(cluster_D(:, 2)), max(cluster_B(:, 2)), max(cluster_E(:, 2))]), grid_count);
[X, Y] = meshgrid(x, y);

% Case 2 MED, GED & MAP
figure(5)
hold on;
scatter(cluster_C(:,1), cluster_C(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_D(:,1), cluster_D(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
scatter(cluster_E(:,1), cluster_E(:,2), "MarkerEdgeColor",[0.4660 0.6740 0.1880]);

plot_ellipse(u_c(1),u_c(2),theta_c,sqrt(eig_vals_c(2,2)),sqrt(eig_vals_c(1,1)),'r')
plot_ellipse(u_d(1),u_d(2),theta_d,sqrt(eig_vals_d(2,2)),sqrt(eig_vals_d(1,1)),'b')
plot_ellipse(u_e(1),u_e(2),theta_e,sqrt(eig_vals_e(2,2)),sqrt(eig_vals_e(1,1)),'g')

plot(u_c(1), u_c(2), 'or', 'MarkerFaceColor','r');
plot(u_d(1), u_d(2), 'ob', 'MarkerFaceColor','b');
plot(u_e(1), u_e(2), 'og', 'MarkerFaceColor','g');

med_line_x_1 = []; med_line_y_1 = [];
med_line_x_2 = []; med_line_y_2 = [];
med_line_x_3 = []; med_line_y_3 = [];
ged_line_x_1 = []; ged_line_y_1 = [];
ged_line_x_2 = []; ged_line_y_2 = [];
ged_line_x_3 = []; ged_line_y_3 = [];
map_line_x_1 = []; map_line_y_1 = [];
map_line_x_2 = []; map_line_y_2 = [];
map_line_x_3 = []; map_line_y_3 = [];

for i = 1:grid_count
    for j = 1:grid_count
        point = [X(i, j), Y(i, j)];
        % MED
        d_c = sqrt((point - u_c) * (point - u_c).');
        d_d = sqrt((point - u_d) * (point - u_d).');
        d_e = sqrt((point - u_e) * (point - u_e).');
        if abs(d_c - d_d) < epsilon && d_c < d_e
            med_line_x_1 = [med_line_x_1 point(1)];
            med_line_y_1 = [med_line_y_1 point(2)];
        end
        if abs(d_c - d_e) < epsilon && d_c < d_d
            med_line_x_2 = [med_line_x_2 point(1)];
            med_line_y_2 = [med_line_y_2 point(2)];
        end
        if abs(d_e - d_d) < epsilon && d_e < d_c
            med_line_x_3 = [med_line_x_3 point(1)];
            med_line_y_3 = [med_line_y_3 point(2)];
        end

        % GED
        d_c = sqrt((point - u_c) * inv(cov_c) * (point - u_c).');
        d_d = sqrt((point - u_d) * inv(cov_d) * (point - u_d).');
        d_e = sqrt((point - u_e) * inv(cov_e) * (point - u_e).');
        if abs(d_c - d_d) < epsilon && d_c < d_e
            ged_line_x_1 = [ged_line_x_1 point(1)];
            ged_line_y_1 = [ged_line_y_1 point(2)];
        end
        if abs(d_c - d_e) < epsilon && d_c < d_d
            ged_line_x_2 = [ged_line_x_2 point(1)];
            ged_line_y_2 = [ged_line_y_2 point(2)];
        end
        if abs(d_e - d_d) < epsilon && d_e < d_c
            ged_line_x_3 = [ged_line_x_3 point(1)];
            ged_line_y_3 = [ged_line_y_3 point(2)];
        end

        % MAP
        d_cd = d_d^2 - d_c^2 - 2*log(n_d/n_c) - log(det(cov_c) / det(cov_d));
        d_ce = d_e^2 - d_c^2 - 2*log(n_e/n_c) - log(det(cov_c) / det(cov_e));
        d_ed = d_d^2 - d_e^2 - 2*log(n_d/n_e) - log(det(cov_e) / det(cov_d));
        if abs(d_cd) < epsilon && d_ce > 0
            map_line_x_1 = [map_line_x_1 point(1)];
            map_line_y_1 = [map_line_y_1 point(2)];
        end
        if abs(d_ce) < epsilon && d_cd > 0
            map_line_x_2 = [map_line_x_2 point(1)];
            map_line_y_2 = [map_line_y_2 point(2)];
        end
        if abs(d_ed) < epsilon && d_ce < 0
            map_line_x_3 = [map_line_x_3 point(1)];
            map_line_y_3 = [map_line_y_3 point(2)];
        end
    end
end

plot(med_line_x_1, med_line_y_1, '-k', med_line_x_2, med_line_y_2, '-k', med_line_x_3, med_line_y_3, '-k', 'LineWidth', 3);
plot(ged_line_x_1, ged_line_y_1, '-r', ged_line_x_2, ged_line_y_2, '-r', ged_line_x_3, ged_line_y_3, '-r', 'LineWidth', 3);
plot(map_line_x_1, map_line_y_1, '-m', map_line_x_2, map_line_y_2, '-m', map_line_x_3, map_line_y_3, '-m', 'LineWidth', 3);

% Case 2 NN, 5NN
figure(6)
hold on;
scatter(cluster_C(:,1), cluster_C(:,2), "MarkerEdgeColor", [0.8500 0.3250 0.0980]);
scatter(cluster_D(:,1), cluster_D(:,2), "MarkerEdgeColor", [0 0.4470 0.7410]);
scatter(cluster_E(:,1), cluster_E(:,2), "MarkerEdgeColor",[0.4660 0.6740 0.1880]);

plot_ellipse(u_c(1),u_c(2),theta_c,sqrt(eig_vals_c(2,2)),sqrt(eig_vals_c(1,1)),'r')
plot_ellipse(u_d(1),u_d(2),theta_d,sqrt(eig_vals_d(2,2)),sqrt(eig_vals_d(1,1)),'b')
plot_ellipse(u_e(1),u_e(2),theta_e,sqrt(eig_vals_e(2,2)),sqrt(eig_vals_e(1,1)),'g')

plot(u_c(1), u_c(2), 'or', 'MarkerFaceColor','r');
plot(u_d(1), u_d(2), 'ob', 'MarkerFaceColor','b');
plot(u_e(1), u_e(2), 'og', 'MarkerFaceColor','g');

nn_line_x = []; nn_line_y = [];
knn_line_x = []; knn_line_y = [];

for i = 1:grid_count
    for j = 1:grid_count
        point = [X(i, j), Y(i, j)];
        d_C = sqrt(diag((cluster_C - point) * (cluster_C - point).'));
        d_D = sqrt(diag((cluster_D - point) * (cluster_D - point).'));
        d_E = sqrt(diag((cluster_E - point) * (cluster_E - point).'));
        
        % NN
        d_c = min(d_C); d_d = min(d_D); d_e = min(d_E);
        if abs(d_c - d_d) < epsilon && d_c < d_e
            nn_line_x = [nn_line_x point(1)];
            nn_line_y = [nn_line_y point(2)];
        end
        if abs(d_c - d_e) < epsilon && d_c < d_d
            nn_line_x = [nn_line_x point(1)];
            nn_line_y = [nn_line_y point(2)];
        end
        if abs(d_e - d_d) < epsilon && d_e < d_c
            nn_line_x = [nn_line_x point(1)];
            nn_line_y = [nn_line_y point(2)];
        end

        % 5NN
        d_c = mean(mink(d_C, 5)); d_d = mean(mink(d_D, 5)); d_e = mean(mink(d_E, 5));
        if abs(d_c - d_d) < epsilon && d_c < d_e
            knn_line_x = [knn_line_x point(1)];
            knn_line_y = [knn_line_y point(2)];
        end
        if abs(d_c - d_e) < epsilon && d_c < d_d
            knn_line_x = [knn_line_x point(1)];
            knn_line_y = [knn_line_y point(2)];
        end
        if abs(d_e - d_d) < epsilon && d_e < d_c
            knn_line_x = [knn_line_x point(1)];
            knn_line_y = [knn_line_y point(2)];
        end
    end
end

plot(nn_line_x, nn_line_y, '.k', 'LineWidth', 3);
plot(knn_line_x, knn_line_y, '.r', 'LineWidth', 3);

function clusters = generate_clusters(n, u, cov)
    clusters = repmat(u,n,1) + randn(n,2)*chol(cov);
end