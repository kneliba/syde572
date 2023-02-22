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

% Part 2 Generating Clusters
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

function clusters = generate_clusters(n, u, cov)
    clusters = repmat(u,n,1) + randn(n,2)*chol(cov)
end
