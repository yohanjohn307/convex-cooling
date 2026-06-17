
clear; clc; close all;

%% Inputs
N = 50; % horizon
x_left = 0; x_right = 1; dx = 0.001; 
x_vec = x_left:dx:x_right; % state discretization
delta = 0.01; % disturbance support
w_left_vec = dx:dx:(N-1)*dx; % disturbance lower bound trajectory
w_right_vec = w_left_vec + delta; % disturbance upper bound trajectory

% plot disturbance support trajectory
figure
plot(1:N-1,w_left_vec,1:N-1,w_right_vec)
xlabel('Time Step'); ylabel('Disturbance')
legend('q_{min}', 'q_{max}');

% set disturbance model
W_vals = zeros(N-1,(delta/dx)+1); % disturbance support discretization
W_probs = zeros(N-1,(delta/dx)+1); % disturbance probabilities
for i = 1:N-1
    W_vals(i,:) = w_left_vec(i):dx:w_right_vec(i);
    W_probs(i,:) = ( 1 / ((delta/dx)+1) ); % discrete uniform
    % W_probs(i,:) = binopdf(0:length(W_vals(i,:))-1,length(W_vals(i,:))-1,0.5); % binomial
    % W_probs(i,:) = hygepdf(0:length(W_vals(i,:))-1,100,75,length(W_vals(i,:))-1); % hypergeometric
    % W_probs(i,:) = betapdf((1:length(W_vals(i,:))) ./ (length(W_vals(i,:)) + 1),0.1,0.1); % beta
    % W_probs(i,:) = W_probs(i,:) / sum(W_probs(i,:));
end

% plot first time step disturbance model
figure
bar(W_vals(1,:),W_probs(1,:))
xlabel('Disturbance'); ylabel('Probability');

%% Simulations
N_runs = 1e4;
myopic_costs = zeros(N_runs,1);
optimal_costs = zeros(N_runs,1);
f = waitbar(0);
for run = 1:N_runs
    % generate disturbance trajectory from model
    w_vec = zeros(1,N-1);
    for i = 1:N-1
        w_bins = cumsum(W_probs(i,:));
        w_vec(i) = W_vals(i,find(rand(1) < w_bins,1));
    end

    % corresponding ordering trajectory
    myopic_ordering = w_vec;
    optimal_ordering = load_shift_VS(w_vec);
    
    % calculate costs
    myopic_costs(run) = sum(quadratic_order_cost(myopic_ordering));
    optimal_costs(run) = sum(quadratic_order_cost(optimal_ordering));
    waitbar(run/N_runs)
end
close(f)

%% Post-processing
figure
plot(1:N-1,optimal_ordering,1:N-1,myopic_ordering)
xlabel('Time Step'); ylabel('Ordering Quantity');
legend('Optimal','Myopic')

figure
boxplot([optimal_costs, myopic_costs],...
    'Labels',{'Optimal','Myopic'});
ylabel('Cost')
title(['N = ' num2str(N_runs)])

figure
violinplot([optimal_costs, myopic_costs]);
ylabel('Cost')
title(['N = ' num2str(N_runs)])

%% Helper functions
function cost = quadratic_order_cost(u)
    cost = 100 * u.^2;
end