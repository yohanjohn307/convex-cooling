
clear; clc; close all;

%% Inputs
N = 50; % horizon
gamma = 1; % discount factor
x_left = 0; x_right = 1; dx = 0.001; 
x_vec = x_left:dx:x_right; % state discretization
U = 0:dx:x_right; % control discretization
delta = 0.01; % disturbance support
w_left_vec = dx:dx:(dx + (N-2)*dx); % disturbance lower bound trajectory
w_right_vec = w_left_vec + delta; % disturbance upper bound trajectory

% plot disturbance trajectory
figure
plot(1:N-1,w_left_vec,1:N-1,w_right_vec)
xlabel('Time Step'); ylabel('Disturbance')
legend('q_{min}', 'q_{max}');

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

% compute analytical bounds
trace_Sigma_exclude_last = 0;
for i = 1:N-2
    trace_Sigma_exclude_last = trace_Sigma_exclude_last + sum( W_probs(i,:) .* (W_vals(i,:) - mean(W_vals(i,:))).^2 );
end
mu = (w_right_vec + w_left_vec) / 2; % symmetric distributions
M = ( sum(mu) + delta/2 ) / N;
S_mu = (mu(1) + delta/2 - M)^2 + sum((mu - M).^2);
u = load_shift_VS(mu(2:end));
u = [w_right_vec(1), u];
S_u = sum((u - M).^2);
% l = 4e5; L = 5.5e5;
l = 200; L = 200;
lb = 0.5 * ( l * (S_mu + trace_Sigma_exclude_last) - L * (S_u + trace_Sigma_exclude_last) );
ub = (L/2) * (S_u + trace_Sigma_exclude_last);

% plot first time step disturbance model
figure
bar(W_vals(1,:),W_probs(1,:))
xlabel('Disturbance'); ylabel('Probability');

%% DP
% no terminal cost
J_dp = zeros(length(x_vec),N);
U_dp = zeros(length(x_vec),N-1);
% costs = order_cost(S_coolprop,U);
costs = quadratic_order_cost(U);
f = waitbar(0,'DP');
tic;
for k = N-1:-1:1
    % Get disturbance support and probabilities for this time step
    W_k = W_vals(k,:);  % [num_w, 1]
    W_probs_k = W_probs(k,:); % [num_w, 1]
    % For all states, all actions, and all disturbances at once
    [X, U_mat, W_mat] = ndgrid(x_vec, U, W_k);  % produce grids
    X_next = X + U_mat - W_mat;                 % next state calculation
    X_next(X_next > x_right) = x_right;         % apply right boundary
    valid = X_next >= x_left;                   % logical for hard constraint

    % For each triple (x, u, w), find the index of next state using interpolation
    J_next = interp1(x_vec, J_dp(:,k+1), X_next, 'linear', NaN);
    J_next(~valid) = Inf; % set to inf if constraint violated

    % Weighted sum over disturbances [num_x, num_u]
    J_vals = ones(length(U),1)*costs + gamma * tensorprod(J_next,W_probs_k,3,2);

    % Choose min cost action
    [J_dp(:,k), idx] = min(J_vals, [], 2);
    U_dp(:,k) = U(idx);

    waitbar((N-k)/N,f,'DP');
end
toc;
close(f);

%% Post-processing
figure
h = heatmap(flip(U_dp'));
h.GridVisible = 'off';
xlabel('x'); ylabel('k');
ax = gca;
ax.XData = x_vec; ax.YData = fliplr(1:N-1);
title('U');

figure
h = heatmap(flip(J_dp'));
h.GridVisible = 'off';
xlabel('x'); ylabel('k');
ax = gca;
ax.XData = x_vec; ax.YData = fliplr(1:N);
title('J');

%% Monte Carlo simulations
N_runs = 1e4;
w_prime = precompute_analytical_heuristic(W_vals,delta);
dp_costs = zeros(N_runs,1);
myopic_costs = zeros(N_runs,1);
empirical_heuristic_costs = zeros(N_runs,1);
analytical_heuristic_costs = zeros(N_runs,1);
f = waitbar(0,'MC');
tic;
for run = 1:N_runs
    % always initialize at zero
    dp_state = zeros(1,N);
    myopic_state = zeros(1,N);
    empirical_heuristic_state = zeros(1,N);
    analytical_heuristic_state = zeros(1,N);

    dp_control = zeros(1,N-1);
    myopic_control = zeros(1,N-1);
    empirical_heuristic_control = zeros(1,N-1);
    analytical_heuristic_control = zeros(1,N-1);
    for k = 1:N-1
        % choose control
        % state_idx = find(abs(x_vec - dp_state(k)) < 1e-9);
        % dp_control = U_dp(state_idx,k);
        dp_control(k) = interp1(x_vec,U_dp(:,k),dp_state(k));
        myopic_control(k) = myopic_policy(myopic_state(k),W_vals(k,end));
        empirical_heuristic_control(k) = empirical_heuristic_policy(empirical_heuristic_state(k),W_vals(k:end,end),delta);
        analytical_heuristic_control(k) = analytical_heuristic_policy(analytical_heuristic_state(k),W_vals(k,end),w_prime(k));

        % update cost
        dp_costs(run) = dp_costs(run) + quadratic_order_cost(dp_control(k));
        myopic_costs(run) = myopic_costs(run) + quadratic_order_cost(myopic_control(k));
        empirical_heuristic_costs(run) = empirical_heuristic_costs(run) + quadratic_order_cost(empirical_heuristic_control(k));
        analytical_heuristic_costs(run) = analytical_heuristic_costs(run) + quadratic_order_cost(analytical_heuristic_control(k));

        % sample disturbance
        w_bins = cumsum(W_probs(k,:));
        w = W_vals(k,find(rand(1) < w_bins,1));

        % evolve state
        dp_state(k+1) = clip(dp_state(k) + dp_control(k) - w, x_left, x_right);
        myopic_state(k+1) = clip(myopic_state(k) + myopic_control(k) - w, 0, x_right);
        empirical_heuristic_state(k+1) = clip(empirical_heuristic_state(k) + empirical_heuristic_control(k) - w, 0, x_right);
        analytical_heuristic_state(k+1) = clip(analytical_heuristic_state(k) + analytical_heuristic_control(k) - w, 0, x_right);
    end
    waitbar(run/N_runs,f,'MC');
end
toc;
close(f);

% calculate performance gaps
myopic_opt_gap = mean(myopic_costs) - mean(dp_costs);
heuristic_opt_gap = mean(analytical_heuristic_costs) - mean(dp_costs);

%% Post-processing
figure
subplot(2,1,1)
plot(1:N,dp_state,1:N,empirical_heuristic_state,1:N,analytical_heuristic_state,1:N,myopic_state)
xlabel('Time Step'); ylabel('Buffer')
legend('DP','Empirical Heuristic','Analytical Heuristic','Myopic')
subplot(2,1,2)
plot(1:N-1,dp_control,1:N-1,empirical_heuristic_control,1:N-1,analytical_heuristic_control,1:N-1,myopic_control)
xlabel('Time Step'); ylabel('Order Quantity')

figure
boxplot([dp_costs, empirical_heuristic_costs, analytical_heuristic_costs, myopic_costs],...
    'Labels',{'DP','Empirical Heuristic','Analytical Heuristic','Myopic'});
ylabel('Cost')
title(['N = ' num2str(N_runs)])

figure
violinplot([dp_costs, empirical_heuristic_costs, analytical_heuristic_costs, myopic_costs]);
ylabel('Cost')
title(['N = ' num2str(N_runs)])

%% Helper functions
function cost = quadratic_order_cost(u)
    cost = 100 * u.^2;
end

function u = myopic_policy(x,w_max)
    u = max([0, w_max - x]);
end

function w_prime = precompute_analytical_heuristic(W_vals,delta)
    mu = W_vals(:,1) + delta/2;
    w_star = load_shift_VS(mu(2:end));
    w_star = [0; w_star];
    w_prime = zeros(size(mu));
    w_prime(1) = mu(1) + delta/2;
    for i = 2:length(mu)
        w_prime(i) = w_prime(i-1) + w_star(i) - mu(i-1);
    end
end

function u = analytical_heuristic_policy(x,w_max,w_prime)
    u = max([w_prime, w_max, x]) - x;
    % u = max([w_prime, x]) - x;
end

function u = empirical_heuristic_policy(x,w_max_remaining,delta)
    w = w_max_remaining - delta / 2;
    w(1) = w(1) + delta / 2;
    q = load_shift_with_buffer(w, x);
    u = max(0, q(1));
end
