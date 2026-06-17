function q = load_shift_with_buffer(q, buffer)

m = 1;
T = length(q);
sz = size(q);

k = 1:T-m+1;
if sz(1) == T
    avg_loads = (cumsum(q(m:end)) - buffer) ./ (1:length(q(m:end)))';
else
    avg_loads = (cumsum(q(m:end)) - buffer) ./ (1:length(q(m:end)));
end
[~,idx] = max(avg_loads);
n = m + k(idx);
q(m:n-1) = (sum(q(m:n-1)) - buffer) / (n - m);

end