function q = load_shift_VS(q)

m = 1;
T = length(q);
sz = size(q);

while m <= T
    k = 1:T-m+1;
    if sz(1) == T
        avg_loads = cumsum(q(m:end)) ./ (1:length(q(m:end)))';
    else
        avg_loads = cumsum(q(m:end)) ./ (1:length(q(m:end)));
    end
    [~,idx] = max(avg_loads);
    n = m + k(idx);
    q(m:n-1) = sum(q(m:n-1)) / (n - m);
    m = n;
end

end