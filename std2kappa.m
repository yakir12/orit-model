Ns = [1, 10, 100, 500];
for Ni = 1:length(Ns)
    kappa_array = [0.0000001,0.00001,0.0001,0.001,0.1:0.1:699.9];
    std_array = zeros(size(kappa_array));
    for kappa_i=1:length(kappa_array)
        [p1, alpha1] = circ_vmpdf(-pi:0.01:pi, 0, kappa_array(kappa_i));
        random_vm_angles=randpdf(p1, alpha1,[1000*Ns(Ni),1]);
        std_array (kappa_i) = circ_std(random_vm_angles);
    end
    % remove_indexes = find(isnan(std_array)==1);
    % std_array(remove_indexes) = [];
    % kappa_array(remove_indexes) = [];
%     figure(); plot(rad2deg(std_array), kappa_array,'.-'); xlabel('std, aka theta*'); ylabel('kappa');
    dlmwrite(strcat(string(Ns(Ni)), '.csv'), std_array')
end