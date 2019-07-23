% This function is used to optimize the SOPSRL model at time t. All
% parameters are described briefly as follows:
% X     -   the Bt-by-nt sample matrix at time T.
% W     -   the Bt-by-Bt coefficient matrix associated with X at time T.
% S     -   the nt-by-nt coefficient matrix accociated with X at time T.
% q     -   the Bt-dimensional vector associated with W at time T.
% param -   the set of other parameters
% Written by Xiaohui Wei <xh_wei@hnu.edu.cn>
% Please cite the following paper:
% <Xiaohui Wei, Wen Zhu, Bo Liao, Lijun Cai. Scalable one-pass
% self-representation learning for hyperspectral band selection. IEEE
% Transactions on Geoscience and Remote Sensing, VOL. 57, NO. 7, pages:
% 4369-4374, Jul. 2019.>
function [W, phi, S] = SOPSRL(X, S, W, phi, q, param)
%% initialization
tau1 = param.tau1; tau2 = param.tau2; tau3 = param.tau3; r = param.r;
k = param.k; % the number of neighbouring pixels.
maxIter = param.maxIter; tol = param.tol;
iter = 0;
% precoessing
[Bt, nt] = size(X);
%% main loop
Shat = (S+S') ./ 2;
L = diag(sum(Shat)) - Shat;
while iter < maxIter
    iter = iter + 1;
    %% update D, G, W   
    in_iter = 0;
    while in_iter < maxIter
        in_iter = in_iter + 1;
        D = diag((tau1 - 2*tau2.*q) ./ (2*sqrt(sum(W.^2,2)+10^-6)));
        E = X' - X' * W;
        G = diag((phi.^r) ./ (2*sqrt(sum(E.^2,2)+10^-6)));  % equivalent to the \mathbf{B} in the paper
        W = pinv(X*(G+2*tau3.*L)*X' + tau2 .* eye(Bt) + D) * (X*G*X');
        if param.flag == 0
            break;
        end
        in_obj = sum(sum((diag(phi.^r)*(X'-X'*W)).^2, 2).^(1/2)) + tau2*norm(W,'fro')^2+sum(sum((diag(tau1-2*tau2.*q)*W).^2,2).^(1/2)) + 2*tau3*trace(W'*X*L*X'*W);
        if in_iter > 1
            in_delta = abs(in_oldObj-in_obj) / abs(in_obj);
            if in_delta < tol
                break;
            end
        end
        in_oldObj = in_obj;
    end
    %% update phi
    beta = zeros(nt,1);
    for i = 1:nt
        beta(i) = norm(X(:,i) - W'*X(:,i));
    end
    phi = beta.^(1/(1-r)) ./ sum(beta.^(1/(1-r)));
    %% update S
    beta = zeros(nt, nt);
    for i = 1:nt
        tmpBeta = bsxfun(@minus, repmat(X(:,i),1,nt), X);
        beta(:,i) = tau3 .* sum((W'*tmpBeta).^2)';
    end
    S = zeros(nt, nt);
    upsilon = zeros(nt,1);
    for i = 1:nt
        [iY, iI] = sort(beta(:,i), 'ascend');
        upsilon(i) = 0.5 * k * iY(k+1) - 0.5 * sum(iY(1:k));
        iS = (iY(k+1) - iY(1:k)) ./ (k*iY(k+1) - sum(iY(1:k)));
        S(iI(1:k), i) = iS;
    end
    Shat = (S+S') ./ 2;
    L = diag(sum(Shat)) - Shat;
    %%
    Phi = diag(phi.^r);
    Q = diag(tau1 - 2*tau2.*q);
    obj = sum(sum((Phi*E).^2,2).^(1/2)) + tau2 * norm(W,'fro')^2 + sum(sum((Q*W).^2,2).^(1/2)) + ...
        + upsilon' * sum(S.^2, 1)' + 2 * tau3 * trace(W'*X*L*X'*W);
    if iter > 1
        delta = abs(oldObj-obj) / abs(obj);
        if delta < tol
            break;
        end
    end
    oldObj = obj;
end

end