function preds = KNN_kernel(y, X, KA, k, Xt)
% function preds = KNN(y, X, ka, k, Xt)
% perform knn classification on each row of Xt
% y, X is training data, Xt is test data

add1 = 0;
if (min(y) == 0),
    y = y + 1;
    add1 = 1;
end
[n,m] = size(X); % n is training data number 
[nt, m] = size(Xt); % nt is test data number

K=X*X'; % X is n by p, K is n by n matrix

a=1;
S=pinv(K)*(KA-a*K)*pinv(K);

D = zeros(n, nt);

for i=1:n
    for j=1:nt
        xtemp=X(i,:)-Xt(j,:);
        k_x=xtemp*X'; %1 by n
        D(i,j)=a*xtemp*xtemp'+k_x*S*k_x' ;      
    end
end
% sort each oolumn of D, each colum is the Xt to each of the X
[V, Inds] = sort(D);

preds = zeros(nt,1);
for (i=1:nt),
    counts = [];
    for (j=1:k),        
        if (y(Inds(j,i)) > length(counts)),
            counts(y(Inds(j,i))) = 1;
        else
            counts(y(Inds(j,i))) = counts(y(Inds(j,i))) + 1;
        end
    end
    [v, preds(i)] = max(counts);
end
if (add1 == 1),
    preds = preds - 1;
end