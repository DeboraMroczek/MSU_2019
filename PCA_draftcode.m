% Import data, build a matrix with dimensions (T,muB) containing P(T,muB) in each entry

function MUB = pca(filename)
    
data = importdata(filename);
X = vec2mat(data(:,3), max(data(:,1)+1));

%mean center data
for i=1:size(X,1)
    for j=1:size(X,2)

        X(i,j) = X(i,j) - mean(X(:,j));
    end
end

X = zscore(X);

%Compute SVD
[U,S,V] = svd(X);

% muB Basis
MUB = zeros(size(X,1),1);
for i=1:size(X,1)
MUB(i,1) = dot(V(:,1).',X(i,:))/dot(V(:,1).',V(:,1).');
end

MUB = normalize(MUB);

end




