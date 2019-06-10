% Import data, build a matrix with dimensions (T,muB) containing P(T,muB) in each entry
data = importdata("/Users/deboramroczek/Desktop/TrainingFiles/AcceptableSets/Press_Final_PAR_143_350_3_93_286_1002_3D.dat");
X = vec2mat(data(:,3), max(data(:,1)+1));

%mean center data
for i=1:size(X,1)
    for j=1:size(X,2)
        X(i,j) = X(i,j) - mean(X(:,j));
    end
end

X = zscore(X);

%Compute SVD

smub = (svd(X)).^2;
Nfac = sum(smub);
smub = (1/Nfac).*smub;
[U,S,V] = svd(X);

% muB Basis
MUB = zeros(size(X,1),2);
for i=1:size(X,1)

MUB(i,1) = dot(V(:,1).',X(i,:))/dot(V(:,1).',V(:,1).');
MUB(i,2) = dot(V(:,2).',X(i,:))/dot(V(:,2).',V(:,2).');

end


% MUB = normalize(MUB);


%%Plots
figure;
plot(MUB(:,1))
title('\mu_B Basis')




