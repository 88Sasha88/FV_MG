% For building up a Fourier operator for AMR grids
clear;

% Max level
maxL=1;
baseN=4;

% Use cell structure to create a mask at each level
amrMask = cell(maxL,1);
amr{1}.N = baseN;
amr{1}.dx = 1/baseN;
% amr{1}.mask = [1 0 0 1]'; % left and rightmost cells refined
amr{1}.mask = [0 1 1 1]'; % leftmost cell not refined

amr{2}.N = baseN*2;
amr{2}.dx = 1/(baseN*2);
amr{2}.mask = zeros(amr{2}.N,1); % no refinement

% Create a composite grid from the masks
amr{1}.ix = find(amr{1}.mask == 0); % index for coarse cells, 1:N
amr{1}.xflo = (amr{1}.ix-1) * amr{1}.dx;
amr{1}.xfhi = amr{1}.ix * amr{1}.dx;

ref_ix = find(amr{1}.mask == 1); % index for refined coarse cells, 1:N
amr{2}.ix = unique(sort([2*ref_ix-1; 2*ref_ix]));
amr{2}.xflo = (amr{2}.ix-1) * amr{2}.dx;
amr{2}.xfhi = amr{2}.ix * amr{2}.dx;

% The solution lives on a composite grid
N = amr{2}.N;
dx = amr{2}.dx;
xf_lo = [0:N-1]'*amr{2}.dx;
xf_hi = xf_lo + amr{2}.dx;

% Fourier bases on fine grid
f = zeros(N,N); % rows = k, col = j
% Order is cos0, sin1, cos1, ... , sin3, cos3, sin4
for k=0:N-1
    if k==0 % cos modes avg value
        f(k+1,:) = 1;
%         disp("cos0");
    elseif mod(k,2)==0 % cos modes avg value
        kval = k/2;
%         msg = sprintf("cos%d", kval);
%         disp(msg);
        f(k+1,:) = (sin(2*pi*kval*xf_hi) - sin(2*pi*kval*xf_lo))' / (2*pi*k*dx);
    else % sin modes avg value
        kval = floor(k/2) + 1;
%         msg = sprintf("sin%d", kval);
%         disp(msg);
        f(k+1,:) = -(cos(2*pi*kval*xf_hi) - cos(2*pi*kval*xf_lo))' / (2*pi*k*dx); 
    end
end

% Forward Fourier needs to be scaled
f8 = f; % all 8 fine modes
d = diag(f8*f8');
F8 = diag(1./d)*f8;
F8inv = inv(F8);

% Create the coarse version
T8_4 = kron(eye(4,4),[.5 .5]); % Restriction op from 8 to 4
f4 = (T8_4 * f8(1:4,:)')';
d = diag(f4*f4');
F4 = diag(1./d)*f4;
F4inv = inv(F4);

% Now create mixed AMR version
xf_lo = [];
xf_lo = [xf_lo; amr{1}.xflo];
xf_lo = [xf_lo; amr{2}.xflo];
xf_lo = sort(xf_lo);
xf_hi = [];
xf_hi = [xf_hi; amr{1}.xfhi];
xf_hi = [xf_hi; amr{2}.xfhi];
xf_hi = sort(xf_hi);
dx = xf_hi - xf_lo;
% Pick a test vector
% v8 = f8(5,:)' + f8(2,:)';
v8 = f8(2,:)';
% v8 = f8(5,:)';

% Prolong fourier coefs from 4 to 8 with zero padding
That4_8 = [eye(4,4); zeros(4,4)];


if (0) % This is for [1 0 0 1] amr mask
% TODO - this is hard-wired, should automate this from AMR data
% Restrict f8 to our mixed mesh
T8_m = eye(6,8);
T8_m(3, 3:4) = [.5 .5];
T8_m(4, 4:6) = [0 .5 .5];
T8_m(5:6, 5:6) = 0;
T8_m(5:6, 7:8) = eye(2);

vm = T8_m * v8;

% Restrict f8 to our mixed mesh
% TODO - automate this from AMR data
Tm_4 = [0.5 0.5 0 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0.5 0.5];

end

if (1) % This is for [0 1 1 1] amr mask
% TODO - this is hard-wired, should automate this from AMR data
% Restrict f8 to our mixed mesh
T8_m = [T8_4(1,1:2) zeros(1,6); zeros(6,2) eye(6)];

vm = T8_m * v8;

% Restrict f8 to our mixed mesh
% TODO - automate this from AMR data
Tm_4 = [eye(4,1) [zeros(1,6); T8_4(1:3,1:6)]];

end

% This is the coarse projection operator on m
Pc = T8_m*F8inv*That4_8*F4*Tm_4;
Pc = round(Pc,15);


