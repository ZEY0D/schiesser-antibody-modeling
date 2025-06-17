function antibody_binding_kinetics()
    % Clear previous files
    clear all
    clc
    
    % Parameters shared with the ODE routine
    global zl zu z dz dz2 D kf cbsat kr n cbulk ndss ncall
    
    % Parameter numerical values
    D = 1.0e-10; 
    kf = 1.0e+05; 
    cbulk = 4.48e-05;
    h = 5.0e-05; 
    c0 = 0; 
    cb0 = 0;
    
    % Variation in interface binding saturation%
    cbsat = 1.66e-08;
    cbsat = 1.66e-09;
    
    % Variation in interface unbinding rate%
    kr = 1.0e-01;
    kr = 1.0e+01;
    
    % Spatial grid
    zl = 0; 
    zu = 5.0e-05; 
    n = 21; 
    dz = (zu-zl)/(n-1); 
    dz2 = dz^2;
    z = linspace(zl, zu, n);
    
    % Initial condition
    u0 = zeros(n+1, 1);
    for i = 1:n
        u0(i) = c0;
    end
    u0(n+1) = cb0;
    
    % Independent variable for ODE integration
    t0 = 0.0;
    tf = 100;
    tout = linspace(t0, tf, 51);
    
    ncall = 0;
    
    % ODE integration
    % Variation in error tolerances
    reltol = 1.0e-06; 
    abstol = 1.0e-06;
     reltol = 1.0e-07; abstol = 1.0e-07;
    
    options = odeset('RelTol', reltol, 'AbsTol', abstol);
    mf = 1;
    
    [t, u] = ode15s(@pde_1, tout, u0, options); 

    
    % Store numerical solutions at z=0
    c_plot = u(:, 1);
    cb_plot = u(:, n+1);
    theta_plot = cb_plot / cbsat;
    rate_plot = kf * c_plot .* (cbsat - cb_plot) - kr * cb_plot;
    
    % Display selected output
    fprintf('\n mf = %2d abstol = %8.1e reltol = %8.1e\n', mf, abstol, reltol);
    fprintf('\n t c(0,t) cb(t) theta rate\n');
    
    for it = 1:length(t)
        fprintf('%6.0f%12.3e%12.3e%12.3e%12.3e\n', ...
                t(it), c_plot(it), cb_plot(it), theta_plot(it), rate_plot(it));
    end
    fprintf('\n ncall = %4d\n', ncall);
    
    % Plot numerical solutions at z=0
    figure(1);
    subplot(2,2,1)
    plot(t, c_plot); axis tight
    title('c(0,t) vs t'); xlabel('t'); ylabel('c(0,t)')
    
    subplot(2,2,2)
    plot(t, cb_plot); axis tight
    title('cb(t) vs t'); xlabel('t'); ylabel('cb(t)')
    
    subplot(2,2,3)
    plot(t, theta_plot); axis tight
    title('theta(t) vs t'); xlabel('t'); ylabel('theta(t)')
    
    subplot(2,2,4)
    plot(t, rate_plot); axis tight
    title('rate(t) vs t'); xlabel('t'); ylabel('rate(t)')
    
    % Store numerical solution for 3D plot
    c_3D = u(:, 1:n);
    
    figure(2)
    surf(z, t, c_3D)
    xlabel('z (m)'); ylabel('t (s)'); zlabel('c(z,t) (moles/m^3)');
    title('c(z,t) (moles/m^3), z=0,2.5\times10^{-6},..., 5\times10^{-5} (m), t=0,2,...,100 (s)')
end

function ut = pde_1(t, u)
    % Problem parameters
    global zl zu z dz dz2 D kf cbsat kr n cbulk ndss ncall
    
    % ODE and PDE
    c = u(1:n);
    cb = u(n+1);
    
    % BCs
    cf = c(2) - (2*dz/D) * (kf * c(1) * (cbsat - cb) - kr * cb);
    c(n) = cbulk;
    
    % Initialize derivatives
    ct = zeros(n, 1);
    cbt = 0;
    
    % PDE
    for i = 1:n
        if i == 1
            ct(1) = D * (c(2) - 2 * c(1) + cf) / dz2;
        elseif i == n
            ct(n) = 0;
        else
            ct(i) = D * (c(i+1) - 2 * c(i) + c(i-1)) / dz2;
        end
    end
    
    % ODE
    cbt = kf * c(1) * (cbsat - cb) - kr * cb;
    
    % Derivative vector
    ut = [ct; cbt];
    
    % Increment calls to pde_1
    ncall = ncall + 1;
end

% Dummy functions for other methods (needed for the code to run)
function ut = pde_2(t, u)
    ut = pde_1(t, u);
end

function ut = pde_3(t, u)
    ut = pde_1(t, u);
end