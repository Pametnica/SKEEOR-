%% Robust IMC PID auto-tuner (single-read, last-row)
% Reads last row of tf_data.csv, estimates FOPDT if possible, else uses pidtune.
clear; clc; close all;
disp("=== Robust IMC PID auto-tuner (single read) ===");

%% Config
csv_path = "C:\Users\DD\OneDrive\Desktop\tf_data.csv";       % input CSV (must contain columns b..., a..., optional delay)
pid_export = "pid_results.csv";
s = tf('s');

if ~isfile(csv_path)
    error("Input file '%s' not found.", csv_path);
end

% Read table and take last row
T = readtable(csv_path);
if isempty(T)
    error("CSV is empty.");
end
row = T(end, :);
vars = row.Properties.VariableNames;

% extract numerator and denominator coefficients (any b* and a* columns)
bcols = startsWith(vars, 'b');
acols = startsWith(vars, 'a');
if ~any(bcols) || ~any(acols)
    error("CSV must contain columns starting with 'b' and 'a' (e.g. b0,b1,... a0,a1,...)");
end
b = table2array(row(:, bcols));
a = table2array(row(:, acols));
b = b(~isnan(b));
a = a(~isnan(a));
if isempty(b) || isempty(a)
    error("No valid b or a coefficients found.");
end

% optional delay column names (try several common names)
delayNames = {'delay','L','TimeDelay','time_delay'};
L = 0;
for k=1:numel(delayNames)
    if ismember(delayNames{k}, vars)
        L = double(row.(delayNames{k}));
        if isnan(L), L = 0; end
        break;
    end
end

% Build transfer function (note: assume coefficients are in descending power order)
try
    G_nom = tf(b, a);      % nominal (without explicit delay from CSV)
catch ME
    error("Error building tf: %s", ME.message);
end
if L > 0
    G = G_nom * exp(-L*s);
else
    G = G_nom;
end

disp("Detected process (nominal, no explicit delay shown below):");
G_nom
if L>0
    fprintf("Explicit delay L (from CSV) = %.6g s\n", L);
end

%% Try to estimate FOPDT (K, T, L_est) from step response if appropriate
K_dc = dcgain(G_nom);   % DC gain of nominal (without explicit delay)
K_thresh = 1e-6;

useFOPDT = true;
if ~isfinite(K_dc) || abs(K_dc) < K_thresh
    fprintf("Warning: nominal DC gain K = %.3g (≈0) -> FOPDT formulas will fail. Will fall back to pidtune.\n", K_dc);
    useFOPDT = false;
end

% Function to estimate FOPDT via tangent method (uses G_nom step)
if useFOPDT
    try
        [K_est, T_est, L_est, success] = estimateFOPDT_from_step(G_nom);
        if ~success
            fprintf("FOPDT estimation failed or unreliable -> fallback to pidtune.\n");
            useFOPDT = false;
        else
            % If CSV provided explicit delay L, prefer to use that as L_total (additive)
            if L > 0
                L_total = L;      % trust explicit CSV delay
            else
                L_total = L_est;
            end
            fprintf("FOPDT estimate: K=%.6g, T=%.6g, L=%.6g (using %s delay)\n", K_est, T_est, L_total, (L>0)*"CSV" + (~(L>0))*"estimated");
        end
    catch ME
        fprintf("Error during FOPDT estimation: %s\n", ME.message);
        useFOPDT = false;
    end
end

%% Compute PID
C = []; info = [];
if useFOPDT
    % IMC rule for lambda (tunable): choose lambda = max(L_total, 0.5*T_est)
    lambda = max(L_total, 0.5*T_est);
    lambda = max(lambda, 1e-3);
    % Heuristic IMC-PID formulas (common approximations)
    % avoid division by zero
    if K_est == 0 || T_est <= 0
        useFOPDT = false;
    else
        Kp = (T_est / (K_est * (L_total + lambda)));
        Ki = Kp / T_est;           % Ki = Kp / Ti where Ti = T_est
        % heuristic derivative (small) — often Kd = Kp * L/ (2 + L/T)
        Kd = Kp * (L_total) / (2 + (L_total / T_est));
        % check finiteness
        if any(~isfinite([Kp,Ki,Kd]))
            fprintf("Computed IMC-PID parameters are not finite -> fallback to pidtune.\n");
            useFOPDT = false;
        else
            % build controller
            try
                C = pid(Kp, Ki, Kd);
                % set derivative filter time constant if available
                if isprop(C, 'Tf')
                    C.Tf = lambda;
                end
                info = struct('Method','IMC-FOPDT','Lambda',lambda);
                fprintf("IMC-PID (FOPDT) -> Kp=%.6g, Ki=%.6g, Kd=%.6g, lambda=%.6g\n", Kp, Ki, Kd, lambda);
            catch ME
                fprintf("Failed to create pid object: %s -> fallback to pidtune.\n", ME.message);
                useFOPDT = false;
            end
        end
    end
end

% Fallback: use pidtune directly on G (tries G including delay if possible)
if ~useFOPDT
    opt = pidtuneOptions('DesignFocus','reference-tracking','PhaseMargin',60);
    try
        % try tuning on delayed model first
        [C, info] = pidtune(G, 'PID', opt);
    catch
        % if tuning with explicit delay fails, try nominal
        try
            [C, info] = pidtune(G_nom, 'PID', opt);
        catch ME
            error("pidtune failed on both delayed and nominal models: %s", ME.message);
        end
    end
    % If pidtune returned controller object, extract numeric values (works for PID object)
    try
        Kp = C.Kp; Ki = C.Ki; Kd = C.Kd;
    catch
        % some older/newer versions may be different: try conversion
        tmp = pid(C);
        Kp = tmp.Kp; Ki = tmp.Ki; Kd = tmp.Kd;
    end
    fprintf("pidtune fallback -> Kp=%.6g, Ki=%.6g, Kd=%.6g\n", Kp, Ki, Kd);
end

%% Save results to CSV
ts = datetime("now");
new_entry = table(ts, Kp, Ki, Kd, double(getfield_ifexists(info,'Lambda',NaN)), ...
    'VariableNames', {'Time','Kp','Ki','Kd','Lambda'});
if ~isfile(pid_export)
    writetable(new_entry, pid_export);
else
    writetable(new_entry, pid_export, 'WriteMode','Append');
end
fprintf("Saved PID to '%s'\n", pid_export);

%% Plotting: closed-loop step and bode (try to include delay)
figure('Name','IMC PID AutoTune Results','NumberTitle','off');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

% Closed-loop step
nexttile;
try
    T_cl = feedback(C * G, 1);   % includes delay
    % choose t_final based on dominant time constant if available
    if exist('T_est','var') && T_est>0
        t_final = min(max(8*(T_est + L_total), 5), 200);
    else
        t_final = 20;
    end
    step(T_cl, linspace(0, t_final, 400));
    title('Closed-loop step (with tuned PID)');
    grid on;
catch ME
    warning("Unable to simulate closed-loop step: %s", ME.message);
end

% Bode of loop (C*G without delay for bode if needed)
nexttile;
try
    bode(C * G_nom); grid on;
    title('Open-loop Bode (C*G_{nom})');
catch
    warning("Unable to plot bode: %s", ME.message);
end

drawnow;

disp("=== Finished ===");

%% --- Helper functions ---
function val = getfield_ifexists(s, name, default)
    if isempty(s) || ~isfield(s, name)
        val = default;
    else
        val = s.(name);
    end
end

function [K, T, L, success] = estimateFOPDT_from_step(Gnom)
% Estimate FOPDT (K, T, L) from step response of Gnom (no explicit delay).
% Uses tangent method: find max slope point and tangent intercept.
success = false;
K = NaN; T = NaN; L = NaN;
% get poles to pick a reasonable simulation time
p = pole(Gnom);
realp = real(p(isfinite(p)));
if isempty(realp)
    t_final = 50;
else
    tcands = abs(1./realp(realp<0));
    if isempty(tcands)
        t_final = 50;
    else
        tc = max(tcands);
        t_final = max(8*tc, 5);
        t_final = min(t_final, 200);
    end
end

t = linspace(0, t_final, 2000);
[y, tt] = step(Gnom, t);
y = squeeze(y);
% steady-state approx
y_inf = y(end);
if ~isfinite(y_inf) || abs(y_inf) < 1e-6
    % no steady-state gain -> cannot approximate FOPDT
    return;
end
% derivative
dy = gradient(y) ./ gradient(tt);
[slope, idx] = max(dy);
if slope <= 1e-8
    return;
end
% tangent intercept (time where tangent crosses zero)
t_at = tt(idx);
y_at = y(idx);
t_intercept = t_at - y_at / slope;
if ~isfinite(t_intercept)
    return;
end
L = max(t_intercept, 0);
% find time where y reaches 63.2% of final
y63 = 0.632 * y_inf;
idx63 = find(y >= y63, 1, 'first');
if isempty(idx63)
    % not reached within t_final -> not reliable
    return;
end
t63 = tt(idx63);
T = t63 - L;
if T <= 0
    return;
end
K = y_inf;  % since input is unit step, dcgain = final value
success = true;
end
