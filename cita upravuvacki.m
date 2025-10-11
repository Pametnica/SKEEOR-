% Претпоставуваме дека сигналот е зачуван како структура со полиња Time и Data
t = rpm_data.time;   % време
y = rpm_data.signals.values;  % вредности на сигналот

% Комбинирај во една табела
T = table(t, y);

% Запиши во CSV
writetable(T, 'rpm_output.csv');
