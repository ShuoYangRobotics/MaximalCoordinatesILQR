clc,clear,close('all')
%% MC
figure

Itr_un = [
14
27
48];

Itr_con = [
30
48
61];

Solve_time_un = [
2.23309552
12.25661066
33.85835322];

Solve_time_con = [
4.36469008
18.23444996
46.87333194];

yyaxis left
b = bar([Itr_un, Itr_con], "LineWidth",2,'FaceColor','flat');
b(2).FaceColor = [.2 .6 .5];
ylabel("Iterations","FontSize", 15)
hold on

yyaxis right
plot(1:length(Solve_time_un),Solve_time_un, "LineWidth",2)
plot(1:length(Solve_time_con),Solve_time_con, "LineWidth",2)

set(gca,'XTickLabel',{'1','2','3'});
xlabel("Arm Number","FontSize", 15)
ylabel("Solve time (s)","FontSize", 15)
% title("MC constrained v.s. unconstrained","FontSize", 18)
legend("unconstrained","constrained","unconstrained","constrained","Location","NorthWest","FontSize",14)

hold off
%% RBD
figure

Itr_un = [
7
13
24
39
];

Itr_con = [
16
46
61
154
];

Solve_time_un = [
0.4810624
1.1309278
2.5373505
7.4312715
];

Solve_time_con = [
1.405940041
4.8811873
8.76121806
32.6478757
];

yyaxis left
b = bar([Itr_un, Itr_con], "LineWidth",2,'FaceColor','flat');
b(2).FaceColor = [.2 .6 .5];
ylabel("Iterations","FontSize", 15)
hold on
yyaxis right
plot(1:length(Solve_time_un),Solve_time_un, "LineWidth",2)
plot(1:length(Solve_time_con),Solve_time_con, "LineWidth",2)
set(gca,'XTickLabel',{'1','2','3', '4'});
xlabel("Arm Number","FontSize", 15)
ylabel("Solve time (s)","FontSize", 15)
% title("RBD constrained v.s. unconstrained","FontSize", 18)
legend("unconstrained","constrained","unconstrained","constrained","Location","NorthWest","FontSize",14)
hold off