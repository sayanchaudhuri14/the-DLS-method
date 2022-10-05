
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

#####################################################################################
################################Data Loading and Cleanup#############################
#####################################################################################

df = pd.read_csv(r"C:\Users\sayan\OneDrive\Desktop\Data Analysis Projects\Data\04_cricket_1999to2011.csv")

df = df[['Match', 'Innings', 'Over', 'Runs.Remaining', 'Wickets.in.Hand']]
df = df[df['Innings'] == 1]
df.groupby('Match')
Unique_Matches = df.Match.unique()
new_data = pd.DataFrame()
for full_innings in range(len(Unique_Matches)):
    if df.groupby('Match').last().to_numpy()[:, 1][full_innings] > 40:
        new_data = pd.concat(
            [new_data, df[df['Match'] == Unique_Matches[full_innings]]],
            axis=0)
        new_data = pd.concat([
            new_data,
            pd.Series([
                Unique_Matches[full_innings], 1, 0,
                max(df[df['Match'] ==
                       Unique_Matches[full_innings]].loc[:, 'Runs.Remaining']),
                10
            ])


        ])
        new_data = pd.concat([
            new_data,
            pd.Series([
                Unique_Matches[full_innings], 1, 50,
                0,
                10
            ])
        ])
df = new_data

#####################################################################################
###################################Data Abstraction##################################
#####################################################################################

Overs = np.array(df.loc[:, 'Over'])
Wicket_in_hand = df.loc[:, 'Wickets.in.Hand']
Runs_Remaining = df.loc[:, 'Runs.Remaining']
Overs_to_go = 50 - Overs


#####################################################################################
#################################Function to Predict Score###########################
#####################################################################################


def pred_score(Overs, L, Z_not):
    pred_score = Z_not * (1 - np.exp(-L * Overs / Z_not))
    return pred_score


#####################################################################################
##################################Initializing Variables#############################
#####################################################################################

L = []
Z = []
for wkt_left in range(1, 11):
    popt, _ = curve_fit(pred_score, Overs_to_go[Wicket_in_hand == wkt_left],
                        Runs_Remaining[Wicket_in_hand == wkt_left])
    L.append(popt[0])
    Z.append(popt[1])

Shared_L = [np.mean(L)]
Z_not = Z
Init_Variables = np.array(Shared_L + Z_not)


#####################################################################################
#################################Cost Function to Optimize###########################
#####################################################################################


def squared_loss(Variables, Overs_to_go, Runs_Remaining, Wicket_in_hand):
    pred_score = np.zeros([
        len(Runs_Remaining),
    ])
    number_of_data_points = len(Runs_Remaining)
    for i in range(1, 11):
        pred_score[Wicket_in_hand == i] = Variables[i] * (1 - np.exp(
            -Variables[0] * Overs_to_go[Wicket_in_hand == i] / Variables[i]))
    return np.sum((pred_score - Runs_Remaining) ** 2) / number_of_data_points


arguments = (Overs_to_go, Runs_Remaining, Wicket_in_hand)

#####################################################################################
################################Optimizing the Cost Function#########################
#####################################################################################

res = minimize(squared_loss,
               Init_Variables,
               args=(arguments),
               method='Nelder-Mead',
               options={
                   'disp': True,
                   'maxiter': 20000
               })

#####################################################################################
###################Abstracting the values of the Optimized Parameters################
#####################################################################################

L = res.x[0]
Z_not = res.x[1:]

#####################################################################################
################################Visualizing the Results##############################
#####################################################################################

overs = np.arange(1, 51, 1)
prediction = np.zeros([50, 10])
wkt: int
for wkt in range(10):
    prediction[:, wkt] = pred_score(overs, L, Z_not[wkt])
    plt.plot(overs, prediction[:, wkt], label=wkt + 1)
plt.xlabel('Overs Remaining')
plt.ylabel('Average runs obtainable')

plt.legend(title='Wickets in Hand', prop={'size': 7})
plt.title('Average Runs Obtainable vs Overs Remaining')
plt.show()
