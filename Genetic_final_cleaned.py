import numpy as np
import pandas as pd
import csv
import random
import itertools
import os
from datetime import datetime


###########################################################
####################Globale Variable#######################
###########################################################

# Global Variable
GROUPE_NUMBER_OF_ITERATION = np.arange(25, 100 + 1, 25)
GROUPE_POPULATION_SIZE = np.arange(50, 300 + 1, 50)
GROUPE_MUTATION_PROB = np.arange(0.1, 1 + 0.1, 0.1)
GROUPE_TAUSCH = np.arange(3, 5 + 1, 1)
GROUPE_CLASSIFIER_SIZE = np.arange(5, 10 + 1, 1)
ALL_COMBINATION = []
for xs in itertools.product(
    GROUPE_NUMBER_OF_ITERATION,
    GROUPE_POPULATION_SIZE,
    GROUPE_MUTATION_PROB,
    GROUPE_TAUSCH,
    GROUPE_CLASSIFIER_SIZE,
):
    ALL_COMBINATION.append(xs)
ALL_COMBINATION = random.sample(ALL_COMBINATION, 100)



###########################################################
#######################Funktionen##########################
###########################################################

## This function drops all the miRNa columns containing only zeros or ones
def drop_unwichte_daten(df):

    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    return df



## This function computes TN, FP, TP, FN of two vectors
def calculate_fit(
    data_row, random_row
):  # datarow unsere Training data die erste; randomw row Inizial popukation zeile
    data_list = data_row.iloc[2:]

    match = sum(a == b for a, b in zip(data_list, random_row))
    unmatch = sum(a != b for a, b in zip(data_list, random_row))
    TN, FP, TP, FN = 0, 0, 0, 0

    if data_row["Annots"] == 0:
        # berechnet Tn u. FP
        if unmatch >= match:
            TN = 1
        else:
            FP = 1
    else:

        # berechnet TP FN
        if unmatch <= match:
            TP = 1
        else:
            FN = 1

    return TN, FP, TP, FN


## This function calculates F1 score between each random individuals and row in the test dataset.
def calculate_fit_test(
    data_row, random_row, annots
):

    match = sum(a == b for a, b in zip(data_row, random_row))
    unmatch = sum(a != b for a, b in zip(data_row, random_row))
    TN, FP, TP, FN = 0, 0, 0, 0

    if annots == 0:
        # berechnet Tn u. FP
        if unmatch >= match:
            TN = 1
        else:
            FP = 1
    else:

        # berechnet TP FN
        if unmatch <= match:
            TP = 1
        else:
            FN = 1

    return TN, FP, TP, FN


## This function calculates the final Fittnes score, what is called F1
def Anzahl_calculate_fit(data, df):
    F1_list = []
    for random_index, random_row in df.iterrows():

        Result = []
        for data_index, data_row in data.iterrows():
            TN, FP, TP, FN = calculate_fit(data_row, random_row)
            Result.append([TN, FP, TP, FN])

        Result = np.asarray(Result)
        TN, FP, TP, FN = Result.sum(axis=0)
        F1 = 2 * TP / (2 * TP + FP + FN)
        F1_list.append(F1)

    df["F1"] = F1_list
    return df


## This function swappes elements (miRNAs) at position_list in both A and B vectors
def uniform_crossover(A, B, position_list):
    for position in position_list:
        A[position], B[position] = B[position], A[position]
    return A, B


## This function creates the crossover population
def to_df_uniform_crossover(df, selectiongroup, position_list):
    df_crossover = pd.DataFrame()
    df_crossover = pd.concat([df_crossover, pd.DataFrame(selectiongroup)], axis=1)
    for j in range(0, len(selectiongroup), 2):  # wir erzeugen Kinde von je zwei Parents
        A, B = selectiongroup[j : j + 2][0], selectiongroup[j : j + 2][1]
        ErgebnisA, ErgebnisB = uniform_crossover(A, B, position_list)

        df_crossover.loc[len(df_crossover)] = ErgebnisA
        df_crossover.loc[len(df_crossover)] = ErgebnisB

    df_crossover = df_crossover.iloc[0 : df.shape[0], :]
    return df_crossover


## This function swapps two positions within one vector
def swap_mutation(A, position1, position2):
    B = A.copy()

    B[position1], B[position2] = B[position2], B[position1]
    return B


## This function create the mutation population
def swap_allmutation(df_crossover, MUTATION_PROB):
    for index, row in df_crossover.iterrows():
        prob = random.uniform(0, 1)
        # genrate random value between 0 and 1
        if prob <= MUTATION_PROB:
            position1, position2 = random.sample(
                list(np.arange(df_crossover.shape[1])), 2
            )
            A = swap_mutation(list(row), position1, position2)
            df_crossover.loc[index:index, :] = A
    return df_crossover


## This function read csv data
def lese_csv(dateiname, sep=","):
    data = pd.read_csv(dateiname, sep=sep)
    return data


## This function runs the developed genetic algorithm for a specific dataset and parameters combination
def genetic_one_run(data, df, NUMBER_OF_ITERATION, TAUSCH, MUTATION_PROB):

    F1_all_scores = []
    df_mutation = None

    for step in range(NUMBER_OF_ITERATION):

        if df_mutation is None:
            Ergebnis_Anzahl_calculate_fit = Anzahl_calculate_fit(data, df)
        else:
            Ergebnis_Anzahl_calculate_fit = Anzahl_calculate_fit(data, df_mutation)

        Ergebnis_Anzahl_calculate_fit = Ergebnis_Anzahl_calculate_fit.sort_values(
            by="F1", ascending=False
        )  # wir sortieren nach den besten f1 score
        colums_name = data.columns.to_list()[2:]  # change column name
        colums_name.append("F1")
        Ergebnis_Anzahl_calculate_fit.columns = colums_name  # change column name

        F1_all_scores.append(Ergebnis_Anzahl_calculate_fit["F1"].mean())
        selectiongroup = Ergebnis_Anzahl_calculate_fit.to_numpy()[:, :-1]
        haelfte = int(selectiongroup.shape[0] / 2)
        if haelfte % 2 == 1:
            haelfte = haelfte + 1  # hälfte ist ungerade

        selectiongroup = selectiongroup[
            :haelfte
        ]  # wählen hälfte +1  nach berechnen von fittnes

        position_list = random.sample(list(np.arange(selectiongroup.shape[1])), TAUSCH)

        df_crossover = to_df_uniform_crossover(df, selectiongroup, position_list)

        df_mutation = swap_allmutation(df_crossover, MUTATION_PROB)

    Ergebnis_Anzahl_calculate_fit = Anzahl_calculate_fit(data, df_mutation)
    return Ergebnis_Anzahl_calculate_fit, F1_all_scores


## This function runs the developed genetic algorithm for a specific dataset and parameters combination
def genetic_one_combination(data, COMBINATION, train_best=False):

    (
        NUMBER_OF_ITERATION,
        POPULATION_SIZE,
        MUTATION_PROB,
        TAUSCH,
        CLASSIFIER_SIZE,
    ) = COMBINATION

    data_index = data.iloc[:, :2]  # alle Zeilen,ersten 2 Spalten(ID,Annots)
    data = data.iloc[:, 2:].sample(
        CLASSIFIER_SIZE, axis=1
    )  # select randomly two columns
    data = pd.concat([data_index, data], axis=1, sort=False)  # merge two dataframes

    ## create random Dataframe with size (POPULATION_SIZE, CLASSIFIER_SIZE)
    df = pd.DataFrame(np.random.randint(0, 2, size=(POPULATION_SIZE, CLASSIFIER_SIZE)))
    df = df.mask(np.random.choice([True, False], size=df.shape, p=[0.5, 0.5]))
    df.columns = data.columns.to_list()[2:]  # change column name
    miRNAs = df.columns.tolist()

    Ergebnis_Anzahl_calculate_fit, F1_all_scores = genetic_one_run(
        data, df, NUMBER_OF_ITERATION, TAUSCH, MUTATION_PROB
    )

    Ergebnis_Anzahl_calculate_fit = Ergebnis_Anzahl_calculate_fit.sort_values(
        by="F1", ascending=False
    )

    if train_best:
        result_name = "/Result_10_times_"
    else:
        result_name = "/Result_"
    Ergebnis_Anzahl_calculate_fit.to_csv(
        "Ergebnisse/"
        + OUTPUT
        + result_name
        + str(COMBINATION)
        + "_miRNA_"
        + str(miRNAs)
        + ".csv",
        sep=";",
        index=False,
    )

    classifiers = " ".join(map(str, Ergebnis_Anzahl_calculate_fit.iloc[0, :-1]))
    F1 = Ergebnis_Anzahl_calculate_fit.iloc[0, -1]
    classifiers_name = " ".join(map(str, data.columns.to_list()[2:]))
    row_ENDRESULT = [
        NUMBER_OF_ITERATION,
        POPULATION_SIZE,
        MUTATION_PROB,
        TAUSCH,
        CLASSIFIER_SIZE,
        classifiers,
        F1,
        np.array(F1_all_scores).mean(),
        classifiers_name,
    ]

    return row_ENDRESULT


###########################################################
#######################Main Train##########################
###########################################################

print("\n\n\nMain Train - All Combinations\n\n\n")


OUTPUT = "Genetic2"
train_data = "liver_train_2.csv"


if not os.path.exists("Ergebnisse"):
    os.makedirs("Ergebnisse")
if not os.path.exists("Ergebnisse/" + OUTPUT):
    os.makedirs("Ergebnisse/" + OUTPUT)


df_ENDRESULT = pd.DataFrame(
    columns=[
        "number of Iteration",
        "Population size",
        "mutation_prob",
        "tausch",
        "classifier size",
        "classifiers",
        "F1",
        "AVG F1",
        "miRNAs",
    ]
)


for COMBINATION in ALL_COMBINATION:  # generate all Kombination parameters
    print(COMBINATION)
    data = lese_csv(train_data, sep=",")  # ; bei Datei 1 , bei datei 2

    data = drop_unwichte_daten(data)

    row_ENDRESULT = genetic_one_combination(data, COMBINATION)
    df_ENDRESULT.loc[len(df_ENDRESULT)] = row_ENDRESULT


df_ENDRESULT = df_ENDRESULT.sort_values(by="F1", ascending=False)
df_ENDRESULT.to_csv("Ergebnisse/" + OUTPUT + "/df_ENDRESULT.csv", sep=";", index=False)


df_ENDRESULT = lese_csv("Ergebnisse/" + OUTPUT + "/df_ENDRESULT.csv", sep=";")



###########################################################
###########10 times run for the best indivium##############
###########################################################


classifiers_name = df_ENDRESULT.iloc[0, 8]  # andere classifier 3 erste stelle ändern

NUMBER_OF_ITERATION = int(
    df_ENDRESULT.iloc[0, 0]
)
POPULATION_SIZE = int(df_ENDRESULT.iloc[0, 1])  # int(df_ENDRESULT["Population size"])
MUTATION_PROB = float(df_ENDRESULT.iloc[0, 2])  # float(df_ENDRESULT["mutation_prob"])
TAUSCH = int(df_ENDRESULT.iloc[0, 3])  # int(df_ENDRESULT["TAUSCH"])
CLASSIFIER_SIZE = int(df_ENDRESULT.iloc[0, 4])  # int(df_ENDRESULT["classifier size"])

print("\n\n\nChoose Best Individum and Run it 10 times\n\n\n")

print("-- BEST Individum")
print("NUMBER_OF_ITERATION = ", NUMBER_OF_ITERATION)
print("POPULATION_SIZE = ", POPULATION_SIZE)
print("MUTATION_PROB = ", MUTATION_PROB)
print("TAUSCH = ", TAUSCH)
print("CLASSIFIER_SIZE = ", CLASSIFIER_SIZE)


COMBINATION = [
    NUMBER_OF_ITERATION,
    POPULATION_SIZE,
    MUTATION_PROB,
    TAUSCH,
    CLASSIFIER_SIZE,
]


df_ENDRESULT = pd.DataFrame(
    columns=[
        "number of Iteration",
        "Population size",
        "mutation_prob",
        "tausch",
        "classifier size",
        "classifiers",
        "F1",
        "AVG F1",
        "miRNAs",
    ]
)


print("\n\n\nTrain 10 times\n\n\n")


for i in range(10):

    print("Training again " + str(i + 1))

    data = lese_csv(train_data, sep=",")  # ; bei Datei 1 , bei datei 2

    data = drop_unwichte_daten(data)

    start = datetime.now()
    row_ENDRESULT = genetic_one_combination(data, COMBINATION, train_best=True)
    df_ENDRESULT.loc[len(df_ENDRESULT)] = row_ENDRESULT

    # measure time now
    stop = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    stop_time = stop.strftime("%H:%M:%S")
    # calculate difference time
    difference = stop - start
    print("Start Time =", start_time)
    print("Stop Time =", stop_time)
    print("Difference Time =", difference)
    print("\n\n")

df_ENDRESULT = df_ENDRESULT.sort_values(by="F1", ascending=False)
df_ENDRESULT.to_csv(
    "Ergebnisse/" + OUTPUT + "/df_ENDRESULT_10_times.csv", sep=";", index=False
)


###########################################################
#######################Main Test###########################
###########################################################

print("\n\n\nMain Test\n\n\n")

test_data = "liver_test_2.csv"
data = lese_csv(test_data, sep=",")  # [;] bei Datei 1 | [,] bei datei 2

classifiers_name = df_ENDRESULT.iloc[0, 8]
classifiers_name = ["ID", "Annots"] + list(classifiers_name.split(" "))
F1 = int(df_ENDRESULT.iloc[0, 6])
classifiers = df_ENDRESULT.iloc[0, 5]
classifiers = [float(x) for x in list(classifiers.split(" "))]
print("F1 = ", F1)
print("classifiers = ", classifiers)
print("classifiers_name = ", classifiers_name)

selected_data = data[classifiers_name]
print("selected_data = ", selected_data)

df_TESTRESULT_columns = classifiers_name + ["prediction_int", "prediction_str"]
print(df_TESTRESULT_columns)
df_TESTRESULT = pd.DataFrame(columns=df_TESTRESULT_columns)

# position of zeros in classifiers
classifiers_zero_position = np.where(np.array(classifiers) == 0.0)[0]
print("classifiers_zero_position= ", classifiers_zero_position)


Result = []
for index, row in selected_data.iterrows():
    print(list(row))

    row_negation = []
    row_classifiers = list(row)[2:]
    for i in range(len(classifiers)):
        if classifiers[i] == 1.0:
            row_negation.append(float(row_classifiers[i]))
        else:
            if row_classifiers[i] == 0.0:
                row_negation.append(1.0)
            else:
                row_negation.append(0.0)

    row_negation = np.array(row_negation)

    count1 = np.count_nonzero(row_negation == 1.0)
    count0 = np.count_nonzero(row_negation == 0.0)

    if count1 > count0:
        prediction_int = 1
        prediction_str = "cancer"
    else:
        prediction_int = 0
        prediction_str = "healthy"

    blat = list(row) + [prediction_int, prediction_str]
    df_TESTRESULT.loc[len(df_TESTRESULT)] = blat

    TN, FP, TP, FN = calculate_fit_test(
        np.array(classifiers), row_negation, list(row)[1]
    )
    Result.append([TN, FP, TP, FN])


Result = np.asarray(Result)
print("Result=", Result)
TN, FP, TP, FN = Result.sum(axis=0)
print("TN,FP,TP,FN=", TN, FP, TP, FN)
F1 = 2 * TP / (2 * TP + FP + FN)
print("F1=", F1)


#print(df_TESTRESULT)
df_TESTRESULT.to_csv(
    "Ergebnisse/" + OUTPUT + "/df_TESTRESULT2.csv", sep=";", index=False
)
