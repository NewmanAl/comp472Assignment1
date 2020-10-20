https://github.com/NewmanAl/comp472Assignment1

Assignment is run by executing 'assignment1.py'.
Script assumes that the following csv files provided in assignment are present in the same directory:
- info_1.csv
- info_2.csv
- train_1.csv
- train_2.csv
- test_with_label_1.csv
- test_with_label_2.csv

Execution of the script will generate the desired output csv files for each model into the directory output/, relative to the current working directory. The script will also generate diagrams for each model in the directory diagrams/, relative tot he current working directory.

dataObservation.py can be run separately to generate diagrams describing the contents of the training data and test data.

plotConfusion.py is used by the main script to generate the confusion matrix diagrams. Code for this function was retrieved from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix, with minor adjustments to allow saving the diagram to file.
