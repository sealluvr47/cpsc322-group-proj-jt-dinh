import csv
import matplotlib.pyplot as plt
import pandas as pd

def compute_euclidean_distance(p1,p2):
    result = 0

    for index in range(len(p1)):
        difference = p1[index] - p2[index]
        result += difference ** 2
    return result ** 0.5

def mpg_to_rating(mpg):
    cutoffs = [14, 15, 17, 20, 24, 27, 31, 37, 45]

    for j in range(len(cutoffs)):
        if mpg < cutoffs[j]:
            return j + 1

    return 10

def normalize_train_test(X_train_knn, X_test_knn):

    minimums = []
    maximums = []
    for i in range(3):
        columns_values = []
        for row in X_train_knn:
            columns_values.append(row[i])
        minimums.append(min(columns_values))
        maximums.append(max(columns_values))


    X_train_normalized = []
    for row in X_train_knn:
        normalized_row = []
        for i in range(3): 
            difference = maximums[i] - minimums[i]
            if difference == 0:
                value = 0
            else:
                value = (row[i] - minimums[i]) / difference
            normalized_row.append(value)
        X_train_normalized.append(normalized_row)


    X_test_normalized = []
    for row in X_test_knn:
        normalized_row = []
        for i in range(3):
            difference = maximums[i] - minimums[i]
            if difference == 0:
                value = 0
            else:
                value = (row[i] - minimums[i]) / difference
            normalized_row.append(value)
        X_test_normalized.append(normalized_row)


    return X_train_normalized, X_test_normalized


def create_box_plot(groupings, labels, title, xlabel, ylabel):
    plt.figure()
    plt.boxplot(groupings)
    position = list(range(1, len(labels) + 1))

    plt.xticks(position, labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def bar_chart(x, y, title, xlabel, ylabel):
    plt.figure()

    positions = list(range(len(x)))
    plt.bar(positions, y)
    plt.xticks(positions, x, rotation = 15, ha ="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def scatter_chart_example(x, y):
    plt.figure()
    plt.scatter(x, y, marker="x", s=100, c="purple")


    plt.tight_layout()
    plt.savefig("scatter_chart.pdf")
    plt.show()

def line_chart_example(x, y2, y3):
    plt.figure()
    plt.plot(x, y2, label="$y=x^2$")
    plt.plot(x, y3, c="green", lw=2, marker='o', label="$y=x^3$")

    plt.legend()
    plt.title("Our First Line Chart")
    plt.xlabel("X Units")
    plt.ylabel("Y Units")
    plt.grid(True)

    xtick_labels = ["zero", "one", "two", "three super long label to overlap", "four"]
    plt.xticks(x, xtick_labels, rotation=15, ha="right")
    plt.show()

def histogram_example(data, data2):

    plt.figure()
    plt.hist(data, bins=10, facecolor="green", edgecolor="black")
    plt.hist(data2, bins=10)
    plt.show()

def create_histogram(data, title, xlabel, ylabel):
    plt.figure()
    plt.hist(data, bins=10, facecolor="blue", edgecolor="black") 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def create_scatter_plot(x, y, title, xlabel, ylabel):
    plt.figure() 
    plt.scatter(x, y, marker="x", s=100, c="purple")
    plt.tight_layout() 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def find_rows(table, text):
    """
    returns a row that matches text
    """
    index = table.column_names.index("car name")
    matches = []
    for row in table.data:
        if text in row[index]:
            matches.append(row)
    return matches

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    return [row[col_index] for row in table]

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
    frequencies = {}

    for value in col:
        frequencies[value] = frequencies.get(value, 0) + 1 
    unique_col_values = sorted(frequencies)
    counts = []

    for val in unique_col_values:
        counts.append(frequencies[val])

    return unique_col_values, counts

def remove_missing_values(table, header, col_name):
    """Makes a new table that is a copy of the table parameter (without modifying the original
    table), but with rows removed from the table that have missing values in the column with
    label col_name

    Args:
        table (list of list): Data in 2D table format
        header (list of str): Column names corresponding to the table (in the same order)
        col_name (str): Represents the name of the column to check for missing values

    Returns:
        list of list: The new table with removed rows
    """

    clean_data = []

    index = header.index(col_name)
    for row in table:
        if row[index] != "":
            clean_data.append(row.copy())

    return clean_data

def load_data(filename):
    with open(filename, "r", newline='') as file: 
        reader = csv.reader(file)
        table = list(reader) 
        header = table.pop(0)
    
    return table, header


def partition_instances(instances, attribute, header, attribute_domains):


    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class(instances):

    first_class = instances[0][-1]
    for instance in instances:

        if instance[-1] != first_class:
            return False
        

    return True


def tdidt_predict(tree, instance, header):
    data_type = tree[0]


    if data_type == "Leaf":
        label = tree[1]
        return label
    

    attribute_name = tree[1]
    attribute_index = header.index(attribute_name)
    instance_value = instance[attribute_index]


    for values in tree[2:]:
        value = values[1]
        subtree = values[2]
        
        if instance_value == value:
            return tdidt_predict(subtree, instance, header)


def fill_df_missing_values(df, fill_str="NA"):
    """Utility function
    """
    df_joined_as_list = df.values.tolist()
    for i, _ in enumerate(df_joined_as_list):
        for j, _ in enumerate(df_joined_as_list[i]):
            df_joined_as_list[i][j] = fill_str if pd.isnull(df_joined_as_list[i][j]) \
                else df_joined_as_list[i][j]
    return df_joined_as_list


def discretizer(y):
    if y >= 100:
        return "high"
    else: 
        return "low"


def get_min_label_counts(y, label, n_splits):
    """Utility function
    """
    label_counts = sum([1 for yval in y if yval == label])
    min_test_label_count = label_counts // n_splits
    min_train_label_count = (n_splits - 1) * min_test_label_count
    return min_train_label_count, min_test_label_count
