import copy
import csv

from tabulate import tabulate


class MyPyTable:

    def __init__(self, column_names=None, data=None):
        if column_names is None:
            column_names = []

        if data is None:
            data = []

        self.column_names = copy.deepcopy(column_names)
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        rows = len(self.data)
        cols = len(self.column_names)
        return rows, cols

    def get_column(self, col_identifier, include_missing_values=True):
        if isinstance(col_identifier, str):
            try:
                col_index = self.column_names.index(col_identifier)
            except ValueError:
                raise ValueError("Unknown Column Name: " + col_identifier)
        else:
            col_index = col_identifier

        new_column = []

        for row in self.data:
            if include_missing_values == True or row[col_index] != "NA":
                new_column.append(row.copy()[col_index])

        return new_column

    def convert_to_numeric(self):
        for row in self.data:
            for i in range(len(row)):
                if row[i] != "NA":
                    try:
                        row[i] = float(row[i])
                    except (ValueError, TypeError):
                        pass

    def drop_rows(self, row_indexes_to_drop):
        row_indexes_to_drop = set(row_indexes_to_drop)
        remainder = []

        for i, row in enumerate(self.data):
            if i not in row_indexes_to_drop:
                remainder.append(row)
        self.data = remainder

    def load_from_file(self, filename):
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            self.column_names = next(reader)
            self.data = list(reader)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        indexes = []
        for names in key_column_names:
            indexes.append(self.column_names.index(names))

        seen = []
        duplicates = []

        for i, rows in enumerate(self.data):
            key = tuple(rows[index] for index in indexes)

            if key in seen:
                duplicates.append(i)
            else:
                seen.append(key)
        return duplicates

    def remove_rows_with_missing_values(self):
        pass

        reaminder = []

        for rows in self.data:
            if "NA" not in rows:
                reaminder.append(rows)
        self.data = reaminder

    def replace_missing_values_with_column_average(self, col_name):
        col_index = self.column_names.index(col_name)
        values = self.get_column(col_index, include_missing_values=False)

        column_avg = sum(values) / float(len(values))

        for rows in self.data:
            if rows[col_index] == "NA":
                rows[col_index] = column_avg

    def compute_summary_statistics(self, col_names):
        results = []
        for name in col_names:
            values = self.get_column(name, include_missing_values=False)

            if not values:
                continue

            values = sorted(values)
            minimum = min(values)
            maximum = max(values)
            mid = (minimum + maximum) / 2
            avg = sum(values) / float(len(values))

            middle = len(values) // 2

            if len(values) % 2 == 0:
                median = (values[middle - 1] + values[middle]) / 2
            else:
                median = values[middle]

            results.append([name, minimum, maximum, mid, avg, median])
        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], results)

    def perform_inner_join(self, other_table, key_column_names):
        left = []
        right = []

        for data in key_column_names:
            left.append(self.column_names.index(data))
            right.append(other_table.column_names.index(data))

        columns = self.column_names.copy()
        waste = []
        for i, name in enumerate(other_table.column_names):
            if name not in key_column_names:
                columns.append(name)
                waste.append(i)

        result = []

        for a in self.data:
            for b in other_table.data:
                match = True

                for i, name in enumerate(key_column_names):
                    if a[left[i]] != b[right[i]]:
                        match = False
                        break
                if match:
                    row = a.copy()
                    for i, name in enumerate(other_table.column_names):
                        if name not in key_column_names:
                            row.append(b[i])
                    result.append(row)

        return MyPyTable(columns, result)

    def perform_full_outer_join(self, other_table, key_column_names):
        result = self.perform_inner_join(other_table, key_column_names)

        left = []
        right = []

        for data in key_column_names:
            left.append(self.column_names.index(data))
            right.append(other_table.column_names.index(data))

        left_key = []
        right_key = []
        for rows in self.data:
            temp = []
            for i in left:
                temp.append(rows[i])
            left_key.append(temp)

        for rows in other_table.data:
            temp = []
            for i in right:
                temp.append(rows[i])
            right_key.append(temp)

        for i, rows in enumerate(self.data):
            if left_key[i] not in right_key:
                new_row = rows.copy()

                while len(new_row) < len(result.column_names):
                    new_row.append("NA")
                result.data.append(new_row)

        for i, rows in enumerate(other_table.data):
            if right_key[i] not in left_key:
                new_row = []
                for name in result.column_names:
                    if name in other_table.column_names:
                        new_row.append(rows[other_table.column_names.index(name)])
                    else:
                        new_row.append("NA")
                result.data.append(new_row)

        return result
