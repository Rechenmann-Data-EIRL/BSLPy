from datetime import datetime
from datetime import timedelta
import os
import re

import xlrd


class LabBookLoader:
    def __init__(self):
        pass

    def load_notebook_from_fileobj(self, fileobj):
        information = {}
        wb = xlrd.open_workbook(file_contents=fileobj.read())
        sheet = wb.sheet_by_index(0)
        information["Experiment"] = self.__load_project(sheet)
        information["Mouse"] = self.__load_mouse(sheet)
        information["Anaesthesia"] = self.__load_anaesthesia(sheet)
        information["Electrophy"] = self.__load_electrode(sheet)
        information["Trials"] = self.__load_trials(sheet)
        row, col = self.__find_cell("Craniotomy", sheet)
        information["Craniotomy"] = {"Brain Area": self.__load_value_right(sheet, "Brain Area", [row, col]),
                                     "Size": self.__load_value_right(sheet, "Size (mmxmm)", [row, col])}
        return information

    def load_notebook(self, filepath, filename):
        information = {}
        full_path = os.path.join(filepath, filename)
        wb = xlrd.open_workbook(full_path)
        sheet = wb.sheet_by_index(0)
        information["Experiment"] = self.__load_project(sheet)
        information["Mouse"] = self.__load_mouse(sheet)
        information["Anaesthesia"] = self.__load_anaesthesia(sheet)
        information["Electrophy"] = self.__load_electrode(sheet)
        information["Trials"] = self.__load_trials(sheet)
        row, col = self.__find_cell("Craniotomy", sheet)
        information["Craniotomy"] = {"Brain Area": self.__load_value_right(sheet, "Brain Area", [row, col]),
                                     "Size": self.__load_value_right(sheet, "Size (mmxmm)", [row, col])}
        return information

    def __load_trials(self, sheet):
        start_row_electrophy, start_col_electrophy = self.__find_cell("Electrophy", sheet)
        start_row, start_col = self.__find_cell("Penetration #", sheet,
                                                range(start_row_electrophy, start_row_electrophy + 100))
        end_row = start_row + 100 if start_row + 100 < sheet.nrows else sheet.nrows
        column_names = sheet.row_values(start_row, start_col, sheet.ncols)
        column_names = [col for col in column_names if col != ""]
        column_index = start_col
        information = {}
        for column in column_names:
            information[column] = sheet.col_values(column_index, start_row + 1, end_row)
            if "time" in column:
                information[column] = self.__convert_array_to_time_array(information[column])
            column_index += 1
        return information

    def __load_electrode(self, sheet):
        start_row_type, start_col_type = self.__find_cell("electrode type", sheet)
        start_row, start_col = self.__find_cell("electrode ID", sheet)
        electrode_type = sheet.cell_value(start_row_type, start_col_type + 1)
        reg = re.compile("[0-9]{1,2}x[0-9]{1,2}")
        channel_config = reg.search(electrode_type).group().split("x")
        nb_channels = int(channel_config[0]) * int(channel_config[1])
        electrode_info = {"Electrode Type": electrode_type,
                          "Electrode ID": sheet.cell_value(start_row, start_col + 1),
                          "Nb channels": nb_channels,
                          "Channel Config": channel_config,
                          "Cortical region": self.__try_loading_column_from_name(sheet, "Cortical region", auto_resize=True),
                          "Tip depth": self.__try_loading_column_from_name(sheet, "Tip depth (um)", auto_resize=True),
                          "ML coordinates": self.__try_loading_column_from_name(sheet, "ML coordinates (um)",
                                                                         auto_resize=True),
                          "RC coordinates": self.__try_loading_column_from_name(sheet, "RC coordinates (um)",
                                                                         auto_resize=True),
                          "Penetration time": self.__convert_array_to_time_array(
                              self.__try_loading_column_from_name(sheet, "Pen time", auto_resize=True))}
        return electrode_info

    def __convert_array_to_time_array(self, array):
        return [(datetime.min + timedelta(seconds=x * 24 * 3600)).time().strftime("%H:%M") if isinstance(x, int) else ""
                for x in array]

    def __load_value_right(self, sheet, names, start_position=None, size=None, exact=True):
        if not isinstance(names, list):
            names = [names]
        for name in names:
            try:
                if start_position is None:
                    start_position = [0, 0]
                if size is None:
                    size = [sheet.nrows, sheet.ncols]
                row, col = self.__find_cell(name, sheet, list(range(start_position[0], size[0])),
                                            list(range(start_position[1], size[1])),
                                            exact=exact)
                return sheet.cell_value(row, col + 1)
            except AttributeError:
                pass
        raise AttributeError("Value " + " ,".join(names) + " not found in sheet.")

    def __load_column_from_position(self, sheet, start_position, height=None, auto_resize=False):
        if height is None or auto_resize:
            values = sheet.col_values(start_position[1], start_position[0], sheet.nrows - 1)
            last_index = 0
            for index in range(len(values)):
                if values[index] == "":
                    last_index = index
                    break
            return values[0:last_index]
        return sheet.col_values(start_position[0], start_position[1], height)

    def __try_loading_column_from_name(self, sheet, column_name, start_position=None, height=None, auto_resize=False):
        try:
            return self.__load_column_from_name(sheet, column_name, start_position, height, auto_resize)
        except AttributeError:
            return []

    def __load_column_from_name(self, sheet, column_name, start_position=None, height=None, auto_resize=False):
        if start_position is None or len(start_position) != 2:
            start_position = [0, 0]
        row, col = self.__find_cell(column_name, sheet, range(start_position[0], sheet.nrows),
                                    range(start_position[1], sheet.ncols - 1))
        if height is None:
            height = sheet.nrows - row
        return self.__load_column_from_position(sheet, [row + 1, col], height, auto_resize)

    def __load_anaesthesia(self, sheet):
        information = dict()
        start_row, start_column = self.__find_cell(["Anaesthesia", "Anesthesia"], sheet)
        information["What"] = sheet.col_values(start_column, start_row + 2, 100)
        last_row = information["What"].index("")
        information["What"] = information["What"][0:last_row]
        start_time = self.__convert_array_to_time_array(
            sheet.col_values(start_column + 1, start_row + 2, 100)[0:last_row])
        information["When"] = start_time
        information["Quantity"] = sheet.col_values(start_column + 2, start_row + 2, 100)[0:last_row]
        return information

    def __find_cell(self, values, sheet, row_range=None, column_range=None, exact=True):
        if not isinstance(values, list):
            values = [values]
        for value in values:
            row_range = row_range if row_range is not None else range(sheet.nrows)
            column_range = column_range if column_range is not None else range(sheet.ncols)
            for row in row_range:
                for col in column_range:
                    if (sheet.cell_value(row, col) == value and exact) or (
                            not exact and isinstance(sheet.cell_value(row, col), str) and value in sheet.cell_value(row,
                                                                                                                    col)):
                        return row, col
        raise AttributeError("Value " + " ,".join(values) + " not found in sheet.")

    def __load_mouse(self, sheet):
        information = dict()
        row, col = self.__find_cell("Mouse", sheet)
        information["Age"] = self.__load_value_right(sheet, "Age", start_position=[row + 1, col], exact=False)
        information["DateOfBirth"] = datetime.strptime(
            str(int(self.__load_value_right(sheet, ["DoB", "DateOfBirth", "Dateofbirth(YearMonthDay)"],
                                            start_position=[row + 1, col], exact=False))), "%y%m%d")
        information["Weight"] = self.__load_value_right(sheet, "Weight (g)", start_position=[row + 1, col], exact=False)
        information["Strain"] = self.__load_value_right(sheet, "Strain", start_position=[row + 1, col], exact=False)
        information["Gender"] = self.__load_value_right(sheet, "Gender", start_position=[row + 1, col], exact=False)
        information["ID"] = self.__load_value_right(sheet, "ID", start_position=[row + 1, col], exact=False)
        return information

    def __load_project(self, sheet):
        return {"Date": sheet.cell_value(2, 1), "Experimenter": sheet.cell_value(3, 1), "ID": sheet.cell_value(2, 3)}
