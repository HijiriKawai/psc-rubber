from xlsx2csv import Xlsx2csv
import os

home = os.environ['HOME'];

for number in range(1,11):
  xlsx_path = f"{home}/data/isodisplacement{number}.xlsx"
  csv_path = f"{home}/data/isodisplacement{number}.csv"
  Xlsx2csv(xlsx_path, outputencoding="utf-8").convert(csv_path)