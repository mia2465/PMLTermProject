
import os.path



EntireCovidSet = "/Users/amelianelson/Desktop/Imaging/manifest-1628608914773/COVID-19-NY-SBU"

with os.scandir(EntireCovidSet) as entries:
    for entry in entries:
        print(entry.name)