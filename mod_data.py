import sys
import webbrowser
from glob import glob
from os.path import isfile
from urllib.parse import urlparse
import pandas as pd

for datafile in glob('DATA/*.csv'):
    print(datafile)
    newfile = datafile.replace(".csv", ".csv_")
    if not isfile(newfile):
        with open(datafile, 'r') as data_input:
            print("Processing", datafile)
            data = pd.read_csv(data_input, index_col="Ord")
            print(data)
            go_on = input("Action needed?")
            if go_on.lower() == "true":
                data['last'] = ["" for i in data.index]
                for i in data.index:
                    webbrowser.open(data.loc[i, 'url'])
                    to_exclude = input("Copy url for last: ")
                    to_exclude = urlparse(to_exclude).fragment
                    assert isinstance(to_exclude, str)
                    data.loc[i, "last"] = to_exclude
                print(data)
                with open(newfile, 'w') as data_output:
                    data.to_csv(data_output)
            else:
                sys.exit(1)
