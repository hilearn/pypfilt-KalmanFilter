import argparse
import pandas
import os
import numpy as np
import xarray


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-v', '--variable', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args(*argument_array)
    return args


def convert_csv_to_netcdf(csv_path: str, netcdf_path: str, var='rets'):
    csv_data = pandas.read_csv(csv_path)[[var]]
    data = np.array(csv_data)
    nc_data = xarray.Dataset({'time': ('nr',
                                       [float(i) for i in range(0, data.shape[0])]), # noqa
                              var: (('nr', 'np'), data)})
    nc_data.to_netcdf(netcdf_path)


def convert_netcdf_to_csv(csv_path: str, netcdf_path: str, var='rets'):
    nc_data = xarray.open_dataset(netcdf_path).to_dataframe()
    nc_data[[var]].to_csv(csv_path)


if __name__ == '__main__':
    args = parse_args()
    filename, file_extension = os.path.splitext(args.input)
    if file_extension == '.nc':
        convert_netcdf_to_csv(csv_path=args.output,
                              netcdf_path=args.input,
                              var=args.variable)
    else:
        convert_csv_to_netcdf(csv_path=args.input,
                              netcdf_path=args.output,
                              var=args.variable)
