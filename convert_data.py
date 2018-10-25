import argparse
import netCDF4


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('--rename', nargs=2, required=True)
    parser.add_argument('-o', '--outfile', required=True)
    args = parser.parse_args(*argument_array)
    return args


if __name__ == '__main__':
    args = parse_args()

    with netCDF4.Dataset(args.infile) as source:
        with netCDF4.Dataset(args.outfile, "w") as destination:
            # copy global attributes all at once via dictionary
            destination.setncatts(source.__dict__)
            # copy dimensions
            for name, dimension in source.dimensions.items():
                print('name', name)
                destination.createDimension(
                    name,
                    len(dimension) if not dimension.isunlimited() else None)
            # copy all file data except for the excluded
            for s_name, variable in source.variables.items():
                # Rename if necessary
                d_name = s_name if s_name != args.rename[0] else args.rename[1]
                destination.createVariable(d_name, variable.datatype,
                                           variable.dimensions)
                print(s_name, d_name)
                destination[d_name][:] = source[s_name][:]
                # copy variable attributes all at once via dictionary
                destination[d_name].setncatts(source[s_name].__dict__)
