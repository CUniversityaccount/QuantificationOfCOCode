import numpy as np
from netCDF4 import Dataset

import copy
import sys


def makes_grid(place, nlon_t, nlat_t):
    """
    Makes the grid to put the data in and get the correct mask for the parsing
    It will only the grid for the area, so there is little ram used
    """

    lon = (np.arange(nlon_t) + 0.5) * 360./np.float(nlon_t) - 180.
    lat = (np.arange(nlat_t) + 0.5) * 180./np.float(nlat_t) - 90.
    lon = lon[(lon >= place[0]) & (lon <= place[1])]
    lat = lat[(lat >= place[2]) & (lat <= place[3])]
    lon, lat = np.meshgrid(lon, lat)
    return lon, lat

class WRF_data:
    """
    loads data
    needs to know for where the data is from
    """
    def __init__(self, resolution, place):
        self.CO = {}
        self.kernel = None
        self.ps = None
        self.p_levels = None

        self.lon, self.lat = makes_grid(place=place, nlon_t=resolution[0], nlat_t=resolution[1])
        self.place = place

        self.time = None
        self.max = None

    def load_data(self, file, day, time):
        """
        Loads the time of the TROPOMI dataset in the correct array
        """
        print("Load", file)

        CO = {}

        try:
            fid = Dataset(file)
            lon = fid.variables["lon"][:].data[time, :, :].ravel()
            lat = fid.variables["lat"][:].data[time, :, :].ravel()
            ps = fid.variables["ps"][:].data[time, :, :].ravel()
            self.kernel = fid.variables["eta"][:].data[time, :]
            self.time = day
            for name in list(fid.variables.keys())[4:]:
                CO[name] = copy.deepcopy(fid.variables[name][:].data[time, :, :, :])
                CO[name] = CO[name].reshape(CO[name].shape[0], -1)

                # get the max
                maximum = np.max(CO[name])
                if self.max is None or self.max < maximum:
                    self.max = copy.deepcopy(maximum)

        except KeyboardInterrupt:
            sys.exit(1)
        # except:
        #     print("Error")
        #     return False
        else:
            fid.close()

        self.conform_data(lat=lat, lon=lon, CO=CO, ps=ps)

    # conforms the data toward the TROPOMI dataset
    def conform_data(self, lat, lon, CO, ps):

        # makes empty arrays
        self.ps, count = (np.zeros((self.lon.shape[0], self.lon.shape[1])) for _ in range(2))
        CO_shape = None

        # makes an empty array for the different keys in CO
        for key in CO.keys():
            self.CO[key] = copy.deepcopy(np.zeros((self.lon.shape[0], self.lon.shape[1], CO[key].shape[0])))

            if CO_shape is None:
                CO_shape = self.CO[key].shape

        # puts the data at the right coordinates
        for iobs in range(len(lon)):

            ilon = np.int((lon[iobs] - self.place[0]) * self.lon.shape[0] / (self.place[1] - self.place[0]))
            ilat = np.int((lat[iobs] - self.place[2]) * self.lon.shape[1] / (self.place[3] - self.place[2]))

            try:
                self.ps[ilat, ilon] += ps[iobs]

                for key in CO.keys():
                    self.CO[key][ilat, ilon, :] += CO[key][:, iobs]

                count[ilat, ilon] += 1
            except IndexError:
                pass

        mask_count = (count != 0)
        self.ps[mask_count] = self.ps[mask_count] / count[mask_count]
        mask_count = np.reshape(mask_count, (mask_count.shape[0], mask_count.shape[1], 1)) * np.ones(CO_shape)
        mask_count = mask_count.astype(bool)
        count = np.reshape(count, (mask_count.shape[0], mask_count.shape[1], 1)) * np.ones(CO_shape)

        for key in self.CO.keys():
            self.CO[key][mask_count] = self.CO[key][mask_count] / count[mask_count]

    def make_p_levels(self):
        top = 5000  # in Pa
        p_levels = np.zeros((self.ps.shape[0], self.ps.shape[1], self.kernel.shape[0]))

        # calculates the pressure on every height
        for height, kernel in enumerate(self.kernel):
            p_levels[:, :, height] = kernel * (self.ps - top) + top

        self.p_levels = p_levels


class TROPOMI_data:
    def __init__(self, resolution, place):
        self.CO = None
        self.kernel = None
        self.ppm = None

        self.ps = None
        self.p_levels = None
        self.dry_air = None

        self.place = place
        self.lon, self.lat = makes_grid(place=place, nlon_t=resolution[0], nlat_t=resolution[1])
        self.time = []

        self.u10 = None
        self.v10 = None
        self.count = None

        size_array_vertical = 50
        self.initiate_array(size_array_vertical)

    def get_information(self, file):
        try:
            fid = Dataset(file)
            pcq = fid.groups["diagnostics"].variables["processing_quality_flags"][:]
            qf = fid.groups["diagnostics"].variables["qa_value"][:].data

            mask = fid.groups['target_product'].variables['co_column'][:].mask
            field = fid.groups['target_product'].variables['co_column'][:].data
            kernel = np.flip(fid.groups["target_product"].variables['co_column_averaging_kernel'][:].data, 1)
            print(fid.groups["meteo"].variables["dry_air_subcolumns"])
            sys.exit()
            dry_air = np.flip(fid.groups["meteo"].variables["dry_air_subcolumns"][:].data, 1)
            surface_pressure = fid.groups["meteo"].variables["surface_pressure"][:].data
            pressure_column = np.flip(fid.groups["meteo"].variables["pressure_levels"][:].data, 1)
            u10 = fid.groups["meteo"].variables["u10"][:].data
            v10 = fid.groups["meteo"].variables["v10"][:].data

            lon = fid.groups['instrument'].variables['longitude_center'][:].data
            lat = fid.groups['instrument'].variables['latitude_center'][:].data
            time = fid.groups["instrument"].variables["time"][:].data
        except KeyboardInterrupt:
            sys.exit(1)
        # except:
        #     print("Error")
        #     return False
        else:
            fid.close()

        # mask the correct data out
        mask = ~mask & (pcq < 2) & (self.place[0] <= lon) & (self.place[1] >= lon) \
            & (self.place[2] <= lat) & (self.place[3] >= lat)
        field = field[mask]
        kernel = kernel[mask]
        surface_pressure = surface_pressure[mask]
        pressure_column = pressure_column[mask]
        dry_air = dry_air[mask]
        u10 = u10[mask]
        v10 = v10[mask]

        lon = lon[mask]
        lat = lat[mask]

        # checks if there is still data after mask
        if len(field) > 0:

            # reduce data to resolution
            for iobs in range(len(lon)):

                ilon = np.int((lon[iobs] - self.place[0]) * self.lon.shape[0] / (self.place[1] - self.place[0]))
                ilat = np.int((lat[iobs] - self.place[2]) * self.lon.shape[1] / (self.place[3] - self.place[2]))

                try:
                    # place the data on the correct place
                    self.CO[ilat, ilon] += field[iobs]
                    self.kernel[ilat, ilon, :] += kernel[iobs, :]

                    # places correct meteo data
                    self.dry_air[ilat, ilon, :] += dry_air[iobs, :]
                    self.ps[ilat, ilon] += surface_pressure[iobs]
                    self.p_levels[ilat, ilon, :] += pressure_column[iobs, :]
                    self.u10[ilat, ilon] += u10[iobs]
                    self.v10[ilat, ilon] += v10[iobs]

                    self.count[ilat, ilon] += 1
                except IndexError:
                    pass

            # add time in hours
            self.time.append({"first": time[0, 2], "last": time[-1, 2]})
        else:
            return False
        print("FINISH")
        return True

    def initiate_array(self, shape):
        self.u10, self.ps, self.v10, self.CO, self.count = (np.zeros(self.lon.shape) for _ in range(5))
        self.kernel, self.dry_air = (np.zeros((self.lon.shape[0], self.lon.shape[1], shape)) for _ in range(2))
        self.p_levels = np.zeros((self.lon.shape[0], self.lon.shape[1], shape + 1))

    def mean_CO(self):
        count_mask = (self.count > 0)

        # makes from variables a list to mask over
        variables_masking_2d = [self.CO, self.u10, self.v10, self.ps]

        for variable in variables_masking_2d:
            variable[count_mask] = variable[count_mask] / self.count[count_mask]

        del variables_masking_2d

        # applies mask over multiple 3D arrays and get the mean of all the data
        count_mask = np.reshape(count_mask, (count_mask.shape[0], count_mask.shape[1], 1))
        count = np.reshape(self.count, count_mask.shape)
        variables_masking_3d = [self.dry_air, self.kernel, self.p_levels]

        for variable in variables_masking_3d:
            new_count_mask = count_mask * np.ones(variable.shape[-1])
            new_count = count * np.ones(variable.shape[-1])
            new_count_mask = new_count_mask.astype("bool")

            variable[new_count_mask] = variable[new_count_mask] / new_count[new_count_mask]

        del self.count

    def calculation_ppm(self):

        # calculates the ppm over the whole column not the parially
        ppm_kernel = self.dry_air * self.kernel
        ppb_kernel = np.sum(ppm_kernel, axis=2)
        ppb = (self.CO / ppm_kernel) * 10**9
        self.parse_co_data(ppb)

    def parse_co_data(self, ppm):
        new_ppm = np.zeros(self.kernel.shape)
        total_pressure = self.ps

        # mask if there is data in that pixel
        ctrl_mask = (ppm != 0)

        # goes through the tropomi layers
        for height in range(self.kernel.shape[-1]):
            fraction_p = np.zeros(self.lon.shape)
            fraction_p[ctrl_mask] = (self.p_levels[:, :, height][ctrl_mask] - self.p_levels[:, :, height + 1][ctrl_mask]) \
                                    / total_pressure[ctrl_mask]

            new_ppm[:, :, height] = copy.deepcopy(fraction_p * ppm)

        self.ppm = new_ppm
