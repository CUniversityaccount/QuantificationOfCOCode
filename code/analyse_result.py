from classes_analyse import WRF_data, TROPOMI_data
from optimization_tools import OPTIM

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import sys
import copy
import os


def read_files_TROPOMI(file_list, place, resolution):
    """
    Does read the files and gets the correct data_array
    """
    files = {}
    date = None
    data = None
    print(file_list)
    # goes through fileList
    for file in file_list:

        if "20180715" in file and int(file[6:8]) <= 23:
            if date is None:
                data = TROPOMI_data(resolution = resolution, place  = place)
                date = file[:8]
            elif date != file[:8] or file == file_list[-1]:
                print("New day", file[:8])
                data.mean_CO()
                data.calculation_ppm()
                files[date] = copy.deepcopy(data)
                date = file[:8]
                data = TROPOMI_data(resolution = resolution, place  = place)

            check = data.get_information(file = file)
            if check:
                print("New Data", file)
            else:
                print("No data in", file)
        else:
            # if data is not None:
            #     data.mean_CO()
            #     files[date] = copy.deepcopy(data)

            break

    if data is not None:
        data.mean_CO()
        data.calculation_ppm()
        files[date] = copy.deepcopy(data)

    return files


def plot_TROPOMI(files):
    """
    Plot the days of TROPOMI
    """
    fig, axs = plt.subplots(nrows=3, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.ravel()

    for count, day in enumerate(files.keys()):
        data = files[day]
        ax = axs[count]
        print("min", np.nanmin(data.CO))
        gl = ax.gridlines(crs=ccrs.PlateCarree(),  draw_labels=True)
        print(data.ppm)
        color = ax.pcolormesh(data.lon, data.lat, np.nansum(data.ppm, axis=2), cmap="jet", vmin=65, vmax=170, transform=ccrs.PlateCarree())

        # add the Gridlines, make picture up
        gl.xlines, gl.ylines, gl.xlabels_top, gl.ylabels_right = (False for _ in range(4))
        gl.xlocator = mticker.FixedLocator(np.arange(data.place[0] + 1, data.place[1], 2))
        gl.ylocator = mticker.FixedLocator(np.arange(data.place[2] + 1 , data.place[3] + 1, 2))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style, gl.ylabel_style  = ({'size': 6, 'color': 'black'} for _ in range(2))
        ax.set_title(day[:4] + "-" + day[4:6] + "-" + day[6:], size=6)

    fig.tight_layout()
    cbar = fig.colorbar(color, ax=axs, orientation="vertical", extend="both", aspect=20, fraction=.12, pad=.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("CO [ppb]", size=6)

    plt.savefig(day + "_test.png", dpi=400 )


def read_file_simulation(file_map, place, resolution, tropomi):
    os.chdir(file_map)
    data_points = np.array(os.listdir())
    data_points = data_points[:]
    files = {}

    # load the wrf files into a dict with the correct days
    for file in data_points[:1]:
        data = WRF_data(place=place, resolution=resolution)
        day = file[11:-4]
        print("File", tropomi)

        time_list = tropomi[day].time
        max_time = None

        # get the right time to get for the WRF data for a better comparison
        for time_object in time_list:
            time = np.max(list(time_object.values()))
            if max_time is None or max_time < time:
                max_time = time

        data.load_data(file=file, day=day, time=max_time)
        data.make_p_levels()
        files[day] = copy.deepcopy(data)

    return files


def plot_wrf(files):
    """
    plot the days in a 3D graph
    """

    for day in files.keys():
        fig, axs = plt.subplots(nrows=3, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()})
        axs = axs.ravel()
        data = files[day]

        max = None

        for count, lvl in enumerate(data.CO.keys()):
            ax = axs[count]
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            CO = copy.deepcopy(data.CO[lvl])

            print(CO.shape)

            if max is None:
                max = np.nanmax(CO)

            color = ax.pcolormesh(data.lon, data.lat, np.nansum(CO, axis=2), cmap="jet", vmin=0, vmax=0.005, transform=ccrs.PlateCarree())

            # add the Gridlines, make picture up
            gl.xlines, gl.ylines, gl.xlabels_top, gl.ylabels_right = (False for _ in range(4))
            gl.xlocator = mticker.FixedLocator(np.arange(data.place[0] + 1, data.place[1], 2))
            gl.ylocator = mticker.FixedLocator(np.arange(data.place[2] + 1, data.place[3] + 1, 2))
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style, gl.ylabel_style = ({'size': 10, 'color': 'black'} for _ in range(2))

        fig.tight_layout()
        cbar = fig.colorbar(color, ax=axs, orientation="vertical", extend="both")
        cbar.set_label("CO [ppb]", size=10)
        cbar.ax.tick_params(labelsize=10)
        plt.show()
        plt.savefig(day + "_test.png" ,bbox_inches='tight', dpi=900)


def fit_pressure_column(wrf_files, tropomi_files):
    """
    Composite the WRF air column towards TROPOMI_data
    """
    print("BEGIN LOADING")

    for day in wrf_files.keys():
        tropomi_file = tropomi_files[day]
        wrf_file = wrf_files[day]

        p_levels_wrf = wrf_file.p_levels
        p_levels_TROPOMI = tropomi_file.p_levels
        CO_WRF = wrf_file.CO

        # makes the boundary  where to begin and where to stop
        top_boundary = (p_levels_TROPOMI > 5000)

        # makes new array for the WRF CO data to conform to the data format of TROPOMI
        new_WRF_co = {}
        for date in CO_WRF.keys():
            new_WRF_co[date] = np.zeros(tropomi_file.dry_air.shape)

        # sets the new pressure layers for WRF model this 1:1 with TROPOMI
        new_p_levels = np.zeros(tropomi_file.p_levels.shape)
        new_p_levels[:, :, 0] = tropomi_file.ps

        # makes the tropomi height grid
        lvl_tropo= np.zeros(tropomi_file.ps.shape, dtype="intc")
        lat, lon = np.ogrid[:p_levels_TROPOMI.shape[0], :p_levels_TROPOMI.shape[1]]

        # # checks if the TROPOMI is lower than the surface of the WRF model is
        for height in range(p_levels_TROPOMI.shape[-1]):
            wrf_fraction = np.zeros(tropomi_file.lon.shape)
            lvl_mask = (p_levels_TROPOMI[:, :, height + 1] > wrf_file.ps) & (tropomi_file.ps != 0)
            old_lvl_mask = (p_levels_TROPOMI[:, :, height] > wrf_file.ps) & (tropomi_file.ps != 0)

            # checks if the whole column
            double_layer = old_lvl_mask & lvl_mask

            # checks if that a full layer difference between TROPOMI ground pressure and WRF ground pressure
            if np.any(double_layer):
                lvl_tropo[double_layer] = lvl_tropo[double_layer] + 1
                wrf_fraction += p_levels_TROPOMI[:, :, height][double_layer] - \
                                p_levels_TROPOMI[:, :, height + 1][double_layer]

            # checks how much of TROPOMI column is part of the difference
            single_layer = old_lvl_mask & ~lvl_mask
            if np.any(single_layer):
                wrf_fraction[single_layer] += (p_levels_TROPOMI[:, :, height][single_layer] - wrf_file.ps[single_layer])
                wrf_fraction = wrf_fraction / (p_levels_wrf[:, :, 0] - p_levels_wrf[:, :, 1])

                for date in CO_WRF.keys():
                    new_WRF_co[date][:, :, 0][single_layer] = CO_WRF[date][:, :, 0][single_layer] * \
                                                              wrf_fraction[single_layer]

            if np.all(~double_layer):
                print("END LOOP FOR BENEATH WRF SURFACE")
                break

        # redefines the air column from WRF format to TROPOMI format
        for h in range(0, wrf_file.p_levels.shape[-1]):
            wrf_fraction = np.zeros(tropomi_file.lon.shape)
            lvl_mask = (p_levels_TROPOMI[lat, lon, lvl_tropo] > p_levels_wrf[:, :, h]) & (tropomi_file.ps != 0)

            # checks if the mask has an influence on the dataset
            if np.any(lvl_mask):

                # applies the correct equations for the different situation
                if h != 0:
                    wrf_fraction[lvl_mask] = (p_levels_TROPOMI[lat, lon, lvl_tropo][lvl_mask] - p_levels_wrf[:, :, h][lvl_mask]) / \
                                             (p_levels_wrf[:, :, h - 1][lvl_mask] - p_levels_wrf[:, :, h][lvl_mask])
                elif h != wrf_file.p_levels.shape[-1] - 1:
                    wrf_fraction[~lvl_mask] = (p_levels_wrf[:, :, h][~lvl_mask] - p_levels_TROPOMI[lat, lon, lvl_tropo][~lvl_mask]) / \
                                             (p_levels_wrf[:, :, h][~lvl_mask] - p_levels_wrf[:, :, (h + 1)][~lvl_mask])

                # will put the correct amount of CO in the correct tray
                for key_CO in new_WRF_co.keys():
                    new_WRF_co[key_CO][lat, lon, lvl_tropo] += copy.deepcopy(CO_WRF[key_CO][:, :, h - 1] * wrf_fraction)

            if h != 0:
                lvl_tropo[lvl_mask] = lvl_tropo[lvl_mask] + 1

            # checks if the top boundary true
            if np.all(~top_boundary[lat, lon, lvl_tropo]):
                print("END LOOP, TOP BOUNDARY")
                break

        # adapt the averaging kernel at towards the WRF data
        for key_CO in new_WRF_co.keys():
            new_WRF_co[key_CO] = copy.deepcopy(new_WRF_co[key_CO] * tropomi_file.kernel)
            new_WRF_co[key_CO][new_WRF_co[key_CO] == 0] = np.NaN

        wrf_file.CO = copy.deepcopy(new_WRF_co)

    return wrf_files


def statistical(tropomi, wrf):
    """"
    Calculates the best fit between the tropomi data and wrf data
    """
    results = {}

    for day in tropomi.keys():
        tropomi_day = tropomi[day]
        co_tropomi = copy.deepcopy(np.nansum(tropomi_day.ppm, axis=2))
        mask = (co_tropomi == 0)
        co_tropomi[mask] = np.NaN
        statistical_tropo = (co_tropomi > np.nanpercentile(co_tropomi, q=70)) & (co_tropomi < np.nanpercentile(co_tropomi, q=100))
        co_tropomi[~statistical_tropo] = np.NaN

        mask_nan = ~np.isnan(co_tropomi)
        co_tropomi = co_tropomi[mask_nan]
        print(len(co_tropomi))

        # makes the array for and loads the correct day
        wrf_day = wrf[day]
        wrf_levels = np.zeros((len(co_tropomi.ravel()), len(wrf_day.CO.keys()) + 1))
        best_fit = None

        # loads the WRF data in for the correct format
        for count, co_lvl in enumerate(wrf_day.CO.keys()):
            wrf_levels[:, count] = np.nansum(wrf_day.CO[co_lvl], axis=2)[mask_nan]

        # backgroud values added
        wrf_levels[:, -1] = np.ones(len(co_tropomi)) * np.percentile(co_tropomi, q=5)

        dobs = co_tropomi
        xapri = np.append(np.ones(len(wrf_day.CO.keys())) * (1.03 * 1e3), np.array([1]))

        jacob = wrf_levels
        rdiag = np.ones(len(co_tropomi)) * np.std(co_tropomi)  # weight of the observed data
        bdiag = np.ones(len(wrf_day.CO.keys()) + 1 )
        optim = OPTIM()
        result = optim.rls_matinv_diag(dobs=dobs, xapri=xapri, jacob=jacob, rdiag=rdiag, bdiag=bdiag)

        if best_fit is None or best_fit["chi2_apri"] > result["chi2_apri"]:
            best_fit = copy.deepcopy(result)

        plt.plot(dobs, "ro")
        plt.plot(np.matmul(jacob, xapri), linestyle="--")
        plt.plot(result['Dapos'])
        plt.show()

        ax = plt.axes(projection=ccrs.PlateCarree())
        new_values = np.zeros(mask_nan.shape)

        print(new_values.shape, len(result["Dapos"]))
        new_values[mask_nan] = result["Dapos"]
        new_values[~mask_nan] = np.NaN
        # print(result["Capos"])
        print(best_fit["chi2_apri"])
        print(best_fit["chi2_apos"])

        ax.pcolormesh(tropomi_day.lon, tropomi_day.lat, new_values, cmap="jet", transform=ccrs.PlateCarree())
        plt.show()
        print(result["Xapos"])

        # print(best_fit["Japos"])
        plt.show()
        sys.exit()
        results[day][co_lvl] = result

    return


def main():
    original_map = os.getcwd()
    resolution = (3600, 1800)
    # pc
    # os.chdir("D:\\Universiteit\\Collegejaar 2018 -2019\\Aardwetenschappen\\new_data")

    # laptop
    os.chdir("C:\\Users\\Coen\\Documents\\Universiteit\\Aardwetenschappen 2019\\Bachelorthesis\\new_data")

    tropomi_files = read_files_TROPOMI(os.listdir(), place=[-122, -118, 36, 39.], resolution=resolution)
    # plot_TROPOMI(tropomi_files)
    # print("Load the files of the WRF files")

    # pc
    wrf_list = "D:\\Universiteit\\Collegejaar 2018 -2019\\Aardwetenschappen\\simulation"

    # laptop
    wrf_list = "C:\\Users\\Coen\\Documents\\Universiteit\\Aardwetenschappen 2019\\Bachelorthesis\\simulation"

    wrf_files = read_file_simulation(wrf_list, place=[-122, -118, 36, 39], resolution=resolution, tropomi=tropomi_files)
    # print("Plot the files")
    plot_wrf(wrf_files)
    sys.exit()
    wrf_files = fit_pressure_column(wrf_files, tropomi_files)
    statistical(tropomi=tropomi_files, wrf=wrf_files)


if __name__ == "__main__":
    main()
