from optimization_tools import OPTIM

import numpy as np
import os
import copy
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def load_files(files):
    results = {}

    for count, file in enumerate(files[:3]):

        data = np.load(file).item()

        # load the files in
        wrf = data.get("wrf")
        tropomi = data.get("tropomi")
        tropomi = calculate_ppm(tropomi=tropomi)

        co_tropomi = tropomi.ppm
        mask = (co_tropomi == 0)
        co_tropomi[mask] = np.NaN

        # checks wat is the base
        statistical_tropo = (co_tropomi > np.nanpercentile(co_tropomi, q=0)) & (co_tropomi < np.nanpercentile(co_tropomi, q=100))
        co_tropomi[~statistical_tropo] = np.NaN

        mask_nan = ~np.isnan(co_tropomi)
        co_tropomi = co_tropomi[mask_nan]

        # makes the array for and loads the correct day
        wrf_levels = np.zeros((len(co_tropomi.ravel()), len(wrf.CO.keys()) + 1))

        # loads the WRF data in for the correct format
        for count_wrf, co_lvl in enumerate(wrf.CO.keys()):
            wrf_levels[:, count_wrf] = np.nansum(wrf.CO[co_lvl], axis=2)[mask_nan]

        # backgroud values added
        wrf_levels[:, -1] = np.ones(len(co_tropomi)) * np.percentile(co_tropomi, q=0)

        dobs = co_tropomi

        best_fit = None
        final_scale = None
        xapri_array = np.array([[0.0001, 0.0001, 0.0001, 0.01, 0.01, 16., 8, 70., 70],
                               [0.55, 0.001, 0.001, 0.001, 3.5, 0.001, 6, 2, 2.5 ],
                               [0.1, 0.001, 0.001, 0.3, 0.1, 1.75, 0.1, 0.5, 2.5],
                               [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                               [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
                               [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6],
                               [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

        for scale in np.arange(1, 100.01, 0.01):

            xapri = np.append(xapri_array[count] * (scale * 1e3), np.array([1]))

            jacob = wrf_levels
            rdiag = np.ones(len(co_tropomi)) * np.std(co_tropomi)  # weight of the observed data
            bdiag = np.ones(len(wrf.CO.keys()) + 1)
            optim = OPTIM()
            result = optim.rls_matinv_diag(dobs=dobs, xapri=xapri, jacob=jacob, rdiag=rdiag, bdiag=bdiag)

            if best_fit is None or best_fit["chi2_apos"] > result["chi2_apos"]:
                best_fit = result
                final_scale = scale
        best_fit["Xapos"][-1] *= np.percentile(co_tropomi, q=0)
        uncertainty = (np.abs(co_tropomi - best_fit["Dapos"]) / co_tropomi) * 100
        print(np.sum(uncertainty) / len(uncertainty))
        print(uncertainty)

        results[file[:8]] = {"data": best_fit, "scale": final_scale, "mask": mask_nan,
                             "percentage_error": np.sum(uncertainty) / len(uncertainty),
                             "emission": np.sum(best_fit["Xapos"][:-1] * 16 * 28.1) / 1000}

    return results


def calculate_ppm(tropomi):
    """"
    Recalculates to ppm
    """
    molecular_weight_air = 28.96
    gravity = 9.807
    const_avogrado = 6.02214076e23
    recalculation_si = 0.1

    ppb = (tropomi.CO / const_avogrado) / ((((tropomi.ps - tropomi.p_levels[:, :, -1]) * recalculation_si) / gravity) /
                                           molecular_weight_air)

    tropomi.ppm = copy.deepcopy(ppb * 1e9)

    return tropomi


def correlation_gfed(tropomi, expected):
    max_lat = np.zeros(len(tropomi))
    max_lon = np.zeros(len(tropomi))

    for count, file in enumerate(tropomi):

        data = np.load(file).item()["tropomi"]

        max_co = np.unravel_index(np.nanargmax(np.nansum(data.ppm, axis=2), axis=None), data.lon.shape)
        max_lon[count] = data.lon[max_co]
        max_lat[count] = data.lat[max_co]

    chi_square_lon = ((max_lon - expected[1]) ** 2) / np.abs(expected[1])
    chi_square_lat = ((max_lat - expected[0]) ** 2) / expected[0]
    chi_square = np.sum(chi_square_lon) + np.sum(chi_square_lat)
    print(chi_square)


def plot(files, results):
    print(results.keys())
    result_fig, results_axs = plt.subplots(nrows=3, ncols=1, sharey="col")
    results_axs = results_axs.ravel()
    fig, axs = plt.subplots(nrows=3, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()})

    count_t = 0
    for count, file in enumerate(files[:3]):
        axs = axs.ravel()

        day = file[:8]
        data = np.load(file).item()
        result = results[day]
        tropomi = data.get("tropomi")
        tropomi = calculate_ppm(tropomi)
        print(tropomi.ppm.shape)
        print(np.max(tropomi.CO))

        # plot wrf
        wrf_ax = axs[count_t + 1]
        # wrf_fig, wrf_ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
        wrf_ax.set_title(day + " Optimized WRF")
        wrf_ax.set_extent([-122, -118 - 0.85, 36, 39])
        wrf_gl = wrf_ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
        new_CO = np.zeros(result["mask"].shape)

        new_CO[result["mask"]] = result["data"]["Dapos"]
        new_CO[~result["mask"]] = np.NaN
        color = wrf_ax.pcolormesh(tropomi.lon, tropomi.lat, new_CO, cmap="jet", transform=ccrs.PlateCarree(),
                                  vmin=60, vmax=200)

        # plot tropomi
        tropo_ax = axs[count_t]
        # tropo_fig, tropo_ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
        tropo_ax.set_extent([-122, np.max(tropomi.lon) - 0.85, 36, 39])
        co_tropomi = tropomi.ppm
        print(tropomi.ppm.shape)
        tropo_ax.set_title(day + " TROPOMI")
        tropo_ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
        color = tropo_ax.pcolormesh(tropomi.lon, tropomi.lat, co_tropomi, cmap="jet", transform=ccrs.PlateCarree(),
                            vmin=60, vmax=200)

        # Results scaling
        result_ax = results_axs[count]

        result_ax.plot(co_tropomi[result["mask"]], 'ro', markersize=2)
        result_ax.plot(result["data"]['Dapos'])
        print(result["data"]["Xapos"])
        print(result["scale"])
        print(result["data"]["chi2_apri"])
        print(result["data"]["chi2_apos"])
        result_ax.set_xlabel("observation (n)", size=6)
        result_ax.set_ylabel("CO [ppm]", size=6)
        result_ax.set_title(file[6:8] + "-" + file[4:6] + "-" + file[:4], size=4)

        if count == 0:
            result_ax.legend(["TROPOMI", "WRF"])

        count_t += 2

    fig.subplots_adjust(right=0.4)
    cbar = fig.colorbar(color, ax=axs, orientation="vertical", extend="both")
    cbar.set_label("CO [ppb]", size=6)

    # fig.savefig(day + "_data.png", dpi=400)

    result_fig.tight_layout()
    # result_fig.savefig("results.png", dpi=400)
    # plt.show()
    return


def save_scale(results):

    final_array = None
    for count, day in enumerate(results.keys()):
        result = results[day]
        if final_array is None:
            print(result["data"]["Xapos"].shape)
            final_array = np.zeros((len(result["data"]["Xapos"]) + 2, len(results.keys())))

        final_array[:9, count] = result["data"]["Xapos"][:-1] * 16 * 28.1
        final_array[9, count] = result["data"]['Xapos'][-1]
        final_array[-1, count] = result["data"]["chi2_apos"]
        final_array[-2, count] = np.sum(result["data"]["Xapos"][:-1] * 16 * 28.1) / 1000
        print(count)

    np.savetxt("table.csv", final_array, delimiter=",")
    return


def plot_wrf(files):
    """
    Plot the scaled and unscaled values
    """

    for day in files[1:2]:
        print(day)
        fig, axs = plt.subplots(nrows=3, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()})
        axs = axs.ravel()
        data = np.load(day).item()["wrf "]

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


if __name__ == "__main__":
    files = "C:\\Users\\Coen\\Documents\\Universiteit\\Aardwetenschappen 2019\\Bachelorthesis\\BachelorThesis2019\\parsed_data"
    os.chdir(files)
    results = load_files(files=os.listdir(files))
    save_scale(results)
    # plot_wrf(os.listdir(files))
    # correlation_gfed(tropomi=os.listdir(files), expected=[37.65, -119.875])S
    plot(results=results, files=os.listdir())
    sys.exit()

    old_scale = np.array([[0, 0, 0., 0., 0, 14., 4, 2., 75],
                       [0.3, 0.15, 0.025, 0.01, 5.5, 1.5, 3.5, 2, 8.75],
                       [0.01, 0.05, 0.01, 0.4, 0.01, 1.75, 0.1, 0.5, 2.5],
                       [0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
                       [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6],
                       [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

    new_scale = np.array([[0.0001, 0.0001, 0.0001, 0.01, 0.01, 16., 8, 70., 70],
                               [0.55, 0.001, 0.001, 0.001, 3.5, 0.001, 6, 2, 2.5 ],
                               [0.1, 0.001, 0.001, 0.3, 0.1, 1.75, 0.1, 0.5, 2.5],
                               [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                               [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
                               [0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6],
                               [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])