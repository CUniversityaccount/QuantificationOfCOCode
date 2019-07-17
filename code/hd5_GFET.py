import h5py
import numpy as np
import os
import copy
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def show_gfed(gfed, tropomi, place):

    # makes figure
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

    color = ax.pcolormesh(gfed["lon"], gfed["lat"], gfed["emission"], cmap="jet", \
            vmin=np.nanmin(gfed["emission"]), vmax=np.nanmax(gfed["emission"]), \
            transform=ccrs.PlateCarree())
    ax.coastlines(color="black", linewidth=0.5)
    ax.set_extent(place, crs=ccrs.PlateCarree())

    # plots tropomi
    print(tropomi)
    for file in tropomi.keys():
        data = tropomi[file]
        mask = (data["data"].ppm == data["max"])
        print(data["data"].lon[mask])
        print(data["data"].lat[mask])
        ax.scatter(data["data"].lon[mask], data["data"].lat[mask], transform=ccrs.PlateCarree(), color="red",
                   edgecolor="black")

    gfed_mask = (gfed["emission"] == np.nanmax(gfed["emission"]))
    ax.scatter([gfed["lon"][gfed_mask]], [gfed["lat"][gfed_mask]], edgecolor="black", transform=ccrs.PlateCarree())

    # makes the gridline
    gl.xlines, gl.ylines, gl.xlabels_top, gl.ylabels_right = (False for _ in range(4))
    gl.xlocator = mticker.FixedLocator(np.arange(place[0], place[1], 1))
    gl.ylocator = mticker.FixedLocator(np.arange(place[2], place[3], 1))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style, gl.ylabel_style = ({'size': 8, 'color': 'black'} for _ in range(2))

    # plot the x, y point to pinpoint the import points
    # ax.scatter(lat.ravel(), lon.ravel(), transform = ccrs.PlateCarree(), marker="o", color="blue")
    cbar = plt.colorbar(color, ax=ax, orientation="vertical", extend = "both")
    cbar.set_label("[kg C m-2 month-1]")

    plt.savefig("GFED_America_20170720.png", bbox_inches='tight', dpi=300)


def concentration_gfed(file, place):
    """"
    Calculates how much CO is emitted with help of the GFED model
    """
    file = h5py.File(file)

    days = file["emissions"]["07"]["daily_fraction"]
    landscape = file["emissions"]["07"]["partitioning"]
    lat = np.array(file["lat"])
    lon = np.array(file["lon"])
    mask = (lat == 37.625) & (lon == -119.875)
    emission = np.array(file["emissions"]["07"]["DM"])
    max_emission = emission[mask][0]
    grid_area = np.array(file["ancill"]["grid_cell_area"])[mask][0]

    CO = 0
    co_emission = {"DM_SAVA": 63/1000, "DM_AGRI": 102/1000}

    days_sorted = np.zeros(3)

    # calculates the CO fraction of the burning vegetation
    for land in np.array(landscape):
        if np.array(landscape[land])[mask][0] != 0:
            CO += (np.array(landscape[land])[mask][0] * max_emission) * co_emission[land]

    begin_date = 15

    # calculates the area of GFED
    for count in range(3):
        day_frac = np.array(days["day_" + str(begin_date + count)])[mask][0] * CO
        days_sorted[count] = day_frac * grid_area

    # emission in total area
    mask_area = (lon >= place[0]) & (lon <= place[1]) & (lat >= place[2]) & (lat <= place[3]) & (emission > 0)
    lon_area = lon[mask_area]
    lat_area = lat[mask_area]

    lon, lat = makes_grid(place=place, nlon_t=lon.shape[1], nlat_t=lat.shape[0])

    emission_area = emission[mask_area]
    new_emission = np.zeros(lon.shape)

    for count in range(len(emission_area)):
        mask = (lon_area[count] == lon) & (lat_area[count] == lat)
        new_emission[mask] = emission_area[count]
    new_emission[new_emission == 0] = np.NaN
    file.close()

    data_tropomi = {}
    os.chdir("C:\\Users\\Coen\\Documents\\Universiteit\\Aardwetenschappen 2019\\Bachelorthesis\\BachelorThesis2019\\parsed_data")

    for count, file in enumerate(os.listdir()[:3]):
        data = np.load(file).item()
        tropomi = data.get("tropomi")
        tropomi = calculate_ppm(tropomi)
        location_max = np.nanmax(tropomi.ppm)
        data_tropomi[file] = {"data": tropomi, "max": location_max}

    save_gfed(gfed={"lon": lon, "lat": lat, "emission": new_emission, "days": days_sorted})
    # show_gfed(place=place, gfed={"lon": lon, "lat": lat, "emission": new_emission}, tropomi=data_tropomi)


def save_gfed(gfed):
    print(gfed["days"])
    return

def calculate_ppm(tropomi):
    """"
    Recalculates from molec cm2 of TROPOMI to ppm
    """
    molecular_weight_air = 28.96
    gravity = 9.807
    const_avogrado = 6.02214076e23
    recalculation_si = 0.1

    ppb = (tropomi.CO / const_avogrado) / ((((tropomi.ps - tropomi.p_levels[:, :, -1]) * recalculation_si) / gravity) /
                                           molecular_weight_air)

    tropomi.ppm = copy.deepcopy(ppb * 1e9)

    return tropomi


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


if __name__ == "__main__":
    concentration_gfed("GFED4.1s_2018_beta.hdf5", place = [-122, -118, 35.5, 39.5])

