"""
Library for Positive Cell Cluster Detection (PCCD)

Author: C. Godin Inria, RDP, Juil 2020 - Sept 2021

Design contribution and data:
    Jonathan Enriquez, CNRS, IGFL
    Wenyue Guan, IGFL

Licence: Open source LGPL
"""

import os
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

def load_TF_dataframe(df_filename):
    """
    Loads the main databasis and adds a field "norm" that contains the distance of the entry defined by ['X','Y','Z']
    to the neuroblast [0,0,0] considered as the origin.
    """
    #df = pd.read_csv('Runt.csv',sep = ';')
    df = pd.read_csv(df_filename,sep = ';')

    # add a norm column to df
    # incrementing df with the distance to the origin
    xyz = df[['X','Y','Z']].values

    # norm of the xyz vectors
    xyznorm = linalg.norm(xyz, axis = 1)

    #augment df with wyznorms
    df['norm']=xyznorm

    return df

def pca(df):
    """
    Computes the pca of a dataframe df related to [X,Y,Z]
    Returns a dataframe [X,Y,Z] whose point coordinates are expressed in the pca basis
    """
    from sklearn.decomposition import PCA

    # creates an object able to compute PCA (with 3 output coordinates)
    pca = PCA(n_components=3)

    ptfeatures = ['X','Y','Z']
    # extracts the sub dataframe with specific column labels and the corresponding numpy array
    points = df.loc[:,ptfeatures].values

    # points with coordinates expressed in the pca basis (= array)
    PCAPoints = pca.fit_transform(points)

    # transform the resulting numpy array into dataframe
    PCAdf = pd.DataFrame(data = PCAPoints)

    # change column names of pca coordinates
    PCAdf.columns = ['XX','YY','ZZ']

    return PCAdf

def region_name(region):
    region_names = ['T1','T2','T3']
    return region_names[region-1]

def create_experiment_name(TF_name, id, region, side):
    return TF_name + ' ' + id + '_' + region_name(region) + side

def plot_experiment(df, exp_name, view = 'original'):
    """
    Plots the 3D cell positions (X,Y,Z) corresponding to the experiment in which
    the transcription factor TFname has been labeled (available in the dataframe df)
    and colors the points (cells) according to whether they expresss or not
    the transcription factor TFname (points expresssing the TF in green the others in blue)

    options:
    - view = original (original 3D points)  or pca (pca 3D points)
    """

    print("Plotting ", exp_name, " (Active cells are in green)")

    TFdf = df[df['TFname'] == exp_name].copy()

    # reset the TFdf dataframe index
    lTF = len(TFdf.index)
    TFdf.index=range(lTF)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')

    if view == 'original':

        onTFdf = TFdf[TFdf['TF']== '*']
        offTFdf = TFdf[TFdf['TF']!= '*']

        xs = TFdf.loc[:, 'X'].values
        ys = TFdf.loc[:, 'Y'].values
        zs = TFdf.loc[:, 'Z'].values

        xon = onTFdf.loc[:, 'X'].values
        yon = onTFdf.loc[:, 'Y'].values
        zon = onTFdf.loc[:, 'Z'].values

        xoff = offTFdf.loc[:, 'X'].values
        yoff = offTFdf.loc[:, 'Y'].values
        zoff = offTFdf.loc[:, 'Z'].values

        ax.set_xlabel('X coord ', fontsize = 15)
        ax.set_ylabel('Y coord', fontsize = 15)
        ax.set_title(exp_name +' Original coords', fontsize = 20)

    else:

        # Computes the principal component axes
        PCdf = pca(TFdf)
        #plot(PCdf, name)

        # change column names of pca coordinates
        PCdf.columns = ['XX','YY','ZZ']

        onTFdf = PCdf[TFdf['TF']== '*']
        offTFdf = PCdf[TFdf['TF']!= '*']

        xs = PCdf.loc[:, 'XX'].values
        ys = PCdf.loc[:, 'YY'].values
        zs = PCdf.loc[:, 'ZZ'].values

        xon = onTFdf.loc[:, 'XX'].values
        yon = onTFdf.loc[:, 'YY'].values
        zon = onTFdf.loc[:, 'ZZ'].values

        xoff = offTFdf.loc[:, 'XX'].values
        yoff = offTFdf.loc[:, 'YY'].values
        zoff = offTFdf.loc[:, 'ZZ'].values

        ax.set_xlabel('Principal Component 1 (XX)', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title(exp_name +' 3 component PCA', fontsize = 20)

    minx = min(xs)
    maxx = max(xs)
    miny = min(ys)
    maxy = max(ys)

    globmax = max(maxx, maxy)
    globmin = min(minx, miny)

    ax.set_xlim3d(globmin, globmax)
    ax.set_ylim3d(globmin, globmax)
    ax.set_zlim3d(0, globmax-globmin)

    ax.scatter(xon, yon, zon, c = 'g')
    ax.scatter(xoff, yoff, zoff, c = 'b')
    #ax.legend(targets)
    plt.show()

def TF_histogram(df, TFnames, sort_column = 'XX', ascending_order = False, TFsamplesize = 0):
    """
    Computes the histogram of number of active TF at rank k in pool of sequences.
    Sequences are first sorted according to cell position according to a specific criterion
    (determined by the sort_column parameter).

    Input:
    - df = overall DataFrame
    - TFnames = array of TF names (indicated by their names and experience id) to pull together

    Returns:
    - TFdist = array of number of active values detected at each position across all the sorted sequences

    Options:
    - sort_column: column used to sort the data: 'XX' or 'norm' (column 'norm' already exists in df, but not 'XX')
    - ascending_order = True/False
    - samplesize = 0 (for all TFnames in the list or index !=0 if one specific TFname is required)

    """
    max_size = 55
    TFdist = np.zeros(max_size)
    TFdata = []

    if TFsamplesize == 0:
        TFlist = TFnames
    else:
        TFlist = TFnames[:TFsamplesize]

    # Loops on all the considered experiments
    # e.g. 'Brc 01_T1L', 'Brc 01_T1R', 'Brc 02_T1R', 'Brc 03_T1L', 'Brc 03_T1R', ...
    # to collect for each rank the number of active TF (this defines TFdist)
    nb_active_cells = [] # stores the found number of active cells for each experiment

    for name in TFlist : # TFnames[:k] to select only a sub array of TFnames with k entries

        # create a dataframe that is a subdataframe copy of df
        # for the required experiment (e.g. name = 'Brc 01_T1L')
        TFdf = df[df['TFname'] == name].copy()

        # reset the TFdf dataframe index so that the index goes from 0 to len(TFdf.index)
        # (index translation)
        lTF = len(TFdf.index)
        TFdf.index=range(lTF)

        # Computes the principal component axes
        # Note: the output dataframe only contains XX, YY, ZZ coordinates
        PCdf = pca(TFdf)

        # add the column TF to PCdf for the case pca 'XX' is used.
        TFdf['XX']=PCdf['XX']

        # Sort according to the selected sort method ('norm' or 'XX').
        # Here the df is readonly. Makes a deepcopy
        sortedTFdf = TFdf.sort_values(by=[sort_column],ascending=[ascending_order])
        #print(name, len(sortedTFdf),lTF)

        # create an array with kth value True if TF is on at the kth line of sortedTFdf,
        # False otherwise
        isTFon = sortedTFdf['TF'].values == '*'

        # Scans the array
        nb = 0
        for i in range(lTF):
          if isTFon[i]:
              TFdist[i] +=1
              nb += 1
              TFdata.append(i)
        nb_active_cells.append(nb)

    average_nb_active_cells = np.mean(nb_active_cells)

    # TFdist is an array containing the number of active TF for each rank of the observed pooled experiments
    return TFdist, average_nb_active_cells

def TF_average_span_length(df, TFnames, peak_spans, sort_column = 'XX', ascending_order = False, TFsamplesize = 0):
    """
    Computes the average spans of the different experiments involving TFname in the selected region_list.
    The number of peaks and a first estimation of their maximal span is given as an input using peak_spans,
    which is an array containing [[start0,end0],[start1,end1]] of the different detected peaks (max 2)
    (non overlapping intervals).

    Input:
    - df = complete databasis DataFrame
    - TFnames = array of TF names (indicated by their names and experience id) to pull together to create the histogram
    - peak_spans = array Kx2 giving the start and end of each peak (K peaks in total)

    Returns:
    - an array of 1 or 2 values corresponding to the average # of active cells spanned by each TF peak (1 or 2)

    Options:
    - sort_column: column used to sort the data: 'XX' or 'norm' (column 'norm' already exists in df, but not 'XX')
    - ascending_order = True/False
    - samplesize = 0 (for all TFnames in the list or index !=0 if one specific TFname is required)

    """
    TFdist = np.zeros(55)
    TFdata = []

    if TFsamplesize == 0:
        TFlist = TFnames
    else:
        TFlist = TFnames[:TFsamplesize]

    peaknb = len(peak_spans)

    # Two methods will be used to compute the span of each peak:
    # 1. number of active cells in the guessed region of the peak
    # 2. minimal size of a region containing all active cells within the guessed region

    # These two arrays will store the computed span (method 1 or 2)
    # computed for each peak of each TF experiment (each defined by a TFname)
    active_peak_len_list = []
    active_peak_len_list2 = []

    for name in TFlist :

        TFdf = df[df['TFname'] == name].copy()

        lTF = len(TFdf.index)
        TFdf.index=range(lTF)

        PCdf = pca(TFdf)

        # add the column TF to PCdf
        TFdf['XX']=PCdf['XX']

        # Sort according to x values (column identified by index 0).
        # Here the df is readonly. Makes a deepcopy
        # Note that ascending_order=False means the NB (ventral side) will be
        # put on the right of the arrays (at the end)
        sortedTFdf = TFdf.sort_values(by=[sort_column],ascending=[ascending_order])

        sortedTFdf = sortedTFdf[sortedTFdf['Type'] != 'NB']
        sortedTFdf = sortedTFdf[sortedTFdf['Type'] != 'GMC']

        #print (sortedTFdf)
        #print(name, len(sortedTFdf),lTF)

        # create an array with kth value True if TF is on at the kth line of sortedTFdf,
        # False otherwise
        isTFon = sortedTFdf['TF'].values == '*'

        # These two lists will store the computed span values for
        # each peak of the current TF experiment (defined by name)
        active_peak_len = []
        active_peak_len2 = []

        # computes the maximal span of True values in the array isTFon
        for k in range(peaknb):
            start = peak_spans[k][0]
            end = min(peak_spans[k][1], len(isTFon)-1)
            #print ("peak span at h = 1: [", start, ",",end,"]" )

            # method with # of active cells over a span
            # (WARNING: values of start and end are not affected by this method)
            nb_active_cells = 0
            for i in range(start,end+1):
                if isTFon[i]:
                    nb_active_cells +=1
            # updates the peak list for this method
            active_peak_len2.append(nb_active_cells)

            # method with minimal span including all True values
            # (WARNING: this method affects the values of start and end)
            for i in range(start,end+1):
                if isTFon[i]:
                    start = i
                    break
            for i in range(end,start-1,-1):
                if isTFon[i]:
                    end = i
                    break
            # updates the peak list for this method
            if start < end:
                active_peak_len.append(end-start+1)
            else:
                active_peak_len.append(0)

        # update the span lists with the computed values before going
        # to next name
        active_peak_len_list.append(active_peak_len)
        active_peak_len_list2.append(active_peak_len2)

    active_spans = np.array(active_peak_len_list)
    active_spans2 = np.array(active_peak_len_list2)

    average_list = [] # list of mean active spans by peak
    for k in range(peaknb):
        m = np.mean(active_spans[:,k])
        average_list.append(m)
    #print("averages spans of active cells: ", average_list)

    average_list2 = [] # list of mean active spans by peak
    for k in range(peaknb):
        m = np.mean(active_spans2[:,k])
        average_list2.append(m)
    #print("averages nb of active cells: ", average_list2)

    return  average_list2 # Only the # of active cells over a span is returned

def TF_names(df):
    """
    computes the list of unique TF names from the column TFname of df
    """
    return df['TFname'].apply(lambda x: x.split()[0]).unique()

def sub_dataframefromTFname(df,name):
    """
    extracts the subdataframe if TFname contains name

    WARNING: this is based on the fact that the TF's name is followed
    by a white-space character (to make differences between Mamo and MamoS for instance ...)
    """
    return df[df.TFname.str.contains(name+' ',case=True)]

def TF_distribution(df, TFname, region_list, ordering_method = 'XX', ascending_order = False, TFsamplesize = 0, normalized = False, smoothed = False, show = False, printcsv = False):
    """
    This function creates the subdataframe that pools sequences from df defined by TFname and region_list.

    Then it computes the histogram of TF active on the sequence axis (defined by ordering choice) on the set of sequences,
    and possibbly smoothes it out or normalizes it before returning according to options set.

    The function can also display the resulting spatial distribution before returning.

    Returns:
    - TFdist: the spatial distribution of active TF along the chosen axis on the selected set of experiments
    - nb_experiments: size of the experiment sample that was used
    """

    from scipy.signal import savgol_filter

    # sub dataframe containing all the lines corresponding to TFname experiments
    subdf = sub_dataframefromTFname(df,TFname)

    # get the different TF names in column TFname
    # e.g. if TFname = 'Brc', one would get 'Brc 01_T1L', 'Brc 01_T1R', 'Brc 02_T1R', ...
    # df['TFname'].unique()
    all_expe_names = subdf.TFname.unique()
    #print("all_expe_names = ", all_expe_names)

    # Filters out the names corresponding to the given list of regions
    expe_names_list = []
    for region in region_list:
        expe_names_list.extend([name for name in all_expe_names if region_name(region) in name])

    #print("expe_names_list = ", expe_names_list)

    # Computes the histogram of TF active on the sequence axis (defined by ordering choice)
    # on the pooled set of sequences defined by expe_names_at_region
    TFdist, average_nb_active_cells  = TF_histogram(subdf, expe_names_list, ordering_method, ascending_order, TFsamplesize)


    # normalizes or smoothes the histogram out according to options
    nb_experiments = len(expe_names_list)
    if normalized:
        TFdist = TFdist / float(nb_experiments)
    if smoothed:
        TFdist = savgol_filter(TFdist, 11, 2) # window size 5 (must be odd), polynomial order 3

    # plots and returns everything shifted after insertion of a 0 (useful after to extract peaks)
    TFdist_shifted = np.zeros(len(TFdist)+2)
    TFdist_shifted[1:len(TFdist)+1] = TFdist

    if show :
        xvals = range(len(TFdist_shifted))
        h1 = plt.bar(xvals, TFdist_shifted, color = 'blue', label = TFname + ' in regions ' + str(region_list) + ' (' + str(nb_experiments) + ' experiments)' + 'av.length=' + "{:.1f}".format(average_nb_active_cells))

        if normalized:
            ymax = 1
        else:
            ymax = 25
        plt.ylim((0,ymax))
        plt.title('TF activity along the corpus ordered according to ' + ordering_method)
        plt.legend()
        plt.show()

    if printcsv:
        tf_dist_df = pd.DataFrame(TFdist_shifted)
        if smoothed:
            txt = '_smoothed'
        else :
            txt = ''
        result_dir = 'Results/'
        if not os.path.exists(result_dir):
          os.makedirs(result_dir)
        filename = result_dir + 'TFdist_' + TFname + txt +'.csv'
        colname=['TF intensity']
        tf_dist_df.to_csv(filename,header = colname)

    return TFdist_shifted, nb_experiments

def detect_peaks(X2, TFname, region_list, nb_experiments, h, show = False):
    """
    Detects peaks in a signal using specified properties (minimal height, minimal prominence, etc.).

    Uses the scipy implementation:
    - height defines the minimal absolute height for a point to be a peak
    - distance is the minimum distance separating two peaks.
    - the prominence of a peak measures how much a peak stands out from the surrounding baseline
    of the signal and is defined as the vertical distance between the peak and its lowest contour
    line (a contour line in 1D is simply the x-region subtending the peak)

    peak_widths detect the width centered on its peak argument in the original signal
    it return a matrix (4 x nb_peaks matrix) whose lines resp. corresponds to:
    1st line: width of the jth peaks
    2nd line: height of the jth peak (from bottom at the x-axis)
    3rd line: left xvalue of the horizontal span for the jth peak
    4th line: right xvalue the horizontal span for the jth peak
    """

    from scipy.signal import find_peaks, peak_widths, peak_prominences

    # parameters used for identifying peaks
    min_height = 0.02    # in the normalized histogram (max height = 1)
    min_dist = 15        # in cells
    min_prominence = 0.1 # local altitude of the peak

    peaks, _ = find_peaks(X2, height = min_height, distance = min_dist, prominence = min_prominence)

    width_half = peak_widths(X2, peaks, rel_height=h)
    width_full = peak_widths(X2, peaks, rel_height=1)

    # peak_widths returns = [width, altitude, seg_xleft, seg_xrigth].
    # note: width=seg_xrigth-seg_xleft

    #print('peaks', peaks)
    #print('width_half', width_half)
    #print('width_full', width_full)

    if show:
        fig, axis = plt.subplots(1, 1, figsize=(7, 7))

        axis.plot(X2, color = 'b', label = 'Freq. of cells expressing '+TFname)
        axis.plot(peaks, X2[peaks], 'x',color='r', label = 'Significant max values')

        #hlines args: [y, x1, x2]. The starred array flattens the array
        plt.hlines(*width_half[1:], color='g', label='width half peak')
        plt.hlines(*width_full[1:], color='darkorange', label='width full peak',linewidth=6)
        plt.ylim((0,1))

        axis.set_xlabel('cell indexes')
        axis.set_ylabel('TF activity')
        axis.set_title('TF: '+TFname + ' - regions: ' + str(region_list) + ' (' + str(nb_experiments) + ' experiments)')

        axis.legend(loc = 'upper right', fontsize='small') # bbox_to_anchor=(0.5, 0.5))
        plt.show()

    return peaks, width_half, width_full,X2

def find_TF_span_automatic(df, TFname,region_list,ordering_method, span_conv_eps = 0.01, plot=False, printcsv = False, MAX_ITER=40, verbose = True):
    """
    This function computes the span of a particular TF (TFname) activity on the set of sequences defined by a list of regions.

    - It first computes the TF activity distribution histogram TFdist
    - It then detects the main peaks of the spatial distribution together with their width at relative height 1 and 0.5.
    - It deduces from it a max span of the peak that corresponds to the basis of the peak (h=1), possibly truncated at the position where the next peak starts.
    - This defines span regions on the x-axis where the TF factors are expected to be found
    - Then for each experiment, these span regions are used to determine an estimated actual span corresponding to the minimal span embedding active cells within the span region
    - For each TF, these estimated actual spans are averaged for each peak.
    --> this defines an average estimated span for each peak of each TF.
    - Then, the algorithm finds the percent (h < 1) of relative height whose peak span best matches the average estimated span and the average location of this average estimated span on the axis.
    """
    from scipy.signal import find_peaks, peak_widths, peak_prominences

    TFdist, nb_expe = TF_distribution(df, TFname, region_list, ordering_method, ascending_order = False, TFsamplesize = 0, normalized = True, smoothed = True, show = False)

    h = 0.5
    peaks, width_half, width_full, TFdist_modified = detect_peaks(TFdist, TFname, region_list, nb_expe, h)
    peak_nb = len(peaks)

    width0 = int(round(width_full[0][0]))
    start0 = int(round(width_full[2][0]))
    end0   = int(round(width_full[3][0]))
    #print (TFname, "mask : [",start0,end0,"]")

    if peak_nb > 1:

       # Here we assume that
       # - there are two peaks
       # - one peak dominates the other (which might not be always the case ...)

        width1 = int(round(width_full[0][1]))
        start1 = int(round(width_full[2][1]))
        end1   = int(round(width_full[3][1]))

        if width0 > width1: # peak 0 dominates peak 1
            assert (start0 <= start1)
            assert (start1 <= end1)
            assert (end1 <= end0)
            # depending on where peak 0 is with respect to [start1,end1]:
            if peaks[0] <= start1:
                end0 = start1
            else:
                start0 = end1
            width0 = end0-start0
        else: # peak 1 dominates peak 0
            assert (start1 <= start0)
            assert (start0 <= end0)
            assert (end0 <= end1)
            # depending on where peak 1 is with respect to [start0,end0]:
            if peaks[1] <= start0:
                end1 = start0
            else:
                start1 = end0
            width1 = end1-start1

    subdf = sub_dataframefromTFname(df,TFname)
    all_expe_names = subdf.TFname.unique()

    if peak_nb == 1:
        spans = [[start0,end0]]
    else:
        spans = [[start0,end0],[start1,end1]]
    #print (TFname, "mask : [",spans,"]")

    # Estimate the average actual estimated spans of the collection of experiments
    average_span_list = TF_average_span_length(df, all_expe_names, spans, sort_column = ordering_method)
    #print(" --> Computed average spans for ", TFname, " : ", average_span_list)

    h_list = []

    for k in range(peak_nb):
        i = 0
        hmin = 0
        hmax = 1
        span_min = 0
        span_max = width_full[0][k]      # initial width of the peak at height 1
        computed_span = width_half[0][k] # gets the initial peak width at h = 0.5
        h = 0.5                          # h currently being tested

        if verbose:
            print ("{:7s}".format(TFname),": PEAK #", "{:1d}".format(k+1))
        peak_width = span_max
        peak_start = width_full[2][k]
        peak_end = width_full[3][k]
        if verbose:
            print ("\ttot basis (h=1): {:.1f}".format(peak_width), " [{:.1f}".format(peak_start), ",{:.1f}".format(peak_end),"]")
        if k == 0:
            width00 = width0
            start00 = start0
            end00 = end0
            if verbose:
                print ("\trel basis = {:.1f}".format(width00), " [{:.1f}".format(start00), ",{:.1f}".format(end00),"]")
        else:
            width11 = width1
            start11 = start1
            end11 = end1
            if verbose:
                print ("\trel basis = {:.1f}".format(width11), " [{:.1f}".format(start11), ",{:.1f}".format(end11),"]")

        if k >= len(average_span_list):
            if verbose:
                print("... WARNING: DETECTED PEAK #", k+1, " NOT EXPECTED --> take width at relative height h = 0.4")
            # Arbitrarily take the width at height h=0.4
            width_half = peak_widths(TFdist_modified, peaks, rel_height=0.4)
            h_list.append(0.4)
        else:
            width_half_tmp = peak_widths(TFdist_modified, peaks, rel_height=h)
            if k==0:
                width_half = width_half_tmp
            else: # avoid destroying the values for the first peak as they are for different h
                width_half[0][k] = width_half_tmp[0][k]
                width_half[1][k] = width_half_tmp[1][k]
                width_half[2][k] = width_half_tmp[2][k]
                width_half[3][k] = width_half_tmp[3][k]
            computed_span = width_half[0][k]
            target_span = average_span_list[k]

            while np.abs(computed_span - target_span) > span_conv_eps and i < MAX_ITER:

                if computed_span < target_span:
                    hmin = h
                    span_min = computed_span
                else:
                    hmax = h
                    span_max = computed_span
                h = hmin + (hmax-hmin)/2.
                width_half_tmp = peak_widths(TFdist_modified, peaks, rel_height=h)
                if k==0:
                    width_half = width_half_tmp
                else: # avoid destroying the values for the first peak as they are for different h
                    width_half[0][k] = width_half_tmp[0][k]
                    width_half[1][k] = width_half_tmp[1][k]
                    width_half[2][k] = width_half_tmp[2][k]
                    width_half[3][k] = width_half_tmp[3][k]

                computed_span = width_half[0][k]
                i+=1
            h_list.append(h)

            if verbose:
                print("\tspan (h=","{:.1f}".format(h), ") [{:.1f}".format(width_half[2][k]),",{:.1f}".format(width_half[3][k]),"]" , "average width = {:.1f}".format(average_span_list[k]),", computed width at h = {:.2f}".format(computed_span), " at y ={:.2f}".format( width_half[1][k]))
                print("\t(converg. criterion = {:.3f})".format(np.abs(computed_span - target_span)))

    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(7, 7))

        axis.plot(TFdist_modified, color = 'b', label = 'Freq. of cells expressing '+TFname)
        axis.plot(peaks, TFdist_modified[peaks], 'x',color='r', label = 'Significant max values')

        plt.hlines(*width_half[1:], color='g', label='width half peak')

        plt.ylim((0,1))

        axis.set_xlabel('cell indexes')
        axis.set_ylabel('TF activity')
        nb_experiments = len(all_expe_names)
        axis.set_title('TF: '+TFname + ' - regions: ' + str(region_list) + ' (' + str(nb_experiments) + ' experiments)')

        axis.legend(loc = 'upper right', fontsize='small')
        plt.show()

    return peaks, width_half, width_full, TFdist_modified, h_list, average_span_list

def plot_TF_distributions_basedon_average(df, regions, TFlist, h = 0.5, ordering_method = 'XX', verbose = True):
    """
    Plots all the TF distributions for the given TFlist and regions with the detected peaks and their spans.
    - Spans are estimated from TF distributions finding the position on the x-axis that best fits the peak width
    """

    #fig, axis = plt.subplots(1, 1, figsize=(7, 7))
    fig = plt.figure(figsize=(24, 20))

    # scans the list of all TFs and computes their peaks and spans
    i = 1
    for name in TFlist:

        peaks, width_half, width_full, TFdistribution,best_h_list, average_estimated_span_list = find_TF_span_automatic(df, name,regions,ordering_method,plot = False, verbose = verbose)

        nbpeaks = len(peaks)

        axis = fig.add_subplot(6, 3, i)

        if i == 1:   # if i = 1, print legend
            axis.plot(TFdistribution, color = 'b', label = 'Freq. of cells expressing TF')
            axis.plot(peaks, TFdistribution[peaks], 'x',color='r', label = 'Significant max values')
            #hlines args: [y, x1, x2]. The starred array flattens the array
            plt.hlines(*width_half[1:], color='g', label='width half peak with h='+str(h))
            plt.hlines(*width_full[1:], color='darkorange', label='width full peak')
        else:
            axis.plot(TFdistribution, color = 'b')
            axis.plot(peaks, TFdistribution[peaks], 'x',color='r')
            #hlines args: [y, x1, x2]. The starred array flattens the array
            plt.hlines(*width_half[1:], color='g')
            plt.hlines(*width_full[1:], color='darkorange')

        plt.ylim((0,1))

        axis.set_title('TF: '+ name + ' h = ' +  "{:.2f}".format(best_h_list[0]))

        axis.set_xlabel('')
        axis.set_ylabel('TF activity')

        i += 1

    fig.legend(loc = 'upper right', fontsize='small') # bbox_to_anchor=(0.5, 0.5))

    plt.show()

def plot_TF_spans_basedon_average(df, TFlist, regions, ordering_method = 'XX'):
    """
    Plots all detected TF spans for the given TFlist and regions, using the average values automatically detected.
    - Spans are estimated from TF distributions finding the position on the x-axis that best fits the peak width
    """

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 1, 1)

    i = 1

    lenTF = len(TFlist)
    II = lenTF # II is the number of horizontal lines to draw

    # scans the list of all TFs and computes their peaks and spans
    for name in TFlist:

        # Create a list of average span values for peaks
        peaks, width_half, width_full, TFdistribution,best_h_list,average_estimated_span_list = find_TF_span_automatic(df, name,regions,ordering_method,plot = False)
        nbpeaks = len(peaks)

        for k in range(nbpeaks):
            # Assess peak spans
            if k > 0: print (" ",)
            # print("Processing ", name, " peak ",  k+1)

            # Plot horizontal lines corresponding to peak spans
            #hlines args: [y, x1, x2]. The starred array flattens the array
            ii = II - i - 1 # to change i coordinates backward
            a = [ii, width_half[2][k], width_half[3][k]]
            c = 'C' + str(i)
            if k == nbpeaks-1:
                plt.hlines(*a, label=name+" h= "+ "{:.2f}".format(best_h_list[k]), color = c, linewidth=4)
            else:
                plt.hlines(*a, color = c, linewidth=4)
        i += 1

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 51, 10)
    minor_ticks = np.arange(0, 51, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_title('Estimated TF activation zones')
    ax.grid(which='both') # alpha = 0.5
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
