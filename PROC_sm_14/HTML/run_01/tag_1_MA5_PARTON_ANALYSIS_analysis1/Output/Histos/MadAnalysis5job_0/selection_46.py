def selection_46():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(0.0,10.0,41,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([0.125,0.375,0.625,0.875,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875,4.125,4.375,4.625,4.875,5.125,5.375,5.625,5.875,6.125,6.375,6.625,6.875,7.125,7.375,7.625,7.875,8.125,8.375,8.625,8.875,9.125,9.375,9.625,9.875])

    # Creating weights for histo: y47_DELTAR_0
    y47_DELTAR_0_weights = numpy.array([7205.198388960005,43831.626282839796,63946.134577020144,80458.04317672053,92166.4921837804,83460.2129221202,113481.89037611874,135397.68851753994,142002.48795741703,196041.3833746249,268994.07718783984,323033.0726050392,366564.4689133402,243175.4793773973,175026.28515681947,164818.88602246242,113481.89037611874,90965.6222856207,76555.2335077,63645.91460248043,46833.786028240334,30622.08740308051,23717.107988660304,20114.50829418034,10507.579108900167,8105.848312579996,6004.3314908000475,3902.815669020013,2401.732796320002,2101.5158217800335,1801.2998472399802,600.4331490800047,900.6497236200071,0.0,0.0,0.0,0.0,300.21657454000234,0.0,0.0])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y47_DELTAR_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$\Delta R$ $[ p_{1}, p_{2} ]$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y47_DELTAR_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y47_DELTAR_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_46.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_46.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_46.eps')

# Running!
if __name__ == '__main__':
    selection_46()
