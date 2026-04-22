def selection_48():

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

    # Creating weights for histo: y49_DELTAR_0
    y49_DELTAR_0_weights = numpy.array([3002.165728200026,24617.757771240318,71151.33355834009,112581.18980750324,151008.98632845675,162116.98532279814,161816.78534997662,189736.88282224085,191237.98268633932,214354.68059347756,230866.57909857886,237471.27850062482,242274.77806574173,173825.38426278252,154611.58600229674,133596.38790489998,102974.29067726033,78956.9628516603,76255.01309628034,56740.934862980255,56140.50491733996,43231.186086080415,32423.387064560535,21915.808015860355,19213.858260480385,15311.048613819823,11108.008994340476,11108.008994340476,7505.414320500065,4203.032619479983,3302.382701019992,1501.082864100013,2401.7327825600028,900.6497184600078,1200.8658912800465,900.6497184600078,300.2165728200026,300.2165728200026,0.0,0.0])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y49_DELTAR_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$\Delta R$ $[ p_{2}, p_{3} ]$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y49_DELTAR_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y49_DELTAR_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_48.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_48.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_48.eps')

# Running!
if __name__ == '__main__':
    selection_48()
