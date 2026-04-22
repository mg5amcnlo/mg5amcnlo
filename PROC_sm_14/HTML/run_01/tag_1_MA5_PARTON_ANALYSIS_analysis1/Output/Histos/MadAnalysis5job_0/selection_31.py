def selection_31():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(0.0,500.0,41,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([6.25,18.75,31.25,43.75,56.25,68.75,81.25,93.75,106.25,118.75,131.25,143.75,156.25,168.75,181.25,193.75,206.25,218.75,231.25,243.75,256.25,268.75,281.25,293.75,306.25,318.75,331.25,343.75,356.25,368.75,381.25,393.75,406.25,418.75,431.25,443.75,456.25,468.75,481.25,493.75])

    # Creating weights for histo: y32_M_0
    y32_M_0_weights = numpy.array([0.0,0.0,1801.299904599984,8105.848570699982,28820.788473600278,55540.067058500215,81658.91567519998,109579.09419649816,120687.09360819895,144103.99236799873,146205.4922566996,135998.09279730145,139300.49262240055,123388.99346510156,117084.49379899897,111080.09411700256,105075.79443500085,102373.89457809822,85861.94545260012,89764.76524589992,84961.29550030014,84060.64554800013,68449.38637479993,71451.54621580026,60343.5368041,61544.40674049981,50736.607312899905,49535.73737650009,45332.70759909995,35725.77810789986,42330.53775810016,38727.93794890019,35425.55812380004,37827.2879966002,30922.308362300082,29421.228441799918,24017.328727999968,21615.59885519981,20414.728918799996,19814.298950599827])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y32_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y32_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y32_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_31.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_31.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_31.eps')

# Running!
if __name__ == '__main__':
    selection_31()
