def selection_35():

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

    # Creating weights for histo: y36_M_0
    y36_M_0_weights = numpy.array([0.0,64246.35550599979,128492.69101200097,151609.38939499954,180129.9873999981,187635.38687499918,181030.58733700158,173224.98788299932,163918.28853399825,144103.98991999848,132695.69071800326,108678.39239800118,106576.89254500002,94268.01340599994,87963.463847,84961.29405700027,75654.5847079999,68149.16523300021,60643.755757999825,54939.63615700012,43531.406955000006,43231.186976000245,41129.6771229998,36926.63741700031,34224.68760600034,30922.30783700014,28820.787984000395,27319.708089000174,26419.058152000187,28520.578004999934,23717.108341000214,20714.94855099978,19213.858656000262,18012.99873999981,18313.20871900027,21315.37850900001,19514.078635000027,20114.50859300025,11408.229202000111,12909.309097000329])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y36_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ p_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y36_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y36_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_35.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_35.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_35.eps')

# Running!
if __name__ == '__main__':
    selection_35()
