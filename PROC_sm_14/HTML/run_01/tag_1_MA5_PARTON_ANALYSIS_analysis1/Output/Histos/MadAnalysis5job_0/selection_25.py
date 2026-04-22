def selection_25():

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

    # Creating weights for histo: y26_M_0
    y26_M_0_weights = numpy.array([1801.2999579999914,18913.64955899991,30021.659300000014,38727.93909700005,51036.81881000007,48334.86887300009,51337.03880299999,49235.518852000074,116483.997284001,283404.49339199945,289108.59325899975,249179.79418999958,208350.29514200054,181030.59577900032,140801.59671699972,126991.59703900057,113481.89735399945,93367.36782299986,80758.26811699993,68449.38840399991,60943.96857900002,54339.2087329999,46233.358921999934,42630.75900599996,42630.75900599996,36626.4291459999,31222.529271999927,27619.929355999946,31522.739265000084,30622.08928600009,23717.109447000043,21015.159510000056,17112.349600999918,17412.559594000075,16511.909615000077,18313.20957300007,15010.829650000007,12909.309699000094,12909.309699000094,11108.009741000104])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y26_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y26_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y26_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_25.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_25.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_25.eps')

# Running!
if __name__ == '__main__':
    selection_25()
