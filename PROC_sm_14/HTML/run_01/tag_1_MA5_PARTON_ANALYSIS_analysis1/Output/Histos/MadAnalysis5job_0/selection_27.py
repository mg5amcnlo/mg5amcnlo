def selection_27():

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

    # Creating weights for histo: y28_M_0
    y28_M_0_weights = numpy.array([600.4331865600004,18913.649576639917,29421.229341439943,43231.189032320035,59142.668676160036,60043.31865600003,49835.95888447993,63946.13856863994,263590.1940998396,386078.591358079,320030.8928364801,248279.09444256077,189436.69575967954,146205.4967273597,125490.49719104094,102073.59771520105,91265.84795711997,84661.08810495984,61844.618615680025,58842.458682879886,47134.00894495994,45332.70898527995,42930.979039039885,32423.38927424008,25818.62942207996,27319.709388480027,24317.54945567989,27619.929381759954,18012.99959679992,17712.779603519994,19514.07956319999,18613.429583359994,13809.959690880087,12909.309711040092,10507.579764800028,9606.931784959988,10507.579764800028,12609.099717759944,11108.0097513601,8706.28180511999])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y28_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y28_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y28_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_27.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_27.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_27.eps')

# Running!
if __name__ == '__main__':
    selection_27()
