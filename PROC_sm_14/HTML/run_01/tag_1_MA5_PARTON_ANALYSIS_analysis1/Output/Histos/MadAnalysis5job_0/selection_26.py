def selection_26():

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

    # Creating weights for histo: y27_M_0
    y27_M_0_weights = numpy.array([0.0,0.0,600.4331749600011,3302.382862279989,7205.198699519987,15611.25934896016,27619.928848159932,29121.008785560058,34524.90856020002,33624.258597760025,74753.93688251986,121587.69492940117,136898.79429087896,160315.6933143188,149507.89376503887,139900.89416568173,150408.49372748096,119185.9950295598,119185.9950295598,105976.49558043851,96369.5259810801,95168.66603115984,84060.64649440006,74453.71689504,67248.51719552005,68749.59713292019,59142.66753356011,51337.03785908003,53438.557771439875,51637.25784655989,46833.788046880065,43531.40818459995,40529.23830980011,42930.97820963982,34224.68857272016,30021.658748000053,38127.50840995999,26419.05889824008,27319.70886068007,25818.628923279946])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y27_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} p_{2} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y27_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y27_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_26.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_26.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_26.eps')

# Running!
if __name__ == '__main__':
    selection_26()
