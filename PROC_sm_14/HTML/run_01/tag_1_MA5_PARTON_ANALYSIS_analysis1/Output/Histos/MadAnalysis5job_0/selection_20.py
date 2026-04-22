def selection_20():

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

    # Creating weights for histo: y21_M_0
    y21_M_0_weights = numpy.array([123389.01519055919,173525.22136288343,99071.49219680182,56740.94698544123,43831.63539616149,38727.94476784045,195741.2240979203,2129436.2621572716,79857.62983136182,20114.512476320047,13509.751663200585,6604.766813120204,5704.116702240165,3602.599443520033,2101.516258720009,2101.516258720009,1200.86614783997,1200.86614783997,600.4332739200096,1501.0831848000241,300.2166369600048,0.0,300.2166369600048,300.2166369600048,300.2166369600048,0.0,0.0,300.2166369600048,0.0,0.0,0.0,0.0,300.2166369600048,300.2166369600048,0.0,300.2166369600048,0.0,0.0,0.0,300.2166369600048])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y21_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y21_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y21_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_20.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_20.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_20.eps')

# Running!
if __name__ == '__main__':
    selection_20()
