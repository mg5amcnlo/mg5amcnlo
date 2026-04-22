def selection_17():

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

    # Creating weights for histo: y18_M_0
    y18_M_0_weights = numpy.array([54639.415286200514,120386.88961409716,178028.4846412965,266592.3770007969,277700.3760424982,292410.9747733995,261188.47746699696,200244.4827246991,174125.58497800372,143503.48761980407,126991.58904430285,103274.49109040167,86162.16256670015,74753.9335509,65447.214353800395,55540.065208500506,43831.62621859978,42030.3263739998,51637.25554519998,37827.28673660042,30622.087358200508,29121.007487700237,27019.497668999687,20714.94821289976,21015.158187000332,15911.478627300103,13209.528860400134,15611.258653200393,11708.448989899865,13509.748834499844,12909.308886300425,8406.064274800046,8406.064274800046,6004.331482000046,8706.281248900013,6304.548456100013,5704.115507899991,6004.331482000046,5103.681559700056,3902.815663300012])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y18_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y18_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y18_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_17.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_17.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_17.eps')

# Running!
if __name__ == '__main__':
    selection_17()
