def selection_30():

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

    # Creating weights for histo: y31_M_0
    y31_M_0_weights = numpy.array([0.0,0.0,0.0,0.0,300.2165855400007,1801.2999132399848,9006.49756620002,20414.729016719986,23717.10885766012,41129.678018979816,54339.20738273986,65147.00686217977,75354.36637054,76855.44629824015,91566.06558969988,82259.34603796012,92466.71554631986,81959.13605241978,102674.09505467914,87363.02579214022,81058.4860957998,84060.64595120009,78656.74621148013,85861.94586444007,73252.84647176019,74153.49642838018,69950.46663082005,64846.78687663993,59743.10712245981,59442.88713691998,50436.38757072006,57341.36723814016,52237.68748396004,52537.90746949988,48635.087657480064,42330.53796114012,47434.21771532024,44732.27784545978,35125.33830818018,33924.478366019874])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y31_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{1} p_{2} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y31_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y31_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_30.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_30.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_30.eps')

# Running!
if __name__ == '__main__':
    selection_30()
