def selection_19():

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

    # Creating weights for histo: y20_M_0
    y20_M_0_weights = numpy.array([60043.316460000206,160615.89053049943,262989.78449479747,343147.5797688996,343147.5797688996,293011.38272480114,241374.18576919768,181030.58932690122,157613.69070750143,123989.49268989783,105676.1937696029,82559.56513249999,73553.06566350008,75054.14557500025,49235.51709720031,45933.137291900144,39328.37768129982,31222.528159199894,25518.408495500145,20414.7287964,28520.578318499924,20414.7287964,18613.42890260002,16511.909026500234,13509.74920349987,12008.659292000277,12008.659292000277,13509.74920349987,11108.009345100287,8105.848522099981,10507.579380500096,9006.49746900003,6604.765610599975,5704.115663699984,5704.115663699984,4503.248734500015,4803.465716799993,3602.5987876000245,6004.3316460000215,3902.8157699000017])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y20_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y20_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y20_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_19.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_19.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_19.eps')

# Running!
if __name__ == '__main__':
    selection_19()
