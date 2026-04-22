def selection_37():

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

    # Creating weights for histo: y38_M_0
    y38_M_0_weights = numpy.array([0.0,49535.73950500001,169922.59830200006,227864.3977230001,271695.99728500034,268093.3973210004,256084.7974409997,208350.2979180003,168721.6983140004,150108.29850000006,134797.2986529996,111380.39888699964,98471.049016,84360.86915699998,75954.79924100004,60943.96939100003,55239.859447999974,47134.00952899999,36326.209637,38727.93961300004,35425.559646,34224.68965800005,32123.17967899998,24617.759754000024,26719.279732999985,22516.24977499996,25218.19974799996,17412.559826000037,13209.52986800001,18012.999819999968,13209.52986800001,12308.879877000012,12008.659880000047,10207.35989800005,9306.714907000001,7805.631922,10507.579895000015,9006.497910000004,6604.765933999995,8105.848918999996])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y38_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ p_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y38_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y38_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_37.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_37.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_37.eps')

# Running!
if __name__ == '__main__':
    selection_37()
