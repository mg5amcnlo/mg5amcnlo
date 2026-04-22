def selection_34():

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

    # Creating weights for histo: y35_M_0
    y35_M_0_weights = numpy.array([51036.82005099998,166320.00016619996,261788.90026159995,330538.5003302999,332039.60033179994,299015.7002987999,232968.10023279994,190337.3001901999,159415.00015929993,120086.60011999993,111080.10011099993,90665.42009059998,71751.77007169998,68449.39006839998,48334.870048299985,48334.870048299985,39028.160038999995,33624.260033599996,25818.630025799994,31222.530031199993,26118.850026099997,21915.810021899993,22516.250022499997,18613.430018599996,11708.450011699999,13509.7500135,11408.230011399995,13509.7500135,9006.498008999997,10507.580010499996,9907.148009899996,5403.899005399999,9907.148009899996,5704.116005699999,7205.199007199998,8406.065008399999,4503.249004499999,6304.549006299999,3302.383003299999,3302.383003299999])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y35_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y35_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y35_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_34.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_34.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_34.eps')

# Running!
if __name__ == '__main__':
    selection_34()
