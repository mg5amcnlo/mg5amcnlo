def selection_29():

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

    # Creating weights for histo: y30_M_0
    y30_M_0_weights = numpy.array([0.0,0.0,1200.8659624000138,9006.49771800001,20414.729360799985,34825.128909599895,60643.758101199855,81058.48746199984,96669.74697319996,113782.09643739986,105075.79671000043,128792.89596740081,125190.29608020083,114082.29642800038,105976.49668179886,108678.3965972004,106276.69667239937,98771.26690739984,88864.1172175999,96369.52698260006,84360.86735859992,81358.69745260004,80458.04748080006,69950.46780980001,58242.01817640007,56140.50824219988,56140.50824219988,48334.868486600135,49835.95843959992,44131.838618200054,50736.60841139991,46833.78853360004,41730.10869339996,35425.5588908,36626.42885319989,30321.879050599928,31822.959003600023,33624.25894720001,29721.439069400138,27319.70914460005])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y30_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y30_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y30_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_29.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_29.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_29.eps')

# Running!
if __name__ == '__main__':
    selection_29()
