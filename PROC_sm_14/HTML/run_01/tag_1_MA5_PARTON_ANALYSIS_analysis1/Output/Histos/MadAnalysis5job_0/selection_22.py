def selection_22():

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

    # Creating weights for histo: y23_M_0
    y23_M_0_weights = numpy.array([0.0,0.0,0.0,0.0,0.0,2401.732978399999,7205.1989351999955,9306.714916299998,15911.479856900001,16812.1298488,26719.27975969998,44732.27959769995,67548.73939249998,76555.23931149996,93667.57915760002,104475.39906039981,103874.8990658004,103574.69906850025,107177.29903610026,99371.69910629997,99671.91910359994,91566.06917649996,90665.41918459996,86462.37922240004,86162.16922509996,79557.3992845,80157.83927909994,73252.84934120002,64246.359422199945,61544.40944649995,64846.78941679998,56740.93948969999,54639.41950860002,45032.48959500001,49535.739554499996,48034.659567999974,53138.33952209999,45332.70959229998,42030.329621999954,41730.10962469999])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y23_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} p_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y23_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y23_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_22.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_22.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_22.eps')

# Running!
if __name__ == '__main__':
    selection_22()
