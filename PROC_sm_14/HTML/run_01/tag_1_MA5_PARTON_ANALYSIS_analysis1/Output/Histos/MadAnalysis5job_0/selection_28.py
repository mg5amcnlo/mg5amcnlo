def selection_28():

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

    # Creating weights for histo: y29_M_0
    y29_M_0_weights = numpy.array([21615.599459999896,57041.1585749999,95769.09760749998,157913.89605500092,173224.99567249962,216155.994599999,190637.4952375012,199643.99501250114,200544.6949899999,179229.29552250038,144404.19639249975,138700.09653499935,120987.29697749985,98771.26753249987,98471.04753999996,81959.13795249986,75654.5881099999,71751.7682075,63045.48842499995,56140.5085974999,50136.168747500094,47134.00882249995,38427.729039999904,35425.559115,31522.739212500102,30922.309227500024,22816.459430000057,24617.75938500005,23416.89941499989,23416.89941499989,13509.749662499937,17112.34957249992,12909.309677500105,15010.829625000013,9306.714767499998,11108.009722500115,10807.799729999948,9306.714767499998,10207.35974500012,7505.4148125000065])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y29_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{1} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y29_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y29_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_28.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_28.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_28.eps')

# Running!
if __name__ == '__main__':
    selection_28()
