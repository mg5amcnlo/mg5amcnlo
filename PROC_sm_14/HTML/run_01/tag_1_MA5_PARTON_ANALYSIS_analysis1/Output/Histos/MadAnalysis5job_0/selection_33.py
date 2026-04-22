def selection_33():

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

    # Creating weights for histo: y34_M_0
    y34_M_0_weights = numpy.array([0.0,1200.8660008000002,6004.332004000003,31822.960021200008,70851.12004720002,116484.00007760001,148307.00009880005,166920.40011120003,168421.50011220004,170222.80011340004,165119.10011000003,144404.20009620007,137199.00009140006,128492.70008560004,115583.40007700004,110179.50007340004,84961.30005660003,81358.70005420002,82259.35005480003,72051.99004800004,61544.410041000025,59743.11003980003,61844.620041200025,55540.07003700002,51036.82003400001,39628.59002640001,38727.94002580002,37827.29002520001,31222.53002080001,34825.130023200014,31822.960021200008,24317.55001620001,27619.93001840001,22516.25001500001,24917.98001660001,19213.860012800003,20114.510013400002,21615.60001440001,18913.65001260001,17712.780011800005])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y34_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{2} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y34_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y34_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_33.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_33.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_33.eps')

# Running!
if __name__ == '__main__':
    selection_33()
