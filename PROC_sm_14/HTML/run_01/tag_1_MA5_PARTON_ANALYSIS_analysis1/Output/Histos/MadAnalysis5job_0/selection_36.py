def selection_36():

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

    # Creating weights for histo: y37_M_0
    y37_M_0_weights = numpy.array([0.0,0.0,0.0,6604.76607480001,27619.93031280004,61544.4106970001,81358.70092140003,109278.80123759955,127892.30144840035,131795.10149260017,119786.40135659976,125790.80142460053,107177.30121379973,114682.70129879955,111080.10125799956,110479.70125119992,98771.27111860013,89164.33100980002,92766.93105060002,83460.22094520008,79557.40090100003,75654.59085680009,62445.06070720009,58242.020659600006,54939.64062220004,56140.510635800085,51637.26058480007,44432.060503200046,43231.190489600005,41730.11047260004,41429.8904692,33324.04037739998,36326.210411400025,38427.73043520007,31522.740356999977,24617.76027879999,32123.180363800053,24317.550275400066,28220.360319600004,23717.11026859999])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y37_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ p_{1} p_{2} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y37_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y37_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_36.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_36.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_36.eps')

# Running!
if __name__ == '__main__':
    selection_36()
