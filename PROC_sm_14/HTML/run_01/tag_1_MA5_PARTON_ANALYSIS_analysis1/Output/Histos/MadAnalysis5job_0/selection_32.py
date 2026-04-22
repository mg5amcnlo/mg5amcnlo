def selection_32():

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

    # Creating weights for histo: y33_M_0
    y33_M_0_weights = numpy.array([51036.81522300063,126090.98819799846,207149.48061099747,246778.076901797,270495.17468189826,267192.77499099984,242875.17726710485,207149.48061099747,172024.0838987026,149808.08597809976,129993.78783269998,102373.89041789719,88263.6717386008,74453.71303120034,67248.51370560043,53438.55499819997,52838.11505440061,48935.305419700024,34224.68679660052,35125.33671230051,36025.98662800049,28220.357358600282,21315.378004900052,20114.50811730038,18313.2082859004,18313.2082859004,22516.247892499727,21315.378004900052,10807.798988399867,10507.579016500184,9606.93110080001,9306.714128900043,9606.93110080001,7205.198325600006,7805.63126940003,8706.281185100019,6904.981353700041,5704.115466099993,7205.198325600006,5103.681522300062])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y33_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l-_{1} p_{2} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y33_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y33_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_32.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_32.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_32.eps')

# Running!
if __name__ == '__main__':
    selection_32()
