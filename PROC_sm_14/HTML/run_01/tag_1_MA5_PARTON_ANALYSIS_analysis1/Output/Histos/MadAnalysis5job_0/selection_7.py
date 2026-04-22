def selection_7():

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

    # Creating weights for histo: y8_PT_0
    y8_PT_0_weights = numpy.array([0.0,127291.797125281,432011.69024358015,469238.5894028592,411296.7907113991,308922.8930233799,258786.69415564046,199043.59550486034,148306.99665068014,130894.39704392097,99071.47776260004,69650.24842704009,64846.78853551997,48635.088901640025,38727.93912538007,29421.229335559958,24317.5494508199,22516.249491499908,15911.47964066001,14110.17968134002,13809.959688120094,9306.71478982,7505.414830500007,7805.6318237199985,7805.6318237199985,6904.981844060002,4503.248898300004,5403.898877960001,3902.8159118599992,3002.165932200003,2101.5159525400068,2101.5159525400068,1801.2999593199927,2101.5159525400068,1200.8659728800103,1801.2999593199927,1501.0829661000016,1200.8659728800103,300.21659322000033,600.4331864400007])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y8_PT_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$p_{T}$ $[ p_{1} ]$ $(GeV/c)$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y8_PT_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y8_PT_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_7.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_7.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_7.eps')

# Running!
if __name__ == '__main__':
    selection_7()
