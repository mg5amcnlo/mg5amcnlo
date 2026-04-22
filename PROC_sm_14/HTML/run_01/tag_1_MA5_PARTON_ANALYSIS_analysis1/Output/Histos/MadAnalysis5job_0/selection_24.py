def selection_24():

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

    # Creating weights for histo: y25_M_0
    y25_M_0_weights = numpy.array([0.0,0.0,0.0,0.0,0.0,5403.898677800008,10507.579373500097,15911.479051300044,18913.64887229982,24617.758532200158,35425.55788780005,57341.36658110024,78656.74531020023,102373.89389609802,110479.69341280092,115283.19312639888,119486.19287580083,113481.8932337989,121887.89273260279,113181.69325169791,95468.87430780027,98471.04412880004,93367.3644330999,91265.84455840012,95769.09428990008,88864.11470159993,75654.58548919986,81959.13511329981,63045.486240999984,57041.15659899985,59142.66647370022,58242.01652740023,51036.8169570003,53438.55681379988,44732.277332899765,43231.18742240018,43531.40740449998,39028.15767300002,37827.28774460023,30321.878192099906])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y25_M_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$M$ $[ l+_{1} l-_{1} p_{1} p_{3} ]$ $(GeV/c^{2})$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y25_M_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y25_M_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_24.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_24.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_24.eps')

# Running!
if __name__ == '__main__':
    selection_24()
