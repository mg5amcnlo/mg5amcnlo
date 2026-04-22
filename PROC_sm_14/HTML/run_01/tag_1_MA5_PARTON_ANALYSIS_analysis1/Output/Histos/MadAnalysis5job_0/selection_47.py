def selection_47():

    # Library import
    import numpy
    import matplotlib
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec

    # Library version
    matplotlib_version = matplotlib.__version__
    numpy_version      = numpy.__version__

    # Histo binning
    xBinning = numpy.linspace(0.0,10.0,41,endpoint=True)

    # Creating data sequence: middle of each bin
    xData = numpy.array([0.125,0.375,0.625,0.875,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875,4.125,4.375,4.625,4.875,5.125,5.375,5.625,5.875,6.125,6.375,6.625,6.875,7.125,7.375,7.625,7.875,8.125,8.375,8.625,8.875,9.125,9.375,9.625,9.875])

    # Creating weights for histo: y48_DELTAR_0
    y48_DELTAR_0_weights = numpy.array([1501.0828908000076,15911.478842480066,52838.11615616039,95468.8730548804,117384.6914605599,137799.3899754421,163017.58814088182,159414.98840296187,198142.9855855978,232667.88307399862,244076.08224408093,282804.07942671684,276199.2799071993,211952.8845809625,160315.68833743825,142302.68964783844,104475.39239967884,92166.49329512019,69049.8149768002,60943.96556648029,50136.166352720415,31522.73770680038,25818.628121759957,18012.9986895998,18913.648624079793,9606.93130111999,7805.63143216001,7505.414454000038,6004.33156320003,3002.165781600015,3002.165781600015,600.433156320003,1200.865912640035,300.2165781600015,0.0,300.2165781600015,0.0,0.0,0.0,0.0])

    # Creating a new Canvas
    fig   = plt.figure(figsize=(8.75,6.25),dpi=80)
    frame = gridspec.GridSpec(1,1)
    pad   = fig.add_subplot(frame[0])

    # Creating a new Stack
    pad.hist(x=xData, bins=xBinning, weights=y48_DELTAR_0_weights,\
             label="$run\_01$", histtype="stepfilled", rwidth=1.0,\
             color="#5954d8", edgecolor="#5954d8", linewidth=1, linestyle="solid",\
             bottom=None, cumulative=False, density=False, align="mid", orientation="vertical")


    # Axis
    plt.rc('text',usetex=False)
    plt.xlabel(r"$\Delta R$ $[ p_{1}, p_{3} ]$ ",\
               fontsize=16,color="black")
    plt.ylabel(r"$\mathrm{Events}$ $(\mathcal{L}_{\mathrm{int}} = 10\ \mathrm{fb}^{-1})$ ",\
               fontsize=16,color="black")

    # Boundary of y-axis
    ymax=(y48_DELTAR_0_weights).max()*1.1
    #ymin=0 # linear scale
    ymin=min([x for x in (y48_DELTAR_0_weights) if x])/100. # log scale
    plt.gca().set_ylim(ymin,ymax)

    # Log/Linear scale for X-axis
    plt.gca().set_xscale("linear")
    #plt.gca().set_xscale("log",nonpositive="clip")

    # Log/Linear scale for Y-axis
    #plt.gca().set_yscale("linear")
    plt.gca().set_yscale("log",nonpositive="clip")

    # Saving the image
    plt.savefig('../../HTML/MadAnalysis5job_0/selection_47.png')
    plt.savefig('../../PDF/MadAnalysis5job_0/selection_47.png')
    plt.savefig('../../DVI/MadAnalysis5job_0/selection_47.eps')

# Running!
if __name__ == '__main__':
    selection_47()
