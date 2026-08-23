# Complete the merge and fix the conflicts

Hi Claude.
In this repository, I merged two branches, one that branched off the latest commit of `https://github.com/zeniheisser/mg5amcnlo/tree/nlo` (the current `HEAD`), and the other being the *master* branch of the `https://github.com/mg5amcnlo/mg5amcnlo` repository (branch `3.x`).

The current `HEAD` is the result of the work implementing a multi-event interface to perform NLO computation.
You can read the paper, in tex mode, by reading the file `paper/main.tex` in the current working directory.
It will give you enough information on the reasoning behind it and of what it meant to create this interface.

The important thing is that in the future we will be able to run the plugin CUDACPP from this version, so that is why we need to merge the branches, so that these changes brought in by this `HEAD` will be available in the main version that can support CUDACPP plugin for hardware acceleration of matrix element computation.

As you can see, the merge has stopped with conflicts.
Given I'm not sure how to solve them, please you can try on my behalf.

Later, we can test if everything is fine, I'd like to obtain similar plots to the one that are present in the paper.
So, after the merge we'll see what to do.

In the meanwhile you can test it by trying to generate a process:
```
generate p p > t t~ [QCD]
output MY_PROC
launch
```

