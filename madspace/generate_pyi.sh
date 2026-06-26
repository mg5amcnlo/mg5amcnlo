pybind11-stubgen madspace._madspace_py \
    -o . \
    --enum-class-locations=TChannelMode:PhaseSpaceMapping.TChannelMode \
    --enum-class-locations=Activation:MLP.Activation \
    --enum-class-locations=CutMode:Cuts.CutMode \
    --enum-class-locations=LRSchedule:AdamOptimizer.LRSchedule
