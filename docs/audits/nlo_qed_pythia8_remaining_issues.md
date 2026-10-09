# Remaining NLO QED MC@NLO issues with Pythia8

Status: 9 October 2026; Pythia 8.318. NLO QED shower launches remain disabled.

- **ISR conversions:** Pythia 8.318's Simple shower lacks backward quark-to-photon evolution in proton beams and lepton conversion channels. `SpaceShower:QEDshowerByQ = on` does not enable them. Zero raw MC subtraction plus the existing G replacement leaves divergent H weights; a consistent treatment is required.
- **Pythia source patch:** [pythia8318-global-qed.patch](../../Template/NLO/MCatNLO/srcPythia8/patches/pythia8318-global-qed.patch) is already supplied. It changes `src/SimpleTimeShower.cc` in **Pythia 8.318** to recognize purely colourless hard systems and include their particles in global recoil. Apply it from a separate Pythia source root with `patch -p1 < /path/to/pythia8318-global-qed.patch`, rebuild Pythia, and link the shower driver to that build. Application is manual; the patch adds neither ISR conversions nor mode-2 support for colourless radiators.
- **Global-recoil settings:** the current QED workaround requires `TimeShower:globalRecoil = on`, `TimeShower:globalRecoilMode = 0` and `TimeShower:nMaxGlobalRecoil = N`, where `N` counts all Born final-state particles. This configuration still needs production integration; the launch script retains the QCD mode-2 settings.
- **Coupling:** the launcher fixes the shower's electromagnetic coupling but does not synchronize `StandardModel:alphaEM0` with the model coupling used in subtraction.
- **Mass conventions:** massless conversion counterterms differ from Pythia's massive-daughter boundaries, pair thresholds and heavy-flavour restrictions. Validate this approximation; shower infrared cutoffs must not simply be copied into singular subtraction.
- **Matching validation:** establish the shower's first-order normalization, S/H cancellation and differential NLO accuracy, including competing ISR/FSR and G/spin terms. Current kernel and recoil tests are insufficient.
- **Scope:** decays, resonance assignments and varying Born multiplicities remain unvalidated. QED radiation from BSM particles still needs explicit routing support. QED MC@NLO-Delta and FxFx are unsupported.
- **Launch guard:** retain it until the intended supported configurations pass the matching checks above.

Recoil patch and current restrictions: [implementation notes](../../Template/NLO/MCatNLO/srcPythia8/patches/README).
