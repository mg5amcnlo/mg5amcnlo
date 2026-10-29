// One-step MAcNLOPS proposals. The final physics veto is performed separately.
// Build with: make MacNLOPS PYTHIA8LOCATION=/path/to/pythia8
#include "MacNLOPS.h"

int main(int argc, char** argv) {
  try {
    const Options o = options(argc, argv);
    Pythia pythia;
    configure(pythia, o);
    if (!o.probe.empty()) {
      probeConfiguration(o, pythia);
      return 0;
    }
    RawReader raw(o.input);
    // Pythia exposes a const view of its mutable Info, whereas its LHAup
    // constructor requires the mutable pointer used by the native LHE reader.
    std::shared_ptr<CountingReader> reader = std::make_shared<CountingReader>(
      const_cast<Info*>(&pythia.info), o.input, pythia, o.mcatnloDelta);
    std::shared_ptr<FirstEmission> hook = std::make_shared<FirstEmission>();
    configureHook(pythia, o, *hook);
    pythia.setLHAupPtr(reader);
    pythia.setUserHooksPtr(hook);
    if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");
    validateQCDOnlySettings(pythia);
    validateSourceMatchingSettings(pythia, o);
    hook->captureBeamLimits();
    if (!pythia.settings.writeFile(o.settingsOutput, true))
      throw std::runtime_error("Cannot write Pythia settings snapshot");

    std::ofstream output(o.output.c_str()), metadata(o.metadata.c_str());
    if (!output || !metadata) throw std::runtime_error("Cannot open output files");
    output << std::scientific << std::setprecision(17);
    metadata << std::setprecision(17);
    std::size_t count = 0, emitted = 0, hard = 0;
    InputEvent input;
    while (raw.next(input)) {
      ++count;
      int kind = o.kind == "S" ? 1 : o.kind == "H" ? 2 : input.kind;
      if (!kind) throw std::runtime_error("Missing #aMCatNLO kind; use --event-kind");
      if (input.hasHard && kind != input.hardMetadata[1])
        throw std::runtime_error("Explicit kind conflicts with hard-event annotation");
      if (!input.kind) input.kind = kind;
      hook->reset();
      hook->source = &input;
      std::vector<LHEParticle> particles = input.particles;
      reader->expect(input);
      const std::size_t before = reader->readCount;
      if (kind == 2) {
        if (!reader->skipOne()) throw std::runtime_error("Cannot consume H event");
        ++hard;
      } else {
        const auto warningStart = nativeTrialCounts(pythia);
        const bool success = pythia.next();
        validateNominalWeights(pythia);
        hook->nativeTrialWarnings = warningDifference(nativeTrialCounts(pythia), warningStart);
        if (!success && !hook->finished)
          throw std::runtime_error("Shower failed before first emission at event " +
                                   std::to_string(count));
        if (success && hook->finished)
          throw std::runtime_error("First-emission technical stop did not abort");
        if (hook->seen) {
          particles = hardParticles(*hook, input);
          if (particles.size() != input.particles.size() + 1)
            throw std::runtime_error("Shower changed external multiplicity unexpectedly");
          ++emitted;
        }
      }
      if (reader->readCount != before + 1)
        throw std::runtime_error("Pythia consumed more than one source event");
      validateQCDOnlyEndpoint(particles, input);
      output << input.prefix;
      writeLHE(output, input, particles, *hook);
      recordJSON(metadata, count, kind, input, *hook, particles);
      if (!output || !metadata) throw std::runtime_error("Output write failure");
    }
    output << input.prefix;
    if (!output || !metadata) throw std::runtime_error("Output write failure");
    std::cout << "MAcNLOPS proposals: " << count << " source events, "
              << emitted << " S emissions, " << hard << " H pass-throughs\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MAcNLOPS: " << error.what() << '\n';
    return 1;
  }
}
