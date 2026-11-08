#include "madspace/driver/event_utils.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

using namespace madspace;

DataLayout
madspace::combined_event_layout(int extra_event_flags, int extra_particle_flags) {
    return DataLayout(
        EventRecord::layout(
            EventRecord::f_weight | EventRecord::f_subproc_index |
            EventRecord::f_event_data | (extra_event_flags & ~EventRecord::f_bias)
        ),
        ParticleRecord::layout(ParticleRecord::f_particle_data | extra_particle_flags)
    );
}

int madspace::mlm_particle_flags(
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels
) {
    int flags = 0;
    for (auto& channel : channels) {
        flags |= channel->particle_layout_extra_flags() &
            (ParticleRecord::f_alphas_scales | ParticleRecord::f_pdf_ladder);
    }
    return flags;
}

namespace {

// the pdf ladder fields of one particle, or zeros (an inactive slot)
void copy_ladder(ParticleRecord in, ParticleRecord out, bool from_in) {
    out.ladder_x() = from_in ? in.ladder_x().value() : 0.;
    out.ladder_pdg() = from_in ? in.ladder_pdg().value() : 0.;
    out.ladder_q() = from_in ? in.ladder_q().value() : 0.;
    out.ladder_info() = from_in ? in.ladder_info().value() : 0.;
    out.ladder_lead_q() = from_in ? in.ladder_lead_q().value() : 0.;
}

} // namespace

void madspace::copy_channel_event(
    EventBuffer& in_buffer,
    std::size_t in_index,
    EventBuffer& out_buffer,
    std::size_t out_index,
    double weight,
    int subprocess_index,
    int extra_flags,
    int in_particle_flags,
    int out_particle_flags
) {
    bool has_clustering = out_particle_flags & ParticleRecord::f_clustering;
    bool has_alphas_scales = out_particle_flags & ParticleRecord::f_alphas_scales;
    bool has_ladder = out_particle_flags & ParticleRecord::f_pdf_ladder;
    bool channel_has_alphas_scales =
        in_particle_flags & ParticleRecord::f_alphas_scales;
    bool channel_has_ladder = in_particle_flags & ParticleRecord::f_pdf_ladder;
    auto event_in = in_buffer.event(in_index);
    auto event_out = out_buffer.event(out_index);
    // unweighted with the bias b, so the event carries 1 / b
    event_out.weight() =
        extra_flags & EventRecord::f_bias ? weight / event_in.bias().value() : weight;
    event_out.subprocess_index() = extra_flags & EventRecord::f_subproc_index
        ? event_in.subprocess_index().value()
        : subprocess_index;
    event_out.diagram_index() = event_in.diagram_index();
    event_out.color_index() = event_in.color_index();
    event_out.flavor_index() = event_in.flavor_index();
    event_out.helicity_index() = event_in.helicity_index();
    event_out.ren_scale() = event_in.ren_scale();
    event_out.alpha_qcd() = event_in.alpha_qcd();

    if (extra_flags & EventRecord::f_beam1) {
        event_out.x1() = event_in.x1();
        event_out.fact_scale1() = event_in.fact_scale1();
    }
    if (extra_flags & EventRecord::f_beam2) {
        event_out.x2() = event_in.x2();
        event_out.fact_scale2() = event_in.fact_scale2();
    }
    if (extra_flags & EventRecord::f_partial_weights) {
        event_out.partial_weight_product() = event_in.partial_weight_product();
    }
    if (extra_flags & EventRecord::f_matrix_element) {
        event_out.matrix_element() = event_in.matrix_element();
    }

    std::size_t i = 0;
    for (; i < in_buffer.particle_count(); ++i) {
        auto particle_in = in_buffer.particle(in_index, i);
        auto particle_out = out_buffer.particle(out_index, i);
        particle_out.energy() = particle_in.energy();
        particle_out.px() = particle_in.px();
        particle_out.py() = particle_in.py();
        particle_out.pz() = particle_in.pz();
        if (has_clustering) {
            particle_out.cluster_scale() = particle_in.cluster_scale();
        }
        if (has_alphas_scales) {
            // zero from a channel without vertices (2 -> 1)
            particle_out.alphas_scale() =
                channel_has_alphas_scales ? particle_in.alphas_scale().value() : 0.;
        }
        if (has_ladder) {
            copy_ladder(particle_in, particle_out, channel_has_ladder);
        }
    }
    for (; i < out_buffer.particle_count(); ++i) {
        auto particle_out = out_buffer.particle(out_index, i);
        particle_out.energy() = 0.;
        particle_out.px() = 0.;
        particle_out.py() = 0.;
        particle_out.pz() = 0.;
        // the buffer is reused: a padding slot must not keep the scales of
        // an earlier, higher-multiplicity event
        if (has_clustering) {
            particle_out.cluster_scale() = 0.;
        }
        if (has_alphas_scales) {
            particle_out.alphas_scale() = 0.;
        }
        if (has_ladder) {
            copy_ladder(particle_out, particle_out, false);
        }
    }
}

void madspace::fill_lhe_event(
    LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    MixMaxRandom& rand_gen,
    int extra_event_flags,
    int extra_particle_flags,
    const GeneratorConfig& config
) {
    EventRecord event_in = buffer.event(event_index);
    lhe_event.weight = event_in.weight();
    lhe_event.process_id = 0;
    lhe_event.alpha_qed = 0; // TODO: populate this
    lhe_event.alpha_qcd = event_in.alpha_qcd();
    // Both scales, as LHEF3 <scales> attributes: SCALUP holds only one number,
    // so write them out explicitly.
    lhe_event.mu_r = event_in.ren_scale();
    lhe_event.mu_f1 = (extra_event_flags & EventRecord::f_beam1)
        ? double(event_in.fact_scale1())
        : double(event_in.ren_scale());
    lhe_event.mu_f2 = (extra_event_flags & EventRecord::f_beam2)
        ? double(event_in.fact_scale2())
        : lhe_event.mu_f1;
    lhe_event.has_scales = true;
    // SCALUP is the scale the PDFs were evaluated at (Les Houches accord), the
    // larger mu_F as madevent's unwgt.f writes it. A shower that reads SCALUP
    // starts from it; mu_R is only in <scales> and alpha_qcd. Without a PDF
    // both mu_F fall back to mu_R.
    lhe_event.scale = std::max(lhe_event.mu_f1, lhe_event.mu_f2);
    lhe_event.particles.clear();
    for (std::size_t i = 0; i < buffer.particle_count(); ++i) {
        auto particle_in = buffer.particle(event_index, i);
        if (particle_in.energy() == 0.) {
            break;
        }
        lhe_event.particles.push_back(
            LHEParticle{
                .px = particle_in.px(),
                .py = particle_in.py(),
                .pz = particle_in.pz(),
                .energy = particle_in.energy(),
            }
        );
    }
    bool has_clustering = extra_particle_flags & ParticleRecord::f_clustering;
    lhe_event.has_cluster_scales = has_clustering;
    if (has_clustering) {
        for (std::size_t i = 0; i < lhe_event.particles.size(); ++i) {
            lhe_event.particles.at(i).cluster_scale =
                buffer.particle(event_index, i).cluster_scale();
        }
    }
    lhe_completer.complete_event_data(
        lhe_event,
        event_in.subprocess_index(),
        event_in.diagram_index(),
        event_in.color_index(),
        event_in.flavor_index(),
        event_in.helicity_index(),
        rand_gen
    );
    // after the completion, which sets the pdg ids it needs
    if (has_clustering && config.mlm_hardest_shower_scale) {
        apply_mlm_hardest_shower_scale(
            lhe_event, config.mlm_collider_energy, config.mlm_max_jet_flavor
        );
    } else if (has_clustering && config.mlm_collider_energy > 0.) {
        apply_mlm_collider_shower_scale(
            lhe_event, config.mlm_collider_energy, config.mlm_max_jet_flavor
        );
    }
}

void madspace::write_lhe_particles(
    const LHEEvent& lhe_event, EventBuffer& buffer, std::size_t event_index
) {
    std::size_t i = 0;
    for (; i < lhe_event.particles.size(); ++i) {
        buffer.particle(event_index, i)
            .from_lhe_particle(lhe_event.particles[i], lhe_event.has_cluster_scales);
    }
    for (; i < buffer.particle_count(); ++i) {
        buffer.particle(event_index, i)
            .from_lhe_particle(LHEParticle{}, lhe_event.has_cluster_scales);
    }
}

void madspace::sum_channel_status(
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    GeneratorStatus& status
) {
    double total_mean = 0., total_var = 0.;
    double total_mean_abs = 0., total_var_abs = 0.;
    std::size_t total_count = 0, total_count_opt = 0;
    std::size_t total_count_after_cuts = 0, total_count_after_cuts_opt = 0;
    std::size_t total_integ_count = 0;
    std::size_t iterations = 0;
    std::size_t count_invalid_step_values = 0;
    bool optimized = true;
    const ChannelEventGenerator* bad_channel = nullptr;
    for (auto& channel : channels) {
        auto& chan_status = channel->status();
        auto integral = channel->result_integral();
        // special case for channels with 0 samples, as they have nan variance
        if (bad_channel == nullptr && integral.count > 0 &&
            (!std::isfinite(integral.mean) || !std::isfinite(integral.variance) ||
             !std::isfinite(integral.mean_abs) ||
             !std::isfinite(integral.variance_abs))) {
            bad_channel = channel.get();
        }
        if (channel->uses_integral_prior()) {
            // not sampled in this run (no events assigned): use the earlier estimate
            auto [mean, error] = channel->integral_prior();
            auto [mean_abs, error_abs] = channel->abs_integral_prior();
            total_mean += mean;
            total_var += error * error;
            total_mean_abs += mean_abs;
            total_var_abs += error_abs * error_abs;
        } else {
            total_mean += integral.mean;
            total_mean_abs += integral.mean_abs;
            if (integral.count > 0) {
                total_var += integral.variance / integral.count;
                total_var_abs += integral.variance_abs / integral.count;
            }
        }
        total_count += chan_status.count;
        total_count_opt += chan_status.count_opt;
        total_count_after_cuts += chan_status.count_after_cuts;
        total_count_after_cuts_opt += chan_status.count_after_cuts_opt;
        total_integ_count += integral.count;
        iterations = std::max(chan_status.iterations, iterations);
        count_invalid_step_values += chan_status.count_invalid_step_values;
        if (!chan_status.optimized) {
            optimized = false;
        }
    }
    if (bad_channel != nullptr || !std::isfinite(total_mean) ||
        !std::isfinite(total_mean_abs)) {
        std::string where = bad_channel != nullptr
            ? std::format(
                  "channel '{}' after {} samples (mean={}, variance={})",
                  bad_channel->status().name,
                  bad_channel->result_integral().count,
                  bad_channel->result_integral().mean,
                  bad_channel->result_integral().variance
              )
            : std::format("the sum over channels (mean={})", total_mean);
        throw std::runtime_error(
            std::format(
                "non-finite integral in {}. A non-finite weight cannot be recovered "
                "from by sampling further, so the integration is aborted. This usually "
                "means the matrix element or the phase-space mapping returned nan/inf "
                "for "
                "some phase-space point -- check the process for a zero or negative "
                "width, "
                "a parameter point outside the model's validity, or a kinematic "
                "configuration at a threshold.",
                where
            )
        );
    }

    status.mean = total_mean;
    status.error = std::sqrt(total_var);
    status.mean_abs = total_mean_abs;
    status.error_abs = std::sqrt(total_var_abs);
    status.rel_std_dev = std::sqrt(total_var_abs * total_integ_count) / total_mean_abs;
    status.count = total_count;
    status.count_opt = total_count_opt;
    status.count_after_cuts = total_count_after_cuts;
    status.count_after_cuts_opt = total_count_after_cuts_opt;
    status.iterations = iterations;
    status.count_invalid_step_values = count_invalid_step_values;
    status.optimized = optimized;
}
