#pragma once

#include <cstring>
#include <fstream>

#include "madspace/driver/lhe_output.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

Tensor load_tensor(const std::string& file);
void save_tensor(const std::string& file, Tensor tensor);

struct FieldLayout {
    static constexpr const char* i32 = "<i4";
    static constexpr const char* f64 = "<f8";

    const char* name;
    const char* type;
    int group;
};

template <typename T>
class UnalignedRef {
public:
    UnalignedRef(void* ptr) : _ptr(ptr) {}
    T value() const {
        T value;
        std::memcpy(&value, _ptr, sizeof(T));
        return value;
    }
    operator T() const { return value(); }
    UnalignedRef<T> operator=(const T& value) {
        std::memcpy(_ptr, &value, sizeof(T));
        return *this;
    }
    UnalignedRef<T> operator=(const UnalignedRef<T>& value) {
        std::memcpy(_ptr, value._ptr, sizeof(T));
        return *this;
    }

private:
    void* _ptr;
};

class ParticleRecord {
public:
    static constexpr int f_none = 0;
    static constexpr int f_particle_data = 1;
    static constexpr int f_lhe_particle = 2;
    static constexpr int f_clustering = 4;

    static std::vector<FieldLayout> layout(int fields) {
        std::vector<FieldLayout> ret;
        if (fields & f_particle_data) {
            ret.insert(
                ret.end(),
                {
                    {"energy", FieldLayout::f64, 0},
                    {"px", FieldLayout::f64, 0},
                    {"py", FieldLayout::f64, 0},
                    {"pz", FieldLayout::f64, 0},
                }
            );
        } else if (fields & f_lhe_particle) {
            ret.insert(
                ret.end(),
                {
                    {"pdg_id", FieldLayout::i32, 0},
                    {"status_code", FieldLayout::i32, 0},
                    {"mother1", FieldLayout::i32, 0},
                    {"mother2", FieldLayout::i32, 0},
                    {"color", FieldLayout::i32, 0},
                    {"anti_color", FieldLayout::i32, 0},
                    {"px", FieldLayout::f64, 0},
                    {"py", FieldLayout::f64, 0},
                    {"pz", FieldLayout::f64, 0},
                    {"energy", FieldLayout::f64, 0},
                    {"mass", FieldLayout::f64, 0},
                    {"lifetime", FieldLayout::f64, 0},
                    {"spin", FieldLayout::f64, 0},
                }
            );
        }
        if (fields & f_clustering) {
            ret.push_back({"cluster_scale", FieldLayout::f64, 1});
        }
        return ret;
    }

    ParticleRecord(char* data, const std::size_t* offsets) :
        _data(data), _offsets(offsets) {}

    // raw particle data
    UnalignedRef<double> energy() { return &_data[0]; }
    UnalignedRef<double> px() { return &_data[8]; }
    UnalignedRef<double> py() { return &_data[16]; }
    UnalignedRef<double> pz() { return &_data[24]; }

    // LHE particle data
    UnalignedRef<int> lhe_pdg_id() { return &_data[0]; }
    UnalignedRef<int> lhe_status_code() { return &_data[4]; }
    UnalignedRef<int> lhe_mother1() { return &_data[8]; }
    UnalignedRef<int> lhe_mother2() { return &_data[12]; }
    UnalignedRef<int> lhe_color() { return &_data[16]; }
    UnalignedRef<int> lhe_anti_color() { return &_data[20]; }
    UnalignedRef<double> lhe_px() { return &_data[24]; }
    UnalignedRef<double> lhe_py() { return &_data[32]; }
    UnalignedRef<double> lhe_pz() { return &_data[40]; }
    UnalignedRef<double> lhe_energy() { return &_data[48]; }
    UnalignedRef<double> lhe_mass() { return &_data[56]; }
    UnalignedRef<double> lhe_lifetime() { return &_data[64]; }
    UnalignedRef<double> lhe_spin() { return &_data[72]; }

    // clustering data
    UnalignedRef<double> cluster_scale() { return &_data[_offsets[0] + 0]; }

    void from_lhe_particle(const LHEParticle& particle) {
        lhe_pdg_id() = particle.pdg_id;
        lhe_status_code() = particle.status_code;
        lhe_mother1() = particle.mother1;
        lhe_mother2() = particle.mother2;
        lhe_color() = particle.color;
        lhe_anti_color() = particle.anti_color;
        lhe_px() = particle.px;
        lhe_py() = particle.py;
        lhe_pz() = particle.pz;
        lhe_energy() = particle.energy;
        lhe_mass() = particle.mass;
        lhe_lifetime() = particle.lifetime;
        lhe_spin() = particle.spin;
    }

private:
    char* _data;
    const std::size_t* _offsets;
};

class EventRecord {
public:
    static constexpr int f_none = 0;
    static constexpr int f_weight = 1;
    static constexpr int f_subproc_index = 2;
    static constexpr int f_event_data = 4;
    static constexpr int f_lhe_event = 8;
    static constexpr int f_beam1 = 16;
    static constexpr int f_beam2 = 32;
    static constexpr int f_partial_weights = 64;

    static std::vector<FieldLayout> layout(int fields) {
        std::vector<FieldLayout> ret;
        if (fields & f_weight) {
            ret.push_back({"weight", FieldLayout::f64, 0});
        }
        if (fields & f_subproc_index) {
            ret.push_back({"subprocess_index", FieldLayout::i32, 1});
        }
        if (fields & f_event_data) {
            ret.insert(
                ret.end(),
                {
                    {"diagram_index", FieldLayout::i32, 2},
                    {"color_index", FieldLayout::i32, 2},
                    {"flavor_index", FieldLayout::i32, 2},
                    {"helicity_index", FieldLayout::i32, 2},
                    {"ren_scale", FieldLayout::f64, 2},
                    {"alpha_qcd", FieldLayout::f64, 2},
                }
            );
        }
        if (fields & f_lhe_event) {
            ret.insert(
                ret.end(),
                {
                    {"process_id", FieldLayout::i32, 3},
                    {"weight", FieldLayout::f64, 3},
                    {"scale", FieldLayout::f64, 3},
                    {"alpha_qed", FieldLayout::f64, 3},
                    {"alpha_qcd", FieldLayout::f64, 3},
                }
            );
        }
        if (fields & f_beam1) {
            ret.insert(
                ret.end(),
                {
                    {"x1", FieldLayout::f64, 4},
                    {"fact_scale1", FieldLayout::f64, 4},
                }
            );
        }
        if (fields & f_beam2) {
            ret.insert(
                ret.end(),
                {
                    {"x2", FieldLayout::f64, 5},
                    {"fact_scale2", FieldLayout::f64, 5},
                }
            );
        }
        if (fields & f_partial_weights) {
            ret.push_back({"partial_weight_product", FieldLayout::f64, 6});
        }
        return ret;
    }

    EventRecord(char* data, const std::size_t* offsets) :
        _data(data), _offsets(offsets) {}

    // event weight
    UnalignedRef<double> weight() { return &_data[0]; }

    // subprocess index
    UnalignedRef<int> subprocess_index() { return &_data[_offsets[0] + 0]; }

    // raw event data
    UnalignedRef<int> diagram_index() { return &_data[_offsets[1] + 0]; }
    UnalignedRef<int> color_index() { return &_data[_offsets[1] + 4]; }
    UnalignedRef<int> flavor_index() { return &_data[_offsets[1] + 8]; }
    UnalignedRef<int> helicity_index() { return &_data[_offsets[1] + 12]; }
    UnalignedRef<double> ren_scale() { return &_data[_offsets[1] + 16]; }
    UnalignedRef<double> alpha_qcd() { return &_data[_offsets[1] + 24]; }

    // LHE event data
    UnalignedRef<int> lhe_process_id() { return &_data[_offsets[2] + 0]; }
    UnalignedRef<double> lhe_weight() { return &_data[_offsets[2] + 4]; }
    UnalignedRef<double> lhe_scale() { return &_data[_offsets[2] + 12]; }
    UnalignedRef<double> lhe_alpha_qed() { return &_data[_offsets[2] + 20]; }
    UnalignedRef<double> lhe_alpha_qcd() { return &_data[_offsets[2] + 28]; }

    // partial weight data for beam 1
    UnalignedRef<double> x1() { return &_data[_offsets[3] + 0]; }
    UnalignedRef<double> fact_scale1() { return &_data[_offsets[3] + 8]; }

    // partial weight data for beam 2
    UnalignedRef<double> x2() { return &_data[_offsets[4] + 0]; }
    UnalignedRef<double> fact_scale2() { return &_data[_offsets[4] + 8]; }

    // combined partial weight
    UnalignedRef<double> partial_weight_product() { return &_data[_offsets[5] + 0]; }

    void from_lhe_event(const LHEEvent& event) {
        lhe_process_id() = event.process_id;
        lhe_weight() = event.weight;
        lhe_scale() = event.scale;
        lhe_alpha_qed() = event.alpha_qed;
        lhe_alpha_qcd() = event.alpha_qcd;
    }

private:
    char* _data;
    const std::size_t* _offsets;
};

class DataLayout {
public:
    DataLayout(
        const std::vector<FieldLayout>& event_layout,
        const std::vector<FieldLayout>& particle_layout
    ) :
        _event_layout(event_layout), _particle_layout(particle_layout) {
        auto size_and_offsets =
            [](const std::vector<FieldLayout>& layout,
               std::size_t& size,
               std::vector<std::size_t>& offsets) {
                size = 0;
                std::size_t group_index = 0;
                for (auto& field : layout) {
                    std::size_t field_size;
                    if (field.type == FieldLayout::i32) {
                        field_size = 4;
                    } else if (field.type == FieldLayout::f64) {
                        field_size = 8;
                    } else {
                        std::logic_error("unknown type");
                    }
                    if (field.group > group_index) {
                        offsets.resize(field.group);
                        offsets.back() = size;
                        group_index = field.group;
                    }
                    size += field_size;
                }
            };
        size_and_offsets(event_layout, _event_size, _event_offsets);
        size_and_offsets(particle_layout, _particle_size, _particle_offsets);
    }

    const std::vector<FieldLayout>& event_layout() const { return _event_layout; }
    const std::vector<FieldLayout>& particle_layout() const { return _particle_layout; }
    std::size_t event_size() const { return _event_size; }
    std::size_t particle_size() const { return _particle_size; }
    const std::vector<std::size_t>& event_offsets() const { return _event_offsets; }
    const std::vector<std::size_t>& particle_offsets() const {
        return _particle_offsets;
    }

private:
    std::vector<FieldLayout> _event_layout;
    std::vector<FieldLayout> _particle_layout;
    std::size_t _event_size;
    std::size_t _particle_size;
    std::vector<std::size_t> _event_offsets;
    std::vector<std::size_t> _particle_offsets;
};

inline const DataLayout weight_file_layout = DataLayout(
    EventRecord::layout(EventRecord::f_weight),
    ParticleRecord::layout(ParticleRecord::f_none)
);

class EventBuffer {
public:
    EventBuffer(
        std::size_t event_count, std::size_t particle_count, const DataLayout& layout
    ) :
        _event_count(event_count),
        _particle_count(particle_count),
        _layout(layout),
        _data(
            event_count *
            (layout.event_size() + particle_count * layout.particle_size())
        ) {}
    char* data() { return _data.data(); }
    const char* data() const { return _data.data(); }
    std::size_t size() const { return _data.size(); }
    std::size_t event_count() const { return _event_count; }
    std::size_t particle_count() const { return _particle_count; }
    const DataLayout& layout() const { return _layout; }

    std::size_t event_size() const {
        return _layout.event_size() + _particle_count * _layout.particle_size();
    }

    std::size_t event_offset(std::size_t event_index) const {
        return event_index * event_size();
    }

    std::size_t
    particle_offset(std::size_t event_index, std::size_t particle_index) const {
        return event_offset(event_index) + _layout.event_size() +
            particle_index * _layout.particle_size();
    }

    ParticleRecord particle(std::size_t event_index, std::size_t particle_index) {
        return {
            &_data.data()[particle_offset(event_index, particle_index)],
            _layout.particle_offsets().data()
        };
    }

    EventRecord event(std::size_t event_index) {
        return {
            &_data.data()[event_offset(event_index)], _layout.event_offsets().data()
        };
    }

    void resize(std::size_t event_count) {
        _data.resize(event_count * event_size());
        _event_count = event_count;
    }

    void copy_and_pad(EventBuffer& buffer) {
        if (buffer.particle_count() > particle_count()) {
            throw std::runtime_error("Given buffer contains too many particles");
        }
        resize(buffer.event_count());
        for (std::size_t i = 0; i < event_count(); ++i) {
            std::memcpy(
                &_data[event_offset(i)],
                &buffer._data[buffer.event_offset(i)],
                buffer.event_size()
            );
            std::memset(
                &_data[event_offset(i) + buffer.event_size()],
                0,
                event_size() - buffer.event_size()
            );
        }
    }

private:
    std::size_t _event_count;
    std::size_t _particle_count;
    const DataLayout& _layout;
    std::vector<char> _data;
};

class EventFile {
public:
    enum Mode { create, append, load };

    EventFile(
        const std::string& file_name,
        const DataLayout& layout,
        std::size_t particle_count = 0,
        Mode mode = create,
        bool delete_on_close = false
    );

    EventFile(EventFile&& other) noexcept = default;
    EventFile& operator=(EventFile&& other) noexcept = default;
    void seek(std::size_t index);
    void clear();
    std::size_t particle_count() const { return _particle_count; }
    std::size_t event_count() const { return _event_count; }
    ~EventFile();

    void write(EventBuffer& buffer) {
        if (_mode == EventFile::load) {
            throw std::runtime_error("Event file opened in read mode.");
        }
        if (buffer.particle_count() != _particle_count) {
            throw std::invalid_argument("Wrong number of particles");
        }
        _file_stream.write(buffer.data(), buffer.size());
        _current_event += buffer.event_count();
        if (_current_event > _event_count) {
            _event_count = _current_event;
        }
    }

    bool read(EventBuffer& buffer, std::size_t count) {
        if (_current_event == _event_count) {
            return false;
        }
        count = std::min(count, _event_count - _current_event);
        buffer.resize(count);
        if (buffer.particle_count() == _particle_count) {
            _file_stream.read(buffer.data(), buffer.size());
        } else if (buffer.particle_count() > _particle_count) {
            EventBuffer tmp_buffer(count, _particle_count, buffer.layout());
            _file_stream.read(tmp_buffer.data(), tmp_buffer.size());
            buffer.copy_and_pad(tmp_buffer);
        } else {
            throw std::invalid_argument("Wrong number of particles");
        }
        _current_event += count;
        return true;
    }

private:
    std::string _file_name;
    std::size_t _event_count;
    std::size_t _current_event;
    std::size_t _capacity;
    std::size_t _particle_count;
    std::size_t _shape_pos;
    std::fstream _file_stream;
    std::size_t _header_size;
    std::size_t _event_size;
    Mode _mode;
    bool _delete_on_close;
};

} // namespace madspace
