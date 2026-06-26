#include "madspace/driver/io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "madspace/util.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
    case DataType::dt_float:
        return "<f8";
        break;
    case DataType::dt_int:
        return "<i4";
        break;
    default:
        throw std::invalid_argument("Unsupported data type");
    }
}

DataType str_to_dtype(std::string dtype) {
    if (dtype == "<f8") {
        return DataType::dt_float;
    } else if (dtype == "<i4") {
        return DataType::dt_int;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
}

json parse_header(std::fstream& file_stream) {
    char magic_num[8];
    file_stream.read(magic_num, 8);
    if (std::memcmp(magic_num, "\x93NUMPY\x01\x00", 8) != 0) {
        throw std::runtime_error("invalid header of npy file");
    }
    union {
        uint16_t size;
        char chars[2];
    } header_size;
    file_stream.read(&header_size.chars[0], 2);
    std::string header(header_size.size, '\0');
    file_stream.read(header.data(), header_size.size);
    header.erase(
        std::remove_if(
            header.begin(), header.end(), [](char x) { return std::isspace(x); }
        ),
        header.end()
    );
    std::replace(header.begin(), header.end(), '(', '[');
    std::replace(header.begin(), header.end(), ')', ']');
    std::size_t pos = 0;
    while ((pos = header.find(",}", pos)) != std::string::npos) {
        header.erase(pos, 1);
    }
    pos = 0;
    while ((pos = header.find(",]", pos)) != std::string::npos) {
        header.erase(pos, 1);
    }
    pos = 0;
    while ((pos = header.find("False", pos)) != std::string::npos) {
        header[pos] = 'f';
    }
    pos = 0;
    while ((pos = header.find("True", pos)) != std::string::npos) {
        header[pos] = 't';
    }
    pos = 0;
    while ((pos = header.find("'", pos)) != std::string::npos) {
        header[pos] = '"';
    }
    return json::parse(header);
}

std::vector<std::pair<std::string, std::string>>
full_descr(std::size_t particle_count, const DataLayout& layout) {
    std::vector<std::pair<std::string, std::string>> field_descr;
    field_descr.reserve(
        layout.event_layout().size() + particle_count * layout.particle_layout().size()
    );
    for (auto& field : layout.event_layout()) {
        field_descr.emplace_back(field.name, field.type);
    }
    for (std::size_t i = 1; i <= particle_count; ++i) {
        for (auto& field : layout.particle_layout()) {
            field_descr.push_back(
                {std::format("part{}_{}", i, field.name), field.type}
            );
        }
    }
    return field_descr;
}

std::tuple<std::size_t, std::size_t, std::size_t>
read_event_header(std::fstream& file_stream, const DataLayout& layout) {
    json header = parse_header(file_stream);
    if (!header.is_object()) {
        throw std::runtime_error("Invalid header");
    }
    json descr = header.at("descr");
    json fortran_order = header.at("fortran_order");
    json header_shape = header.at("shape");

    std::size_t event_field_count = layout.event_layout().size();
    std::size_t particle_field_count = layout.particle_layout().size();
    if (!descr.is_array() || descr.size() < event_field_count ||
        (descr.size() - event_field_count) % particle_field_count != 0 ||
        !fortran_order.is_boolean() || fortran_order.get<bool>() ||
        header_shape.is_array() || header_shape.size() != 1) {
        throw std::runtime_error("Invalid header for event file");
    }
    std::size_t particle_count = (descr.size() - event_field_count) / 5;
    std::size_t event_count = header_shape.at(0).get<std::size_t>();

    auto field_descr = full_descr(particle_count, layout);
    for (auto [descr_item, descr_expected] : zip(descr, field_descr)) {
        if (!descr_item.is_array() || descr_item.size() != 2 ||
            descr_item.at(0) != descr_expected.first ||
            descr_item.at(1) != descr_expected.second) {
            throw std::runtime_error("Invalid header for event file");
        }
    }
    std::size_t header_size = 0;
    return {header_size, particle_count, event_count};
}

std::tuple<std::size_t, std::size_t> write_event_header(
    std::fstream& file_stream,
    std::size_t particle_count,
    const DataLayout& layout,
    std::size_t header_size = 0
) {
    using namespace std::string_literals;
    file_stream << "\x93NUMPY\x01\x00\x00\x00{'descr':["s;
    auto field_descr = full_descr(particle_count, layout);
    for (auto [field_name, field_type] : field_descr) {
        file_stream << std::format("('{}','{}'),", field_name, field_type);
    }
    file_stream << "],'fortran_order':False,'shape':(";
    std::size_t shape_pos = file_stream.tellp();
    if (header_size == 0) {
        header_size = (shape_pos + 100) / 64 * 64;
    }
    for (int i = shape_pos; i < header_size - 1; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
    file_stream.seekp(8);
    uint16_t header_size_short = header_size - 10;
    file_stream.write(reinterpret_cast<char*>(&header_size_short), 2);
    file_stream.seekp(header_size);
    return {header_size, shape_pos};
}

} // namespace

Tensor madspace::load_tensor(const std::string& file) {
    std::fstream file_stream(file, std::ios::binary | std::ios::in);
    if (file_stream.fail()) {
        throw std::runtime_error(std::format("Could not open file '{}'", file));
    }
    json header = parse_header(file_stream);
    if (!header.is_object()) {
        throw std::runtime_error("Invalid header");
    }
    json descr = header.at("descr");
    json fortran_order = header.at("fortran_order");
    json header_shape = header.at("shape");
    if (!descr.is_string() || !fortran_order.is_boolean() ||
        !fortran_order.get<bool>() || !header_shape.is_array()) {
        throw std::runtime_error("Invalid file header");
    }
    DataType dtype = str_to_dtype(descr);
    SizeVec shape;
    for (auto& size : header_shape) {
        if (!size.is_number_unsigned()) {
            throw std::runtime_error("Invalid file header");
        }
        shape.push_back(size.get<std::size_t>());
    }
    Tensor tensor(dtype, shape);
    file_stream.read(static_cast<char*>(tensor.data()), tensor.byte_size());
    if (file_stream.fail()) {
        throw std::runtime_error("Failed to read file");
    }
    return tensor;
}

void madspace::save_tensor(const std::string& file, Tensor tensor) {
    using namespace std::string_literals;
    Tensor cpu_tensor = tensor.cpu().contiguous();
    std::ofstream file_stream(file, std::ios::binary);
    file_stream << "\x93NUMPY\x01\x00\x76\x00{'descr':'"s
                << dtype_to_str(tensor.dtype()) << "','fortran_order':True,'shape':(";
    for (std::size_t size : tensor.shape()) {
        file_stream << size << ",";
    }
    file_stream << ")}";
    for (int i = file_stream.tellp(); i < 127; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
    file_stream.write(
        static_cast<const char*>(cpu_tensor.data()), cpu_tensor.byte_size()
    );
}

EventFile::EventFile(
    const std::string& file_name,
    const DataLayout& layout,
    std::size_t particle_count,
    Mode mode,
    bool delete_on_close
) :
    _file_name(file_name),
    _event_count(0),
    _current_event(0),
    _capacity(0),
    _particle_count(particle_count),
    _event_size(layout.event_size() + particle_count * layout.particle_size()),
    _mode(mode),
    _delete_on_close(delete_on_close) {
    auto file_mode = std::ios::binary | std::ios::in;
    if (mode == EventFile::create) {
        file_mode |= std::ios::out | std::ios::trunc;
    } else if (mode == EventFile::append) {
        file_mode |= std::ios::out;
    }
    _file_stream.open(file_name, file_mode);
    if (!_file_stream) {
        throw std::runtime_error(std::format("Could not open file {}", file_name));
    }
    if (mode == EventFile::create) {
        std::tie(_header_size, _shape_pos) =
            write_event_header(_file_stream, particle_count, layout);
    } else {
        std::tie(_header_size, _particle_count, _event_count) =
            read_event_header(_file_stream, layout);
        if (mode == EventFile::load) {
            std::tie(_header_size, _shape_pos) =
                write_event_header(_file_stream, particle_count, layout, _header_size);
        }
        _capacity = _event_count;
    }
}

void EventFile::seek(std::size_t index) {
    _current_event = index;
    _file_stream.seekp(_header_size + index * _event_size);
}

void EventFile::clear() {
    if (_mode == EventFile::load) {
        throw std::runtime_error("Event file opened in read mode.");
    }

    _capacity = _current_event;
    seek(0);
    _event_count = 0;
}

EventFile::~EventFile() {
    if (!_file_stream.is_open() || _mode == EventFile::load) {
        return;
    }
    if (_delete_on_close) {
        _file_stream.close();
        std::filesystem::remove(_file_name);
        return;
    }

    _file_stream.seekp(_shape_pos);
    _file_stream << _event_count << ",)}";
    if (_event_count < _capacity) {
        _file_stream.close();
        std::filesystem::resize_file(
            _file_name, _header_size + _event_count * _event_size
        );
    }
}
