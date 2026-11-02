#include "madspace/driver/io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "madspace/util.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {

std::string dtype_to_str(DataType dtype) {
    switch (dtype) {
    case DataType::dt_float32:
        return "<f4";
        break;
    case DataType::dt_float64:
        return "<f8";
        break;
    case DataType::dt_int32:
        return "<i4";
        break;
    default:
        throw std::invalid_argument("Unsupported data type");
    }
}

DataType str_to_dtype(std::string dtype) {
    if (dtype == "<f4") {
        return DataType::dt_float32;
    } else if (dtype == "<f8") {
        return DataType::dt_float64;
    } else if (dtype == "<i4") {
        return DataType::dt_int32;
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
}

// The header-length field's width depends on the npy format version: 1.0
// uses 2 bytes (uint16, so the header itself must fit in 65535 bytes), 2.0/3.0
// use 4 bytes (uint32) -- write_event_header() switches to 2.0 whenever the
// header would otherwise be too long to describe in 2 bytes (eg several "add
// process" channels of different final-state multiplicity combined into one
// binary EventFile can easily need a header longer than 65535 bytes' worth of
// field descriptors). Only ever reading the 2-byte case silently truncated the
// *declared* length for any such oversized header, cutting a field name off
// mid-string for every reader (this one and numpy alike).
constexpr std::size_t npy_preamble_size = 10;
constexpr std::size_t npy_wide_preamble_size = 12;

/// Check the magic number and version (the first 8 bytes) of an npy file and
/// return the size of its preamble: 10 bytes for format 1.0, 12 for 2.0/3.0.
std::size_t preamble_size(const char* bytes) {
    if (std::memcmp(bytes, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid header of npy file");
    }
    switch (static_cast<unsigned char>(bytes[6])) {
    case 1:
        return npy_preamble_size;
    case 2:
    case 3:
        return npy_wide_preamble_size;
    default:
        throw std::runtime_error("invalid header of npy file");
    }
}

/// Return the length of the header that follows the preamble of an npy file.
/// bytes must hold the full preamble (see preamble_size()).
std::size_t parse_preamble(const char* bytes) {
    // little-endian 16 or 32 bit length
    std::size_t length_bytes = preamble_size(bytes) - 8;
    std::size_t length = 0;
    for (std::size_t i = 0; i < length_bytes; ++i) {
        length |= static_cast<std::size_t>(static_cast<unsigned char>(bytes[8 + i]))
            << (8 * i);
    }
    return length;
}

json parse_header_text(std::string header) {
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

json parse_header(std::fstream& file_stream) {
    char preamble[npy_wide_preamble_size];
    file_stream.read(preamble, 8);
    if (file_stream.fail()) {
        throw std::runtime_error("invalid header of npy file");
    }
    file_stream.read(preamble + 8, preamble_size(preamble) - 8);
    if (file_stream.fail()) {
        throw std::runtime_error("invalid header of npy file");
    }
    std::string header(parse_preamble(preamble), '\0');
    file_stream.read(header.data(), header.size());
    return parse_header_text(std::move(header));
}

/// The dtype and shape described by the parsed header of a tensor npy file.
std::pair<DataType, SizeVec> parse_tensor_header(json header) {
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
    return {dtype, shape};
}

/// The npy header of `tensor`: 128 bytes, padded with spaces.
std::string tensor_header(const Tensor& tensor) {
    using namespace std::string_literals;
    // magic, version 1.0, little-endian header length of 118
    std::string header = "\x93NUMPY\x01\x00"s;
    header.push_back(static_cast<char>(118));
    header.push_back('\0');
    header += std::format(
        "{{'descr':'{}','fortran_order':True,'shape':(", dtype_to_str(tensor.dtype())
    );
    for (std::size_t size : tensor.shape()) {
        header += std::format("{},", size);
    }
    header += ")}";
    if (header.size() > 127) {
        throw std::runtime_error("Tensor has too many dimensions to save");
    }
    header.resize(127, ' ');
    header.push_back('\n');
    return header;
}

std::vector<std::pair<std::string, std::string>>
full_descr(std::size_t particle_count, const DataLayout& layout) {
    std::vector<std::pair<std::string, std::string>> field_descr;
    field_descr.reserve(
        layout.event_layout().size() + particle_count * layout.particle_layout().size()
    );
    for (auto& field : layout.event_layout()) {
        field_descr.emplace_back(field.name, FieldLayout::dtype(field.type));
    }
    for (std::size_t i = 1; i <= particle_count; ++i) {
        for (auto& field : layout.particle_layout()) {
            field_descr.push_back(
                {std::format("part{}_{}", i, field.name),
                 FieldLayout::dtype(field.type)}
            );
        }
    }
    return field_descr;
}

} // namespace

std::vector<std::string> madspace::peek_npy_field_names(const std::string& file) {
    std::fstream file_stream(file, std::ios::binary | std::ios::in);
    if (!file_stream) {
        throw std::runtime_error(std::format("Could not open file {}", file));
    }
    json header = parse_header(file_stream);
    json descr = header.at("descr");
    std::vector<std::string> names;
    names.reserve(descr.size());
    for (auto& item : descr) {
        names.push_back(item.at(0).get<std::string>());
    }
    return names;
}

namespace {

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
        !fortran_order.is_boolean() || fortran_order.get<bool>() ||
        !header_shape.is_array() || header_shape.size() != 1) {
        throw std::runtime_error("Invalid header for event file");
    }
    std::size_t particle_descr_count = descr.size() - event_field_count;
    std::size_t particle_count = 0;
    if (particle_field_count == 0) {
        if (particle_descr_count != 0) {
            throw std::runtime_error("Invalid header for event file");
        }
    } else {
        if (particle_descr_count % particle_field_count != 0) {
            throw std::runtime_error("Invalid header for event file");
        }
        particle_count = particle_descr_count / particle_field_count;
    }
    std::size_t event_count = header_shape.at(0).get<std::size_t>();

    auto field_descr = full_descr(particle_count, layout);
    for (auto [descr_item, descr_expected] : zip(descr, field_descr)) {
        if (!descr_item.is_array() || descr_item.size() != 2 ||
            descr_item.at(0) != descr_expected.first ||
            descr_item.at(1) != descr_expected.second) {
            throw std::runtime_error("Invalid header for event file");
        }
    }
    // parse_header leaves the stream at the end of the header
    std::size_t header_size = file_stream.tellg();
    return {header_size, particle_count, event_count};
}

std::tuple<std::size_t, std::size_t> write_event_header(
    std::fstream& file_stream,
    std::size_t particle_count,
    const DataLayout& layout,
    std::size_t header_size = 0
) {
    using namespace std::string_literals;
    // Built into a string first (rather than written straight to
    // file_stream, as this function used to) so its true length is known
    // before deciding whether npy format 1.0's 2-byte header-length field
    // (max 65535) is wide enough, or format 2.0's 4-byte field is needed
    // instead -- see parse_header()'s matching version-selection logic for
    // why: silently wrapping an oversized length into 16 bits (the old
    // behavior here) corrupts the file for every reader, this project's own
    // and numpy's alike. A large particle_count (eg several "add process"
    // channels of different final-state multiplicity combined into one
    // binary EventFile) easily needs a header longer than 65535 bytes'
    // worth of field descriptors.
    std::string content = "{'descr':["s;
    auto field_descr = full_descr(particle_count, layout);
    for (auto [field_name, field_type] : field_descr) {
        content += std::format("('{}','{}'),", field_name, field_type);
    }
    content += "],'fortran_order':False,'shape':(";

    // prefix_len: magic (6) + major/minor version (2) + header-length field
    // (2 for v1.0, 4 for v2.0) -- header_size is always the *absolute* file
    // offset where the header ends and event data begins (matching
    // EventFile's own _header_size-based seek math), regardless of format.
    bool wide = false;
    std::size_t prefix_len = 10;
    if (header_size == 0) {
        header_size = (prefix_len + content.size() + 100) / 64 * 64;
        if (header_size - prefix_len > 0xFFFF) {
            wide = true;
            prefix_len = 12;
            header_size = (prefix_len + content.size() + 100) / 64 * 64;
        }
    } else {
        wide = (header_size - prefix_len > 0xFFFF);
        if (wide) prefix_len = 12;
    }

    file_stream << (wide ? "\x93NUMPY\x02\x00"s : "\x93NUMPY\x01\x00"s);
    if (wide) {
        uint32_t zero32 = 0;
        file_stream.write(reinterpret_cast<char*>(&zero32), 4);
    } else {
        uint16_t zero16 = 0;
        file_stream.write(reinterpret_cast<char*>(&zero16), 2);
    }
    file_stream << content;
    std::size_t shape_pos = file_stream.tellp();
    for (std::size_t i = shape_pos; i < header_size - 1; ++i) {
        file_stream.put(' ');
    }
    file_stream.put('\n');
    file_stream.seekp(8);
    // little-endian 16 bit (v1.0) or 32 bit (v2.0) length
    std::size_t header_length = header_size - prefix_len;
    for (std::size_t i = 0; i < prefix_len - 8; ++i) {
        file_stream.put(static_cast<char>((header_length >> (8 * i)) & 0xff));
    }
    file_stream.seekp(header_size);
    return {header_size, shape_pos};
}

} // namespace

Tensor madspace::load_tensor(const std::string& file) {
    std::fstream file_stream(file, std::ios::binary | std::ios::in);
    if (file_stream.fail()) {
        throw std::runtime_error(std::format("Could not open file '{}'", file));
    }
    auto [dtype, shape] = parse_tensor_header(parse_header(file_stream));
    Tensor tensor(dtype, shape);
    file_stream.read(static_cast<char*>(tensor.data()), tensor.byte_size());
    if (file_stream.fail()) {
        throw std::runtime_error("Failed to read file");
    }
    return tensor;
}

void madspace::save_tensor(const std::string& file, Tensor tensor) {
    Tensor cpu_tensor = tensor.cpu().contiguous();
    std::ofstream file_stream(file, std::ios::binary);
    std::string header = tensor_header(tensor);
    file_stream.write(header.data(), header.size());
    file_stream.write(
        static_cast<const char*>(cpu_tensor.data()), cpu_tensor.byte_size()
    );
}

Tensor madspace::load_tensor(TarReader& tar, const std::string& name) {
    std::vector<char> data = tar.read(name);
    if (data.size() < npy_preamble_size ||
        data.size() < preamble_size(data.data())) {
        throw std::runtime_error(std::format("Invalid npy file '{}'", name));
    }
    std::size_t preamble_length = preamble_size(data.data());
    std::size_t header_length = parse_preamble(data.data());
    std::size_t data_offset = preamble_length + header_length;
    if (data.size() < data_offset) {
        throw std::runtime_error(std::format("Invalid npy file '{}'", name));
    }
    auto [dtype, shape] = parse_tensor_header(
        parse_header_text(std::string(data.data() + preamble_length, header_length))
    );
    Tensor tensor(dtype, shape);
    if (data.size() - data_offset != tensor.byte_size()) {
        throw std::runtime_error(std::format("Wrong data size in npy file '{}'", name));
    }
    std::memcpy(tensor.data(), data.data() + data_offset, tensor.byte_size());
    return tensor;
}

void madspace::save_tensor(TarWriter& tar, const std::string& name, Tensor tensor) {
    Tensor cpu_tensor = tensor.cpu().contiguous();
    std::string header = tensor_header(tensor);
    std::vector<char> data(header.size() + cpu_tensor.byte_size());
    std::memcpy(data.data(), header.data(), header.size());
    std::memcpy(data.data() + header.size(), cpu_tensor.data(), cpu_tensor.byte_size());
    tar.add(name, data.data(), data.size());
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
    _delete_on_close(delete_on_close),
    _layout(layout) {
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
        _event_size = layout.event_size() + _particle_count * layout.particle_size();
        _capacity = _event_count;
        if (mode == EventFile::append) {
            // rewrite the header with an open shape, filled in on close
            _file_stream.seekp(0);
            std::tie(_header_size, _shape_pos) =
                write_event_header(_file_stream, _particle_count, layout, _header_size);
            seek(_event_count);
        } else {
            seek(0);
        }
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
