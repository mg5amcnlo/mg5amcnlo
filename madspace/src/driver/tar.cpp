#include "madspace/driver/tar.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t block_size = 512;
using Block = std::array<char, block_size>;

// ustar header layout
constexpr std::size_t name_pos = 0, name_len = 100;
constexpr std::size_t mode_pos = 100, mode_len = 8;
constexpr std::size_t uid_pos = 108, uid_len = 8;
constexpr std::size_t gid_pos = 116, gid_len = 8;
constexpr std::size_t size_pos = 124, size_len = 12;
constexpr std::size_t mtime_pos = 136, mtime_len = 12;
constexpr std::size_t chksum_pos = 148, chksum_len = 8;
constexpr std::size_t typeflag_pos = 156;
constexpr std::size_t magic_pos = 257, magic_len = 6;
constexpr std::size_t version_pos = 263;
constexpr std::size_t prefix_pos = 345, prefix_len = 155;

/// Write `value` as zero-padded octal into a field of `len` bytes, ending
/// with a NUL.
void put_octal(Block& block, std::size_t pos, std::size_t len, std::uint64_t value) {
    for (std::size_t i = len - 1; i-- > 0;) {
        block[pos + i] = static_cast<char>('0' + (value & 7));
        value >>= 3;
    }
    if (value != 0) {
        throw std::runtime_error("Value too large for tar header field");
    }
    block[pos + len - 1] = '\0';
}

/// Read an octal field, ignoring leading spaces and stopping at the first NUL
/// or space after the digits.
std::uint64_t get_octal(const Block& block, std::size_t pos, std::size_t len) {
    if (static_cast<unsigned char>(block[pos]) & 0x80) {
        throw std::runtime_error("Unsupported binary number in tar header");
    }
    std::uint64_t value = 0;
    bool digits = false;
    for (std::size_t i = 0; i < len; ++i) {
        char c = block[pos + i];
        if (c == ' ' && !digits) {
            continue;
        }
        if (c < '0' || c > '7') {
            break;
        }
        value = (value << 3) | static_cast<std::uint64_t>(c - '0');
        digits = true;
    }
    return value;
}

std::uint64_t checksum(const Block& block) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < block_size; ++i) {
        bool in_chksum = i >= chksum_pos && i < chksum_pos + chksum_len;
        sum += in_chksum ? 32 : static_cast<unsigned char>(block[i]);
    }
    return sum;
}

bool is_zero_block(const Block& block) {
    return std::all_of(block.begin(), block.end(), [](char c) { return c == '\0'; });
}

std::string get_string(const Block& block, std::size_t pos, std::size_t len) {
    std::size_t end = 0;
    while (end < len && block[pos + end] != '\0') {
        ++end;
    }
    return std::string(&block[pos], end);
}

/// Split `name` into the ustar prefix and name fields.
std::pair<std::string, std::string> split_name(const std::string& name) {
    if (name.empty()) {
        throw std::invalid_argument("Empty tar entry name");
    }
    if (name.size() <= name_len) {
        return {"", name};
    }
    // the split has to be at a '/', with the last part fitting in the name field
    for (std::size_t pos = name.find('/'); pos != std::string::npos;
         pos = name.find('/', pos + 1)) {
        if (pos <= prefix_len && name.size() - pos - 1 <= name_len &&
            name.size() - pos - 1 > 0) {
            return {name.substr(0, pos), name.substr(pos + 1)};
        }
    }
    throw std::invalid_argument(std::format("Tar entry name too long: {}", name));
}

} // namespace

TarWriter::TarWriter(const std::string& file_name) :
    _file_name(file_name),
    _stream(file_name, std::ios::binary | std::ios::trunc),
    _closed(false) {
    if (!_stream) {
        throw std::runtime_error(std::format("Could not open file '{}'", file_name));
    }
}

TarWriter::~TarWriter() {
    try {
        close();
    } catch (...) {
    }
}

void TarWriter::add(const std::string& name, const char* data, std::size_t size) {
    if (_closed) {
        throw std::runtime_error("Tar archive is already closed");
    }
    auto [prefix, short_name] = split_name(name);
    Block header{};
    std::memcpy(&header[name_pos], short_name.data(), short_name.size());
    std::memcpy(&header[prefix_pos], prefix.data(), prefix.size());
    put_octal(header, mode_pos, mode_len, 0644);
    put_octal(header, uid_pos, uid_len, 0);
    put_octal(header, gid_pos, gid_len, 0);
    put_octal(header, size_pos, size_len, size);
    put_octal(header, mtime_pos, mtime_len, 0);
    header[typeflag_pos] = '0';
    std::memcpy(&header[magic_pos], "ustar", magic_len); // includes the NUL
    header[version_pos] = '0';
    header[version_pos + 1] = '0';
    // six octal digits, NUL, space
    put_octal(header, chksum_pos, 7, checksum(header));
    header[chksum_pos + 7] = ' ';

    _stream.write(header.data(), block_size);
    _stream.write(data, static_cast<std::streamsize>(size));
    std::size_t padding = (block_size - size % block_size) % block_size;
    static const Block zeros{};
    _stream.write(zeros.data(), static_cast<std::streamsize>(padding));
    if (!_stream) {
        throw std::runtime_error(
            std::format("Failed to write to file '{}'", _file_name)
        );
    }
}

void TarWriter::close() {
    if (_closed) {
        return;
    }
    _closed = true;
    static const Block zeros{};
    _stream.write(zeros.data(), block_size);
    _stream.write(zeros.data(), block_size);
    _stream.close();
    if (_stream.fail()) {
        throw std::runtime_error(
            std::format("Failed to write to file '{}'", _file_name)
        );
    }
}

TarReader::TarReader(const std::string& file_name) :
    _file_name(file_name), _stream(file_name, std::ios::binary) {
    if (!_stream) {
        throw std::runtime_error(std::format("Could not open file '{}'", file_name));
    }
    Block header;
    std::uint64_t pos = 0;
    while (true) {
        _stream.read(header.data(), block_size);
        if (_stream.gcount() == 0 || is_zero_block(header)) {
            break;
        }
        if (_stream.gcount() != static_cast<std::streamsize>(block_size)) {
            throw std::runtime_error("Truncated tar header");
        }
        if (get_octal(header, chksum_pos, chksum_len) != checksum(header)) {
            throw std::runtime_error(
                std::format("Invalid tar header checksum in '{}'", file_name)
            );
        }
        std::uint64_t size = get_octal(header, size_pos, size_len);
        char type = header[typeflag_pos];
        pos += block_size;
        if (type == '0' || type == '\0') {
            std::string name = get_string(header, name_pos, name_len);
            // the prefix field is only defined for ustar archives
            if (std::memcmp(&header[magic_pos], "ustar", 5) == 0) {
                std::string prefix = get_string(header, prefix_pos, prefix_len);
                if (!prefix.empty()) {
                    name = prefix + "/" + name;
                }
            }
            _entries[name] = {pos, size};
        }
        std::uint64_t padded_size = (size + block_size - 1) / block_size * block_size;
        pos += padded_size;
        _stream.seekg(static_cast<std::streamoff>(pos));
    }
    _stream.clear();
}

std::vector<std::string> TarReader::names() const {
    std::vector<std::string> ret;
    ret.reserve(_entries.size());
    for (auto& [name, entry] : _entries) {
        ret.push_back(name);
    }
    return ret;
}

bool TarReader::contains(const std::string& name) const {
    return _entries.contains(name);
}

const TarReader::Entry& TarReader::entry(const std::string& name) const {
    auto search = _entries.find(name);
    if (search == _entries.end()) {
        throw std::runtime_error(
            std::format("File '{}' not found in '{}'", name, _file_name)
        );
    }
    return search->second;
}

std::size_t TarReader::size(const std::string& name) const { return entry(name).size; }

void TarReader::read(const std::string& name, char* data) {
    const Entry& e = entry(name);
    _stream.clear();
    _stream.seekg(static_cast<std::streamoff>(e.offset));
    _stream.read(data, static_cast<std::streamsize>(e.size));
    if (!_stream) {
        throw std::runtime_error(
            std::format("Failed to read '{}' from '{}'", name, _file_name)
        );
    }
}

std::vector<char> TarReader::read(const std::string& name) {
    std::vector<char> data(size(name));
    read(name, data.data());
    return data;
}
