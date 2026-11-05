#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace madspace {

/// Writes a POSIX ustar archive, readable by any standard tar tool. Entries are
/// regular files, added one at a time. Names longer than 100 characters are
/// split over the ustar `prefix` and `name` fields; names that do not fit even
/// then are rejected. The archive is finished by @ref close or the destructor.
class TarWriter {
public:
    /// Create (or truncate) the archive at `file_name`.
    explicit TarWriter(const std::string& file_name);
    TarWriter(const TarWriter&) = delete;
    TarWriter& operator=(const TarWriter&) = delete;
    ~TarWriter();
    /// Append a file called `name` with the given contents.
    void add(const std::string& name, const char* data, std::size_t size);
    /// Write the end-of-archive marker and close the file.
    void close();

private:
    std::string _file_name;
    std::ofstream _stream;
    bool _closed;
};

/// Reads the regular files of a tar archive (ustar or GNU, as written by
/// standard tools). The archive is indexed when opened; entries are then read
/// by name. Entries other than regular files are skipped, and long names
/// stored in GNU or pax extension headers are not interpreted.
class TarReader {
public:
    /// Open and index the archive at `file_name`.
    explicit TarReader(const std::string& file_name);
    /// The names of all files in the archive, in sorted order.
    std::vector<std::string> names() const;
    /// Whether the archive contains a file called `name`.
    bool contains(const std::string& name) const;
    /// The size in bytes of the file called `name`.
    std::size_t size(const std::string& name) const;
    /// Read the file called `name` into `data`, which must hold `size(name)`
    /// bytes.
    void read(const std::string& name, char* data);
    /// The contents of the file called `name`.
    std::vector<char> read(const std::string& name);

private:
    struct Entry {
        std::uint64_t offset;
        std::uint64_t size;
    };
    const Entry& entry(const std::string& name) const;

    std::string _file_name;
    std::ifstream _stream;
    std::map<std::string, Entry> _entries;
};

} // namespace madspace
