#include "madspace/driver/status_file.hpp"

#include <filesystem>
#include <format>
#include <fstream>

using namespace madspace;

StatusFile::StatusFile(const std::string& file_name, double min_interval_sec) :
    _file_name(file_name), _min_interval(min_interval_sec) {}

void StatusFile::write(const nlohmann::json& content, bool force_write) {
    for (auto& [key, value] : content.items()) {
        if (key == "run_times" && value.is_object()) {
            auto& run_times = _content["run_times"];
            if (!run_times.is_object()) {
                run_times = nlohmann::json::object();
            }
            run_times.update(value);
        } else {
            _content[key] = value;
        }
    }

    auto now = std::chrono::steady_clock::now();
    if (_has_written && !force_write && now - _last_write_time < _min_interval) {
        return;
    }
    _has_written = true;
    _last_write_time = now;

    std::string status_tmp_file = std::format("{}.tmp", _file_name);
    std::ofstream f(status_tmp_file);
    f << _content.dump();
    // rename atomically deletes the old file and replaces it with the new one
    // such that the status file exists at all times
    std::filesystem::rename(status_tmp_file, _file_name);
}

StatusFile::~StatusFile() {
    if (_has_written && _content.value("status", "") != "done") {
        write({{"status", "done"}}, true);
    }
}
