#pragma once

#include <chrono>
#include <string>

#include <nlohmann/json.hpp>

namespace madspace {

/**
 * Rate-limited JSON status file, merged from possibly multiple sources.
 *
 * Lets several parts of a run report progress into the same file without
 * flooding the disk: @ref write merges its argument
 * into the accumulated content and only actually rewrites the file at most
 * every @p min_interval_sec, unless forced.
 */
class StatusFile {
public:
    /**
     * @param file_name        Path the status is written to.
     * @param min_interval_sec Minimum time between two file rewrites.
     */
    explicit StatusFile(const std::string& file_name, double min_interval_sec = 10.0);

    /**
     * Merge @p content into the accumulated status.
     *
     * Top-level keys of @p content overwrite the corresponding accumulated
     * ones; the `"run_times"` key is merged one level deeper instead. The
     * file is rewritten if @p force_write is set, if this is the first call,
     * or if @p min_interval_sec has passed since the last rewrite.
     */
    void write(const nlohmann::json& content, bool force_write = false);

    /// If a status was written and its `"status"` field is not `"done"`,
    /// sets it to `"done"` and writes the file out one last time.
    ~StatusFile();

private:
    std::string _file_name;
    std::chrono::duration<double> _min_interval;
    std::chrono::time_point<std::chrono::steady_clock> _last_write_time;
    bool _has_written = false;
    nlohmann::json _content = nlohmann::json::object();
};

} // namespace madspace
