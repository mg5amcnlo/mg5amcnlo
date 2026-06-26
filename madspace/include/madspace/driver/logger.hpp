#pragma once

#include <functional>
#include <string>

#include "madspace/util.hpp"

namespace madspace {

enum class Verbosity { silent, log, pretty };

class Logger {
public:
    enum LogLevel { level_debug, level_info, level_warning, level_error };
    using LogHandlerFunc =
        std::function<void(LogLevel level, const std::string& message)>;

    static void log(LogLevel level, const std::string& message) {
        if (_log_handler) {
            _log_handler.value()(level, message);
            return;
        }
        switch (level) {
        case level_debug:
            println("[DEBUG] {}", message);
            break;
        case level_info:
            println("[INFO] {}", message);
            break;
        case level_warning:
            println("[WARNING] {}", message);
            break;
        case level_error:
            println("[ERROR] {}", message);
            break;
        }
    }
    static void debug(const std::string& message) { log(level_debug, message); }
    static void info(const std::string& message) { log(level_info, message); }
    static void warning(const std::string& message) { log(level_warning, message); }
    static void error(const std::string& message) { log(level_error, message); }
    static void set_log_handler(LogHandlerFunc func) { _log_handler = func; }

private:
    static inline std::optional<LogHandlerFunc> _log_handler = std::nullopt;
};

} // namespace madspace
