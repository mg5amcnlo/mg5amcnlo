#pragma once

#include <functional>
#include <iostream>
#include <string>

#include "madspace/util.hpp"

namespace madspace {

/// Output level for progress reporting during event generation.
///
/// `silent` prints nothing, `log` prints one line per progress update, and
/// `pretty` redraws a live status box in place; see @ref PrettyBox.
enum class Verbosity { silent, log, pretty };

/**
 * Process-wide logging facility with a pluggable handler.
 *
 * A thin wrapper around four severity levels that by default prints to
 * `stdout`. Call @ref set_log_handler to redirect messages elsewhere, for
 * example into a host application's own logging system; @ref clear_log_handler
 * restores the default printer.
 */
class Logger {
public:
    /// Severity of a logged message.
    enum LogLevel { level_debug, level_info, level_warning, level_error };
    /// Signature of a custom log handler; see @ref set_log_handler.
    using LogHandlerFunc =
        std::function<void(LogLevel level, const std::string& message)>;

    /// Dispatch @p message at @p level to the current handler, or print it.
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
        std::cout << std::flush;
    }
    /// Log @p message at `level_debug`.
    static void debug(const std::string& message) { log(level_debug, message); }
    /// Log @p message at `level_info`.
    static void info(const std::string& message) { log(level_info, message); }
    /// Log @p message at `level_warning`.
    static void warning(const std::string& message) { log(level_warning, message); }
    /// Log @p message at `level_error`.
    static void error(const std::string& message) { log(level_error, message); }
    /// Route every future @ref log call through @p func instead of printing.
    static void set_log_handler(LogHandlerFunc func) { _log_handler = func; }
    /// Remove a handler set with @ref set_log_handler and resume printing.
    static void clear_log_handler() { _log_handler = std::nullopt; }

private:
    static inline std::optional<LogHandlerFunc> _log_handler = std::nullopt;
};

} // namespace madspace
