#include "madspace/driver/format.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <span>
#include <sstream>
#include <sys/resource.h>

using namespace madspace;

namespace {

const std::array<std::string, 5> si_prefixes{"", "k", "M", "G", "T"};
const std::array<std::string, 9> progress_symbols{
    " ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"
};

} // namespace

std::size_t madspace::cpu_time_microsec() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return 1000000 * (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
        usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}

std::string madspace::format_run_time(double wall_time_sec, double cpu_time_sec) {
    using namespace std::chrono_literals;
    std::chrono::duration<double> cpu_duration(cpu_time_sec),
        wall_duration(wall_time_sec);
    // we don't use the ratio feature of duration here because it seems to lead
    // to errors in old gcc versions
    double cpu_centisec = std::fmod(cpu_time_sec / 0.01, 100.);
    double wall_centisec = std::fmod(wall_time_sec / 0.01, 100.);
    return std::format(
        "{:%H:%M:%S}.{:02.0f} wall, {:%H:%M:%S}.{:02.0f} cpu",
        wall_duration,
        wall_centisec,
        cpu_duration,
        cpu_centisec
    );
}

std::string madspace::format_si_prefix(double value) {
    value = std::round(value);
    int value_power = std::floor(std::log10(value));
    int value_power3 = value_power / 3;

    if (value_power3 >= 0 && value_power3 < si_prefixes.size()) {
        int digits_after_dot = value_power3 == 0 ? 0 : 2 - value_power % 3;
        double value_scaled = value / std::pow(10, 3 * value_power3);
        return std::format(
            "{:.{}f}{}", value_scaled, digits_after_dot, si_prefixes.at(value_power3)
        );
    } else {
        return std::format("{}", value);
    }
}

std::string madspace::format_with_error(double value, double error) {
    int value_power = std::floor(std::log10(value));
    int sig_power = std::isnan(error) || error <= 0.
        ? 3 - value_power
        : 1 - static_cast<int>(std::floor(std::log10(error)));
    if (sig_power < 0 || sig_power > 5) {
        std::string exp_fmt = std::format("{:.{}e}", value, value_power + sig_power);
        auto e_pos = exp_fmt.find("e");
        double err_val = error * std::pow(10, sig_power);
        return std::format(
            "{}({:.0f})e{}",
            exp_fmt.substr(0, e_pos),
            err_val,
            exp_fmt.substr(e_pos + 1)
        );
    } else {
        int err_prec = sig_power == 1;
        double err_val = error * std::pow(10, sig_power - err_prec);
        return std::format("{:.{}f}({:.{}f})", value, sig_power, err_val, err_prec);
    }
}

std::string madspace::format_progress(double progress, int width) {
    double frac = width * std::min(1.0, std::max(0.0, progress));
    int n_full = frac;
    std::stringstream str;
    for (int i = 0; i < n_full; ++i) {
        str << progress_symbols.back();
    }
    int n_remaining;
    if (n_full >= width) {
        n_remaining = 0;
    } else {
        str << progress_symbols.at(
            static_cast<int>((frac - n_full) * progress_symbols.size())
        );
        n_remaining = width - n_full - 1;
    }
    for (int i = 0; i < n_remaining; ++i) {
        str << progress_symbols.front();
    }
    return str.str();
}

PrettyBox::PrettyBox(
    const std::string& title,
    std::size_t rows,
    const std::vector<std::size_t>& column_sizes,
    std::size_t offset,
    std::size_t box_width
) :
    _rows(rows),
    _columns(column_sizes.size()),
    _offset(offset),
    _content(rows * column_sizes.size()) {

    std::size_t column = 3;
    _column_ends.reserve(column_sizes.size());
    for (const std::size_t& size : column_sizes) {
        column += size;
        if (&size == &column_sizes.back() && column < box_width) {
            _column_ends.push_back(box_width);
        } else {
            _column_ends.push_back(column);
        }
    }
    box_width = _column_ends.at(_column_ends.size() - 1);

    _header = std::format("┌ {} ", title);
    for (std::size_t i = title.size() + 3; i < box_width - 1; ++i) {
        _header += "─";
    }
    _header += "┐";

    _footer = "└";
    for (std::size_t i = 1; i < box_width - 1; ++i) {
        _footer += "─";
    }
    _footer += "┘";
}

void PrettyBox::print_first() const {
    std::cout << _header << "\n";
    for (std::size_t row = 0; row < _rows; ++row) {
        std::cout << "│ ";
        for (std::size_t column = 0; column < _columns; ++column) {
            std::cout << std::format(
                "{}\033[{}G",
                _content.at(row * _columns + column),
                _column_ends.at(column)
            );
        }
        std::cout << "│\n";
    }
    std::cout << _footer << "\n\n" << std::flush;
}

void PrettyBox::print_update() const {
    // save cursor position, go up {} lines, print header
    std::cout << std::format("\0337\033[{}F\033[2K{}\n", _rows + _offset + 3, _header);
    for (std::size_t row = 0; row < _rows; ++row) {
        std::cout << "\033[2K│ ";
        for (std::size_t column = 0; column < _columns; ++column) {
            std::cout << std::format(
                "{}\033[{}G",
                _content.at(row * _columns + column),
                _column_ends.at(column)
            );
        }
        std::cout << "│\n";
    }

    // restore cursor position
    std::cout << "\033[2K" << _footer << "\0338" << std::flush;
}
