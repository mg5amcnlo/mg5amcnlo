#include "madtrex/param_handler.hpp"

#include <fstream>
#include <mutex>
#include <stdexcept>

namespace madtrex {

ParamHandler& ParamHandler::add_api(const madspace::MatrixElementApi& api) {
    if (!api.supports_set_parameter()) {
        throw std::runtime_error(
            "ParamHandler::add_api: matrix element " + api.file_name() +
            " does not implement umami_set_parameter; use REX::tea::param_rwgt's "
            "file-based parameter reweighting for this library instead"
        );
    }
    for (auto* existing : _apis) {
        if (existing == &api) return *this;
    }
    _apis.push_back(&api);
    return *this;
}

ParamHandler& ParamHandler::add_apis(const std::vector<const madspace::MatrixElementApi*>& apis) {
    for (auto* api : apis) add_api(*api);
    return *this;
}

ParamHandler& ParamHandler::add_apis(const WareHouse& house) {
    for (auto& tw : house.processes()) add_apis(tw.apis());
    return *this;
}

ParamHandler& ParamHandler::read_rwgt_card(std::istream& rwgt_in) {
    _card_iter.parse_rwgt_card(rwgt_in);
    return *this;
}

ParamHandler& ParamHandler::read_rwgt_card(const std::string& rwgt_path) {
    std::ifstream rwgt_in(rwgt_path);
    if (!rwgt_in) {
        throw std::runtime_error("ParamHandler::read_rwgt_card: failed to open " + rwgt_path);
    }
    return read_rwgt_card(rwgt_in);
}

std::size_t ParamHandler::launch_count() const { return _card_iter.cards.size(); }

bool ParamHandler::apply_launch(std::size_t idx) {
    if (idx >= _card_iter.cards.size()) {
        throw std::out_of_range("ParamHandler::apply_launch: launch index out of range");
    }
    if (_apis.empty()) {
        throw std::runtime_error("ParamHandler::apply_launch: no matrix element apis registered");
    }

    for (auto& [api, values] : _nominal) {
        std::lock_guard<std::mutex> guard(detail::api_call_mutex(*api));
        for (auto& [location, nominal] : values) {
            detail::set_api_parameter(*api, location, nominal.first, nominal.second);
        }
    }

    auto set_everywhere = [this](const std::string& location, double value) {
        for (auto* api : _apis) {
            std::lock_guard<std::mutex> guard(detail::api_call_mutex(*api));
            auto& cache = _nominal[api];
            if (cache.find(location) == cache.end()) {
                cache[location] = api->get_parameter(location);
            }
            detail::set_api_parameter(*api, location, value, 0.0);
        }
    };
    for (auto& [block_name, block] : _card_iter.cards[idx].blocks) {
        for (auto [param_id, value] : block.params) {
            set_everywhere(block_name + " " + std::to_string(param_id), value);
        }
        // multi-index entries: the UMAMI location is "BLOCK I J ..."
        for (const auto& [param_ids, value] : block.multi_params) {
            std::string location = block_name;
            for (int param_id : param_ids) location += " " + std::to_string(param_id);
            set_everywhere(location, value);
        }
    }
    return true;
}

std::vector<REX::tea::iterator> ParamHandler::iterators() {
    std::vector<REX::tea::iterator> result;
    result.reserve(_card_iter.cards.size());
    for (std::size_t i = 0; i < _card_iter.cards.size(); ++i) {
        result.push_back([this, i]() { return this->apply_launch(i); });
    }
    return result;
}

std::vector<std::string> ParamHandler::launch_names() { return _card_iter.get_launch_names(); }

std::vector<std::string> ParamHandler::weight_context() {
    return _card_iter.get_rwgt_commands();
}

} // namespace madtrex
