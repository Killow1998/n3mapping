#pragma once

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace n3mapping::eval {

inline std::vector<std::string> splitSimpleCsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

inline std::set<std::string> loadEpisodeFrameTokens(const std::filesystem::path& path,
                                                    const std::string& episode_id,
                                                    const std::string& role)
{
    if (path.empty()) return {};
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open episode frame manifest: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("episode frame manifest is empty: " + path.string());
    }
    const auto header = splitSimpleCsv(line);
    if (header.size() < 3 || header[0] != "episode_id" || header[1] != "role" ||
        header[2] != "frame_token") {
        throw std::runtime_error(
            "episode frame manifest must begin with episode_id,role,frame_token");
    }
    std::set<std::string> tokens;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = splitSimpleCsv(line);
        if (fields.size() < 3) {
            throw std::runtime_error("malformed episode frame manifest line " +
                                     std::to_string(line_number));
        }
        if (fields[0] != episode_id || fields[1] != role) continue;
        if (fields[2].empty() || !tokens.insert(fields[2]).second) {
            throw std::runtime_error("empty or duplicate frame token at manifest line " +
                                     std::to_string(line_number));
        }
    }
    if (tokens.empty()) {
        throw std::runtime_error("episode manifest has no frames for episode=" + episode_id +
                                 " role=" + role);
    }
    return tokens;
}

}  // namespace n3mapping::eval
