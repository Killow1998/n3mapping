// Export the native non-degraded dense trajectory from a pbstream as exact-ns CSV.
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <openssl/evp.h>

#include "n3mapping/dense_trajectory_export.h"

namespace {

struct Options {
    std::filesystem::path pbstream;
    std::filesystem::path output;
    std::filesystem::path evidence;
};

void printUsage(std::ostream& stream, const char* argv0) {
    stream
        << "Usage: " << argv0
        << " --pbstream MAP.pbstream --output trajectory.csv"
           " --evidence trajectory.evidence.json\n"
        << "\n"
        << "Reads through readN3NavResource(), requires source=native and "
           "degraded=false, and writes:\n"
        << "  stamp_ns,tx,ty,tz,qx,qy,qz,qw,seq\n";
}

bool parseOptions(int argc, char** argv, Options* options, std::string* error) {
    if (!options) return false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            printUsage(std::cout, argv[0]);
            return false;
        }
        if (i + 1 >= argc) {
            if (error) *error = "missing value for " + argument;
            return false;
        }
        const std::string value(argv[++i]);
        if (argument == "--pbstream") {
            options->pbstream = value;
        } else if (argument == "--output") {
            options->output = value;
        } else if (argument == "--evidence") {
            options->evidence = value;
        } else {
            if (error) *error = "unknown argument: " + argument;
            return false;
        }
    }
    if (options->pbstream.empty() || options->output.empty() ||
        options->evidence.empty()) {
        if (error) {
            *error = "--pbstream, --output, and --evidence are required";
        }
        return false;
    }
    return true;
}

bool validateOutputPaths(const Options& options, std::string* error) {
    if (std::filesystem::absolute(options.output) ==
        std::filesystem::absolute(options.evidence)) {
        if (error) *error = "--output and --evidence must be different files";
        return false;
    }
    for (const auto& path : {options.output, options.evidence}) {
        if (std::filesystem::exists(path)) {
            if (error) *error = "refusing to overwrite existing output: " + path.string();
            return false;
        }
        const auto parent = path.parent_path();
        if (!parent.empty() && !std::filesystem::is_directory(parent)) {
            if (error) *error = "output parent directory does not exist: " + parent.string();
            return false;
        }
    }
    return true;
}

bool writeCsv(const std::filesystem::path& output,
              const std::vector<n3mapping::tools::DenseTrajectoryCsvRow>& rows,
              std::string* error) {
    if (std::filesystem::exists(output)) {
        if (error) *error = "refusing to overwrite existing output: " + output.string();
        return false;
    }
    const auto parent = output.parent_path();
    if (!parent.empty() && !std::filesystem::is_directory(parent)) {
        if (error) *error = "output parent directory does not exist: " + parent.string();
        return false;
    }

    std::ofstream stream(output);
    if (!stream.is_open()) {
        if (error) *error = "failed to open output: " + output.string();
        return false;
    }
    stream << "stamp_ns,tx,ty,tz,qx,qy,qz,qw,seq\n";
    stream << std::setprecision(17);
    for (const auto& row : rows) {
        stream << row.stamp_ns << ','
               << row.translation.x() << ','
               << row.translation.y() << ','
               << row.translation.z() << ','
               << row.orientation.x() << ','
               << row.orientation.y() << ','
               << row.orientation.z() << ','
               << row.orientation.w() << ','
               << row.seq << '\n';
    }
    stream.flush();
    if (!stream.good()) {
        if (error) *error = "failed while writing output: " + output.string();
        return false;
    }
    return true;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream stream;
    for (const unsigned char byte : value) {
        switch (byte) {
        case '\\': stream << "\\\\"; break;
        case '"': stream << "\\\""; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (byte < 0x20) {
                stream << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(byte)
                       << std::dec << std::setfill(' ');
            } else {
                stream << byte;
            }
        }
    }
    return stream.str();
}

bool sha256File(const std::filesystem::path& path,
                std::string* digest_hex,
                std::string* error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        if (error) *error = "failed to open for SHA-256: " + path.string();
        return false;
    }
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) {
        if (error) *error = "failed to allocate SHA-256 context";
        return false;
    }
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
    std::array<char, 4 * 1024 * 1024> buffer{};
    while (ok && stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            ok = EVP_DigestUpdate(
                     context, buffer.data(), static_cast<std::size_t>(count)) ==
                 1;
        }
    }
    if (!stream.eof()) ok = false;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (ok) {
        ok = EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    }
    EVP_MD_CTX_free(context);
    if (!ok) {
        if (error) *error = "failed to compute SHA-256: " + path.string();
        return false;
    }
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        hex << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    *digest_hex = hex.str();
    return true;
}

bool writeEvidence(const Options& options,
                   std::size_t row_count,
                   std::string* error) {
    if (std::filesystem::exists(options.evidence)) {
        if (error) {
            *error = "refusing to overwrite existing evidence: " +
                     options.evidence.string();
        }
        return false;
    }
    const auto parent = options.evidence.parent_path();
    if (!parent.empty() && !std::filesystem::is_directory(parent)) {
        if (error) {
            *error = "evidence parent directory does not exist: " +
                     parent.string();
        }
        return false;
    }
    std::string pbstream_sha256;
    std::string csv_sha256;
    if (!sha256File(options.pbstream, &pbstream_sha256, error) ||
        !sha256File(options.output, &csv_sha256, error)) {
        return false;
    }
    std::ofstream stream(options.evidence);
    if (!stream.is_open()) {
        if (error) {
            *error = "failed to open evidence: " + options.evidence.string();
        }
        return false;
    }
    stream
        << "{\n"
        << "  \"schema\": \"n3mapping_dense_trajectory_evidence_v1\",\n"
        << "  \"pbstream\": \""
        << jsonEscape(std::filesystem::absolute(options.pbstream).string())
        << "\",\n"
        << "  \"pbstream_sha256\": \"" << pbstream_sha256 << "\",\n"
        << "  \"trajectory_csv\": \""
        << jsonEscape(std::filesystem::absolute(options.output).string())
        << "\",\n"
        << "  \"trajectory_csv_sha256\": \"" << csv_sha256 << "\",\n"
        << "  \"row_count\": " << row_count << ",\n"
        << "  \"source\": \"native\",\n"
        << "  \"degraded\": false,\n"
        << "  \"pose_convention\": \"T_world_body\",\n"
        << "  \"timestamp_encoding\": "
           "\"llround(pbstream_double_seconds*1e9)\"\n"
        << "}\n";
    stream.flush();
    if (!stream.good()) {
        if (error) {
            *error = "failed while writing evidence: " +
                     options.evidence.string();
        }
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    if (!parseOptions(argc, argv, &options, &error)) {
        if (!error.empty()) {
            std::cerr << "ERROR: " << error << '\n';
            printUsage(std::cerr, argv[0]);
            return 2;
        }
        return 0;
    }
    if (!std::filesystem::is_regular_file(options.pbstream)) {
        std::cerr << "ERROR: pbstream does not exist: " << options.pbstream << '\n';
        return 1;
    }
    if (!validateOutputPaths(options, &error)) {
        std::cerr << "ERROR: " << error << '\n';
        return 1;
    }

    n3mapping::N3NavResource resource;
    if (!n3mapping::readN3NavResource(options.pbstream.string(), &resource, &error)) {
        std::cerr << "ERROR: failed to read pbstream: " << error << '\n';
        return 1;
    }
    std::vector<n3mapping::tools::DenseTrajectoryCsvRow> rows;
    if (!n3mapping::tools::prepareNativeDenseTrajectoryCsvRows(resource, &rows, &error)) {
        std::cerr << "ERROR: " << error << '\n';
        return 1;
    }
    if (!writeCsv(options.output, rows, &error)) {
        std::cerr << "ERROR: " << error << '\n';
        return 1;
    }
    if (!writeEvidence(options, rows.size(), &error)) {
        std::error_code ignored;
        std::filesystem::remove(options.output, ignored);
        std::filesystem::remove(options.evidence, ignored);
        std::cerr << "ERROR: " << error << '\n';
        return 1;
    }
    std::cout << "exported " << rows.size() << " native dense poses to "
              << options.output << '\n';
    return 0;
}
