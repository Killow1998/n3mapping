#include "graph_solver_execution.h"
#include "n3mapping/async_log.h"
#include <algorithm>
#include <cstring>
#include <link.h>
#include <string>

namespace n3mapping::internal {
namespace {
struct LoadedGraphLibrary {
    int count = 0;
    std::string path, build_id;
};

int inspectGraphLibrary(dl_phdr_info* info, size_t, void* context) {
    const char* name = std::strrchr(info->dlpi_name, '/');
    name = name ? name + 1 : info->dlpi_name;
    if (std::strncmp(name, "libgtsam.so", 11) != 0) return 0;
    auto& loaded = *static_cast<LoadedGraphLibrary*>(context);
    ++loaded.count;
    loaded.path = info->dlpi_name;
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const auto& segment = info->dlpi_phdr[i];
        if (segment.p_type != PT_NOTE) continue;
        bool readable = false;
        for (int j = 0; j < info->dlpi_phnum; ++j) {
            const auto& memory = info->dlpi_phdr[j];
            if (memory.p_type == PT_LOAD && (memory.p_flags & PF_R) &&
                segment.p_vaddr >= memory.p_vaddr &&
                segment.p_vaddr - memory.p_vaddr <= memory.p_memsz &&
                segment.p_memsz <= memory.p_memsz - (segment.p_vaddr - memory.p_vaddr))
                readable = true;
        }
        if (!readable) continue;
        auto cursor = reinterpret_cast<const unsigned char*>(info->dlpi_addr + segment.p_vaddr);
        const auto end = cursor + segment.p_memsz;
        while (static_cast<size_t>(end - cursor) >= sizeof(ElfW(Nhdr))) {
            ElfW(Nhdr) note;
            std::memcpy(&note, cursor, sizeof(note));
            cursor += sizeof(note);
            const size_t name_bytes = (size_t(note.n_namesz) + 3) & ~size_t(3);
            const size_t data_bytes = (size_t(note.n_descsz) + 3) & ~size_t(3);
            if (name_bytes > size_t(end - cursor) ||
                data_bytes > size_t(end - cursor) - name_bytes) break;
            if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                std::memcmp(cursor, "GNU", 4) == 0) {
                static constexpr char hex[] = "0123456789abcdef";
                loaded.build_id.clear();
                for (size_t k = 0; k < note.n_descsz; ++k) {
                    const auto byte = cursor[name_bytes + k];
                    loaded.build_id += hex[byte >> 4];
                    loaded.build_id += hex[byte & 15];
                }
            }
            cursor += name_bytes + data_bytes;
        }
    }
    return 0;
}

bool qualifiedGraphLibrary() {
    LoadedGraphLibrary loaded;
    dl_iterate_phdr(inspectGraphLibrary, &loaded);
    const bool qualified = N3MAPPING_GTSAM_LOCKED_ELIMINATION && loaded.count == 1 &&
        !loaded.build_id.empty() && loaded.build_id == N3MAPPING_GTSAM_BUILD_ID;
    logEvent("graph_solver_dependency", std::string("policy=") +
        (qualified ? "configured_threads" : "serial_containment") +
        " libraries=" + std::to_string(loaded.count) + " path=" + loaded.path +
        " loaded_build_id=" + loaded.build_id + " expected_build_id=" + N3MAPPING_GTSAM_BUILD_ID,
        !qualified && N3MAPPING_GTSAM_LOCKED_ELIMINATION);
    return qualified;
}
} // namespace

int graphSolveConcurrency(int num_threads) {
    static const bool qualified = qualifiedGraphLibrary();
    return qualified ? std::max(1, num_threads) : 1;
}
} // namespace n3mapping::internal
