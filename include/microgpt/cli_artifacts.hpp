#pragma once

#include "microgpt/cli_shared.hpp"

#include <cstdlib>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace microgpt {

inline std::string artifact_root_arg(const std::vector<std::string>& args) {
  return get_arg(args, "--root", "artifacts");
}

inline int run_list_artifacts_command(const std::vector<std::string>& args, std::ostream& out) {
  namespace fs = std::filesystem;
  fs::path root = artifact_root_arg(args);
  if (!fs::exists(root)) {
    out << "artifact_root " << root.string() << " missing\n";
    return 0;
  }
  out << "artifact_root " << root.string() << '\n';
  for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    fs::path p = entry.path();
    if (p.filename() == ".gitkeep") {
      continue;
    }
    out << p.string() << " bytes " << entry.file_size() << '\n';
  }
  return 0;
}

inline int run_clean_artifacts_command(const std::vector<std::string>& args, std::ostream& out) {
  namespace fs = std::filesystem;
  fs::path root = artifact_root_arg(args);
  bool force = has_arg(args, "--yes");
  if (!fs::exists(root)) {
    out << "artifact_root " << root.string() << " missing\n";
    return 0;
  }
  if (!force) {
    out << "dry_run yes\n";
    out << "pass --yes to delete generated artifact files\n";
  }
  int count = 0;
  for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    fs::path p = entry.path();
    if (p.filename() == ".gitkeep") {
      continue;
    }
    out << (force ? "delete " : "would_delete ") << p.string() << '\n';
    if (force) {
      fs::remove(p);
    }
    ++count;
  }
  out << "files " << count << '\n';
  return 0;
}

}  // namespace microgpt
