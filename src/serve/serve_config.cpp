#include "strix/serve/serve_config.hpp"

#include "strix/model_description.hpp"

#include <array>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::serve::config_file {
namespace {

namespace md = strix::model_description;
namespace fs = std::filesystem;

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

// Line of a (dotted) key: the parser validated the syntax already, so a key line is "name = ..." inside its table.
std::size_t line_of(std::string_view text, std::string_view dotted) {
    std::string table;
    std::size_t number = 0;
    for (std::size_t at = 0; at <= text.size();) {
        auto end = text.find('\n', at);
        if (end == std::string_view::npos) end = text.size();
        ++number;
        auto line = trim(text.substr(at, end - at));
        at = end + 1U;
        if (line.empty() || line.front() == '#') continue;
        if (line.front() == '[') {
            const auto close = line.find(']');
            table = std::string(trim(line.substr(1, close == std::string_view::npos ? std::string_view::npos : close - 1)));
            continue;
        }
        const auto equal = line.find('=');
        if (equal == std::string_view::npos) continue;
        std::string key(trim(line.substr(0, equal)));
        std::string full = table.empty() ? key : table + "." + key;
        std::erase(full, ' ');
        if (full == dotted) return number;
        if (end >= text.size()) break;
    }
    return 0;
}

void flatten(const md::Value::Table& table, const std::string& prefix, std::vector<std::pair<std::string, const md::Value*>>& out) {
    for (const auto& [key, value] : table) {
        const auto full = prefix.empty() ? key : prefix + "." + key;
        if (const auto* nested = std::get_if<md::Value::Table>(&value.data)) flatten(*nested, full, out);
        else out.emplace_back(full, &value);
    }
}

std::string scalar(const md::Value& v, bool& ok) {
    ok = true;
    if (const auto* s = std::get_if<std::string>(&v.data)) return *s;
    if (const auto* i = std::get_if<std::int64_t>(&v.data)) return std::to_string(*i);
    if (const auto* d = std::get_if<double>(&v.data)) {
        std::array<char, 64> buffer{};
        const auto r = std::to_chars(buffer.data(), buffer.data() + buffer.size(), *d);
        return std::string(buffer.data(), r.ptr);
    }
    ok = false;
    return {};
}

bool looks_secret(std::string_view key) {
    const auto dot = key.rfind('.');
    const auto last = dot == std::string_view::npos ? key : key.substr(dot + 1U);
    if (last.ends_with("_file") || last.ends_with("_path")) return false;
    for (const std::string_view word : {"key", "token", "secret", "password"})
        if (last == word || last.ends_with("_" + std::string(word))) return true;
    return false;
}

bool cli_has(const std::vector<std::string>& args, std::string_view flag) {
    for (const auto& a : args) {
        if (a == "--") break;
        if (a == flag || (a.size() > flag.size() && std::string_view(a).starts_with(flag) && a[flag.size()] == '=')) return true;
    }
    return false;
}

std::string toml_string(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) { if (c == '"' || c == '\\') out.push_back('\\'); out.push_back(c); }
    return out + "\"";
}

std::string read_trusted(const std::string& path) {
#if !defined(_WIN32)
    // Keep the validated file open: replacing the pathname must not change the bytes read.
    // NONBLOCK lets fstat reject a FIFO without waiting for a writer.
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) throw std::invalid_argument(path + ": cannot open serve config");
    struct Close { int descriptor; ~Close() { ::close(descriptor); } } close_fd{fd};
    struct stat st{};
    if (::fstat(fd, &st) != 0) throw std::invalid_argument(path + ": cannot stat serve config");
    if (!S_ISREG(st.st_mode)) throw std::invalid_argument(path + ": serve config must be a regular file");
    if (st.st_mode & (S_IWGRP | S_IWOTH)) throw std::invalid_argument(path + ": serve config is writable by other users; refusing it");
    if (st.st_uid != ::geteuid() && st.st_uid != 0) throw std::invalid_argument(path + ": serve config is owned by another user; refusing it");
    std::string text;
    std::array<char, 8192> buffer{};
    for (;;) {
        const auto n = ::read(fd, buffer.data(), buffer.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::invalid_argument(path + ": cannot read serve config");
        }
        if (n == 0) return text;
        text.append(buffer.data(), static_cast<std::size_t>(n));
    }
#else
    // Windows: the file ACL is the operator's responsibility (documented).
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument(path + ": cannot open serve config");
    std::ostringstream text;
    text << input.rdbuf();
    if (input.bad()) throw std::invalid_argument(path + ": cannot read serve config");
    return text.str();
#endif
}

}  // namespace

Expanded expand_text(std::string_view text, const std::string& file_path, const std::vector<std::string>& args, std::span<const Option> table) {
    // args: the command line without --serve-config / --print-serve-config; args[0] is the program.
    Expanded out;
    out.used = true;
    std::size_t lead = args.empty() ? 0U : 1U;
    while (lead < args.size() && !args[lead].starts_with("-")) ++lead;
    const std::vector<std::string> cli(args.begin() + static_cast<std::ptrdiff_t>(lead), args.end());
    md::ModelDescription parsed;
    try { parsed = md::parse(text); }
    catch (const std::invalid_argument& e) { throw std::invalid_argument(file_path + ": " + e.what()); }
    std::vector<std::pair<std::string, const md::Value*>> entries;
    flatten(parsed.fields, {}, entries);
    std::map<std::string_view, const Option*, std::less<>> by_key;
    for (const auto& option : table) by_key.emplace(option.key, &option);
    const fs::path base = fs::absolute(file_path).parent_path();
    std::vector<std::string> from_file;
    std::ostringstream effective;
    effective << "# effective serve options (file " << file_path << " + command line; the command line wins)\n";
    std::set<std::string_view> printed;
    for (const auto& [key, value] : entries) {
        const auto where = file_path + ":" + std::to_string(line_of(text, key));
        const auto fail = [&](const std::string& why) { throw std::invalid_argument(where + ": " + key + ": " + why); };
        const auto it = by_key.find(key);
        if (looks_secret(key) && (it == by_key.end() || it->second->kind == Kind::secret))
            fail("inline credentials are refused; put the secret in a file and set the *_file option");
        if (it == by_key.end()) {
            std::string known;
            for (const auto& option : table) known += (known.empty() ? "" : ", ") + std::string(option.key);
            fail("unknown serve-config key (known: " + known + ")");
        }
        const auto& option = *it->second;
        if (!option.conflicts.empty() && cli_has(cli, option.conflicts))
            fail("also set by " + std::string(option.conflicts) + " on the command line; keep one source for these settings");
        std::vector<std::string> values;
        bool ok = true;
        switch (option.kind) {
        case Kind::flag:
        case Kind::on_off: {
            const auto* b = std::get_if<bool>(&value->data);
            if (!b) fail("expects true or false");
            if (option.kind == Kind::on_off) values.emplace_back(*b ? "on" : "off");
            else if (*b) values.emplace_back();
            break;
        }
        case Kind::value: values.push_back(scalar(*value, ok)); if (!ok) fail("expects a string or a number"); break;
        case Kind::path: {
            const auto* s = std::get_if<std::string>(&value->data);
            if (!s || s->empty()) fail("expects a non-empty path string");
            fs::path p(*s);
            values.push_back((p.is_absolute() || base.empty() ? p : (base / p)).lexically_normal().generic_string());
            break;
        }
        case Kind::list: {
            const auto* array = std::get_if<md::Value::Array>(&value->data);
            if (!array) fail("expects an array");
            std::string joined;
            for (const auto& item : *array) {
                auto s = scalar(item, ok);
                if (!ok) fail("expects an array of strings or numbers");
                if (option.join) joined += (joined.empty() ? "" : std::string(1, option.join)) + s;
                else values.push_back(std::move(s));
            }
            if (option.join && !joined.empty()) values.push_back(std::move(joined));
            break;
        }
        case Kind::secret: fail("inline credentials are refused; use the *_file option");
        }
        if (cli_has(cli, option.flag)) continue;  // the command line wins (a list replaces the file's whole list)
        printed.insert(option.key);
        effective << option.key << " = ";
        if (option.kind == Kind::flag) effective << (values.empty() ? "false" : "true");
        else if (option.kind == Kind::list && !option.join) {
            effective << "[";
            for (std::size_t i = 0; i < values.size(); ++i) effective << (i ? ", " : "") << toml_string(values[i]);
            effective << "]";
        } else effective << (values.empty() ? "\"\"" : toml_string(values.front()));
        effective << "  # " << where << "\n";
        for (auto& v : values) {
            from_file.emplace_back(option.flag);
            if (option.kind != Kind::flag) from_file.push_back(std::move(v));
        }
    }
    for (std::size_t i = 0; i < cli.size(); ++i) {
        if (cli[i] == "--") break;
        for (const auto& option : table) {
            const std::string_view arg = cli[i];
            const bool exact = arg == option.flag, with_value = arg.starts_with(option.flag) && arg.size() > option.flag.size() && arg[option.flag.size()] == '=';
            if (!exact && !with_value) continue;
            std::string v = with_value ? std::string(arg.substr(option.flag.size() + 1U))
                : option.kind == Kind::flag ? std::string("true") : (i + 1U < cli.size() ? cli[i + 1U] : std::string());
            if (option.kind == Kind::secret) v = "<redacted>";
            effective << option.key << " = " << (option.kind == Kind::flag ? v : toml_string(v)) << "  # command line\n";
        }
    }
    out.argv.assign(args.begin(), args.begin() + static_cast<std::ptrdiff_t>(lead));
    out.argv.insert(out.argv.end(), from_file.begin(), from_file.end());
    out.argv.insert(out.argv.end(), cli.begin(), cli.end());
    out.effective = effective.str();
    return out;
}

Expanded expand(int argc, const char* const* argv, std::span<const Option> table) {
    std::vector<std::string> args;
    std::string file;
    bool print = false, seen = false;
    for (int i = 0; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (i > 0 && a == "--") {
            for (; i < argc; ++i) args.emplace_back(argv[i]);
            break;
        }
        if (i > 0 && a == "--print-serve-config") { print = true; continue; }
        if (i > 0 && (a == "--serve-config" || a.starts_with("--serve-config="))) {
            if (seen) throw std::invalid_argument("--serve-config given twice");
            seen = true;
            if (a == "--serve-config") {
                if (i + 1 >= argc || std::string_view(argv[i + 1]).starts_with("--")) throw std::invalid_argument("--serve-config requires a file");
                file = argv[++i];
            } else file = std::string(a.substr(15));
            continue;
        }
        args.emplace_back(a);
    }
    if (!seen) {
        Expanded out;
        out.print = print;
        out.argv = std::move(args);
        if (print) out.effective = "# no --serve-config given; every option comes from the command line\n";
        return out;
    }
    if (file.empty()) throw std::invalid_argument("--serve-config requires a file");
    auto out = expand_text(read_trusted(file), file, args, table);
    out.print = print;
    return out;
}

}  // namespace strix::serve::config_file
